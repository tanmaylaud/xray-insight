import pinecone
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os
import tqdm
import openai
import hashlib
import io
# from llama_index.storage.storage_context import StorageContext
# from llama_index.vector_stores import PineconeVectorStore
# from llama_index.llms import OpenAI
# from llama_index import (
#     VectorStoreIndex,
#     SimpleWebPageReader,
#     LLMPredictor,
#     ServiceContext
# )

# from trulens_eval import TruLlama, Feedback, Tru, feedback
# from trulens_eval.feedback import GroundTruthAgreement, Groundedness
from pathlib import Path

#tru = Tru()


index_name = "medical-images"

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

index = pinecone.Index(index_name)
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

transform_image = T.Compose([T.ToTensor(),
                             T.Resize(224),
                             T.CenterCrop(224),
                             T.Normalize([0.5], [0.5])])

def compute_embedding(file) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """

    with torch.no_grad():
            embedding = dinov2_vits14(load_image(file).to(device))
            print(f"embedding shape before: {embedding.shape}")
            embeddings_numpy = np.array(embedding[0].cpu().numpy()).reshape(1, -1)
            padded_embedding = pad_embedding(embeddings_numpy)
            print(f"embedding shape after padding: {padded_embedding.shape}")

    return padded_embedding

def load_image(file) -> torch.Tensor:
    """
    Load a an image and return a tensor that can be used as an input to DINOv2.
    """
    # Assuming it's PNG or JPEG
    img = Image.open(file).convert("RGB")

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def pad_embedding(embedding: np.ndarray, target_dim: int = 512) -> np.ndarray:
    """
    Pad the given embedding with zeros to match the target dimension.
    """
    original_dim = embedding.shape[1]
    padding_dim = target_dim - original_dim
    
    if padding_dim > 0:
        padding = np.zeros((1, padding_dim))
        padded_embedding = np.hstack([embedding, padding])
    else:
        padded_embedding = embedding
    
    return padded_embedding



def add_embedding_to_index(id: str, embedding):
  single_vector = {
      'id': id,
      'values': embedding.flatten().tolist(),
      'metadata': {'modality': 'mri'}
  }
  upsert_response = index.upsert(vectors=[single_vector])
  print(f"Inserted {single_vector}")

def img_to_vector_db(img_path, index):
  embedding = compute_embedding(img_path)
  add_embedding_to_index(id=str(index), embedding=embedding)


def hash_file(image_path: str) -> str:
    """
    Hash the filename to create a unique ID.
    """
    filename = image_path.split("/")[-1]
    unique_id = hashlib.sha256(filename.encode()).hexdigest()
    return unique_id



def predict(file):
  embedding = compute_embedding(file)
  response = index.query(
    vector=embedding.flatten().tolist(),
    top_k=3,
    include_values=True,
    include_metadata=True
  )
  result =[ m["metadata"]["report"] for m in response["matches"]]
  result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You will get description of 3 radiology images which have been retrieved for a given query xray image. Provide a radiology report with two sections Findings and Impressions. Only consider diagnosis that are common between all the descriptions. Ignore outliers. Also consider the user's query text. We do not want the user to know that we have retrieve images. Directly respond to the user naturally with the radiology report."},
        {"role": "assistant", "content": "\n-".join(result)},
        {"role": "user", "content": "What is wrong in the xray?"},
    ]
    )
  return result['choices'][0]['message']['content']

# result = predict(img_path=img_path)
# print(f"ID: {result['matches'][0]['id']} | Similarity score: {round(result['matches'][0]['score'], 2)}")
# new_img
