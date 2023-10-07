import pinecone
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import tqdm
import hashlib
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
from llama_index import (
    VectorStoreIndex,
    SimpleWebPageReader,
    LLMPredictor,
    ServiceContext
)

from trulens_eval import TruLlama, Feedback, Tru, feedback
from trulens_eval.feedback import GroundTruthAgreement, Groundedness
from pathlib import Path

tru = Tru()

os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_ENVIRONMENT"] = ""


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

def load_image(file_path: str) -> torch.Tensor:
    """
    Load a DICOM dataset and return a tensor that can be used as an input to DINOv2.
    """
    if file_path.endswith('.dcm'):
      dataset = pydicom.dcmread(file_path)
      img_1_channel = dataset.pixel_array.astype(np.float16)
      img = np.stack([img_1_channel, img_1_channel, img_1_channel], axis=2)
    else:  # Assuming it's PNG or JPEG
      img = Image.open(file_path).convert("RGB")

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



def predict(img_path):
  embedding = compute_embedding(img_path)
  response = index.query(
    vector=embedding.flatten().tolist(),
    top_k=1,
    include_values=True
  )
  return response
