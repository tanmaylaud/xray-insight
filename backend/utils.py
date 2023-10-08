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
from gradio_client import Client
from monitor import Monitor, monitoring
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
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


from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

tru = Tru()

import numpy as np

# Initialize provider class
openai_tl = OpenAI()

grounded = Groundedness(groundedness_provider=OpenAI())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai_tl.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai_tl.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

index_name = "medical-images"
client = Client("https://42976740ac53ddbe7d.gradio.live/")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

index = pinecone.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=index)
l_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = l_index.as_query_engine()

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])


dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

transform_image = T.Compose([T.ToTensor(),
                             T.Resize(224),
                             T.CenterCrop(224),
                             T.Normalize([0.5], [0.5])])

@Monitor.monitor
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

@Monitor.monitor
def load_image(file) -> torch.Tensor:
    """
    Load a an image and return a tensor that can be used as an input to DINOv2.
    """
    # Assuming it's PNG or JPEG
    img = Image.open(file).convert("RGB")

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img

@Monitor.monitor
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


@Monitor.monitor
def add_embedding_to_index(id: str, embedding):
  single_vector = {
      'id': id,
      'values': embedding.flatten().tolist(),
      'metadata': {'modality': 'mri'}
  }
  upsert_response = index.upsert(vectors=[single_vector])
  print(f"Inserted {single_vector}")

@Monitor.monitor
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


@Monitor.monitor
def retrieve(embedding):
    response = index.query(
    vector=embedding.flatten().tolist(),
    top_k=3,
    include_values=True,
    include_metadata=True
    )
    result =[ m["metadata"]["report"] for m in response["matches"]]
    urls = []
    for m in response["matches"]:
        if "download_path" in m["metadata"]:
            urls.append(m["metadata"]["download_path"])
    return result, urls


@Monitor.monitor
def generate_response(result, query, li_response):
    result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
         "content": 
         """
         Objective: Generate a concise radiologic diagnosis based on SHARED FEATURES from the provided radiology reports.

        Definition of SHARED FEATURES: Features that appear in more than one report. Features unique to a single report are not considered SHARED.

        Instructions:

        Analyze the provided radiology reports.
        Identify any SHARED FEATURES, these should be the diagnosis and not radiologic features.
        If SHARED FEATURES are found, provide a radiologic diagnosis in one sentence.
        If no SHARED FEATURES are identified, simply state: "Radiologic Diagnosis: Diagnosis not possible."
        Return the reports summarized as well.
         """
        },
        {"role": "assistant", "content": "Reports:"+ "\n-".join(result)},
        {"role": "user", "content": query},
    ]
    ,
    temperature=0)
    return result

@Monitor.monitor
def llama_index_response(query, result):
    from llama_index import SummaryIndex
    from llama_index.schema import TextNode
    index = SummaryIndex([TextNode(text=r) for r in result])
    summary_query_engine = index.as_query_engine()

    tru_query_engine_recorder_tmp = TruLlama(summary_query_engine,
        app_id='LlamaIndex_App1',
        feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])


    with tru_query_engine_recorder_tmp as recording:
        li_response = summary_query_engine.query(query)
    return li_response

def predict(file, query):
    embedding = compute_embedding(file)
    retrieved_result, urls = retrieve(embedding)
    li_response = llama_index_response(query, retrieved_result)
    result = generate_response(retrieved_result, query, li_response)
    result = result['choices'][0]['message']['content']
    result = "**Retrieved Reports:** " + ' \n'.join(retrieved_result) + " \n**Images:** " + (' \n').join(urls) + " \n **Final Diagnosis:** " + result
    return result


# result = predict(img_path=img_path)
# print(f"ID: {result['matches'][0]['id']} | Similarity score: {round(result['matches'][0]['score'], 2)}")
# new_img
