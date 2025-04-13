from dotenv import load_dotenv
load_dotenv(dotenv_path="dags/.env")

import os
import json
import ollama
import random
from uuid import uuid4
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))


def create_expert_collection(collection_name: str, vector_size: int):
    # collection_list = [c.name for collection in qdrant_client.get_collections() for c in collection[1]]
    # if collection_name not in collection_list:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        hnsw_config=models.HnswConfigDiff(on_disk=True, m=64, ef_construct=512)
    )


def insert_documents(collection_name: str):
    jsonFile = open('dags/data/expert_data.json','r')
    data = json.load(jsonFile)
    ollama_vector = []

    for i in data:
        print(i)
        vec = ollama.embeddings(
            model="imac/zpoint_large_embedding_zh", 
            prompt=data[i],
            options={"device": "cpu"},
            keep_alive="0s"
        )["embedding"]
        
        ollama_vector.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vec,
                payload={"question": i, "answer": data[i]}
            )
        )
        
    qdrant_client.upsert(
        collection_name=collection_name,
        points=ollama_vector
    )
    
    
if __name__ == "__main__":
    collection_name="squad_expert_zpoint_large_embedding_zh"
    collection = {
        "squad_expert_zpoint_large_embedding_zh": 1024,
    }
    create_expert_collection(collection_name=collection_name, vector_size=collection[collection_name])
    insert_documents(collection_name=collection_name)