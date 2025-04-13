import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="dags/.env")

import ollama
import logging
from qdrant_client import QdrantClient, models

def retrieval(user_question: str, embed_model: str = "imac/zpoint_large_embedding_zh", types: str = "similarity", **kwargs) -> list:
    """Retrieve relevant documents from Qdrant based on the user question.
    
    Args:
        user_question (str): The user's question.
        model (str): The model to be used for embeddings. Defaults to "gemma2:9b".
        type (str): The type of retrieval. Defaults to "similarity".
        **kwargs: Additional arguments.
        
    Returns:
        search_result (list): A list of retrieved documents.
    """    
    
    logging.info(f"Retrieving information for question: {user_question}")
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://10.20.1.95:6333"))
    content = ""
    search_result = []
    ti = kwargs['ti']
    
    if types == "similarity":
        logging.info(f"Using similarity search")
        result = qdrant_client.search(
            collection_name=f"""squad_zpoint_large_embedding_zh""",
            query_vector=ollama.embeddings(
                model=embed_model, 
                prompt=user_question,
                options={"device": "cpu"},
                keep_alive="0s"
            )["embedding"],
            limit=10,
            score_threshold=0
        )
        
    elif types == "keyword":
        logging.info(f"Using keyword search")
        condition = []
        keyword_list = ti.xcom_pull(task_ids='keyword_extract_task', key='return_value')
        # keyword_list = ti.xcom_pull(task_ids='keyword_extract_task', key='keyword')
        logging.info(f"Keyword list: {keyword_list}")
        
        for keyword in keyword_list:
            logging.info(f"Keyword: {keyword}")
            condition.append(
                models.FieldCondition(
                    key="document",
                    match=models.MatchText(text=keyword),
                )
            )
        
        result = qdrant_client.search(
            collection_name=f"""squad_zpoint_large_embedding_zh""",
            query_vector=ollama.embeddings(
                model=embed_model, 
                prompt=user_question,
                options={"device": "cpu"},
                keep_alive="0s"
            )["embedding"],
            limit=10,
            score_threshold=0,
            query_filter=models.Filter(
                must=condition
            ),
        )
    
    for index in range(len(result)):
        content = f"""{result[index].payload["document"]}"""
        search_result.append(content)
    logging.info(f"Retrieval result: {search_result}")
    # ti.xcom_push(key=type, value=search_result)
    
    return search_result
