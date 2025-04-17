import os
import sys
import json
import random
sys.path.append(os.getcwd())
from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import duration
from datetime import datetime, timedelta

from utils import (
    retrieval,
    rerank,
    llm
)


default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 3, 31),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}

def get_user_question():
    """Get a random user question from the qa_pairs.json file."""
    
    qa_data = json.load(open("dags/data/qa_pairs.json", "r"))
    user_question = random.choice(list(qa_data.keys()))
    
    return user_question

with DAG("RAG_Pipeline", default_args=default_args, schedule=timedelta(hours=12), catchup=False) as dag:
    """RAG pipeline DAG for retrieval-augmented generation (RAG) using Ollama and Qdrant."""
    
    random_question_task = PythonOperator(
        task_id="random_question_task",
        python_callable=get_user_question,
    )
            
    similarity_retrieval_task = PythonOperator(
        task_id="similarity_retrieval_task",
        python_callable=retrieval,
        op_kwargs={
            "types": "similarity"
        },
    )
    
    keyword_extract_task = PythonOperator(
        task_id="keyword_extract_task",
        python_callable=llm,
        op_kwargs={
            "type": "keyword"
        },
    )
    
    keyword_retrieval_task = PythonOperator(
        task_id="keyword_retrieval_task",
        python_callable=retrieval,
        op_kwargs={
            "types": "keyword"
        },
    )
    
    rerank_task = PythonOperator(
        task_id="rerank_task",
        python_callable=rerank,
    )
    
    llm_task = PythonOperator(
        task_id="llm_task",
        python_callable=llm,
        op_kwargs={
            "type": "rag"
        },
    )

    random_question_task >> [similarity_retrieval_task, keyword_extract_task]
    similarity_retrieval_task >> rerank_task >> llm_task
    keyword_extract_task >> keyword_retrieval_task >> rerank_task >> llm_task