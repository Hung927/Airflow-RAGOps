import logging
import os
import sys
import json
import random
# sys.path.append("/opt/airflow/dags")
sys.path.append(os.getcwd())
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from pendulum import duration
from datetime import datetime, timedelta

from utils import (
    retrieval,
    rerank,
    llm
)
# from plugins.file_update_sensor import FileUpdateSensor

default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 3, 31),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}


with DAG("RAG_Pipeline", default_args=default_args, schedule=timedelta(hours=12), catchup=False) as dag:
    """RAG pipeline DAG for retrieval-augmented generation (RAG) using Ollama and Qdrant."""
    
    qa_data = json.load(open("dags/data/qa_pairs.json", "r"))
    user_question = random.choice(list(qa_data.keys()))
    standard_answer = qa_data[user_question]
    
    # trigger when file upload
    # file_upload_check_task = FileUpdateSensor(
    #     task_id="file_upload_check_task",
    #     filepath="dags/data/data_file.json",
    #     last_modified_after=datetime.now(),
    #     poke_interval=3,
    #     timeout=600
    # )

    # trigger when file exists
    # file_exists_check_task = FileSensor(
    #     task_id="file_exists_check_task",
    #     filepath="data_file.json",
    #     fs_conn_id="fs_default",
    #     poke_interval=5,
    #     timeout=600
    # )
    
    similarity_retrieval_task = PythonOperator(
        task_id="similarity_retrieval_task",
        python_callable=retrieval,
        op_kwargs={
            "user_question": user_question,
            "types": "similarity"
        },
    )
    
    # keyword_extract_task = PythonOperator(
    #     task_id="keyword_extract_task",
    #     python_callable=llm,
    #     op_kwargs={
    #         "user_question": user_question,
    #         "type": "keyword"
    #     },
    # )
    
    # keyword_retrieval_task = PythonOperator(
    #     task_id="keyword_retrieval_task",
    #     python_callable=retrieval,
    #     op_kwargs={
    #         "user_question": user_question,
    #         "types": "keyword"
    #     },
    # )
    
    rerank_task = PythonOperator(
        task_id="rerank_task",
        python_callable=rerank,
        op_kwargs={
            "user_question": user_question
        },
    )
    
    llm_task = PythonOperator(
        task_id="llm_task",
        python_callable=llm,
        op_kwargs={
            "user_question": user_question,
            "type": "rag"
        },
    )

    similarity_retrieval_task >> rerank_task >> llm_task
    # keyword_extract_task >> keyword_retrieval_task >> rerank_task >> llm_task
    # file_upload_check_task >> similarity_retrieval_task