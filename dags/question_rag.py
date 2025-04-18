import os
import sys
import json
import random
import logging
sys.path.append(os.getcwd())
from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import duration
from datetime import datetime, timedelta

from utils.retrieval import Retrieval
from utils.rerank import Reranker
from utils.llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
try:
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
    rag_config = config_data.get("rag_pipeline_config")
    logging.info(f"Loaded RAG pipeline config: {rag_config}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.warning(f"Could not load or parse {CONFIG_PATH}. Using default RAG config. Error: {e}")
    rag_config = {
        "use_similarity_retrieval": False,
        "use_keyword_retrieval": False,
        "use_rerank": False
    }
USE_SIMILARITY = rag_config.get("use_similarity_retrieval", False)
USE_KEYWORD = rag_config.get("use_keyword_retrieval", False)
if USE_SIMILARITY and USE_KEYWORD:
    USE_RERANK = True
elif (USE_SIMILARITY or USE_KEYWORD):
    USE_RERANK = rag_config.get("use_rerank", False)
else:
    USE_RERANK = False
logging.info(f"RAG pipeline config: USE_SIMILARITY={USE_SIMILARITY}, USE_KEYWORD={USE_KEYWORD}, USE_RERANK={USE_RERANK}")    
    
default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 3, 31),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}

def get_user_question():
    """Get a random user question from the qa_pairs.json file."""
    
    qa_path = os.path.join(os.path.dirname(__file__), "data/qa_pairs.json")
    try:
        with open(qa_path, "r") as f:
            qa_data = json.load(f)
            user_question = random.choice(list(qa_data.keys()))
    except Exception as e:
        logging.error(f"Error loading or parsing {qa_path}: {e}")
        user_question = "What is the current number of electors currently in a Scottish Parliament constituency? "
    
    return user_question

with DAG("RAG_Pipeline", default_args=default_args, schedule=timedelta(hours=12), catchup=False) as dag:
    """RAG pipeline DAG for retrieval-augmented generation (RAG) using Ollama and Qdrant."""
    
    Retrieval = Retrieval(embed_model=config_data.get("embed_model"))
    Reranker = Reranker()
    LLM = LLM(model=config_data.get("llm_model"))
    
    random_question_task = PythonOperator(
        task_id="random_question_task",
        python_callable=get_user_question,
    )
    
    if USE_SIMILARITY:
        logging.info("Using similarity retrieval")        
        similarity_retrieval_task = PythonOperator(
            task_id="similarity_retrieval_task",
            python_callable=Retrieval.retrieval,
            op_kwargs={
                "document_types": config_data.get("document_types"),
                "types": "similarity"
            },
        )
    else:
        logging.info("Not using similarity retrieval")
        similarity_retrieval_task = None
    
    if USE_KEYWORD:
        logging.info("Using keyword retrieval")
        keyword_extract_task = PythonOperator(
            task_id="keyword_extract_task",
            python_callable=LLM.llm,
            op_kwargs={
                "types": "keyword"
            },
        )        
        keyword_retrieval_task = PythonOperator(
            task_id="keyword_retrieval_task",
            python_callable=Retrieval.retrieval,
            op_kwargs={
                "document_types": config_data.get("document_types"),
                "types": "keyword"
            },
        )
    else:
        logging.info("Not using keyword retrieval")
        keyword_extract_task = None
        keyword_retrieval_task = None
    
    if USE_RERANK:
        logging.info("Using rerank")
        rerank_task = PythonOperator(
            task_id="rerank_task",
            python_callable=Reranker.rerank,
        )
    else:
        logging.info("Not using rerank")
        rerank_task = None
    
    llm_task_op_kwargs = {"types": "rag"} if (similarity_retrieval_task or keyword_retrieval_task) else {"types": "general"}
    llm_task = PythonOperator(
        task_id="llm_task",
        python_callable=LLM.llm,
        op_kwargs=llm_task_op_kwargs,
    )
        
    retrieval_tasks = []

    if USE_SIMILARITY:
        random_question_task >> similarity_retrieval_task
        retrieval_tasks.append(similarity_retrieval_task)

    if USE_KEYWORD:
        random_question_task >> keyword_extract_task >> keyword_retrieval_task
        retrieval_tasks.append(keyword_retrieval_task)

    if USE_RERANK:
        for task in retrieval_tasks:
            task >> rerank_task
        rerank_task >> llm_task
    elif len(retrieval_tasks) > 0:
        for task in retrieval_tasks:
            task >> llm_task

    if not retrieval_tasks:
        random_question_task >> llm_task
        