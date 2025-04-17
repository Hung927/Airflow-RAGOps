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

from utils.data_processing import Data_Processing
from utils.data_embedding import Data_Embedding
from plugins import json_update_sensor


default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 4, 10),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}


with DAG("Data_Preprocessing", default_args=default_args, schedule="@once", catchup=False) as dag:
    """Config file upload check and data processing pipeline DAG."""
    
    config_data = json.load(open("dags/config.json", "r"))
        
    uploaded_files_config_json_update_check_task = json_update_sensor.JsonUpdateSensor(
        task_id="uploaded_files_config_json_update_check_task",
        filepath="dags/config.json",
        key="uploaded_files",
        expected_value=config_data["uploaded_files"],
        poke_interval=3,
        timeout=60
    )
    
    data_processing_task = PythonOperator(
        task_id="data_processing_task",
        python_callable=Data_Processing().data_processing
    )
    
    file_list_config_json_update_check_task = json_update_sensor.JsonUpdateSensor(
        task_id="file_list_config_json_update_check_task",
        filepath="dags/config.json",
        key="file_list",
        expected_value=config_data["file_list"],
        poke_interval=3,
        timeout=600
    )
    
    data_embedding_task = PythonOperator(
        task_id="data_embedding_task",
        python_callable=Data_Embedding().documents_embedding
    )
    
    
    uploaded_files_config_json_update_check_task >> data_processing_task
    file_list_config_json_update_check_task >> data_embedding_task
