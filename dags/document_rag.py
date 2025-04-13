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
from plugins import json_update_sensor

default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 4, 10),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}


with DAG("Data_preprocessing_Pipeline", default_args=default_args, schedule="@once", catchup=False) as dag:
    """Config file upload check and data processing pipeline DAG."""
    
    config_data = json.load(open("dags/config.json", "r"))
    
    json_update_check_task = json_update_sensor.JsonUpdateSensor(
        task_id="json_update_check_task",
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
    
    json_update_check_task >> data_processing_task


# with DAG("DATA_Pipeline", default_args=default_args, schedule="@once", catchup=False) as dag:
    
#     raw_data = json.load(open("data/squad.json", "r"))
    
#     # trigger if file upload
#     file_upload_check_task = FileUpdateSensor(
#         task_id="file_upload_check_task",
#         filepath="dags/config.json",
#         last_modified_after=datetime.now(),
#         poke_interval=3,
#         timeout=60
#     )
    
#     data_processing_task = PythonOperator(
#         task_id="data_processing_task",
#         python_callable=data_processing,
#         op_kwargs={
#             "raw_data": raw_data
#         },
#     )
    
#     file_upload_check_task >> data_processing_task
    
    

#     # trigger if file exists
#     # file_exists_check_task = FileSensor(
#     #     task_id="file_exists_check_task",
#     #     filepath="data_file.json",
#     #     fs_conn_id="fs_default",
#     #     poke_interval=5,
#     #     timeout=600
#     # )
    
    
    
    

#     file_upload_check_task