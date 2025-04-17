import os
from datetime import datetime
from airflow.sensors.base import BaseSensorOperator

class FileUpdateSensor(BaseSensorOperator):
    def __init__(self, filepath: str, last_modified_after: datetime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath = filepath
        self.last_modified_after = last_modified_after

    def poke(self, context):
        self.log.info(f"Checking if {self.filepath} has been modified after {self.last_modified_after}")
        if os.path.exists(self.filepath):
            mtime = datetime.fromtimestamp(os.path.getmtime(self.filepath))
            self.log.info(f"Last modified time: {mtime}")
            return mtime > self.last_modified_after
        return False
