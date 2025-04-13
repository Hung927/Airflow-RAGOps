import os
import json
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

class JsonUpdateSensor(BaseSensorOperator):
    @apply_defaults
    def __init__(self, filepath: str, key: str, expected_value: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath = filepath
        self.key = key
        self.expected_value = expected_value

    def poke(self, context):
        self.log.info(f"Checking if {self.filepath} has key '{self.key}' changed from '{self.expected_value}'")
        if not os.path.exists(self.filepath):
            self.log.info("File does not exist yet.")
            return False
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.log.error(f"Failed to read JSON file: {e}")
            return False
        
        current_value = data.get(self.key)
        self.log.info(f"Current value of key '{self.key}': {current_value}")
        return current_value != self.expected_value
