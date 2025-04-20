import json
import logging
from typing import Union, List
from airflow.exceptions import AirflowSkipException
    

class ExpertBranch:
    @staticmethod
    def branch_logic(**kwargs) -> Union[str, List[str], None]:
        """
        Branch logic for expert retrieval.
        
        Args:
            USE_SIMILARITY (bool): Flag to use similarity retrieval.
            USE_KEYWORD (bool): Flag to use keyword retrieval.
            **kwargs: Additional keyword arguments.
        Returns:
            Union[str, List[str], None]: The task ID(s) to branch to.
        """
        try:
            ti = kwargs['ti']
            USE_SIMILARITY = kwargs.get('USE_SIMILARITY', False)
            USE_KEYWORD = kwargs.get('USE_KEYWORD', False)
            
            expert_validation = ti.xcom_pull(task_ids='expert_validate_task', key='return_value')
            if isinstance(expert_validation, str):
                expert_validation = json.loads(expert_validation)
            if not isinstance(expert_validation, dict):
                logging.error("Expert validation result is not a valid JSON object.")
                raise ValueError("Expert validation result is not a valid JSON object.")
            
            if expert_validation["status"] == "COMPLETE":
                logging.info(f"Expert validation result: {expert_validation['useful_information']}")
                raise AirflowSkipException("Expert validation completed. Skipping downstream tasks.")
            elif expert_validation["status"] == "INCOMPLETE":
                if USE_SIMILARITY and USE_KEYWORD:
                    return ["similarity_retrieval_task", "keyword_extract_task"]
                elif USE_SIMILARITY:
                    return "similarity_retrieval_task"
                elif USE_KEYWORD:
                    return "keyword_extract_task"
                else:
                    logging.warning("No valid branch found, defaulting to similarity retrieval.")
                    return "llm_task"
        except Exception as e:
            logging.error(f"Error in branch logic: {e}")
            raise e