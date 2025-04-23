import os
import json
import logging
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
    # answer_correctness
)
from ragas.dataset_schema import EvaluationResult

class Ragas:
    def __init__(self, qa_path: str = "dags/data/qa_pairs.json"):
        try:
            self.qa_path = qa_path
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            logging.error(f"Error initializing Ragas class: {e}")
            raise e
    
    @staticmethod
    def get_user_question(ti: object) -> str:
        """
        Get user question from XCom.
        
        Returns:
            user_question (str): The user's question.
        """
        try:
            user_question = ti.xcom_pull(task_ids='random_question_task', key='return_value')
            logging.info(f"User question: {user_question}")
            return user_question
        except Exception as e:
            logging.error(f"Error retrieving user question: {e}")
            raise e
           
    def get_standard_answer(self, user_question: str) -> str:
        """
        Get the standard answer from the QA pairs.
        
        Returns:
            standard_answer (str): The standard answer.
        """
        try:
            with open(self.qa_path, 'r') as file:
                qa_pairs = json.load(file)
            standard_answer = qa_pairs.get(user_question)
            if standard_answer is None:
                logging.warning(f"No matching question found for: {user_question}")
                return "No matching question found."
            logging.info(f"Found standard answer for question '{user_question}'")
            return standard_answer
        except Exception as e:
            logging.error(f"Error retrieving standard answer: {e}")
            raise e
        
    def get_llm_answer(self, ti: object, types: str = "rag") -> str:
        """
        Get LLM answer from XCom.
        
        Args:
            ti (object): Task instance.
            types (str): Type of LLM answer to retrieve. Default is "rag".
        
        Returns:
            llm_answer (str): The LLM answer.
        """
        try:
            if types == "rag":
                llm_answer = ti.xcom_pull(task_ids='llm_task', key='return_value')
            else:
                raise ValueError("Invalid type for LLM answer.")
            logging.info(f"LLM answer: {llm_answer}")
            return llm_answer
        except Exception as e:
            logging.error(f"Error retrieving LLM answer: {e}")
            raise e
    
    @staticmethod
    def get_reference_answer(ti: object, types: str) -> str:
        """
        Get reference answer from XCom.
        
        Args:
            ti (object): Task instance.
            types (str): Type of reference answer to retrieve.
        
        Returns:
            reference_answer (str): The reference answer.
        """
        try:
            if types == "similarity":
                reference_answer = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
            elif types == "keyword":
                reference_answer = ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')
            elif types == "rerank":
                reference_answer = ti.xcom_pull(task_ids='reranking_task', key='return_value')
            else:
                raise ValueError("Invalid type for reference answer.")
            # reference_answer = [[item] for item in reference_answer]
            logging.info(f"Reference answer: {[reference_answer]}")
            return [reference_answer]
        except Exception as e:
            logging.error(f"Error retrieving reference answer: {e}")
            raise e
    
    def get_rag_results(self, retrieval_types: str, llm_types: str = "rag", **kwargs) -> dict:
        """
        Get RAG results for the specified type.
        
        Args:
            retrieval_types (str): Type of RAG results to retrieve.
            **kwargs: Additional keyword arguments.
        
        Returns:
            rag_results (dict): Dictionary containing RAG results.
        """
        try:
            ti = kwargs.get('ti')
            user_question = self.get_user_question(ti)
            generate_answer = self.get_llm_answer(ti, llm_types)
            standard_answer = self.get_standard_answer(user_question)
            reference_answer = self.get_reference_answer(ti, retrieval_types)
            
            return {
                "question": [user_question],
                "answer": [generate_answer],
                "ground_truth": [standard_answer],
                "contexts": reference_answer
            }
        except Exception as e:
            logging.error(f"Error retrieving RAG results: {e}")
            raise e
    
    def openai_models(self):
        """
        Get OpenAI models for Ragas.
        
        Returns:
            ragas_llm (object): Ragas LLM wrapper.
            ragas_emb (object): Ragas embeddings wrapper.
        """        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_openai.embeddings import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    openai_api_key=self.openai_api_key, 
                    model="gpt-4o-mini"
                )
            )
            ragas_emb = LangchainEmbeddingsWrapper(
                embeddings=OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model="text-embedding-3-small"
                )
            )
            return ragas_llm, ragas_emb
        except ImportError as e:
            logging.error(f"Error importing langchain_openai: {e}")
            raise e
        
    def evaluate_with_ragas(self, dataset: dict) -> dict:
        try:
            dataset=Dataset.from_dict(dataset)
            ragas_llm, ragas_emb = self.openai_models()
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ]
            logging.info("Starting Ragas evaluation...")
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=RunConfig(max_workers=1, 
                                    max_wait=180, 
                                    log_tenacity=True, 
                                    max_retries=3),
            )
            logging.info("Ragas evaluation completed.")
            return evaluation_result
        except Exception as e:
            logging.error(f"Error during Ragas evaluation: {e}")
            raise e
    
    def ragas(self, **kwargs) -> dict:
        retrieval_modes = {
            "similarity": kwargs.get('USE_SIMILARITY', False),
            "keyword": kwargs.get('USE_KEYWORD', False),
            "rerank": kwargs.get('USE_RERANK', False)
        }
        
        if not any(retrieval_modes.values()):
            logging.warning("No retrieval modes are enabled. Skipping RAGAS evaluation.")
            return {}
                
        evaluation_scores_result = {}
        
        for mode, enabled in retrieval_modes.items():
            if not enabled:
                logging.info(f"Skipping {mode} retrieval as it is not enabled.")
                continue
            
            rag_results = self.get_rag_results(retrieval_types=mode, llm_types="rag", **kwargs)
            logging.info(f"RAG with {mode} results: {rag_results}")
            
            if (mode == "similarity" and "no relevant information" in rag_results.get('answer')[0].lower()) or \
               (mode == "keyword" and not rag_results.get('contexts')[0]):
                logging.warning(f"No relevant information found in the {mode} retrieval.")
                evaluation_scores = {"status": "No relevant information found"}
            else:                
                evaluation_results = self.evaluate_with_ragas(rag_results)
                logging.info(f"{mode.capitalize()} evaluation results: {evaluation_results}")
                
                if isinstance(evaluation_results, EvaluationResult):
                    evaluation_scores = evaluation_results._repr_dict
                    logging.info(f"Extracted {mode} evaluation scores: {evaluation_scores}")
                elif isinstance(evaluation_results, dict):
                    evaluation_scores = evaluation_results
                    logging.warning(f"RAGAS evaluation for {mode} failed or returned pre-defined failure dict: {evaluation_scores}")
                else:
                    logging.error(f"Unexpected return type from ragas_evaluate for {mode}: {type(evaluation_results)}")
                    evaluation_scores = {"error": "Unexpected evaluation result type"}
                
            evaluation_scores_result[mode] = evaluation_scores
        
        return evaluation_scores_result
        