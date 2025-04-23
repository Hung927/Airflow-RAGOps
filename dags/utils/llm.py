from dotenv import load_dotenv
load_dotenv(dotenv_path="dags/.env")

import os
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import utils.prompt_config as prompt_config


class LLM:
    def __init__(
        self, 
        model: str = "gemma2:9b",
        ollama_url: str = os.getenv("OLLAMA_URL"),
        temperature: float = 0.0,
        keep_alive: str = "0s",
        num_ctx: int = 8192,
    ):
        """
        Initialize the LLM class.
        
        Args:
            model (str): The model to be used. Defaults to "gemma2:9b".
            ollama_url (str): The URL for the Ollama model. Defaults to the environment variable OLLAMA_URL.
            temperature (float): The temperature for the model. Defaults to 0.0.
            keep_alive (str): The keep-alive setting for the model. Defaults to "0s".
            prompt_config (object): The prompt configuration object.
        """
        try:
            self.model = model
            self.temperature = temperature
            self.ollama_url = ollama_url
            self.keep_alive = keep_alive
            self.num_ctx = num_ctx
            self.prompt_config = prompt_config.Prompt_en().prompt
        except Exception as e:
            logging.error(f"Error initializing LLM class: {e}")
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
            return "What is the current number of electors currently in a Scottish Parliament constituency?"
    
    def get_llm_model(self) -> object:
        """
        Get the LLM model.
        
        Returns:
            llm: The LLM model.
        """
        try:
            llm = ChatOllama(
                model=self.model,
                base_url=self.ollama_url,
                temperature=self.temperature,
                keep_alive=self.keep_alive,
                num_ctx=self.num_ctx
            )
            return llm
        except Exception as e:
            logging.error(f"Error getting LLM model: {e}")
            raise e

    def get_context(self, ti, types: str) -> list:
        """
        Get context from XCom.
        
        Args:
            ti (object): Task instance.
            
        Returns:
            context (list): List of context retrieved from XCom.
        """
        try:
            if types == "validation":
                context = ti.xcom_pull(task_ids='expert_retrieval_task', key='return_value')
                logging.info(f"Expert retrieval task result: {context}")
            else:    
                context = ti.xcom_pull(task_ids='reranking_task', key='return_value')
                logging.info(f"Rerank task result: {context}")
            
                if not context:
                    context = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
                    logging.info(f"Similarity context: {context}")
                    
                    if not context:
                        context = ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')
                        logging.info(f"Keyword context: {context}")
            
            return context if isinstance(context, list) else []
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return []
        
    def generate_response_from_question(self, types: str, user_question: str, llm: object) -> str:
        """
        Generate a response from the LLM based on the user's question.
        
        Args:
            types (str): The type of task to perform.
            user_question (str): The user's question.
            llm (object): The LLM model.
        
        Returns:
            llm_result (str): The result from the LLM chain.
        """
        try:
            if types == "keyword":
                logging.info(f"Using keyword extraction")
                PROMPT = self.prompt_config.KEYWORD
            elif types == "general":
                logging.info(f"Using general ask")
                PROMPT = self.prompt_config.GENERAL_ASK_SYS_PROMPT
            else:
                logging.warning(f"Unknown type: {types}")
                return ""
                
            prompt_template = f"""{PROMPT}
            
The Question: {{user_question}}

Response:"""
            llm_chain = (
                {"user_question": RunnablePassthrough()}
                | ChatPromptTemplate.from_template(prompt_template)
                | llm
                | StrOutputParser()
            )
            llm_result = llm_chain.invoke({"user_question": user_question})
            return llm_result
        except Exception as e:
            logging.error(f"Error in keyword extraction: {e}")
            raise e
    
    def generate_response_with_context(self, types: str, user_question: str, context: list, llm: object):
        """
        Generate a response from the LLM based on the user's question and context.
        
        Args:
            types (str): The type of task to perform.
            user_question (str): The user's question.
            context (list): The context to use.
            llm (object): The LLM model.
        
        Returns:
            llm_result (str): The result from the LLM chain.
        """
        try:
            if types == "rag":
                logging.info(f"Using RAG")
                PROMPT = self.prompt_config.RAG_DETAIL_SYS_PROMPT
            elif types == "validation":
                logging.info(f"Using validation")
                PROMPT = self.prompt_config.VALIDATION
            elif types == "summary":
                logging.info(f"Using summary")
                PROMPT = self.prompt_config.SUMMARY_1
            else:
                logging.warning(f"Unknown type: {types}")
                return ""
                
            prompt_template = f"""{PROMPT}

    Context: 
    {{context}}

    The Question: {{user_question}}

    Response:"""

            llm_chain = (
                {"context": RunnablePassthrough(), "user_question": RunnablePassthrough()}
                | ChatPromptTemplate.from_template(prompt_template)
                | llm
                | StrOutputParser()
            )
                       
            llm_result = llm_chain.invoke({"context": "".join(context) if context else "", "user_question": user_question})
            return llm_result
        except Exception as e:
            logging.error(f"Error in LLM process: {e}")
            raise e       
       
    def llm(self, types: str = "rag", **kwargs) -> str:
        """
        Create a LangChain LLM chain using the specified model.
        Args:
            types (str): The types of the LLM chain. Defaults to "rag".
            **kwargs: Additional arguments.
        
        Returns:
            llm_result (str): The result from the LLM chain.
        """
        try:
            ti = kwargs['ti']
            user_question = self.get_user_question(ti)
            llm = self.get_llm_model()
            
            if types == "keyword" or types == "general":
                llm_result = self.generate_response_from_question(types, user_question, llm)

            else:
                context = self.get_context(ti, types)
                llm_result = self.generate_response_with_context(types, user_question, context, llm)
            
            logging.info(f"LLM result: /n{llm_result}")
            return llm_result

        except Exception as e:
            logging.error(f"Error in LLM class: {e}")
            raise e
