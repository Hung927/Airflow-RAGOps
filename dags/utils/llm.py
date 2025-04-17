from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import os
import ast
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import utils.prompt_config as prompt_config

def llm(model: str = "gemma2:9b", type: str = "rag", **kwargs) -> str:
    """Create a LangChain LLM chain using the specified model.
    
    Args:
        # user_question (str): The user's question.
        model (str): The model to be used. Defaults to "gemma2:9b".
        type (str): The type of the LLM chain. Defaults to "rag".
        **kwargs: Additional arguments.
        
    Returns:
        llm_result (str): The result from the LLM chain.
    """
    
    ti = kwargs['ti']
    user_question = ti.xcom_pull(task_ids='random_question_task', key='return_value')
    
    llm = ChatOllama(
        model=model,
        base_url=os.getenv("OLLAMA_URL"),
        temperature=0.0,
        keep_alive="0s"
    )
    
    if type == "keyword":
        logging.info(f"Using keyword extraction")
        PROMPT = prompt_config.Prompt_en().prompt.KEYWORD
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
        llm_result = ast.literal_eval(f'[{llm_result}]')
        logging.info(f"user_question: {user_question}")
        logging.info(f"LLM result: /n{llm_result}")
        # ti.xcom_push(key=type, value=llm_result)

    else:
        
        if type == "rag":
            logging.info(f"Using RAG")
            PROMPT = prompt_config.Prompt_en().prompt.RAG_DETAIL_SYS_PROMPT
        elif type == "validation":
            logging.info(f"Using validation")
            PROMPT = prompt_config.Prompt_en().prompt.VALIDATION
        elif type == "summary":
            logging.info(f"Using summary")
            PROMPT = prompt_config.Prompt_en().prompt.SUMMARY_1
        else:
            logging.info(f"Using general ask")
            PROMPT = prompt_config.Prompt_en().prompt.GENERAL_ASK_SYS_PROMPT
            
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
    
        context = ti.xcom_pull(task_ids='rerank_task', key='return_value')
        logging.info(f"Rerank task result: {context}") 
        if not context:
            context = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
            logging.info(f"Context is None, using retrieval task result: {context}")
            
        llm_result = llm_chain.invoke({"context": "".join(context), "user_question": user_question})
        logging.info(f"user_question: {user_question}")
        logging.info(f"Context: {context}")        
        logging.info(f"LLM result: /n{llm_result}")
    
    return llm_result
