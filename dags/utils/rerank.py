import atexit
import logging


class Reranker:
    _reranker = None
    
    @staticmethod
    def get_user_question(ti: object) -> str:
        """
        Get user question from XCom.
        
        Returns:
            user_question (str): The user's question.
        """
        try:
            user_question = ti.xcom_pull(task_ids='random_question_task', key='return_value')
            return user_question
        except Exception as e:
            logging.error(f"Error retrieving user question: {e}")
            return "What is the current number of electors currently in a Scottish Parliament constituency?"
    
    def get_context(self, ti: object) -> list:
        """
        Get context from XCom.
        
        Returns:
            context (list): List of context retrieved from XCom.
        """
        try:
            similarity_context = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
            keyword_context = ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')
            
            context_list = []
            if isinstance(similarity_context, list):
                logging.info(f"Similarity context: {similarity_context}")
                context_list.extend(similarity_context)
            if isinstance(keyword_context, list):
                logging.info(f"Keyword context: {keyword_context}")
                context_list.extend(keyword_context)
                
            return list(set(context_list))
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return []

    # Register cleanup function
    @staticmethod
    def cleanup_reranker() -> None:
        """
        Cleanup the reranker model.
        """
        # Add explicit cleanup if available
        if Reranker._reranker and hasattr(Reranker._reranker, 'stop_self_pool'):
            try:
                Reranker._reranker.stop_self_pool()
                Reranker._reranker = None
                logging.info("Reranker model cleaned up.")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
    
    def get_reranker(self) -> object:
        """
        Get the reranker model.
        
        Returns:
            reranker: The reranker model.
        """
        if Reranker._reranker is None:
            from FlagEmbedding import FlagReranker
            try:
                Reranker._reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, cache_dir="dags/.cache")
                atexit.register(self.cleanup_reranker)
                logging.info("Loading reranker model...")
            except Exception as e:
                print(f"Error loading reranker model: {e}")
                raise e
            
        return Reranker._reranker
    
    def rerank_context(self, user_question: str, context: list, top_k: int = 10) -> list:
        """
        Rerank the context based on the user question using a reranker model.
        
        Args:
            user_question (str): The user's question.
            context (list): List of context to be reranked.
            top_k (int): Number of top results to return. Defaults to 10.
            
        Returns:
            sorted_result (list): A list of sorted context based on relevance to the user question.
        """
        try:
            reranker = self.get_reranker()
            logging.info("Reranking context...")
            sentence_pairs = [[user_question, j] for j in context]
            scores = reranker.compute_score(sentence_pairs, normalize=True)
            sorted_result = [point for point, _ in sorted(zip(context, scores), key=lambda x: x[1], reverse=True)][:top_k]
            return sorted_result
        except Exception as e:
            logging.error(f"Error during reranking: {e}")
            return context            
    
    def rerank(self, **kwargs) -> list:
        """
        Rerank the context based on the user question using a reranker model.
        
        Args:
            **kwargs: Additional arguments.
        
        Returns:
            sorted_result (list): A list of sorted context based on relevance to the user question.
        """
        try:
            ti = kwargs['ti']
            user_question = self.get_user_question(ti)
            context = self.get_context(ti)
            if context:
                logging.info(f"Reranking context for question: {user_question}")
                logging.info(f"Context: {context}")
                sorted_result = self.rerank_context(user_question, context)
                logging.info(f"Reranking result: {sorted_result}")
            else:
                logging.warning("No context found for reranking.")
                sorted_result = []
            
            return sorted_result
        except Exception as e:
            logging.error(f"Error retrieving user question: {e}")
            return []
        
        

    

# def rerank(**kwargs) -> list:
#     """
#     Rerank the context based on the user question using a reranker model.
    
#     Args:
#         # user_question (str): The user's question.
#         **kwargs: Additional arguments.
        
#     Returns:
#         sorted_result (list): A list of sorted context based on relevance to the user question.
#     """
    
#     ti = kwargs['ti']
#     user_question = ti.xcom_pull(task_ids='random_question_task', key='return_value')
#     similarity_context = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
#     keyword_context = ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')
    
#     context_list = []
#     if isinstance(similarity_context, list):
#         logging.info(f"Similarity context: {similarity_context}")
#         context_list.extend(similarity_context)
#     if isinstance(keyword_context, list):
#         logging.info(f"Keyword context: {keyword_context}")
#         context_list.extend(keyword_context)
        
#     if not context_list:
#         logging.warning("No context found for reranking.")
#         return []
    
#     context = list(set(context_list))
#     sentence_pairs = [[user_question, j] for j in context]    
    
#     logging.info(f"Reranking context for question: {user_question}")
#     logging.info(f"Context: {context}")
#     sorted_result = []
#     try:
#         reranker = get_reranker()
#         score_result = reranker.compute_score(sentence_pairs, normalize=True)
#         logging.info(f"Score result: {score_result}")
#         sorted_result = [point for point, _ in sorted(zip(context, score_result), key=lambda x: x[1], reverse=True)][:5]
        
#         logging.info(f"Reranking result: {sorted_result}")
        
#         return sorted_result
#     except Exception as e:
#         logging.error(f"Error during reranking: {e}")
#         return context
    