import atexit
import logging
from FlagEmbedding import FlagReranker
    
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, cache_dir="dags/.cache")


_reranker = None

def get_reranker():
    """
    Get the reranker model.
    
    Returns:
        reranker: The reranker model.
    """
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        try:
            _reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, cache_dir="dags/.cache")
            logging.info("Loading reranker model...")
        except Exception as e:
            print(f"Error loading reranker model: {e}")
            raise e
        
    return _reranker
    

# Register cleanup function
def cleanup_reranker():
    global _reranker
    # Add explicit cleanup if available
    if hasattr(reranker, 'stop_self_pool'):
        try:
            reranker.stop_self_pool()
            _reranker = None
            logging.info("Reranker model cleaned up.")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

atexit.register(cleanup_reranker)

def rerank(user_question: str, **kwargs) -> list:
    """
    Rerank the context based on the user question using a reranker model.
    
    Args:
        user_question (str): The user's question.
        **kwargs: Additional arguments.
        
    Returns:
        sorted_result (list): A list of sorted context based on relevance to the user question.
    """
    
    ti = kwargs['ti']
    similarity_context = ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')
    keyword_context = ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')
    
    context_list = []
    if isinstance(similarity_context, list):
        context_list.extend(similarity_context)
    if isinstance(keyword_context, list):
        context_list.extend(keyword_context)
        
    if not context_list:
        logging.warning("No context found for reranking.")
        return []
    
    context = list(set(context_list))
    sentence_pairs = [[user_question, j] for j in context]    
    
    logging.info(f"Reranking context for question: {user_question}")
    logging.info(f"Context: {context}")
    sorted_result = []
    try:
        sentence_pairs = [[user_question, j] for j in context]
        reranker = get_reranker()
        score_result = reranker.compute_score(sentence_pairs, normalize=True)
        logging.info(f"Score result: {score_result}")
        sorted_result = [point for point, _ in sorted(zip(context, score_result), key=lambda x: x[1], reverse=False)][:5]
        
        logging.info(f"Reranking result: {sorted_result}")
        
        return sorted_result
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        return context
    