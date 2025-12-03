# agents/vector_agent.py - Traditional RAG Vector Search
import logging

logger = logging.getLogger(__name__)

def query_vector_search(question: str, top_k: int = 5) -> str:
    """
    Query ChromaDB for relevant log chunks using semantic search.
    Traditional RAG: Simple vector retrieval without graph queries.
    
    Args:
        question: User's question
        top_k: Number of results to retrieve
        
    Returns:
        Formatted string with retrieved log contexts
    """
    logger.info(f"--- Executing Vector Search for: {question} ---")
    
    try:
        from agents.retriever import retrieve
        results = retrieve(question, top_k=top_k)
        
        if not results:
            logger.warning("No results found from vector search")
            return "No relevant log data found."
        
        # Format results into readable context
        if isinstance(results, list):
            formatted_chunks = []
            for i, r in enumerate(results, 1):
                meta = r.get('metadata', {}) or {}
                src = meta.get('source_file', 'unknown')
                text = r.get('text', '')
                score = r.get('score', 'N/A')
                
                formatted_chunks.append(
                    f"[Source: {src} | Score: {score}]\n{text}"
                )
            
            context = "\n\n".join(formatted_chunks)
            logger.info(f"Vector search returned {len(results)} chunks")
            return context
        
        return str(results)
        
    except Exception as e:
        logger.error(f"[Vector Agent] Retrieval failed: {e}", exc_info=True)
        return f"Error during vector search: {str(e)}"