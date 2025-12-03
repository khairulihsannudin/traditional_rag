# workflow.py - Traditional RAG using LangChain
import logging
import time
from typing import Dict, Any
from agents.guardrails_agent import guardrails_router_chain
from agents.synthesizer_agent import synthesis_chain
from agents.vector_agent import query_vector_search

logger = logging.getLogger(__name__)


# workflow.py - Traditional RAG using LangChain
import logging
import time
from typing import Dict, Any
from agents.guardrails_agent import guardrails_router_chain
from agents.synthesizer_agent import synthesis_chain
from agents.vector_agent import query_vector_search

logger = logging.getLogger(__name__)


# --- Instrumentation Utilities ---
INSTRUMENTATION_ENABLED = False

def enable_instrumentation(enabled: bool = True) -> None:
    """Enable or disable workflow instrumentation."""
    global INSTRUMENTATION_ENABLED
    INSTRUMENTATION_ENABLED = enabled
    logger.info(f"Workflow instrumentation {'enabled' if enabled else 'disabled'}")


def run_traditional_rag(question: str) -> Dict[str, Any]:
    """
    Traditional RAG pipeline: Guardrails → Vector Search → Synthesize
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with 'answer', 'log_vector_context', and optional timing data
    """
    result = {
        "question": question,
        "original_question": question,
        "_timing_data": {}
    }
    
    # Step 1: Guardrails
    logger.info("--- Executing: [[Guardrails]] ---")
    start_time = time.time() if INSTRUMENTATION_ENABLED else None
    
    guardrails_result = guardrails_router_chain.invoke({"question": question})
    
    if INSTRUMENTATION_ENABLED:
        result["_timing_data"]["guardrails"] = time.time() - start_time
    
    if guardrails_result.decision == "irrelevant":
        logger.warning(f"[[Guardrails]]: Irrelevant question detected -> '{question}'")
        result["answer"] = "Sorry, I can only answer questions related to log analysis."
        result["is_relevant"] = False
        return result
    
    logger.info("[[Guardrails]]: Question is relevant.")
    result["is_relevant"] = True
    
    # Step 2: Vector Search
    logger.info("--- Executing: [[Vector Agent]] ---")
    start_time = time.time() if INSTRUMENTATION_ENABLED else None
    
    try:
        vector_context = query_vector_search(question)
        logger.info("[[Vector Agent]]: Vector search completed successfully.")
        logger.info(f"[[Vector Agent]]: Context found:\n{vector_context}")
        result["log_vector_context"] = vector_context
    except Exception as e:
        logger.error(f"[[Vector Agent]]: Vector search failed: {e}")
        result["log_vector_context"] = f"Error during vector search: {e}"
    
    if INSTRUMENTATION_ENABLED:
        result["_timing_data"]["vector_agent"] = time.time() - start_time
    
    # Step 3: Synthesize Answer
    logger.info("--- Executing: [[Synthesizer]] ---")
    start_time = time.time() if INSTRUMENTATION_ENABLED else None
    
    log_vector = str(result.get('log_vector_context')) if result.get('log_vector_context') else "Not applicable for this query."
    
    if log_vector == "Not applicable for this query.":
        final_answer = "Sorry, I could not find any relevant information in the logs."
    else:
        final_answer = synthesis_chain.invoke({
            "original_question": question,
            "log_vector_context": log_vector,
        })
    
    result["answer"] = final_answer
    
    if INSTRUMENTATION_ENABLED:
        result["_timing_data"]["synthesizer"] = time.time() - start_time
    
    return result


async def ainvoke(initial_state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
    """
    Async wrapper for compatibility with existing run.py
    
    Args:
        initial_state: Dictionary with 'question' key
        config: Optional configuration (ignored for traditional RAG)
        
    Returns:
        Result dictionary from run_traditional_rag
    """
    question = initial_state.get("question", "")
    result = run_traditional_rag(question)
    
    # Merge timing data from initial state if present
    if "_timing_data" in initial_state:
        result["_timing_data"].update(initial_state["_timing_data"])
    
    return result


class TraditionalRAGApp:
    async def ainvoke(self, initial_state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        return await ainvoke(initial_state, config)
    
    def invoke(self, initial_state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        question = initial_state.get("question", "")
        return run_traditional_rag(question)


app = TraditionalRAGApp()
