"""Retrieval utilities for Traditional RAG.

This module queries a local ChromaDB collection for log data.

Public API:
    retrieve(question, top_k) -> List[dict]

Each returned dict has keys: `id`, `text`, `metadata`, `score`.
"""
from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except Exception as e:
    chromadb = None
    SentenceTransformer = None
    CHROMA_AVAILABLE = False
    import sys
    print(f"WARNING: ChromaDB/sentence-transformers import failed: {e}", file=sys.stderr)


def _query_chroma(question: str, top_k: int, persist_dir: str = None) -> List[Dict]:
    """Query a Chroma collection named 'logs' and return structured results."""
    if not CHROMA_AVAILABLE:
        raise RuntimeError("Chroma or sentence-transformers not available")

    if persist_dir is None:
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR') or os.path.abspath(os.path.join(os.path.dirname(__file__), 'chroma_db'))

    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection("logs")
    except Exception:
        # collection not found
        raise RuntimeError("Chroma collection 'logs' not found in %s" % persist_dir)

    # Use the same embedding model as the ingestion script
    model = SentenceTransformer(os.environ.get('CHROMA_EMBED_MODEL', 'all-MiniLM-L6-v2'))
    query_emb = model.encode([question])[0].tolist()

    resp = collection.query(
        query_embeddings=[query_emb], 
        n_results=top_k, 
        include=['documents', 'metadatas', 'distances']
    )

    results = []
    ids_list = resp.get('ids', [[]])[0]
    docs_list = resp.get('documents', [[]])[0]
    metas_list = resp.get('metadatas', [[]])[0]
    dist_list = resp.get('distances', [[]])[0]

    for i in range(len(ids_list)):
        results.append({
            'id': ids_list[i],
            'text': docs_list[i],
            'metadata': metas_list[i],
            'score': float(dist_list[i]) if i < len(dist_list) else None,
        })
    return results


def retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """Retrieve top-k passages for `question` from ChromaDB.

    Args:
        question: User's query
        top_k: Number of results to retrieve
        
    Returns:
        List of dicts with keys: id, text, metadata, score
    """
    if not CHROMA_AVAILABLE:
        logger.error("[Retriever] ChromaDB or sentence-transformers not installed")
        return []
    
    try:
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR') or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
        )
        results = _query_chroma(question, top_k, persist_dir=persist_dir)
        logger.info("[Retriever] ChromaDB returned %d results", len(results))
        return results
    except Exception as e:
        logger.error("[Retriever] ChromaDB query failed: %s", e, exc_info=True)
        return []

