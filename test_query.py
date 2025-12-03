"""Simple test script to query the log database.

Usage:
    python test_query.py
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

def query_logs(question: str, top_k: int = 5):
    """Query the ChromaDB logs collection."""
    persist_dir = os.path.join(os.path.dirname(__file__), 'chroma_db')
    
    # Load the same model used during ingestion
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("logs")
    
    # Encode the query
    query_emb = model.encode([question])[0].tolist()
    
    # Query the collection
    resp = collection.query(
        query_embeddings=[query_emb], 
        n_results=top_k, 
        include=['documents', 'metadatas', 'distances']
    )
    
    # Format results
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

def main():
    # Contoh queries
    queries = [
        "authentication failures",
        "email delivery errors", 
        "failed login attempts",
        "connection refused",
        "VPN connection",
    ]
    
    print("=" * 80)
    print("LOG QUERY TEST - ChromaDB Semantic Search")
    print("=" * 80)
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)
        
        try:
            results = query_logs(query, top_k=3)
            
            if not results:
                print("  ❌ No results found")
                continue
                
            for i, result in enumerate(results, 1):
                print(f"\n  📄 Result {i}:")
                print(f"     File: {result['metadata'].get('source_file', 'N/A')}")
                print(f"     Lines: {result['metadata'].get('start_line', '?')}-{result['metadata'].get('end_line', '?')}")
                print(f"     Score: {result['score']:.4f}" if result['score'] else "     Score: N/A")
                print(f"     Text preview: {result['text'][:200]}...")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
