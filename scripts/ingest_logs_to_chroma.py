"""Ingest plain-text log files from a sibling `logData` directory into Chroma.

Usage:
    python scripts/ingest_logs_to_chroma.py

Config:
    LOG_DATA_DIR env var (defaults to ../logData)
    CHROMA_PERSIST_DIR env var (defaults to ./chroma_db)

This script is intentionally simple: it chunks by lines, embeds using
sentence-transformers `all-MiniLM-L6-v2`, and writes to a Chroma collection
named 'logs'.
"""
import os
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
import argparse
import json


def chunk_lines(lines, chunk_size=200, overlap=50):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(lines)), step):
        chunk = lines[i:i+chunk_size]
        if not chunk:
            continue
        chunks.append((i+1, i+len(chunk), "\n".join(chunk)))
        if i + chunk_size >= len(lines):
            break
    return chunks


def ingest(log_dir, persist_dir, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("logs")

    ids = []
    docs = []
    metadatas = []

    for fname in sorted(os.listdir(log_dir)):
        fpath = os.path.join(log_dir, fname)
        if not os.path.isfile(fpath):
            continue
        print(f"Processing {fpath}")
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = [l.rstrip() for l in fh.readlines()]

        chunks = chunk_lines(lines)
        for start, end, text in chunks:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            docs.append(text)
            metadatas.append({
                "source_file": fname,
                "start_line": start,
                "end_line": end
            })

    if not docs:
        print("No documents found to ingest.")
        return

    print(f"Embedding {len(docs)} chunks with model {model_name}...")
    embeddings = model.encode(docs, show_progress_bar=True)

    print("Adding to Chroma collection 'logs'...")
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=[em.tolist() for em in embeddings])
    print(f"Ingestion complete. Persisted to: {persist_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default=os.environ.get('LOG_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'logData')))
    parser.add_argument('--persist-dir', default=os.environ.get('CHROMA_PERSIST_DIR', os.path.join(os.path.dirname(__file__), '..', 'chroma_db')))
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    persist_dir = os.path.abspath(args.persist_dir)

    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        raise SystemExit(1)

    os.makedirs(persist_dir, exist_ok=True)
    ingest(log_dir, persist_dir)
