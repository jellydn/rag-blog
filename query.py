#!/usr/bin/env python3
"""
CLI query tool for the RAG pipeline.
Usage: python query.py "your question about Dung's blog"
"""

import json
import pickle
import sys
import time
from pathlib import Path

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_DIR = Path("/opt/data/rag-blog/data/lancedb")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_bm25():
    """Load the serialized BM25 index data."""
    with open(DB_DIR / "bm25_data.pkl", "rb") as f:
        return pickle.load(f)


def search(query: str, top_k: int = 5):
    """Run hybrid search and return results."""
    # Load model
    model = SentenceTransformer(MODEL_NAME)
    
    # Load vector store
    db = lancedb.connect(str(DB_DIR))
    table = db.open_table("rag_chunks")
    
    # Load BM25 data
    bm25_data = load_bm25()
    
    # Vector search
    t0 = time.time()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    vec_results = table.search(query_vec).metric("cosine").limit(top_k * 2).to_list()
    vec_time = time.time() - t0
    
    # Simple BM25 scoring
    t0 = time.time()
    query_terms = set(
        w.lower() for w in query.split()
        if len(w) > 2 and w.isalpha()
    )
    
    bm25_scores = {}
    doc_freqs = bm25_data["doc_freqs"]
    N = bm25_data["total_docs"]
    k1, b_val = 1.5, 0.75
    avgdl = bm25_data["avg_doc_length"]
    
    for doc_id, text, dl in zip(
        bm25_data["doc_ids"],
        bm25_data["doc_texts"],
        bm25_data["doc_lengths"],
    ):
        score = 0.0
        doc_tokens = set(w.lower() for w in text.split() if len(w) > 2 and w.isalpha())
        for term in query_terms:
            if term in doc_tokens:
                tf = text.lower().count(term)
                df = doc_freqs.get(term, 0)
                if df > 0:
                    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b_val + b_val * dl / avgdl))
        if score > 0:
            bm25_scores[doc_id] = score
    
    bm25_time = time.time() - t0
    
    # RRF fusion
    scores = {}
    for rank, r in enumerate(vec_results):
        doc_id = r["id"]
        score = 0.7 * (1.0 / (60 + rank + 1))
        scores[doc_id] = {
            "content": r["content"],
            "title": r["title"],
            "source_url": r["source_url"],
            "category": r["category"],
            "vector_score": float(1 - r["_distance"]),
            "rrf_score": score,
        }
    
    for rank, (doc_id, bm25_score) in enumerate(sorted(bm25_scores.items(), key=lambda x: -x[1])):
        for vid in scores:
            if scores[vid]["source_url"].endswith(doc_id) or scores[vid]["source_url"] == f"https://productsway.com/notes/{doc_id}":
                scores[vid]["rrf_score"] += 0.3 * (1.0 / (60 + rank + 1))
                scores[vid]["bm25_score"] = float(bm25_score)
                break
    
    results = sorted(scores.values(), key=lambda x: -x["rrf_score"])[:top_k]
    
    return {
        "query": query,
        "results": results,
        "timing": {
            "vector_search_ms": round(vec_time * 1000, 1),
            "bm25_search_ms": round(bm25_time * 1000, 1),
            "total_ms": round((vec_time + bm25_time + (time.time() - t0 + bm25_time)) * 1000, 1),
        },
        "total_chunks": len(bm25_data["doc_ids"]),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        print("       python query.py --json \"your question\"")
        sys.exit(1)
    
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    query_text = " ".join(args)
    
    result = search(query_text)
    
    if as_json:
        print(json.dumps(result, indent=2))
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 Query: {result['query']}")
    print(f"📊 {result['total_chunks']} chunks searched in {result['timing']['total_ms']}ms")
    print(f"{'='*60}\n")
    
    for i, r in enumerate(result["results"], 1):
        print(f"─── Result #{i} ───")
        print(f"📌 {r['title']}")
        print(f"🏷️  {r['category']}")
        print(f"🔗 {r['source_url']}")
        print(f"📊 Vector: {r.get('vector_score', 'N/A'):.3f} | RRF: {r['rrf_score']:.3f}")
        print(f"📝 {r['content'][:400]}...")
        print()


if __name__ == "__main__":
    main()
