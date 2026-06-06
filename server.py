#!/usr/bin/env python3
"""
FastAPI streaming RAG server for Productsway.com.
Keeps model + vector store + BM25 in memory for fast queries.

Run:  /opt/hermes/.venv/bin/python3 server.py
Then: curl http://localhost:8000/query?q=how+to+set+up+neovim
      curl -N http://localhost:8000/query/stream?q=what+typescript+tips
"""

import json
import pickle
import time
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Config
DB_DIR = Path("/opt/data/rag-blog/data/lancedb")
MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

app = FastAPI(
    title="Productsway RAG API",
    description="RAG engine for Dung Huynh Duc's blog/TIL content",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Lazy init on first request ──────────────────────────────────────────────

_embedder = None
_table = None
_bm25_data = None

def lazy_init():
    global _embedder, _table, _bm25_data
    if _embedder is not None:
        return
    
    print("Loading embedding model...")
    t0 = time.time()
    _embedder = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    
    print("Opening LanceDB...")
    _table = lancedb.connect(str(DB_DIR)).open_table("rag_chunks")
    print(f"  Opened: {_table.count_rows()} rows")
    
    print("Loading BM25...")
    with open(DB_DIR / "bm25_data.pkl", "rb") as f:
        _bm25_data = pickle.load(f)
    print(f"  Loaded: {_bm25_data['total_docs']} docs, {len(_bm25_data['doc_freqs'])} terms")


def _tokenize(text: str):
    return set(w.lower() for w in text.split() if len(w) > 2 and (w.isalpha() or "#" in w))


def _bm25_score(query_terms, doc_id, text, dl):
    bm25 = _bm25_data
    N = bm25["total_docs"]
    avgdl = bm25["avg_doc_length"]
    k1, b_val = 1.5, 0.75
    score = 0.0
    doc_tokens = _tokenize(text)
    for term in query_terms:
        if term in doc_tokens:
            tf = text.lower().count(term)
            df = bm25["doc_freqs"].get(term, 0)
            if df > 0:
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b_val + b_val * dl / avgdl))
    return float(score)


def hybrid_search(query: str, top_k: int = 5):
    lazy_init()
    embedder = _embedder
    table = _table
    bm25 = _bm25_data
    
    # — Vector search —
    t0 = time.time()
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    vec_results = table.search(query_vec).metric("cosine").limit(top_k * 3).to_list()
    vec_time_ms = (time.time() - t0) * 1000
    
    # — BM25 search —
    t0 = time.time()
    query_terms = _tokenize(query)
    bm25_scores = {}
    for i, (doc_id, text, dl) in enumerate(zip(
        bm25["doc_ids"], bm25["doc_texts"], bm25["doc_lengths"],
    )):
        score = _bm25_score(query_terms, doc_id, text, dl)
        if score > 0:
            bm25_scores[doc_id] = score
    bm25_time_ms = (time.time() - t0) * 1000
    
    # — RRF Fusion —
    scores = {}
    for rank, r in enumerate(vec_results):
        cid = r["id"]
        scores[cid] = {
            "content": r["content"],
            "title": r["title"],
            "doc_id": r["doc_id"],
            "source_url": r["source_url"],
            "category": r["category"],
            "chunk_index": r["chunk_index"],
            "total_chunks": r["total_chunks"],
            "vector_score": float(1 - r["_distance"]),
            "vector_rank": rank + 1,
            "bm25_rank": None,
            "rrf_score": VECTOR_WEIGHT * (1.0 / (RRF_K + rank + 1)),
        }
    
    # Add BM25 scores
    for rank, (doc_id, bm25_score) in enumerate(sorted(bm25_scores.items(), key=lambda x: -x[1])):
        for cid in scores:
            if doc_id in scores[cid]["source_url"] or scores[cid]["doc_id"] == doc_id:
                scores[cid]["bm25_rank"] = rank + 1
                scores[cid]["rrf_score"] += BM25_WEIGHT * (1.0 / (RRF_K + rank + 1))
                scores[cid]["bm25_score"] = bm25_score
                break
    
    results = sorted(scores.values(), key=lambda x: -x["rrf_score"])[:top_k]
    
    # Generate answer context
    context = "\n\n---\n\n".join(
        f"Source: {r['title']}\n{r['content']}"
        for r in results
    )
    
    return {
        "query": query,
        "results": results,
        "context": context,
        "timing": {
            "vector_search_ms": round(vec_time_ms, 1),
            "bm25_search_ms": round(bm25_time_ms, 1),
            "total_ms": round(vec_time_ms + bm25_time_ms, 1),
        },
        "total_chunks": bm25["total_docs"],
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    query: str
    results: list
    context: str
    timing: dict
    total_chunks: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    lazy_init()
    return {
        "chunks": _table.count_rows(),
        "model": MODEL_NAME,
        "dimension": _embedder.get_sentence_embedding_dimension(),
        "bm25_terms": len(_bm25_data["doc_freqs"]),
    }


@app.get("/query", response_model=SearchResult)
def query(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
):
    t0 = time.time()
    result = hybrid_search(q, top_k=top_k)
    result["timing"]["api_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/query/stream")
def query_stream(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
):
    """Streaming search results as SSE (Server-Sent Events)."""
    result = hybrid_search(q, top_k=top_k)
    
    async def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'query': q, 'total_chunks': result['total_chunks'], 'timing': result['timing']})}\n\n"
        
        for i, r in enumerate(result["results"]):
            yield f"data: {json.dumps({'type': 'result', 'index': i + 1, **r})}\n\n"
        
        yield f"data: {json.dumps({'type': 'context', 'context': result['context'][:500]})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
