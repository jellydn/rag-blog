#!/usr/bin/env python3
"""FastAPI hybrid search API for Productsway RAG."""

import json
import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import MODEL_NAME
from rag_pipeline import HybridSearch, create_hybrid_search

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

_hybrid = None


def get_hybrid() -> HybridSearch:
    global _hybrid
    if _hybrid is None:
        _hybrid = create_hybrid_search()
    return _hybrid


def hybrid_search(query: str, top_k: int = 5):
    hybrid = get_hybrid()
    results, timing = hybrid.search(query, top_k=top_k)
    context = "\n\n---\n\n".join(f"Source: {r['title']}\n{r['content']}" for r in results)
    return {
        "query": query,
        "results": results,
        "context": context,
        "timing": timing,
        "total_chunks": hybrid.bm25.total_docs,
    }


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
    hybrid = get_hybrid()
    return {
        "chunks": hybrid.vector_store.count(),
        "model": MODEL_NAME,
        "dimension": hybrid.embedder.dimension,
        "bm25_terms": len(hybrid.bm25.doc_freqs),
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
