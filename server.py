#!/usr/bin/env python3
"""FastAPI hybrid search API for Productsway RAG."""

import json
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentic import AgentOrchestrator
from config import MODEL_NAME
from rag_pipeline import HybridSearch, create_hybrid_search

# Bounds for the top_k query parameter. With multi-pass retrieval
# (primary + 2 secondary), top_k=N triggers ~3*N embeddings + 3*N
# vector + 3*N BM25 ops. Capping at 20 prevents abuse / accidental
# amplification (a single request can't pin the CPU for minutes).
TOP_K_MIN = 1
TOP_K_MAX = 20

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
_agent = None


# Path to the BM25 index file. Used by the preflight check in
# get_hybrid() to surface a clear operational error if the index
# is missing, instead of letting the request fail deep inside the
# retrieval pipeline with a confusing traceback.
_BM25_INDEX_PATH = Path("data/lancedb/bm25_data.json")


def get_hybrid() -> HybridSearch:
    global _hybrid
    if _hybrid is None:
        # Preflight: surface a clear operational error if the BM25
        # index file is missing (e.g. because the user hasn't run
        # `uv run python rag_pipeline.py` to build the index yet).
        # Without this check, the failure happens deep inside the
        # retrieval pipeline with a confusing FileNotFoundError.
        if not _BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {_BM25_INDEX_PATH}. "
                "Build it with: uv run python rag_pipeline.py"
            )
        _hybrid = create_hybrid_search()
    return _hybrid


def get_agent() -> AgentOrchestrator:
    global _agent
    if _agent is None:
        _agent = AgentOrchestrator(get_hybrid())
    return _agent


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


class AgentResult(BaseModel):
    query: str
    route: dict
    plan: dict
    results: list
    reflection: dict
    answer: str
    timing: dict


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


@app.get("/agent/query", response_model=AgentResult)
def agent_query(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(
        5,
        ge=TOP_K_MIN,
        le=TOP_K_MAX,
        description=f"Number of results (1..{TOP_K_MAX})",
    ),
):
    t0 = time.time()
    agent = get_agent()
    run = agent.run(q, top_k=top_k)
    payload = agent.to_dict(run)
    payload["timing"] = {"api_ms": round((time.time() - t0) * 1000, 1)}
    return payload


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


@app.get("/agent/query/stream")
def agent_query_stream(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(
        5,
        ge=TOP_K_MIN,
        le=TOP_K_MAX,
        description=f"Number of results (1..{TOP_K_MAX})",
    ),
):
    agent = get_agent()

    def event_stream():
        # Consume the agent's stream() generator and yield SSE
        # messages as events are produced. The sync generator runs
        # in FastAPI's threadpool; the SSE response is delivered
        # progressively to the client (not buffered until the full
        # agent run completes). Yields a final 'done' event for
        # client-side completion detection.
        for event in agent.stream(q, top_k=top_k):
            yield f"data: {json.dumps(event)}\n\n"
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
