# 4. Stdlib config/chunking and shared hybrid engine

Date: 2026-06-06

## Status

Accepted

## Context

`scrape_content.py` and unit tests should not import `sentence-transformers` or LanceDB. CLI and FastAPI had duplicated hybrid search wiring and inconsistent tokenization before consolidation. Cold-start cost (model load) should happen once per process, not per CLI invocation.

## Decision

- **`config.py`** — Paths, `MODEL_NAME`, RRF constants; `ensure_data_dirs()` (stdlib only).
- **`chunking.py`** — `Document`, `MarkdownChunker`, chunk id helpers (stdlib only).
- **`rag_pipeline.py`** — Ingest, `BM25Index`, `VectorStore`, `HybridSearch`, `create_hybrid_search()`, persistence helpers.
- **`server.py`** — HTTP/SSE; lazy singleton **`get_hybrid()`** holding one `HybridSearch` instance.
- **`query.py`** — CLI that calls `get_hybrid()` from `server` (same engine as API).

Query timing reports measured vector and BM25 phases from `HybridSearch.search()`, not estimated splits.

## Consequences

### Positive

- Scrape and chunker tests run without ML stack installed.
- One tokenizer and one RRF implementation for API and CLI.
- Clear seam for later extraction (`store.py` / `search.py`) if `rag_pipeline.py` grows.

### Negative

- CLI depends on `server` module for initialization (acceptable for Day 1; could move to `rag_engine.py` later).
- Importing `server` pulls FastAPI into CLI process (lightweight relative to the embedding model).
