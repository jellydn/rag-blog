# 2. File-based LanceDB and JSON BM25 index

Date: 2026-06-06

## Status

Accepted

## Context

Day 1 goal: a production-*style* RAG stack that runs locally and on a simple VPS without Postgres, Docker, or a managed vector DB. The index is small (~tens of documents, ~100 chunks). Operators should inspect and back up indexes without opaque binary blobs where possible.

## Decision

- **Vectors:** LanceDB table `rag_chunks` under `data/lancedb/` (or `RAG_BLOG_DATA` / `/opt/data/rag-blog/data`).
- **BM25:** Serialize inverted-index inputs to **`bm25_data.json`** at ingest (`doc_ids`, `doc_texts`, `doc_lengths`, `doc_freqs`, `chunk_meta`). Load via `BM25Index.from_json_dict()`; no pickle for the primary path.
- **Paths:** Resolved in `config.py` with precedence: `RAG_BLOG_DATA` → existing `/opt/data/rag-blog/data` → `./data`.

## Consequences

### Positive

- No database server or container requirement for the vector store.
- JSON BM25 artifact is diff-friendly and avoids pickle security concerns on load.
- Same layout works for clone-local dev and optional production mount.

### Negative

- LanceDB + embedding model still imply non-trivial disk and first-run download.
- Full re-ingest required when chunking or id scheme changes; no incremental versioning in Day 1.
- JSON stores full chunk text twice (LanceDB + BM25 file) for simplicity.

## References

- LanceDB embedded connect/search: [references.md](./references.md).
