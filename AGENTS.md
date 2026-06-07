# AGENTS.md

Python 3.10+ RAG engine for productsway.com. Hybrid search (vector + BM25) via Reciprocal Rank Fusion.

## Quick commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingestion pipeline (scrape → chunk → embed → store)
python scrape_content.py          # fetch markdown from productsway.com → data/content/
python rag_pipeline.py            # chunk + embed → data/lancedb/ + bm25_data.json

# Run API server (port 8000)
python server.py

# CLI search (no server needed)
python query.py "your question"
python query.py --json "neovim folding"

# Tests
python -m unittest discover -s tests -v
```

No lint, typecheck, or formatter is configured. No CI pipeline exists.

## Key gotchas

- **BM25 index must exist before search.** If hybrid search fails with missing BM25 file, run `python rag_pipeline.py` to recreate `data/lancedb/bm25_data.json`.
- **Tests are ML-free.** Only `test_chunker.py` exists; it tests chunking logic (stdlib only). No integration tests requiring sentence-transformers.
- **Embedding model loads lazily.** First `get_hybrid()` call downloads `all-MiniLM-L6-v2` (~80 MB). Subsequent calls use cached model.
- **Data dir resolution** (`config.py`): checks `RAG_BLOG_DATA` env → `/opt/data/rag-blog/data` → `./data` (relative to repo). All generated data is gitignored.
- **Server uses global singleton.** `get_hybrid()` in `server.py` and `query.py` share the same lazy-loaded engine.

## Architecture in 30 seconds

```
scrape_content.py  →  data/content/*.md
                        ↓
rag_pipeline.py    →  chunking.py (MarkdownChunker) → data/chunks/
                        ↓
                    rag_pipeline.py (Embedder + VectorStore + BM25Index)
                        ↓
                    data/lancedb/ (LanceDB) + data/lancedb/bm25_data.json
                        ↓
server.py / query.py → HybridSearch (RRF: 70% vector / 30% BM25)
```

Config constants live in `config.py` (RRF_K=60, VECTOR_WEIGHT=0.7, BM25_WEIGHT=0.3). `chunking.py` is stdlib-only by design (safe for tests without ML deps).

## Conventions

- Chunk IDs are `doc_slug:chunk_index` (stable across runs).
- Chunks are markdown-aware: splits on `##`/`###`, keeps code blocks intact, 512-char max with 64-char overlap.
- ADRs in `doc/adr/` document design decisions.
