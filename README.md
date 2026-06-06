# rag-blog — Production RAG for Productsway.com

A production-grade **Retrieval-Augmented Generation (RAG)** engine that indexes the blog posts and TILs from [productsway.com](https://productsway.com). Built from scratch with hybrid search (vector + BM25) fused via Reciprocal Rank Fusion.

## Features

- **🔍 Hybrid Search** — Combines dense vector search (sentence-transformers, 384-dim) with BM25 keyword matching
- **📖 Markdown-Aware Chunking** — Splits on `##` headings, respects code blocks and logical boundaries
- **⚡ File-Based Storage** — LanceDB vector store, no PostgreSQL/Docker needed
- **🌊 Streaming API** — Server-Sent Events support for real-time UIs
- **📊 Production Ready** — FastAPI server with CORS, health checks, and metrics

## Architecture

```
Scrape → Chunk → Embed → Store → Search
                          ↓
                    LanceDB + BM25
                          ↓
                    Hybrid RRF Fuser
                          ↓
                    Streaming API
```

## Quick Start

```bash
# 1. Install dependencies
uv pip install sentence-transformers lancedb fastapi uvicorn

# 2. Scrape content
python scrape_content.py

# 3. Ingest (chunk + embed + store)
python rag_pipeline.py

# 4. Start the API server
python server.py

# 5. Query it
curl "http://localhost:8000/query?q=how+to+set+up+neovim+folding"

# Or stream results
curl -N "http://localhost:8000/query/stream?q=typescript+absolute+imports"
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /stats` | Stats (chunks, model, BM25 terms) |
| `GET /query?q=...&top_k=5` | Hybrid search JSON response |
| `GET /query/stream?q=...&top_k=5` | SSE streaming results |

## Project Structure

```
├── scrape_content.py    # Scrapes markdown content from productsway.com
├── rag_pipeline.py      # Chunk → Embed → Store into LanceDB + BM25
├── server.py            # FastAPI hybrid search server
├── query.py             # CLI query tool
└── data/
    ├── content/         # Raw scraped markdown files
    ├── chunks/          # Chunks metadata
    └── lancedb/         # LanceDB vector store + BM25 index
```

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, fast CPU inference)
- **Vector Store**: LanceDB (columnar, file-based, no server)
- **BM25**: Custom implementation with k1=1.5, b=0.75, delta=1.0
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with 70/30 vector/BM25 weighting
- **Chunk Size**: 512 chars with 64-char overlap, section-aware boundaries
- **Documents**: 51 pages → 83 chunks (43 TILs + 6 Guides + 1 Homepage)

## Project Status

- [x] Day 1: Production RAG engine — ✅ Complete
- [ ] Day 2: Advanced Agent Patterns
- [ ] Day 3: MCP Server Deep Dive
- [ ] Day 4: Fine-Tuning & Custom Models
- [ ] Day 5: AI Observability & Evaluation
- [ ] Day 6: AI Product from Stack
- [ ] Day 7: Open Source AI Stack
