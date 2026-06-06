# rag-blog — Production RAG engine for Productsway.com 👋

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Twitter: jellydn](https://img.shields.io/twitter/follow/jellydn.svg?style=social)](https://twitter.com/jellydn)

> **Production RAG engine for [Productsway.com](https://productsway.com)** — hybrid search (vector + BM25) over blog and TIL content, powered by [sentence-transformers](https://www.sbert.net/) and [LanceDB](https://lancedb.github.io/lancedb/).

Indexes posts and Today-I-Learned notes from productsway.com. Retrieval combines dense embeddings with BM25 keyword search, fused with Reciprocal Rank Fusion (RRF). No Postgres or Docker required for the vector store.

## Features

- **Hybrid search** — `all-MiniLM-L6-v2` (384-dim) vector similarity plus custom BM25 (k1=1.5, b=0.75)
- **Markdown-aware chunking** — splits on `##` / `###`, keeps code blocks intact, 512-char chunks with 64-char overlap
- **File-based storage** — LanceDB on disk + serialized BM25 index (`bm25_data.pkl`)
- **Streaming API** — FastAPI with JSON and Server-Sent Events (`/query/stream`)
- **CLI** — `query.py` for ad-hoc hybrid search from the terminal
- **Ops-friendly** — `/health`, `/stats`, CORS enabled for browser clients

## Architecture

```
Scrape → Chunk → Embed → Store → Search
                          ↓
                    LanceDB + BM25
                          ↓
                    Hybrid RRF (70% vector / 30% BM25)
                          ↓
              FastAPI (JSON + SSE) · CLI
```

## Install

Python 3.10+ recommended. Scripts default to data under `/opt/data/rag-blog` (see [Data directory](#data-directory)).

```bash
git clone https://github.com/jellydn/rag-blog.git
cd rag-blog

python3 -m venv .venv
source .venv/bin/activate

pip install sentence-transformers lancedb fastapi uvicorn numpy
# or: uv pip install sentence-transformers lancedb fastapi uvicorn numpy
```

### Data directory

Pipeline paths are rooted at `/opt/data/rag-blog`. For local development from a clone:

```bash
sudo mkdir -p /opt/data
sudo ln -sfn "$(pwd)" /opt/data/rag-blog
```

Generated artifacts live under `data/` (gitignored): `content/`, `chunks/`, `lancedb/`.

## Usage

```bash
# 1. Scrape markdown from productsway.com
python scrape_content.py

# 2. Chunk, embed, and build LanceDB + BM25
python rag_pipeline.py

# 3. API server (default http://0.0.0.0:8000)
python server.py

# 4. Hybrid search
curl "http://localhost:8000/query?q=how+to+set+up+neovim+folding&top_k=5"

# SSE stream
curl -N "http://localhost:8000/query/stream?q=typescript+absolute+imports"

# CLI (no server)
python query.py "how to cherry pick from a pull request"
python query.py --json "neovim folding"
```

## API

| Endpoint                          | Description                                   |
| --------------------------------- | --------------------------------------------- |
| `GET /health`                     | Liveness check                                |
| `GET /stats`                      | Chunk count, model name, BM25 vocabulary size |
| `GET /query?q=...&top_k=5`        | Hybrid search (JSON)                          |
| `GET /query/stream?q=...&top_k=5` | Same results as SSE                           |

## Project structure

```
├── scrape_content.py    # Fetch notes/guides from productsway.com → data/content/
├── rag_pipeline.py      # Chunk → embed → LanceDB table + BM25 pickle
├── server.py            # FastAPI hybrid search (lazy model/DB load)
├── query.py             # Standalone CLI hybrid search
├── DAY1_NOTES.md        # Build notes and trade-offs (7-day AI engineer track)
├── git_push.py          # Helper to open Day-1 PR on GitHub
└── data/                # Generated (see .gitignore)
    ├── content/
    ├── chunks/
    └── lancedb/
```

## Technical details

| Area           | Choice                                                       |
| -------------- | ------------------------------------------------------------ |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2`, cosine, normalized |
| Vector store   | LanceDB table `rag_chunks`                                   |
| Fusion         | RRF, k=60, 70% vector rank / 30% BM25 rank                   |
| Corpus (Day 1) | ~51 pages → ~83 chunks (TILs, guides, homepage)              |

Typical latency on CPU (after warm load): vector ~60–90 ms, BM25 ~2–8 ms per query.

## Project status

Part of a 7-day AI engineer journey ([Day 1 notes](./DAY1_NOTES.md)).

- [x] Day 1 — Production RAG engine
- [ ] Day 2 — Advanced agent patterns
- [ ] Day 3 — MCP server deep dive
- [ ] Day 4 — Fine-tuning and custom models
- [ ] Day 5 — AI observability and evaluation
- [ ] Day 6 — AI product from stack
- [ ] Day 7 — Open source AI stack

## References

- [LanceDB](https://lancedb.github.io/lancedb/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Productsway](https://productsway.com/)

## Author

👤 **Huynh Duc Dung**

- Website: [productsway.com](https://productsway.com/)
- Twitter: [@jellydn](https://twitter.com/jellydn)
- GitHub: [@jellydn](https://github.com/jellydn)

## Show your support

[![kofi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/dunghd)
[![paypal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/dunghd)
[![buymeacoffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/dunghd)

Give a ⭐️ if this project helped you!

