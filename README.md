# rag-blog — Production RAG engine for Productsway.com 👋

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Twitter: jellydn](https://img.shields.io/twitter/follow/jellydn.svg?style=social)](https://twitter.com/jellydn)

> **Production RAG engine for [Productsway.com](https://productsway.com)** — hybrid search (vector + BM25) over blog and TIL content, powered by [sentence-transformers](https://www.sbert.net/) and [LanceDB](https://lancedb.github.io/lancedb/).

Indexes posts and Today-I-Learned notes from productsway.com. Retrieval combines dense embeddings with BM25 keyword search, fused with Reciprocal Rank Fusion (RRF). No Postgres or Docker required for the vector store.

## Features

- **Hybrid search** — `all-MiniLM-L6-v2` (384-dim) vector similarity plus custom BM25 (k1=1.5, b=0.75)
- **Markdown-aware chunking** — splits on `##` / `###`, keeps code blocks intact, 512-char chunks with 64-char overlap
- **File-based storage** — LanceDB on disk + serialized BM25 index (`bm25_data.json`)
- **Streaming API** — FastAPI with JSON and Server-Sent Events (`/query/stream`)
- **CLI** — `query.py` uses the same lazy-loaded search engine as the API (model loads on first query)
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

Python 3.10+ recommended.

```bash
git clone https://github.com/jellydn/rag-blog.git
cd rag-blog

# Astral uv — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: mise install uv

uv sync          # .venv + locked deps (ruff, ty in dev group)
# or: just install
```

Legacy pip: `pip install -r requirements.txt` (prefer `uv sync` + `uv.lock`).

### Data directory

Data is stored under `./data` in the repo by default. Override with `RAG_BLOG_DATA` (directory that contains `content/`, `chunks/`, `lancedb/`). If `/opt/data/rag-blog/data` exists, that path is used instead (production layout).

```bash
# optional: production-style path
export RAG_BLOG_DATA=/opt/data/rag-blog/data
```

Generated artifacts (gitignored): `data/content/`, `data/chunks/`, `data/lancedb/` (includes `bm25_data.json` after ingest).

### Tests and quality

```bash
# quality (Ruff + ty + tests) — https://docs.astral.sh/ruff/ https://docs.astral.sh/ty/
just check
# or: uv run ruff check . && uv run ty check && uv run python -m unittest discover -s tests -v

# git hooks
just prek-install && just prek
```

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

# CLI (no server; first run downloads the embedding model)
python query.py "how to cherry pick from a pull request"
python query.py --json "neovim folding"
```

If hybrid search fails with a missing BM25 file, run ingest again so `data/lancedb/bm25_data.json` is created (or updated after chunk-id changes).

### Docker

```bash
# Build image
docker compose build
# or: just docker-build

# First time: scrape + embed into a named volume
docker compose --profile ingest run --rm ingest
# or: just docker-ingest

# Run API (http://localhost:8000)
docker compose up -d
# or: just docker-up-d

curl "http://localhost:8000/health"
curl "http://localhost:8000/query?q=neovim+folding"
```

Data and Hugging Face model cache live in the **`rag-data`** volume (`/data` inside the container). Override the host port with `RAG_BLOG_PORT=8080 docker compose up`.

## API

| Endpoint                          | Description                                   |
| --------------------------------- | --------------------------------------------- |
| `GET /health`                     | Liveness check                                |
| `GET /stats`                      | Chunk count, model name, BM25 vocabulary size |
| `GET /query?q=...&top_k=5`        | Hybrid search (JSON)                          |
| `GET /query/stream?q=...&top_k=5` | Same results as SSE                           |

JSON responses include a `timing` object: `vector_search_ms`, `bm25_search_ms`, `total_ms`, and on `GET /query` also `api_ms` (end-to-end handler time). SSE `meta` events carry the same timing fields.

## Project structure

```
├── config.py            # Paths, RRF weights (stdlib only)
├── chunking.py          # Markdown-aware chunker + chunk ids (stdlib only)
├── scrape_content.py    # Fetch notes/guides from productsway.com → data/content/
├── rag_pipeline.py      # Ingest, LanceDB, BM25, HybridSearch, create_hybrid_search()
├── server.py            # FastAPI routes; lazy singleton via get_hybrid()
├── query.py             # CLI wrapper around the same get_hybrid() engine
├── Dockerfile           # API image (uvicorn)
├── docker-compose.yml   # api + optional ingest profile
├── justfile             # install, test, lint, prek, serve, docker
├── prek.toml            # prek / git hook config (ruff + builtins)
├── pyproject.toml       # project deps + Ruff + ty config
├── uv.lock              # locked deps (uv)
├── requirements.txt     # optional pip fallback
├── tests/               # Unit tests (chunker; no ML deps required)
├── DAY1_NOTES.md        # Build notes and trade-offs (7-day AI engineer track)
├── git_push.py          # Helper to push via GitHub Git Data API
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
| Fusion         | RRF, k=60, 70% vector rank / 30% BM25 rank (`config.py`)     |
| Chunk identity | Stable ids `doc_slug:chunk_index` in LanceDB and BM25 index    |
| Corpus (Day 1) | ~51 pages → ~83 chunks (TILs, guides, homepage)              |

Typical latency on CPU (after warm load): vector ~60–90 ms, BM25 ~2–8 ms per query (reported in API/CLI `timing`).

## Project status

Part of a 7-day AI engineer journey ([Day 1 notes](./DAY1_NOTES.md)).

- [x] Day 1 — Production RAG engine
- [ ] Day 2 — Advanced agent patterns
- [ ] Day 3 — MCP server deep dive
- [ ] Day 4 — Fine-tuning and custom models
- [ ] Day 5 — AI observability and evaluation
- [ ] Day 6 — AI product from stack
- [ ] Day 7 — Open source AI stack

## Architecture decisions

Design rationale is recorded as [ADRs in `doc/adr/`](./doc/adr/README.md) (hybrid search, storage, chunk ids, module layout).

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
