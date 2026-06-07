# Welcome to rag-blog 👋

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Twitter: jellydn](https://img.shields.io/twitter/follow/jellydn.svg?style=social)](https://twitter.com/jellydn)

> **Production RAG engine for [Productsway.com](https://productsway.com)** — hybrid search (vector + BM25) over blog and TIL content, powered by [sentence-transformers](https://www.sbert.net/) and [LanceDB](https://lancedb.github.io/lancedb/).

Indexes posts and Today-I-Learned notes from [productsway.com](https://productsway.com). Retrieval combines dense embeddings with BM25 keyword search, fused with Reciprocal Rank Fusion (RRF). No Postgres or Docker required for the vector store (Docker is optional for deployment).

## Features

- **Hybrid search** — [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` (384-dim) + custom BM25, fused with RRF (70% vector / 30% BM25)
- **Markdown-aware chunking** — splits on `##` / `###`, respects code fences; stable chunk ids `slug:index`
- **File-based index** — [LanceDB](https://lancedb.github.io/lancedb/) on disk + `bm25_data.json` (no DB server)
- **FastAPI** — JSON search and Server-Sent Events (`/query/stream`), `/health`, `/stats`
- **Agent mode** — `/agent/query` and `/agent/query/stream` add routing, planning, and reflection on top of retrieval
- **CLI** — same hybrid engine as the API via lazy-loaded `get_hybrid()`
- **Tooling** — [mise](https://mise.jdx.dev/), [uv](https://docs.astral.sh/uv/), [Ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/), [prek](https://prek.j178.dev/), Docker Compose

## Install

```sh
git clone https://github.com/jellydn/rag-blog.git
cd rag-blog

mise trust && mise install && mise run install
```

Or with [uv](https://docs.astral.sh/uv/) only:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

With `mise activate` in your shell, entering the repo auto-activates `.venv` when `uv.lock` is present.

Data defaults to `./data` (override with `RAG_BLOG_DATA`). See [Architecture decisions](./doc/adr/README.md).

## Usage

```sh
# Scrape + ingest (first time, or after chunking changes)
mise run pipeline

# API — http://localhost:8000
mise run serve

# Query
curl "http://localhost:8000/query?q=how+to+set+up+neovim+folding&top_k=5"
curl -N "http://localhost:8000/query/stream?q=typescript+absolute+imports"
curl "http://localhost:8000/agent/query?q=how+to+fix+session+id+unknown+with+socket.io"

# CLI (no server)
uv run python query.py "how to cherry pick from a pull request"
```

If search fails with a missing BM25 file, run ingest again: `uv run python rag_pipeline.py`.

### Docker

```sh
docker compose build
docker compose --profile ingest run --rm ingest   # first time
docker compose up -d
# or: just docker-build && just docker-ingest && just docker-up-d
```

Port override: `RAG_BLOG_PORT=8080 docker compose up`. Index and HF cache use the `rag-data` volume.

## API

| Endpoint | Description |
| -------- | ----------- |
| `GET /health` | Liveness |
| `GET /stats` | Chunks, model, BM25 term count |
| `GET /query?q=...&top_k=5` | Hybrid search (JSON + `timing`) |
| `GET /query/stream?q=...` | Same results as SSE |

## Run tests

```sh
mise run test
# or: just check   # ruff + ty + tests
```

Git hooks:

```sh
prek install
mise run prek
```

## Architecture

```
Scrape → Chunk → Embed → Store → Search
                          ↓
                    LanceDB + BM25
                          ↓
                    Hybrid RRF
                          ↓
              FastAPI (JSON + SSE) · CLI
```

## Project status

Day 2 of a 7-day AI engineer track — details in [DAY1_NOTES.md](./DAY1_NOTES.md) and [DAY2_NOTES.md](./DAY2_NOTES.md).

- [x] Day 1 — Production RAG engine
- [ ] Day 2 — Advanced agent patterns
- [ ] Days 3–7 — MCP, fine-tuning, observability, product, open stack

## References

- [LanceDB](https://lancedb.github.io/lancedb/) · [Sentence-Transformers](https://www.sbert.net/)
- [mise](https://mise.jdx.dev/) · [uv](https://docs.astral.sh/uv/) · [Ruff](https://docs.astral.sh/ruff/) · [ty](https://docs.astral.sh/ty/)
- [Productsway](https://productsway.com/)

## Author

👤 **Huynh Duc Dung**

- Website: [productsway.com](https://productsway.com/)
- Twitter: [@jellydn](https://twitter.com/jellydn)
- Github: [@jellydn](https://github.com/jellydn)

## Show your support

[![kofi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/dunghd)
[![paypal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/dunghd)
[![buymeacoffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/dunghd)

Give a ⭐️ if this project helped you!

[![Stargazers repo roster for @jellydn/rag-blog](https://reporoster.com/stars/jellydn/rag-blog)](https://github.com/jellydn/rag-blog/stargazers)
