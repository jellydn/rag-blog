# External references (Context7)

Curated library IDs and documentation anchors used when reasoning about **rag-blog** architecture. Resolved via Context7 MCP (`context7_resolve-library-id`, `context7_query-docs`).

## Stack map

| Concern | Library (Context7 ID) | How rag-blog uses it |
|---------|-------------------------|----------------------|
| Vector store | [`/lancedb/lancedb`](https://github.com/lancedb/lancedb) | Embedded DB at `data/lancedb/`; `table.search(query_vector).limit(k)` (ADR-0002) |
| Embeddings | [`/huggingface/sentence-transformers`](https://github.com/huggingface/sentence-transformers) | `all-MiniLM-L6-v2`, 384-dim, cosine-friendly (ADR-0001) |
| HTTP API | [`/fastapi/fastapi`](https://github.com/fastapi/fastapi) | `GET /query`, `/query/stream`, `/agent/*`; `StreamingResponse` SSE (ADR-0004, ADR-0006) |
| Hybrid patterns | [`/lancedb/vectordb-recipes`](https://github.com/lancedb/vectordb-recipes) | RAG examples; we fuse vector + custom BM25 in-app instead of LanceDB FTS-only |

## Snippets (from Context7 query-docs)

**LanceDB — local vector search**

```python
import lancedb
db = lancedb.connect("<PATH_TO_LANCEDB_DATASET>")
table = db.open_table("my_table")
results = table.search([0.1, 0.3]).limit(20).to_list()
```

Source: [lancedb/python README](https://github.com/lancedb/lancedb/blob/main/python/README.md)

**Sentence Transformers — MiniLM embeddings**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences)  # shape [n, 384]
```

Source: [sentence-transformers quickstart](https://github.com/huggingface/sentence-transformers/blob/main/docs/quickstart.rst)

**FastAPI — SSE-style stream**

```python
from fastapi.responses import StreamingResponse

async def event_stream():
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Source: [FastAPI custom responses](https://github.com/fastapi/fastapi/blob/master/docs/en/docs/advanced/custom-response.md)

## Related ADRs

| ADR | Topic |
|-----|--------|
| [0001](./0001-hybrid-search-vector-bm25-rrf.md) | RRF, weights, BM25+ |
| [0002](./0002-file-based-lancedb-and-json-bm25.md) | LanceDB path, `bm25_data.json` |
| [0003](./0003-chunk-identity-and-markdown-chunking.md) | `{doc_id}:{chunk_index}` |
| [0004](./0004-stdlib-config-and-shared-search-engine.md) | Module boundaries, `get_hybrid()` |
| [0005](./0005-curated-static-note-catalog-for-ingestion.md) | `NOTE_SLUGS`, stdlib scrape |
| [0006](./0006-llm-free-deterministic-agent-layer.md) | `agentic.py`, `/agent/*` |

## Refreshing references

From an agent session with Context7 MCP enabled:

1. `context7_resolve-library-id` with `query` (task) + `libraryName` (e.g. `LanceDB`, `FastAPI`).
2. `context7_query-docs` with chosen `libraryId` and a focused `query`.
3. Add durable decisions here or as a new ADR; avoid duplicating upstream docs—link and summarize trade-offs only.
