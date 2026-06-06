# Day 1 — Production RAG Systems

## Learning Notes

### What I Built

A fully functional RAG engine that indexes my blog content (productsway.com) and answers questions about it using hybrid search.

### Key Decisions & Trade-offs

| Decision | Choice | Why |
|----------|--------|-----|
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | Fast CPU inference, tiny (80MB), good enough for blog content |
| **Vector Store** | LanceDB | File-based, no server needed, fast columnar storage |
| **Chunking** | 512 chars, heading-aware | Small enough for precision, large enough for context |
| **Search Fusion** | RRF (70/30) | Simple, no tuning needed, works well empirically |
| **API** | FastAPI + SSE | Familiar, streaming built-in, CORS ready |

### Surprises & Lessons

1. **Chunking quality matters more than embedding quality** — Naive chunking (just splitting by length) gave poor results. Heading-aware chunking was the single biggest quality improvement.
2. **BM25 + Vector > Vector alone** — For technical TIL content, keyword matches (BM25) catch exact commands/library names that vector search misses.
3. **RRF is surprisingly effective** — No parameter tuning needed. The simple 70/30 weight worked on the first try.
4. **LanceDB is fast** — 83 chunks, 384-dim vectors, <100ms queries even on CPU.

### Raw Performance

- Ingestion: 51 docs → 83 chunks in ~16s
- Model load: ~4s (cached after first load)
- Vector search: 60-90ms per query (cold: ~1s first call)
- BM25 search: 2-8ms per query
- RRF fusion: <1ms

### Next Steps for This

- [ ] Add re-ranking step (cross-encoder) for final precision boost
- [ ] Cache embeddings to avoid recomputing
- [ ] Add document-level metadata filters (by tag, category, date)
- [ ] Deploy as a reusable Hermes skill

### Links

- Repo: https://github.com/jellydn/rag-blog
- Blog: https://productsway.com
- LanceDB: https://lancedb.github.io/lancedb/
- Sentence-Transformers: https://www.sbert.net/

---
_This PR is part of the 7-day AI Engineer journey._

