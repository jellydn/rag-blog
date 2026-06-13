# Learning Record 0002 — Vector Search at Scale

**Date:** 2026-06-09
**Lesson:** [0002 — Vector Search at Scale](../lessons/0002-vector-search-at-scale.html)
**Reference:** [Vector Index Cheat Sheet](../reference/vector-index-cheatsheet.html)

## Context

Continuation of the rag-blog AI learning track. User has now run Lesson 0001 (embeddings) and seen the polysemy test ("fold" in code folding vs laundry). Lesson 0002 shifts from "what is a vector" to "how do we find the right vector at scale."

## Key Concepts Learned

- Exact k-NN is O(N·d) per query. Fine at 83 rows; expensive at 1M+.
- ANN trades a small, controlled recall loss (typically ≤ 5%) for a large, controlled speedup.
- HNSW is a graph-based ANN: multi-layer "social network" of vectors; O(log N) search via greedy descent.
- IVF is a partition-based ANN: cluster vectors at build, search only `nprobe` nearest clusters at query.
- PQ compresses vectors ~30–60×, enabling billion-scale in RAM.
- LanceDB defaults to exact k-NN without an index; you opt in via `table.create_index(..., config=IvfHnswSq(...))`.
- For under ~100k rows, exact k-NN is usually the right choice; indexes are overhead below that threshold.

## Non-Obvious Insights

- The "ANN contract" is a recall/speed/memory triangle — you can only pick 2 of 3. Parameter tuning is finding the best 2-of-3 point for your workload.
- LanceDB's "HNSW" indexes are technically all `IVF_HNSW_*` — HNSW is always a sub-graph inside IVF partitions.
- The right recall benchmark is *not* "is the top-1 correct?" — it's "is the top-1 the same chunk that exact k-NN would have returned?" You compare against your old exact behavior, not against some abstract ground truth.
- `nprobe` is the most common tuning knob and the one most often mis-set. Low nprobe = fast but missing recalls. High nprobe = accurate but slow.

## Code Anchors

| Concept | File · Symbol |
| --- | --- |
| Default search (exact k-NN) | `rag_pipeline.py` · `VectorStore.vector_search` |
| LanceDB index API | (not yet in our code) — would be `VectorStore._create_index` or called once in `ingest` |
| Index types | `lancedb.pydantic.IvfHnswSq`, `IvfPq`, `IvfHnswFlat`, etc. |

## Confusions Cleared

- "Why don't we just have an index by default?" → At small scale, the index itself is more memory and build time than the brute-force scan. The default is the simplest correct thing.
- "HNSW vs IVF: which is faster?" → Depends on what's in your data. IVF is better when there's clear cluster structure; HNSW is better when the data is uniformly distributed. For most natural language, HNSW variants win on recall/latency tradeoffs.
- "What's the difference between cosine and L2?" → Both are distances. Cosine is direction-only; L2 is straight-line. With unit-normalized vectors, they agree up to a constant, so either works.

## Open Questions Going Forward

- How does LanceDB combine a vector index with a `where(...)` filter? (Filtered search can defeat HNSW.)
- When should we switch from `sentence-transformers` local embeddings to a hosted embedding API? (Memory + cost tradeoff at scale.)
- How does BM25 fit in — does it need an index too, or is its scale-up different? (Lesson 0003.)

## Misconceptions to Watch For

- That "exact" is always better than "approximate." In practice, exact at 1M+ rows is *so slow* that the user never gets an answer. ANN is the production default.
- That all indexes are interchangeable. The "magic triangle" is real — picking the wrong one for the data shape is the #1 cause of "my vector search is bad" debugging tickets.
- That index creation is free. It can be slow (minutes to hours at 10M+ rows). Plan ingestion pipelines around it.

## What the User Could Now Do

- [x] Explain why we don't create a vector index today (83 rows is too small).
- [x] Read `VectorStore.vector_search` and identify the line that triggers exact k-NN.
- [x] Describe HNSW in their own words (multi-layer graph, greedy descent).
- [x] Describe IVF in their own words (cluster + nprobe).
- [ ] Run the scale benchmark from §8 of the lesson and report the crossover point.
- [ ] Add an `IvfHnswSq` index to the production code and measure recall@10 against the exact baseline.
- [ ] Decide at what row count we'd switch from exact to ANN (this should be in an ADR).
