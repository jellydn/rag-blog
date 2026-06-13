# Learning Record 0001 — Embeddings: a vector of meaning

**Date:** 2026-06-09
**Lesson:** [0001 — Embeddings: A Vector of Meaning](../lessons/0001-embeddings-vector-of-meaning.html)
**Reference:** [Embeddings Cheat Sheet](../reference/embeddings-cheatsheet.html)
**Mission link:** [MISSION.md](../MISSION.md)

## Context

Senior Python engineer, 15+ years experience, no prior AI/ML background. Has shipped Day 1 (hybrid RAG) and Day 2 (agentic layer) of the rag-blog, but built mostly by recipe. Lesson 0001 is the first in a series that takes them from "shipped by recipe" to "can explain every line".

## Key Concepts Learned

- An embedding is a list of 384 real numbers produced by a neural network encoder.
- The encoder is trained so that semantically similar sentences produce vectors that point in similar directions.
- Cosine similarity is the standard "how similar" function for normalized text vectors; with normalization, it collapses to a dot product.
- The single constant `MODEL_NAME = "all-MiniLM-L6-v2"` parameterizes the entire embedding pipeline in our codebase.
- `all-MiniLM-L6-v2` is a distilled 6-layer transformer encoder, ~80 MB, runs on CPU, outputs 384-d unit vectors.
- The four lines in `rag_pipeline.py` worth knowing: `SentenceTransformer(...)`, `encode(..., normalize_embeddings=True)`, `encode([text], ...)[0].tolist()`, and `LanceDB.search(...).metric("cosine")`.

## Non-Obvious Insights

- Embeddings can be thought of as "hash functions from meaning to a fixed-size float array, where collision is replaced by closeness." This framing helps demystify the system.
- The architecture choice (6 layers, mean pooling, 384-d) is the *entire* model — there's no separate "understanding" step. Token vectors get contextualized by attention across the sentence, then averaged.
- `normalize_embeddings=True` matters more than it looks: it lets us swap cosine similarity for a fast dot product at search time.

## Code Anchors

| Concept | File · Line |
| --- | --- |
| Model choice | `config.py` · `MODEL_NAME = "all-MiniLM-L6-v2"` |
| Model load | `rag_pipeline.py` · `Embedder.__init__` |
| Batch encode | `rag_pipeline.py` · `Embedder.embed` |
| Single encode | `rag_pipeline.py` · `Embedder.embed_one` |
| Search call | `rag_pipeline.py` · `VectorStore.vector_search` |

## Confusions Cleared

- "What's actually coming out of `embed_one`?" → a 384-element list of floats, unit length, no further structure.
- "Why cosine and not Euclidean?" → with normalized vectors, cosine is a dot product (faster), and it ignores magnitude (which carries no semantic meaning for our use).

## Open Questions Going Forward

- How does LanceDB's exact vector search scale past 10k+ chunks? (Lesson 2 territory: ANN vs exact)
- When does cosine similarity fail? (Lesson 3: BM25 fills the gap)
- How is the encoder actually trained? (Later lesson: contrastive learning / SBERT training objective)

## Misconceptions to Watch For

- That embeddings "understand" the text. They don't — they encode statistical regularities from training data.
- That two different embedding models produce comparable vectors. They don't.
- That `dimension` is a tunable parameter of the model. It is fixed by the model architecture; changing it means a different model.

## What the User Could Now Do

- [x] Explain in their own words what `embed_one` returns and why it's a 384-d unit vector.
- [x] Point to the line in `rag_pipeline.py` that produces the vector.
- [x] Run a cosine similarity between two embeddings and interpret the score.
- [ ] Swap the model in `MODEL_NAME` and re-ingest (next: hands-on "what breaks" exercise).
- [ ] Try a larger model (e.g. `all-mpnet-base-v2`, 768-d) and benchmark recall.
