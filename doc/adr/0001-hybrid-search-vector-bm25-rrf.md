# 1. Hybrid search with vector + BM25 fused by RRF

Date: 2026-06-06

## Status

Accepted

## Context

Blog and TIL content on productsway.com mixes narrative prose with exact technical tokens (CLI flags, package names, `C#`, commit commands). Dense retrieval alone often misses keyword-critical passages; keyword-only search misses paraphrases and conceptual matches. We need a retrieval layer suitable for a small corpus on CPU without tuning-heavy rankers.

## Decision

Use **dual retrieval** on every query:

1. **Vector search** — `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, cosine similarity in LanceDB.
2. **BM25+** — in-memory index over chunk text (k1=1.5, b=0.75, delta=1.0) with a shared tokenizer (`\b[a-zA-Z][a-zA-Z0-9#]{1,50}\b`).

Fuse ranked lists with **Reciprocal Rank Fusion (RRF)**, k=60, weights **70% vector / 30% BM25** (`config.py`). No cross-encoder reranker in Day 1.

## Consequences

### Positive

- Better recall on technical TILs than vector-only.
- RRF avoids calibrating incompatible score scales between BM25 and cosine distance.
- Weights and k are centralized and easy to adjust.

### Negative

- Two indexes must stay aligned on chunk set and identity (see ADR-0003).
- Heavier query path than single-mode search; BM25 rebuild is O(docs × query terms) without further optimization beyond token caching.