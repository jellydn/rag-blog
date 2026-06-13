# Learning Record 0003 — BM25: Why Keyword Search Still Matters

**Date:** 2026-06-09
**Lesson:** [0003 — BM25: Why Keyword Search Still Matters](../lessons/0003-bm25-keyword-search-still-matters.html)
**Reference:** [BM25 Cheat Sheet](../reference/bm25-cheatsheet.html)

## Context

Lesson 0001 introduced embeddings and the cosine similarity score. Lesson 0002 covered vector search at scale. Lesson 0003 closes the retrieval circle by adding the other half: keyword/lexical search via BM25. The "fold" polysemy test from Lesson 0001 is the running example — embeddings do a soft 0.43 match, BM25 does a hard lexical match, and the two are complementary.

## Key Concepts Learned

- BM25 is a hand-rolled scoring function with three components: term frequency (with saturation), inverse document frequency, and length normalization.
- The two hyperparameters: `k1` (TF saturation rate) and `b` (length normalization strength). `delta` is a BM25+ extension giving a small floor.
- The default formula uses `k1=1.5`, `b=0.75` — empirically tuned on TREC, still the standard 30 years later.
- Our implementation is BM25+ (with `delta=1.0`), not plain BM25. This gives a small non-zero contribution to any document containing a query term.
- The tokenizer is regex-based: word boundary, letter-prefixed, length 2–51, lowercase, allows `#` for identifiers like `C#`.
- BM25 wins on exact technical terms, code identifiers, brand names, error codes. Embeddings win on paraphrases and semantic queries. They are complementary, not competing.
- The "magic triangle" of search: recall, speed, memory. BM25 has very different tradeoffs than vector search — no model, no GPU, simpler storage.

## Non-Obvious Insights

- The `+δ` in BM25+ sounds tiny but has a real effect: it ensures that no document with a query term ever scores exactly 0, which matters when you sum across query terms. Without it, a single missing term kills the document's contribution.
- IDF can be negative in older formulations; the `+1` inside the log (Lucene/BM25+ style) clamps it to ≥ 0. Our code does this.
- BM25 query time is O(|q| · N · L) — quadratic-ish because we rescan every doc counting TF. For our 83 chunks, this is microseconds. For 100k+ chunks, you'd want a real inverted index (tantivy, Lucene) where cost is O(|q| · matches).
- We don't stem and don't remove stop words, but we *don't need to*: the IDF term handles stop words gracefully (low score for common words), and skipping stemming trades recall for predictability. Both are defensible choices.

## Code Anchors

| Concept | File · Symbol |
| --- | --- |
| Hyperparameters | `rag_pipeline.py` · `BM25Index.__init__` |
| Tokenizer | `rag_pipeline.py` · `BM25Index._tokenize` |
| Index build (DF) | `rag_pipeline.py` · `BM25Index.add_documents` |
| Search (the formula) | `rag_pipeline.py` · `BM25Index.search` |
| Persistence | `rag_pipeline.py` · `to_json_dict` / `from_json_dict` |
| Hybrid use | `rag_pipeline.py` · `HybridSearch.search` (next lesson) |

## Confusions Cleared

- "Why is BM25 still used if we have embeddings?" → Embeddings are great at paraphrase but bad at exact technical terms. BM25 is the opposite. Fusing them gives both.
- "What's the difference between TF and DF?" → TF = per-document occurrence count. DF = how many documents contain the term. IDF uses DF; the per-document score uses TF.
- "Why doesn't our tokenizer remove 'the' and 'is'?" → The IDF term downweights them automatically. No list to maintain.
- "What's delta doing?" → It's a small floor that prevents any matching document from getting a 0 score for that term. Useful when queries have multiple terms.

## Open Questions Going Forward

- How does the score normalization work between BM25 and vector cosine scores? (BM25 scores can be 0–20+, cosine is 0–1.) → Lesson 0004: RRF sidesteps this entirely.
- When should we add stemming? → When recall on morphological queries (e.g. "running" vs "ran") is measurably low. For TIL/blog content with mostly English technical terms, not urgent.
- Should we move to a real inverted index (tantivy) at scale? → Yes, but only when the O(|q| · N · L) cost starts to dominate. Probably at 100k+ chunks.

## Misconceptions to Watch For

- That BM25 is "old" or "obsolete" because of transformers. It isn't. Most production search (Elasticsearch, OpenSearch, Algolia) uses BM25 or BM25+ as the default lexical baseline.
- That delta=0 (plain BM25) is "more correct" than delta>0 (BM25+). They are different algorithms with different recall profiles; neither is "correct" in general.
- That the score magnitudes are comparable between BM25 and cosine. They are not — BM25 can score 20+ for a strong match, cosine is bounded to 1. RRF avoids this problem.

## What the User Could Now Do

- [x] Explain the BM25 formula in their own words, including what `k1` and `b` do.
- [x] Identify which lines of `BM25Index.search` correspond to TF, IDF, length normalization, and the BM25+ delta.
- [x] Predict when a query will score 0 under BM25 (when none of the query terms are in the corpus).
- [x] Decide whether stemming would help the rag-blog corpus (probably not, for TIL content).
- [ ] Run `playground_bm25.py` and compare BM25 vs vector top-3 results.
- [ ] Tune `k1` and `b` on a small held-out set and observe the change in ranking.
- [ ] Add a stemming step (e.g. via `nltk.PorterStemmer`) and measure recall change.
