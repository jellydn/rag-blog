# 0004 — Reciprocal Rank Fusion (RRF): the tiebreaker

**Date:** 2026-06-09
**Status:** Active
**Mission anchor:** MISSION.md → "Understand the whole stack"

## Concept

Reciprocal Rank Fusion is a 2009 algorithm (Cormack, Clarke, Buettcher, SIGIR)
that combines multiple ranked retrieval lists into a single ranking. The formula
is:

```
RRF(d) = Σᵢ  wᵢ / (k + rankᵢ(d))
```

Where `i` indexes rankers, `rankᵢ(d)` is the 1-indexed position of `d` in
ranker `i`'s list, `k` is a smoothing constant (we use 60), and `wᵢ` is a
weight per ranker. Missing-from-list = 0 contribution.

## Why ranks, not scores

Vector scores live in [0, 1]. BM25 scores can be any positive number (we
saw 7.49 to 29.98 on real data). You cannot add them, min-max them, or
z-score them without losing information. Ranks are a universally
comparable unit — "rank 1 = best" in every retrieval system, regardless
of underlying score. RRF sidesteps the scale problem by using only the
ordinal position.

## Code anchors

- `config.py:6-9` — the four magic numbers: `RRF_K=60`, `VECTOR_WEIGHT=0.7`, `BM25_WEIGHT=0.3`
- `rag_pipeline.py` → `HybridSearch.search` — the actual fusion loop
  - Walks `vec_results` first, creates dict entries with `rrf_score = VECTOR_WEIGHT / (RRF_K + rank + 1)`
  - Walks `bm25_results` second, either adds to existing entry or starts a new one
  - Sorts by `rrf_score`, takes top-k
  - The `top_k * 2` fudge factor on the inner retrieval exists to ensure RRF can rescue a chunk that was rank 7 in one list and rank 1 in the other

## Non-obvious insights

### 1. Ranks compress more gracefully than scores
At `k=60`, ranks 1-5 contribute 94-100% of the leader. After rank 50,
contributions are 55%. The curve is "graceful" — you can swap in any
ranker and the top of the fused list barely changes.

### 2. The `+1` in the code is a 0→1 index shift
Code uses 0-indexed `enumerate()`, so `rank + 1` makes it 1-indexed.
This is easy to get wrong if you copy the formula from the paper.

### 3. Missing-from-list is implicit, not explicit
There's no `if d not in rank_list` check. The dict structure handles it:
if a chunk never appears in `vec_results`, it never gets a vector
contribution, and when BM25 later adds its contribution, the `cid not
in scores` branch creates a new entry with only the BM25 contribution.
The "missing" case is encoded in the data flow, not the control flow.

### 4. `top_k * 2` is critical
If you fetched only `top_k` from each ranker, the rescue pattern is
impossible: a chunk that's rank 11 in cosine but rank 1 in BM25 would
be excluded from the cosine list entirely. The 2× fudge factor is what
makes RRF worth doing.

### 5. 70/30 is empirical, not theoretical
For a well-tuned dense encoder on a domain corpus, vector search beats
BM25 on top-1 precision most of the time. But BM25 rescues rare-term
queries (product names, error codes), polysemy failures, and queries
where the user's wording is lexically close to a chunk's wording.
70/30 is "trust the semantic ranking first, let lexical break ties."

### 6. The "rescue pattern" is the whole point
In the paraphrase test (Q3 = "collapse code blocks in my editor"),
the target was cosine #3, BM25 #16. RRF: target gets a full vector
contribution (0.7/63 = 0.01111) and a tiny BM25 contribution (0.3/77
= 0.00390 because the chunk does appear in the BM25 list, just at
rank 16). Total 0.01501 — the chunk climbs into the top 3. Without
RRF, you'd have to pick one ranker and lose the other.

## Things I used to think were true but aren't

- "RRF and weighted RRF are different algorithms." Nope, weighted RRF
  is just RRF with non-uniform `wᵢ`. Our code is weighted RRF.
- "k=60 is mathematically optimal." It's empirically robust across many
  corpora, not optimal. Don't tune it before tuning your rankers.
- "You need a re-ranker on top of RRF." Useful but not required. RRF
  is the production default in Elasticsearch 8.8+, OpenSearch, Vespa,
  and Weaviate. It works at scale.

## Open questions / next steps

- **k and weights are untested on our data.** A 5-query eval would be
  enough to see if k=60 is the right choice for *our* 83-chunk corpus
  or if we should drop to k=20.
- **What's the failure surface?** When does RRF actually hurt? We saw
  one rescue case (Q3). Need to find a case where it actively
  demotes the correct answer.
- **Could we add a third ranker?** A code-aware ranker (regex on
  backtick blocks) might rescue queries like "vim foldcolumn setting"
  that neither cosine nor BM25 hit well.

## Sources used

- Cormack, Clarke, Buettcher 2009 — "Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning Methods" (SIGIR)
- Elasticsearch RRF docs (8.8+)
- OpenSearch Score Ranker Processor docs
- Our own `playground_paraphrase_robustness.py` (Q3 motivation)
