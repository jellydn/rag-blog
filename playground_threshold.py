"""Lesson 0004 supplement — see the effect of a cosine-similarity floor.

For four queries (3 paraphrases from Lesson 0004 + 1 adversarial no-match),
compare the production HybridSearch top-5 with `COSINE_THRESHOLD=0.0`
(current behavior) vs `COSINE_THRESHOLD=0.3` (production hardening).

The threshold drops vector hits the encoder considered essentially unrelated
to the query. The chunk can still surface via BM25 (with only the BM25
contribution to its RRF score), but it loses the 0.7*w boost from cosine.
"""
import json

from rag_pipeline import BM25Index, Embedder, VectorStore
from config import COSINE_THRESHOLD

QUERIES = [
    "how to set up neovim folding",
    "how to enable folding in vim",
    "collapse code blocks in my editor",
    # Adversarial: zero lexical overlap, semantically unrelated to the corpus
    "best pizza in naples italy",
]
THRESHOLDS = [0.0, 0.3]


def make_hybrid(threshold: float):
    """Build a fresh HybridSearch with a specific cosine threshold."""
    embedder = Embedder()
    vs = VectorStore("data/lancedb", embedder.dimension)
    with open("data/lancedb/bm25_data.json", encoding="utf-8") as f:
        bm25 = BM25Index.from_json_dict(json.load(f))
    # Wire it up the same way create_hybrid_search does, with the threshold.
    from rag_pipeline import HybridSearch
    return HybridSearch(vs, bm25, embedder, cosine_threshold=threshold)


def fmt_chunk(c: dict) -> str:
    cid = c.get("id") or c.get("doc_id", "?")
    title = c.get("title", "")[:45]
    vrank = c.get("vector_rank")
    brank = c.get("bm25_rank")
    vs = c.get("vector_score", 0)
    rrf = c.get("rrf_score", 0)
    return f"  rrf={rrf:.4f}  vec#{vrank or '—':>3}(sim={vs:.3f})  bm25#{brank or '—':>3}  {cid[:40]:40s}  {title}"


print("=" * 78)
print("Cosine-threshold playground — comparing COSINE_THRESHOLD=0.0 vs 0.3")
print(f"  Default in config.py: {COSINE_THRESHOLD}  (0.0 = off, backward compatible)")
print("=" * 78)

results_by_query = {}
for q in QUERIES:
    print()
    print("─" * 78)
    print(f"  Q: {q!r}")
    print("─" * 78)

    for th in THRESHOLDS:
        hs = make_hybrid(th)
        results, timing = hs.search(q, top_k=5)
        results_by_query[(q, th)] = (results, timing)

        print(f"\n  ── COSINE_THRESHOLD = {th} ──")
        print(f"     vec_dropped_by_threshold: {timing.get('vec_dropped_threshold', 0)}")
        if not results:
            print("     (no results — both rankers returned nothing)")
            continue
        for i, c in enumerate(results, start=1):
            print(f"     {i}." + fmt_chunk(c))

    # Side-by-side diff
    r0, t0 = results_by_query[(q, 0.0)]
    r1, t1 = results_by_query[(q, 0.3)]
    ids0 = [c.get("id") or c.get("doc_id") for c in r0]
    ids1 = [c.get("id") or c.get("doc_id") for c in r1]
    dropped = [c for c in ids0 if c not in ids1]
    promoted = [c for c in ids1 if c not in ids0]
    print()
    print(f"  DIFF (0.0 → 0.3):")
    if dropped:
        print(f"    dropped from top-5: {dropped}")
    else:
        print(f"    dropped from top-5: (none)")
    if promoted:
        print(f"    promoted into top-5: {promoted}")
    else:
        print(f"    promoted into top-5: (none)")
    if not dropped and not promoted:
        print(f"    → identical top-5. Threshold had no effect on this query.")
    else:
        print(f"    → top-5 changed.")

print()
print("=" * 78)
print("Per-query score-stats on the underlying cosine distribution")
print("=" * 78)
print("  (this is the distribution the threshold carves into)")
import numpy as np

embedder = Embedder()
with open("data/lancedb/bm25_data.json", encoding="utf-8") as f:
    bm25 = BM25Index.from_json_dict(json.load(f))
contents = bm25.doc_texts
vecs = np.array(embedder.embed(contents), dtype="float32")

for q in QUERIES:
    qv = np.array(embedder.embed_one(q), dtype="float32")
    sims = vecs @ qv
    above = (sims >= 0.3).sum()
    print(f"  {q!r}")
    print(f"    cosine:  min={sims.min():.3f}  median={float(np.median(sims)):.3f}"
          f"  mean={sims.mean():.3f}  max={sims.max():.3f}")
    print(f"    chunks with cosine ≥ 0.3: {above}/{len(sims)} ({100*above/len(sims):.0f}%)")
