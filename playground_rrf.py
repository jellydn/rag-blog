"""Lesson 0004 playground — RRF on the real rag-blog index.

For three queries, this script:
  1. Runs the query through the production HybridSearch
  2. Shows the per-list rank lists (cosine + BM25) for the TARGET chunk
  3. Computes the RRF contributions by hand, per ranker
  4. Compares the by-hand RRF to HybridSearch's actual RRF score
  5. Shows the final fused top-5

The TARGET is the Neovim-folding article. We expect it to be #1 in cosine
and #1 in BM25 for query 1, and progressively harder for queries 2 and 3.
"""

from rag_pipeline import create_hybrid_search

TARGET_ID = "til-40-how-to-set-up-folding-in-neovim:0"

QUERIES = [
    "how to set up neovim folding",
    "how to enable folding in vim",
    "collapse code blocks in my editor",
]

RRF_K = 60
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def per_rank_contrib(rank_1indexed: int, weight: float) -> float:
    """Pure RRF math: weight / (k + rank)."""
    return weight / (RRF_K + rank_1indexed)


def find_target_rank(rank_list: list[dict], id_key: str) -> tuple[int | None, float]:
    for r, hit in enumerate(rank_list, start=1):
        if hit[id_key] == TARGET_ID:
            return r, hit.get("score", 0)
    return None, 0.0


def find_target_in_results(results: list[dict]) -> tuple[int | None, dict | None]:
    for r, hit in enumerate(results, start=1):
        if hit.get("id") == TARGET_ID or hit.get("doc_id") == TARGET_ID:
            return r, hit
    return None, None


# --- main ------------------------------------------------------------------
print("=" * 78)
print("Loading production HybridSearch (this is the SAME code /query uses)...")
print("=" * 78)
hs = create_hybrid_search()
print(f"  vector store rows: {hs.vector_store.count()}")
print(f"  BM25 docs:         {hs.bm25.total_docs}")
print(f"  embedder dim:      {hs.embedder.dimension}")
print()

for q in QUERIES:
    print("=" * 78)
    print(f"  Q: {q!r}")
    print("=" * 78)

    # ---- 1. Get the two raw rank lists (top_k * 2) -----------------------
    q_vec = hs.embedder.embed_one(q)
    vec_list = hs.vector_store.vector_search(q_vec, top_k=20)
    bm25_list = hs.bm25.search(q, top_k=20)

    vec_rank, vec_score = find_target_rank(vec_list, "id")
    bm25_rank, bm25_score = find_target_rank(bm25_list, "doc_id")

    print(f"\n  TARGET ({TARGET_ID}) raw ranks:")
    print(f"    cosine rank: {vec_rank}  (raw score {vec_score:.3f})")
    print(f"    BM25 rank:   {bm25_rank}  (raw score {bm25_score:.2f})")

    # ---- 2. By-hand RRF computation --------------------------------------
    rrf_vec = per_rank_contrib(vec_rank, VECTOR_WEIGHT) if vec_rank else 0.0
    rrf_bm25 = per_rank_contrib(bm25_rank, BM25_WEIGHT) if bm25_rank else 0.0
    rrf_total = rrf_vec + rrf_bm25

    print(f"\n  RRF math (by hand, k={RRF_K}):")
    print(f"    vector contribution: {VECTOR_WEIGHT:>4}/({RRF_K} + {vec_rank or '—':>4}) = {rrf_vec:.6f}")
    print(f"    BM25   contribution: {BM25_WEIGHT:>4}/({RRF_K} + {bm25_rank or '—':>4}) = {rrf_bm25:.6f}")
    print(f"    RRF total (by hand):  {rrf_total:.6f}")

    # ---- 3. Run through real HybridSearch and verify --------------------
    results, timing = hs.search(q, top_k=5)
    target_pos, target_hit = find_target_in_results(results)

    print(f"\n  HybridSearch.search() returned (top {len(results)}):")
    for r, hit in enumerate(results, start=1):
        marker = "  ← TARGET" if (target_pos == r) else ""
        cid = hit.get("id") or hit.get("doc_id")
        title = hit.get("title", "")[:50]
        vrank = hit.get("vector_rank")
        brank = hit.get("bm25_rank")
        rrf = hit.get("rrf_score", 0)
        print(f"    {r}. rrf={rrf:.5f}  vec#{vrank or '—':>3}  bm25#{brank or '—':>3}  "
              f"{cid[:45]:45s}  {title}{marker}")

    print(f"\n  HybridSearch TARGET position: {target_pos}  "
          f"(rrf_score={target_hit.get('rrf_score', 0):.6f} if in top-5)")

    # ---- 4. Verify by-hand matches HybridSearch -------------------------
    if target_hit is not None:
        actual = target_hit["rrf_score"]
        delta = abs(actual - rrf_total)
        match = "✓ matches" if delta < 1e-4 else f"✗ MISMATCH (delta={delta})"
        print(f"\n  By-hand RRF {rrf_total:.6f}  vs  HybridSearch RRF {actual:.6f}  →  {match}")
    else:
        print("\n  TARGET not in final top-5 for this query.")

    # ---- 5. Verdict -------------------------------------------------------
    print()
    if target_pos == 1 and vec_rank == 1 and bm25_rank == 1:
        print("  → Easy case. Both rankers agree, RRF trivially wins.")
    elif target_pos == 1 and vec_rank == 1 and (bm25_rank is None or bm25_rank > 5):
        print("  → Cosine dominates. RRF mostly reflects cosine ordering.")
    elif target_pos == 1 and bm25_rank == 1 and (vec_rank is None or vec_rank > 5):
        print("  → BM25 dominates. RRF mostly reflects BM25 ordering.")
    elif target_pos and target_pos <= 3:
        print("  → Rescue pattern: RRF combined partial signals from both lists.")
    elif target_pos is None or target_pos > 3:
        print("  → RRF didn't help. Both rankers missed the TARGET meaningfully.")
    print(f"  → Timing: {timing}")

    print()


# --- summary table ---------------------------------------------------------
print("=" * 78)
print("Final summary — where does TARGET land across paraphrases?")
print("=" * 78)
print(f"  {'query':<45s} {'cosine':>8s} {'BM25':>8s} {'RRF':>8s}  {'top-5?':>7s}")
for q in QUERIES:
    results, _ = hs.search(q, top_k=5)
    q_vec = hs.embedder.embed_one(q)
    vec_list = hs.vector_store.vector_search(q_vec, top_k=83)
    bm25_list = hs.bm25.search(q, top_k=83)
    vec_rank, _ = find_target_rank(vec_list, "id")
    bm25_rank, _ = find_target_rank(bm25_list, "doc_id")
    rrf_rank, _ = find_target_in_results(results)
    in_top5 = "yes" if rrf_rank else "no"
    print(f"  {q:<45s} {str(vec_rank or '—'):>8s} {str(bm25_rank or '—'):>8s} "
          f"{str(rrf_rank or '—'):>8s}  {in_top5:>7s}")
