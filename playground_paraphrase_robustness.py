"""Lesson 0003 robustness test — paraphrases should still find the right chunk.

Three queries, scored on the same 83-chunk rag-blog index:
  Q1. "how to set up neovim folding"          — keyword-rich (control)
  Q2. "how to enable folding in vim"          — partial overlap (vim ≠ neovim)
  Q3. "collapse code blocks in my editor"     — zero lexical overlap with chunk

For each query, we report:
  - BM25 top-3 (and the BM25 score of the Neovim-folding chunk, if any)
  - Cosine top-3 (and the cosine score of the Neovim-folding chunk, if any)

The robustness signal: does the Neovim-folding article (`til-40-...-neovim`)
stay in the top-1 as we make the query more abstract?
"""
import json

import numpy as np

from rag_pipeline import BM25Index, Embedder

# --- load -------------------------------------------------------------------
print("Loading models + index...")
embedder = Embedder()
with open("data/lancedb/bm25_data.json", encoding="utf-8") as f:
    bm25 = BM25Index.from_json_dict(json.load(f))
contents = bm25.doc_texts
doc_ids = list(bm25.doc_ids)
vecs = np.array(embedder.embed(contents), dtype="float32")
print(f"  {len(doc_ids)} chunks loaded, vector matrix {vecs.shape}\n")

# The "ground truth" chunk we want to surface
TARGET_ID = "til-40-how-to-set-up-folding-in-neovim:0"
TARGET_IDX = doc_ids.index(TARGET_ID)
target_title = bm25.doc_texts[TARGET_IDX].splitlines()[0].lstrip("#").strip()

QUERIES = [
    "how to set up neovim folding",
    "how to enable folding in vim",
    "collapse code blocks in my editor",
]


def cosine_search(q_text: str, top_k: int = 5) -> list[dict]:
    q = np.array(embedder.embed_one(q_text), dtype="float32")
    sims = vecs @ q
    top = np.argsort(sims)[::-1][:top_k]
    return [(doc_ids[i], float(sims[i])) for i in top]


def bm25_search(q_text: str, top_k: int = 5) -> list[dict]:
    results = bm25.search(q_text, top_k=top_k)
    return [(r["doc_id"], r["score"]) for r in results]


def find_target_rank(rank_list: list[tuple[str, float]]) -> tuple[int | None, float | None]:
    for rank, (cid, score) in enumerate(rank_list, start=1):
        if cid == TARGET_ID:
            return rank, score
    return None, None


# --- main loop -------------------------------------------------------------
print("=" * 70)
print("Target chunk (the one we want to surface):")
print(f"  {TARGET_ID}")
print(f"  Title: {target_title}")
print("=" * 70)

for q in QUERIES:
    print()
    print("─" * 70)
    print(f"  Q: {q!r}")
    print("─" * 70)
    q_tokens = bm25._tokenize(q)
    print(f"  BM25 tokens: {q_tokens}")
    print()

    # --- BM25 ---
    bm25_top = bm25_search(q, top_k=5)
    print("  BM25 top-5:")
    if not bm25_top:
        print("    (no hits — every query word has df=0 in this corpus)")
    else:
        for rank, (cid, score) in enumerate(bm25_top, start=1):
            title = bm25.doc_texts[bm25.doc_ids.index(cid)].splitlines()[0].lstrip("#").strip()
            marker = "  ← TARGET" if cid == TARGET_ID else ""
            print(f"    {rank}. {score:6.2f}  {cid:50s}  {title[:40]}{marker}")
    bm25_target_rank, bm25_target_score = find_target_rank(bm25_top)

    # --- Cosine ---
    cos_top = cosine_search(q, top_k=5)
    print("\n  Cosine top-5:")
    for rank, (cid, score) in enumerate(cos_top, start=1):
        title = bm25.doc_texts[bm25.doc_ids.index(cid)].splitlines()[0].lstrip("#").strip()
        marker = "  ← TARGET" if cid == TARGET_ID else ""
        print(f"    {rank}. {score:6.3f}  {cid:50s}  {title[:40]}{marker}")
    cos_target_rank, cos_target_score = find_target_rank(cos_top)

    # --- Verdict ---
    print()
    print(f"  → TARGET rank in BM25:    {bm25_target_rank}  (score: {bm25_target_score})")
    print(f"  → TARGET rank in cosine:  {cos_target_rank}  (score: {cos_target_score})")
    if bm25_target_rank == 1 and cos_target_rank == 1:
        print("  → Robustness: STABLE on both methods. ✓")
    elif cos_target_rank == 1 and bm25_target_rank is None:
        print("  → Robustness: COSINE rescues a BM25 miss. The vector 'gets' the paraphrase.")
    elif cos_target_rank == 1 and bm25_target_rank and bm25_target_rank > 1:
        print("  → Robustness: BM25 partial miss, cosine wins.")
    elif cos_target_rank is None:
        print("  → Robustness: BOTH methods miss — chunk is below the fold for this query.")
    else:
        print("  → Robustness: see numbers above.")

print()
print("=" * 70)
print("Summary: how does the Neovim-folding chunk rank across paraphrases?")
print("=" * 70)
print(f"  {'query':<45s} {'BM25 rank':>10s} {'cosine rank':>12s}")
for q in QUERIES:
    bm25_top = bm25_search(q, top_k=83)
    cos_top = cosine_search(q, top_k=83)
    br, _ = find_target_rank(bm25_top)
    cr, _ = find_target_rank(cos_top)
    print(f"  {q:<45s} {str(br):>10s} {str(cr):>12s}")
