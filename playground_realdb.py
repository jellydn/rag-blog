"""Lesson 0003 supplement — cosine + BM25 scores on the real rag-blog index.

This script treats a real blog chunk as a "ground-truth" query, embeds it,
and reports the cosine score distribution across all 83 chunks.

It also runs the same query through BM25 so we can see both score scales
side by side — exactly the scale-mismatch problem RRF solves in Lesson 0004.
"""
import json

import numpy as np

from rag_pipeline import BM25Index, Embedder, VectorStore

print("Loading models + index...")
embedder = Embedder()
vstore = VectorStore("data/lancedb", embedder.dimension)

with open("data/lancedb/bm25_data.json", encoding="utf-8") as f:
    bm25 = BM25Index.from_json_dict(json.load(f))

# Pull all chunks into memory by re-embedding the BM25-stored contents
# (BM25 stored the same chunk content that was embedded, so the result is
# byte-identical to what's in LanceDB; this avoids the pylance dependency.)
print("Loading chunk vectors into memory (re-embedding from BM25 contents)...")
contents = bm25.doc_texts
titles = []
for cid in bm25.doc_ids:
    meta = bm25.chunk_meta.get(cid, {})
    titles.append(meta.get("title", cid))
doc_ids = list(bm25.doc_ids)
vecs = np.array(embedder.embed(contents), dtype="float32")
print(f"  {len(doc_ids)} chunks loaded, vector matrix {vecs.shape}\n")


def cosine_distribution(query_vec: np.ndarray) -> np.ndarray:
    """Return the full N-element cosine distribution for one query."""
    # Vectors are pre-normalized → cosine = dot product
    return (vecs @ query_vec).astype("float32")


def describe(name: str, scores: np.ndarray) -> None:
    print(f"  {name:18s}  min={scores.min():.3f}  median={float(np.median(scores)):.3f}"
          f"  mean={scores.mean():.3f}  p90={float(np.percentile(scores, 90)):.3f}"
          f"  max={scores.max():.3f}")


# ---- 1) Self-test: pick a real chunk, use it as the query -----------------
# This is the "0.799-equivalent" experiment — what does a near-perfect match
# look like on production data?
print("=" * 70)
print("PART 1 — Self-test: chunk N as query, ranked against all 83 chunks")
print("=" * 70)

# Pick a chunk that has substantial content (not the homepage stub)
sample_indices = [i for i, c in enumerate(contents) if len(c) > 200][:5]
for idx in sample_indices:
    print(f"\n--- Query = chunk {idx}: {titles[idx][:60]!r} ---")
    q = vecs[idx]
    sims = cosine_distribution(q)
    describe("cosine (vs all)", sims)

    # Top 5 (excluding self at rank 0 with score ~1.0)
    top = np.argsort(sims)[::-1]
    print(f"  Top 5 (excluding self):")
    shown = 0
    for j in top:
        if j == idx:
            print(f"    [self] {sims[j]:.3f}  {doc_ids[j]:25s}  {titles[j][:55]}")
            continue
        shown += 1
        print(f"    {shown}. {sims[j]:.3f}  {doc_ids[j]:25s}  {titles[j][:55]}")
        if shown >= 5:
            break


# ---- 2) Natural-language query against the real index --------------------
print("\n" + "=" * 70)
print("PART 2 — Real query: 'how to set up neovim folding'")
print("=" * 70)
q_text = "how to set up neovim folding"
q_vec = embedder.embed_one(q_text)
q_arr = np.array(q_vec, dtype="float32")
sims = cosine_distribution(q_arr)
describe("cosine (vs all)", sims)

# Top 5 by cosine
print("\n  Top 5 by cosine:")
top = np.argsort(sims)[::-1][:5]
for k, j in enumerate(top, start=1):
    print(f"    {k}. {sims[j]:.3f}  {doc_ids[j]:25s}  {titles[j][:60]}")
    snippet = contents[j][:120].replace("\n", " ")
    print(f"        {snippet}...")

# ---- 3) Same query through BM25 ------------------------------------------
print("\n  Top 5 by BM25 (raw score, NOT normalized to [0,1]):")
bm25_results = bm25.search(q_text, top_k=5)
for k, r in enumerate(bm25_results, start=1):
    title_line = r["content"].split("\n")[0].lstrip("#").strip()
    print(f"    {k}. {r['score']:6.2f}  {r['doc_id']:25s}  {title_line[:60]}")

# ---- 4) Side-by-side score scale comparison ------------------------------
print("\n" + "=" * 70)
print("PART 3 — Score scale comparison (the RRF motivation)")
print("=" * 70)
print(f"  Cosine range observed:  [{sims.min():.3f}, {sims.max():.3f}]  (bounded 0..1)")
if bm25_results:
    bm_scores = [r["score"] for r in bm25_results]
    print(f"  BM25    range observed:  [{min(bm_scores):.2f}, {max(bm_scores):.2f}]  (unbounded positive)")
    print("\n  → These are NOT directly addable. RRF sidesteps this by using ranks, not scores.")
    print("    That's the whole point of Lesson 0004.")
