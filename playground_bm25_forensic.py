"""Lesson 0003 forensic supplement — why did BM25 hit html2pdf at 7.49?

This is the BM25-false-positive autopsy: we take the actual query, the actual
chunk that scored 7.49, and the actual scoring function from `rag_pipeline.py`,
and we show *which query words* produced that score.

Key learning: BM25 is sum-decomposable. The total score for a (query, doc) pair
is just the sum of per-term contributions, where each term's contribution is
  idf(term) * ( tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl)) + delta )

If a query term has df=0 in the corpus, it contributes 0. So when BM25
returns a chunk, it's because at least one query word appears in it — and
the more *unique* (high-IDF) the word is, the more it dominates.
"""
import json
import re

import numpy as np

from rag_pipeline import BM25Index

# --- load -------------------------------------------------------------------
with open("data/lancedb/bm25_data.json", encoding="utf-8") as f:
    bm25 = BM25Index.from_json_dict(json.load(f))

query = "how to set up neovim folding"
q_tokens = bm25._tokenize(query)
print(f"Query:                {query!r}")
print(f"Query tokens (BM25):  {q_tokens}")
print()

# Find the html2pdf chunk 0
target_id = "til-35-fix-blank-image-with-html2pdf:0"
target_idx = bm25.doc_ids.index(target_id)
target_text = bm25.doc_texts[target_idx]
target_toks = bm25._doc_tokens[target_idx]
dl = bm25.doc_lengths[target_idx]
n = bm25.total_docs
avgdl = bm25.avg_doc_length

print(f"Target chunk:  {target_id}")
print(f"  doc_length  = {dl} tokens")
print(f"  avgdl (corpus) = {avgdl:.1f} tokens")
print(f"  k1={bm25.k1}  b={bm25.b}  delta={bm25.delta}")
print()

# --- per-token breakdown ---------------------------------------------------
k1, b, delta = bm25.k1, bm25.b, bm25.delta
total_score = 0.0
print(f"{'q-token':10s} {'tf':>4s} {'df':>5s} {'idf':>7s} {'term_contrib':>14s}  in doc?")
print("-" * 70)
for qt in q_tokens:
    tf = target_toks.count(qt)
    df = bm25.doc_freqs.get(qt, 0)
    if df == 0 or tf == 0:
        print(f"{qt:10s} {tf:>4d} {df:>5d} {'-':>7s} {0.0:>14.4f}  "
              f"{'YES' if tf else 'no (df=0)'}")
        continue
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
    denom = tf + k1 * (1 - b + b * dl / avgdl)
    tf_norm = tf * (k1 + 1) / denom
    contrib = idf * (tf_norm + delta)
    total_score += contrib
    print(f"{qt:10s} {tf:>4d} {df:>5d} {idf:>7.3f} {contrib:>14.4f}  yes")

print("-" * 70)
print(f"{'SUM':10s} {'':>4s} {'':>5s} {'':>7s} {total_score:>14.4f}")
print(f"\nReported score in playground_realdb.py: ~7.49")
print(f"Decomposed score here:                  {total_score:.4f}")
print()

# --- show the actual lexical overlap --------------------------------------
print("=" * 70)
print("The actual text of the html2pdf chunk 0:")
print("=" * 70)
print(target_text)
print()
print("=" * 70)
print("Query words that appear in this chunk:")
print("=" * 70)
for qt in q_tokens:
    if qt in target_toks:
        # show the line where the token appears (case-insensitive)
        matches = re.findall(rf"(\b\w*{qt}\w*\b)", target_text, flags=re.IGNORECASE)
        print(f"  {qt!r:8s}  →  appears {target_toks.count(qt)}×  "
              f"(variants in text: {matches[:3]})")

# --- for comparison, do the same for the true hit -------------------------
print()
print("=" * 70)
print("For comparison — the TRUE BM25 top hit, and its per-token breakdown:")
print("=" * 70)
results = bm25.search(query, top_k=3)
for r in results:
    if r["doc_id"] == target_id:
        continue
    print(f"\n  Hit: {r['doc_id']}  (BM25 score: {r['score']:.2f})")
    print(f"  Title: {r['content'].splitlines()[0].lstrip('#').strip()}")
    cid = r["doc_id"]
    if cid not in bm25.doc_ids:
        continue
    idx = bm25.doc_ids.index(cid)
    tok_list = bm25._doc_tokens[idx]
    dl_i = bm25.doc_lengths[idx]
    for qt in q_tokens:
        tf = tok_list.count(qt)
        df = bm25.doc_freqs.get(qt, 0)
        if tf == 0 or df == 0:
            continue
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf + k1 * (1 - b + b * dl_i / avgdl)
        tf_norm = tf * (k1 + 1) / denom
        contrib = idf * (tf_norm + delta)
        print(f"    {qt:10s}  tf={tf:>2d}  df={df:>2d}  idf={idf:>6.3f}  "
              f"contrib={contrib:>7.3f}")
