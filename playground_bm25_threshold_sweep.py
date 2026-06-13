#!/usr/bin/env python3
"""Empirical sweep of BM25_THRESHOLD across the 4 canary queries.

Goal: pick a corpus-specific floor by observing, for each threshold:
  - how many BM25 hits are dropped from the candidate set
  - of those dropped, how many are the TARGET (false negative) vs
    the html2pdf false positive (true negative) vs noise
  - how the final RRF top-5 shifts (especially the target Neovim chunk)
  - whether the documented "set()" false positive (Lesson 0003) is caught

The target chunk we want to track is the actual Neovim folding article:
  - chunk 0 (the prose intro)
  - chunk 1 (the code + setup)
"""

from pathlib import Path

from rag_pipeline import create_hybrid_search

THRESHOLDS = [0.0, 1.0, 3.0, 5.0, 10.0]

QUERIES = [
    ("Q1_easy", "how to set up neovim folding"),
    ("Q2_paraphrase", "how to enable folding in vim"),
    ("Q3_abstract", "collapse code blocks in my editor"),
    ("Q4_adversarial", "best pizza in naples italy"),
]


# Derive chunk IDs from filenames so the script fails loudly (FileNotFoundError)
# on corpus drift instead of silently reporting "target_in_top5=N" for a slug
# rename. The Neovim article splits into 2 chunks at our 512-char ceiling;
# html2pdf is the Lesson 0003 false positive.
def _chunk_ids(slug: str, n: int) -> set[str]:
    stem = Path(f"data/content/{slug}.md").stem  # raises FileNotFoundError on drift
    return {f"{stem}:{i}" for i in range(n)}


TARGET_CHUNKS = _chunk_ids("til-40-how-to-set-up-folding-in-neovim", 2)
HTML2PDF_CHUNK = next(iter(_chunk_ids("til-35-fix-blank-image-with-html2pdf", 1)))


def render_top5(results, label):
    lines = [f"  {label}"]
    if not results:
        lines.append("    (empty)")
        return lines
    for i, r in enumerate(results, 1):
        cid = r.get("id", r.get("doc_id", "?"))
        rrf = r.get("rrf_score", 0.0)
        vr = r.get("vector_rank")
        br = r.get("bm25_rank")
        vs = r.get("vector_score", 0.0)
        marker = ""
        if cid in TARGET_CHUNKS:
            marker = "  <-- TARGET"
        elif cid == HTML2PDF_CHUNK:
            marker = "  <-- html2pdf false positive"
        lines.append(
            f"    {i}. {cid[:55]:55s}  rrf={rrf:.5f}  "
            f"vec_r={vr!s:>5}  bm25_r={br!s:>5}  cos={vs:.3f}{marker}"
        )
    return lines


def main():
    print("=" * 72)
    print("BM25_THRESHOLD sweep across 4 canary queries")
    print("=" * 72)
    print(f"  TARGET_CHUNKS: {TARGET_CHUNKS}")
    print(f"  HTML2PDF_CHUNK: {HTML2PDF_CHUNK}")

    h = create_hybrid_search()

    # Per-query state across thresholds, so we can emit a diff at the end
    summary_rows = []

    for q_label, q in QUERIES:
        print(f"\n--- {q_label}: {q!r} ---")
        # Compute the pre-threshold raw BM25 candidate set once per query —
        # the threshold is a pure filter on this list, so the post-threshold
        # outcome is just "what got cut". This lets us attribute every drop
        # to a specific chunk id and see if TARGET / HTML2PDF were axed.
        raw = h.bm25.search(q, top_k=20)
        for thr in THRESHOLDS:
            h.bm25_threshold = float(thr)
            results, timing = h.search(q, top_k=5)
            # Note: the search() pipeline also folds in vector hits,
            # so a chunk can survive in the RRF list even if BM25 dropped it.
            # The relevant quantity for the threshold alone is the BM25-only
            # drops — chunks whose raw BM25 hit fell below the floor.
            bm25_only_drops = {r["doc_id"] for r in raw if r["score"] < thr}
            target_dropped = bool(TARGET_CHUNKS & bm25_only_drops)
            html2pdf_dropped = HTML2PDF_CHUNK in bm25_only_drops
            target_seen = any((r.get("id") in TARGET_CHUNKS) for r in results)
            html2pdf_seen = any((r.get("id") == HTML2PDF_CHUNK) for r in results)
            top_id = results[0]["id"] if results else None
            top_cos = timing["top_cosine"]
            conf = timing["confidence"]
            print(
                f"  thr={thr:>5.1f}  bm25_drop={timing['bm25_dropped_threshold']:>2d}  "
                f"target_in_top5={'Y' if target_seen else 'N':1s}  "
                f"html2pdf_in_top5={'Y' if html2pdf_seen else 'N':1s}  "
                f"target_AXED={'Y' if target_dropped else 'N':1s}  "
                f"fp_AXED={'Y' if html2pdf_dropped else 'N':1s}  "
                f"top={top_id[:38] if top_id else 'None':38s}  "
                f"conf={conf:6s} top_cos={top_cos}"
            )
            for line in render_top5(results, f"top-5 at thr={thr}"):
                print(line)
            summary_rows.append(
                {
                    "query": q_label,
                    "threshold": thr,
                    "bm25_dropped": timing["bm25_dropped_threshold"],
                    "target_in_top5": target_seen,
                    "html2pdf_in_top5": html2pdf_seen,
                    "target_axed": target_dropped,
                    "fp_axed": html2pdf_dropped,
                    "top_id": top_id,
                    "rrf_top_score": results[0]["rrf_score"] if results else 0.0,
                    "confidence": conf,
                    "top_cosine": top_cos,
                }
            )

    # Final aggregate: for each (query, threshold), one row.
    print("\n" + "=" * 72)
    print("SUMMARY — what the floor actually does")
    print("=" * 72)
    print(
        f"{'query':<18s} {'thr':>5s} {'drop':>4s} "
        f"{'tgt':>3s} {'fp':>3s} {'tgt↓':>4s} {'fp↓':>4s} "
        f"{'conf':>6s} {'top_cos':>9s}  top_id"
    )
    for row in summary_rows:
        print(
            f"{row['query']:<18s} {row['threshold']:>5.1f} {row['bm25_dropped']:>4d} "
            f"{'Y' if row['target_in_top5'] else 'N':>3s} "
            f"{'Y' if row['html2pdf_in_top5'] else 'N':>3s} "
            f"{'Y' if row['target_axed'] else 'N':>4s} "
            f"{'Y' if row['fp_axed'] else 'N':>4s} "
            f"{row['confidence']:>6s} {row['top_cosine']:>9.4f}  "
            f"{(row['top_id'] or 'None')[:40]}"
        )

    # Notes on the html2pdf false positive: it's only ever a real hit on Q1
    # and Q2, where "set" / "up" appear in the .set({...}) snippet. Confirm
    # by checking the raw BM25 score for that chunk on each query — we
    # re-run BM25 directly so we can show the score the threshold is
    # filtering on (the playground doesn't currently surface it).
    print("\n" + "=" * 72)
    print("BONUS — raw BM25 score of the html2pdf chunk (the false positive)")
    print("=" * 72)
    for q_label, q in QUERIES:
        raw = h.bm25.search(q, top_k=50)
        for r in raw:
            if r["doc_id"] == HTML2PDF_CHUNK:
                print(f"  {q_label}: html2pdf raw score = {r['score']:.4f}")
                break
        else:
            print(f"  {q_label}: html2pdf not in top-50 BM25 hits")

    print("\nDone.")


if __name__ == "__main__":
    main()
