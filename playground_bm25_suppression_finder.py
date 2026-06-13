#!/usr/bin/env python3
"""Improved empirical search for a real query that fires BM25 suppression.

Key insight from the previous iteration + reviewer feedback:

Key insight from the previous iteration + reviewer feedback:
  With delta=1.0, a single-term match contributes 2 * IDF to a chunk's score.
  For the top hit to score < 1.0, we need IDF < 0.5, which means df > ~50
  in an 83-chunk corpus. That's STOPWORD territory.

Strategy (in order):
  1. Pre-screen candidates by df using bm25.doc_freqs — only run queries
     whose tokens all have df >= MIN_DF (default 40).
  2. Try single stopword-grade words first ("in", "is", "with", ...).
  3. Try 2-token stopword phrases ("in this", "is a", ...).
  4. If no winners at BM25_THRESHOLD=1.0, drop the floor to 0.5 and retry.
  5. For comparison, log the known canary (pizza) and known good (neovim).
"""

import re

from rag_pipeline import create_hybrid_search

MIN_DF = 60  # with df=60, IDF ~0.33, per-term at tf=2 ~0.80 → reliably below 1.0 floor

# Single stopword-grade tokens (2+ chars to pass the tokenizer)
STOPWORDS_1 = [
    "in", "is", "on", "to", "of", "at", "by", "as", "an", "be", "do",
    "go", "he", "if", "it", "me", "my", "no", "or", "so", "up", "us",
    "we", "the", "and", "but", "for", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "are", "has", "his", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "got", "let",
    "say", "she", "too", "use", "this", "that", "with", "from", "have",
    "just", "know", "take", "into", "year", "your", "good", "them",
]

# 2-token stopword phrases
STOPWORDS_2 = [
    "in this", "is a", "is the", "to the", "of the", "in the", "for the",
    "on the", "to be", "it is", "that is", "this is", "as a", "as the",
    "with the", "from the", "by the", "at the", "if you", "if we",
    "you can", "we can", "you can use", "how to", "how do", "how can",
    "what is", "what are", "where is", "when to",
]

# For comparison / context
REFERENCE = [
    ("known_good_neovim", "neovim"),       # df moderate, IDF high → score >> 1
    ("known_canary_collapse", "collapse code blocks in my editor"),  # all terms high df
    ("known_adversarial_pizza", "best pizza in naples italy"),  # returns []
]


def tokenize_for_check(text: str) -> list[str]:
    """Mirror BM25Index._tokenize so we can pre-screen candidates by df."""
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9#]{1,50}\b", text.lower())


def main():
    print("=" * 78)
    print("Improved suppression finder — pre-screen by df, focus on stopwords")
    print("=" * 78)
    print(f"  MIN_DF = {MIN_DF}  (tokens below this are skipped — they give IDF > 0.5)")
    print("  Default floor BM25_THRESHOLD = 1.0; will retry at 0.5 if no winners")
    print()

    h = create_hybrid_search()

    # Diagnostic: top-10 highest-df tokens. Uses h.bm25.max_tf and
    # h.bm25.search directly (NOT try_query) — we don't need the full
    # RRF pipeline just to display per-token stats. Saves 10 wasted
    # RRF invocations.
    print("--- Top-10 highest-df tokens (diagnostic) ---")
    df_pairs = sorted(h.bm25.doc_freqs.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print(
        f"{'token':<14s} {'df':>4s}  {'top_bm25':>9s}  {'max_tf':>6s}  {'would_fire?':>11s}"
    )
    print("-" * 60)
    for token, df in df_pairs:
        max_tf = h.bm25.max_tf(token)
        raw = h.bm25.search(token, top_k=1)
        top_bm25 = raw[0]["score"] if raw else 0.0
        would_fire = top_bm25 < 1.0
        print(
            f"{token:<14s} {df:>4d}  {top_bm25:>9.4f}  {max_tf:>6d}  "
            f"{'YES' if would_fire else 'no':>11s}"
        )
    print()

    # Pre-screen: only run queries whose tokens all have df >= MIN_DF
    def all_tokens_have_high_df(q: str) -> bool:
        tokens = tokenize_for_check(q)
        if not tokens:
            return False
        return all(h.bm25.doc_freqs.get(t, 0) >= MIN_DF for t in tokens)

    def get_df(q: str) -> str:
        tokens = tokenize_for_check(q)
        return " ".join(f"{t}={h.bm25.doc_freqs.get(t, 0)}" for t in tokens)

    def try_query(q: str, floor: float) -> tuple[float, int, bool, dict, int]:
        """Returns (top_bm25, n_hits, fires, timing, max_tf).

        max_tf is the highest TF of any query token across the corpus —
        used to explain why a high-df candidate still doesn't fire
        (e.g. "to" has df=51 but a chunk contains "to" 3 times, pushing
        the top score to 1.4 instead of the predicted 1.02).
        """
        h.bm25_threshold = float(floor)
        results, timing = h.search(q, top_k=5)
        raw = h.bm25.search(q, top_k=20)
        top_bm25 = raw[0]["score"] if raw else 0.0
        n_hits = len(raw)
        fires = (
            top_bm25 > 0
            and top_bm25 < floor
            and timing["bm25_threshold_suppressed"]
        )
        # max_tf across the corpus for any of the query tokens
        tokens = tokenize_for_check(q)
        max_tf = max((h.bm25.max_tf(tok) for tok in tokens), default=0)
        return top_bm25, n_hits, fires, timing, max_tf

    # Phase 1: single stopwords at floor=1.0
    print("--- Phase 1: single stopwords at BM25_THRESHOLD=1.0 ---")
    print(f"  (skipping tokens with df < {MIN_DF})")
    print()
    print(f"{'token':<14s} {'df':>4s}  {'top_bm25':>9s}  {'max_tf':>6s}  {'#hits':>5s}  {'fires?':>6s}")
    print("-" * 60)
    winners_phase1 = []
    for token in STOPWORDS_1:
        df = h.bm25.doc_freqs.get(token, 0)
        if df < MIN_DF:
            continue
        top_bm25, n_hits, fires, timing, max_tf = try_query(token, 1.0)
        marker = "  <-- FIRES" if fires else ""
        print(f"{token:<14s} {df:>4d}  {top_bm25:>9.4f}  {max_tf:>6d}  {n_hits:>5d}  {'YES' if fires else 'no':>6s}{marker}")
        if fires:
            winners_phase1.append((token, top_bm25, n_hits, timing, max_tf))

    # Phase 2: 2-token stopwords at floor=1.0
    print()
    print("--- Phase 2: 2-token stopword phrases at BM25_THRESHOLD=1.0 ---")
    print()
    print(f"{'phrase':<24s} {'df':>16s}  {'top_bm25':>9s}  {'max_tf':>6s}  {'#hits':>5s}  {'fires?':>6s}")
    print("-" * 80)
    winners_phase2 = []
    for phrase in STOPWORDS_2:
        if not all_tokens_have_high_df(phrase):
            continue
        top_bm25, n_hits, fires, timing, max_tf = try_query(phrase, 1.0)
        marker = "  <-- FIRES" if fires else ""
        df_str = get_df(phrase)
        print(f"{phrase:<24s} {df_str:>16s}  {top_bm25:>9.4f}  {max_tf:>6d}  {n_hits:>5d}  {'YES' if fires else 'no':>6s}{marker}")
        if fires:
            winners_phase2.append((phrase, top_bm25, n_hits, timing, max_tf))

    # (Phase 3 removed: dropping the floor tightens the winner band, not expands it.
    # Suppression condition is `top_bm25 < floor`; lower floor = stricter condition.
    # Phase 1/2 at floor=1.0 already captures the full (0, 1.0) band.)
    winners = winners_phase1 + winners_phase2

    # Reset floor
    h.bm25_threshold = 1.0

    # Reference comparisons
    print()
    print("--- Reference (for context) ---")
    print(f"{'label':<32s} {'query':<42s} {'top_bm25':>9s}  {'#hits':>5s}  {'suppressed':>10s}")
    print("-" * 110)
    for label, q in REFERENCE:
        h.bm25_threshold = 1.0
        raw = h.bm25.search(q, top_k=20)
        top_bm25 = raw[0]["score"] if raw else 0.0
        _, timing = h.search(q, top_k=5)
        print(
            f"{label:<32s} {q!r:<42s} {top_bm25:>9.4f}  {len(raw):>5d}  "
            f"{'YES' if timing['bm25_threshold_suppressed'] else 'no':>10s}"
        )

    # Summary
    print()
    print("=" * 78)
    if winners:
        print(f"WINNERS ({len(winners)} queries fire BM25 adaptive suppression):")
        print("=" * 78)
        for entry in winners:
            q, top_bm25, n_hits, timing, max_tf = entry
            print(f"\n  Query:          {q!r}")
            print(f"  Top BM25 score: {top_bm25:.4f}  (below the 1.0 floor)")
            print(f"  Max TF:         {max_tf}  (highest occurrence count of any query token)")
            print(f"  # BM25 hits:    {n_hits}    (non-empty, so check runs)")
            print(f"  bm25_threshold_suppressed: {timing['bm25_threshold_suppressed']}")
            print(f"  confidence:                {timing['confidence']}")
            print(f"  top_cosine:                {timing['top_cosine']}")
            # Show the top-5 RRF result
            h.bm25_threshold = 1.0
            results, _ = h.search(q, top_k=5)
            print("  Top-5 RRF results:")
            for i, r in enumerate(results, 1):
                print(
                    f"    {i}. {r.get('id', r.get('doc_id', '?'))[:50]:50s}  "
                    f"rrf={r.get('rrf_score', 0.0):.5f}  "
                    f"vec_r={r.get('vector_rank')!s:>5}  "
                    f"bm25_r={r.get('bm25_rank')!s:>5}"
                )
    else:
        print("NO WINNERS found.")
        print()
        print("Why: the corpus's max-df token is in the ~50 range, and the math")
        print("requires df >= 60 (so IDF < ~0.33) for single-token matches to land")
        print("below the 1.0 floor. The Neovim-folding corpus is too small AND too")
        print("repetition-heavy (long articles repeat common words 5-7x in a single")
        print("chunk, pushing top_bm25 above 1.0 even for the most common tokens).")
        print()
        print("Production insight: BM25 adaptive suppression is a future-proofing")
        print("feature. On this 83-chunk corpus, the per-term contribution floor")
        print("(delta=1.0) is too generous for the math to ever fire. At ~10k+ chunks")
        print("with broader vocabulary, more terms will have df > 60 and the")
        print("suppression will start firing on truly off-topic queries.")
        print()
        print("To see the suppression fire today, drop BM25_THRESHOLD to 0.5 in")
        print("config.py and re-run any 'collapse code blocks in my editor'-style")
        print("query where the top BM25 score is between 0 and 0.5.")
    print()


if __name__ == "__main__":
    main()
