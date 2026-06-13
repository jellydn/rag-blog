"""Unit tests for the BM25_THRESHOLD feature in HybridSearch.search.

These tests use mock BM25Index, mock VectorStore, and mock Embedder — no
sentence-transformers model is loaded, no LanceDB / BM25 files are read.
The goal is to verify the threshold filter behavior in isolation:

  1. score >= floor  → kept
  2. score < floor   → dropped
  3. empty BM25 list → no error, dropped=0
  4. threshold=0.0   → no filtering (backward-compat default)
  5. dropped BM25 hit does NOT surface via vector when no vector match
"""

import unittest
from unittest.mock import MagicMock

from rag_pipeline import HybridSearch


def _make_vec_hit(chunk_id: str, doc_id: str, distance: float) -> dict:
    """Build a vector_search() result dict in the exact shape our pipeline expects."""
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "title": doc_id.replace("-", " ").title(),
        "content": f"content for {chunk_id}",
        "source_url": f"https://example.com/notes/{doc_id}",
        "category": "TIL",
        "chunk_index": 0,
        "total_chunks": 1,
        "score": distance,  # cosine *distance* per LanceDB convention
    }


def _make_bm25_hit(chunk_id: str, doc_id: str, score: float) -> dict:
    """Build a bm25.search() result dict in the exact shape our pipeline expects."""
    return {
        "doc_id": chunk_id,  # bm25 doc_ids are chunk ids (see add_documents)
        "content": f"content for {chunk_id}",
        "score": score,
    }


def _make_hybrid(
    bm25_hits: list[dict],
    bm25_threshold: float = 1.0,
    vector_hits: list[dict] | None = None,
) -> HybridSearch:
    """Wire up a HybridSearch with all three dependencies mocked.

    `bm25.chunk_meta` is needed because the pipeline calls
    `self.bm25.chunk_meta.get(cid, {})` when a BM25 hit has no vector
    counterpart (see `hit_from_bm25_only`). We populate it from the
    BM25 hits so that path can resolve meta.
    """
    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = [0.0] * 4  # 4-dim fake vector

    mock_vector_store = MagicMock()
    mock_vector_store.vector_search.return_value = vector_hits or []

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = bm25_hits
    mock_bm25.chunk_meta = {
        h["doc_id"]: {
            "doc_id": h["doc_id"].split(":")[0],
            "title": h["doc_id"].split(":")[0].replace("-", " ").title(),
            "source_url": "https://example.com/notes/x",
            "category": "TIL",
            "chunk_index": 0,
            "total_chunks": 1,
        }
        for h in bm25_hits
    }

    return HybridSearch(
        vector_store=mock_vector_store,
        bm25_index=mock_bm25,
        embedder=mock_embedder,
        bm25_threshold=bm25_threshold,
    )


class TestBM25ThresholdFilter(unittest.TestCase):
    def test_kept_when_score_at_or_above_floor(self):
        # Contract: score >= floor is kept (non-strict `>=`); score < floor
        # is dropped. The boundary case is `score = floor` (here 1.0) — it
        # must be kept. The below-floor hit (`score = 0.5`) is a regression
        # guard: without it the test would pass even if the entire filter
        # block was deleted, because the kept hits' bm25_rank assertions
        # only prove the BM25 path ran, not that the filter ran.
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 10.0),
            _make_bm25_hit("b:0", "b", 5.0),
            _make_bm25_hit("c:0", "c", 1.0),  # exactly at the floor
            _make_bm25_hit("dropped:0", "dropped", 0.5),  # below floor
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0)
        results, timing = h.search("anything", top_k=5)

        self.assertEqual(timing["bm25_dropped_threshold"], 1)
        result_ids = {r["id"] for r in results}
        self.assertEqual(result_ids, {"a:0", "b:0", "c:0"})
        self.assertNotIn("dropped:0", result_ids)
        # Tighten: each KEPT result must have a bm25_rank set (BM25 path
        # fired) and vector_rank None (came from BM25, not vector).
        for r in results:
            self.assertIsNotNone(
                r.get("bm25_rank"),
                f"{r['id']} missing bm25_rank — BM25 path did not fire",
            )
            self.assertIsNone(
                r.get("vector_rank"),
                f"{r['id']} has vector_rank but no vector mock was provided",
            )
        # Ranks must be 1, 2, 3 in the order the kept hits were returned
        by_rank = sorted(results, key=lambda r: r["bm25_rank"])
        self.assertEqual([r["id"] for r in by_rank], ["a:0", "b:0", "c:0"])

    def test_dropped_when_score_below_floor(self):
        # Scores: 10.0 (kept), 2.0 (kept), 0.5 (dropped), 0.0001 (dropped)
        # Floor = 1.0 → dropped=2. We use 0.0001 instead of 0.0 because
        # production `BM25Index.search` filters out score=0.0 hits before
        # returning, so the pipeline never sees a literal 0.0.
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 10.0),
            _make_bm25_hit("b:0", "b", 2.0),
            _make_bm25_hit("c:0", "c", 0.5),
            _make_bm25_hit("d:0", "d", 0.0001),
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0)
        results, timing = h.search("anything", top_k=5)

        self.assertEqual(timing["bm25_dropped_threshold"], 2)
        result_ids = {r["id"] for r in results}
        self.assertEqual(result_ids, {"a:0", "b:0"})
        # The dropped chunks must NOT appear anywhere
        self.assertNotIn("c:0", result_ids)
        self.assertNotIn("d:0", result_ids)

    def test_empty_bm25_results_does_not_error(self):
        h = _make_hybrid(bm25_hits=[], bm25_threshold=1.0)
        # Should not raise — empty list short-circuits the filter block
        results, timing = h.search("anything", top_k=5)

        self.assertEqual(timing["bm25_dropped_threshold"], 0)
        self.assertEqual(results, [])

    def test_threshold_zero_disables_filter(self):
        # 0.0 floor: every non-negative score passes
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 10.0),
            _make_bm25_hit("b:0", "b", 0.5),
            _make_bm25_hit("c:0", "c", 0.0001),  # tiny positive — passes
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=0.0)
        results, timing = h.search("anything", top_k=5)

        self.assertEqual(timing["bm25_dropped_threshold"], 0)
        result_ids = {r["id"] for r in results}
        self.assertEqual(result_ids, {"a:0", "b:0", "c:0"})
        # Tighten: each result must have come via the BM25 path. If
        # threshold=0.0 was broken and the filter dropped everything,
        # the chunks wouldn't appear at all.
        for r in results:
            self.assertIsNotNone(r.get("bm25_rank"))

    def test_dropped_bm25_hit_does_not_surface_via_vector(self):
        """A chunk that's BM25-only AND below the floor must be truly gone.

        Catches a regression where the filter is applied but the chunk
        still surfaces via some other path. We give the chunk NO vector
        result at all, so the only way it could appear is through BM25
        — and BM25 dropped it. If it shows up in the final list, the
        filter is broken.
        """
        bm25_hits = [
            _make_bm25_hit("kept:0", "kept", 10.0),  # above floor
            _make_bm25_hit("dropped:0", "dropped", 0.5),  # below floor
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0, vector_hits=[])
        results, timing = h.search("anything", top_k=5)

        self.assertEqual(timing["bm25_dropped_threshold"], 1)
        self.assertFalse(timing["bm25_threshold_suppressed"])
        result_ids = {r["id"] for r in results}
        self.assertIn("kept:0", result_ids)
        self.assertNotIn("dropped:0", result_ids)

    def test_adaptive_suppression_when_top_hit_below_floor(self):
        """If the *top* BM25 hit itself is below the floor, suppress the filter.

        Mirrors the cosine-threshold adaptive behavior. The contract: when
        the corpus has no real lexical match for the query (top BM25 score
        is itself below the floor), keep every BM25 hit so vector search +
        RRF can still do useful work. Without this, an adversarial query
        like "best pizza in naples italy" against a Neovim-folding corpus
        would wipe the BM25 list clean.
        """
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 0.5),  # top — below floor of 1.0
            _make_bm25_hit("b:0", "b", 0.3),
            _make_bm25_hit("c:0", "c", 0.1),
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0)
        results, timing = h.search("anything", top_k=5)

        # Suppression fired — nothing was dropped from the BM25 list
        self.assertTrue(timing["bm25_threshold_suppressed"])
        self.assertEqual(timing["bm25_dropped_threshold"], 0)
        # All 3 hits should surface (proves the filter was suppressed,
        # not just that the flag was set)
        result_ids = {r["id"] for r in results}
        self.assertEqual(result_ids, {"a:0", "b:0", "c:0"})
        for r in results:
            self.assertIsNotNone(
                r.get("bm25_rank"),
                f"{r['id']} missing bm25_rank — did not come via BM25 path",
            )

    def test_no_suppression_when_top_hit_at_floor(self):
        """Boundary: top hit AT the floor does not trigger suppression.

        Non-strict `>=` — a top score exactly equal to the floor means
        there *is* a real match, so the filter runs normally. Test 1
        owns the 'at-floor is kept' contract; this test owns only the
        suppression-flag assertion.
        """
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 1.0),  # top — exactly at floor
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0)
        _, timing = h.search("anything", top_k=5)

        # The contract under test: at-floor top hit does NOT trigger suppression.
        self.assertFalse(timing["bm25_threshold_suppressed"])

    def test_mixed_top_above_floor_drops_only_below(self):
        """Asymmetric: top hit above floor, others below.

        The most common production case: a query has a real lexical
        match (top score above the floor) but also returns some weak
        noise hits (below the floor). The contract: top above floor
        means there IS a real match, so the adaptive override does NOT
        fire — but the normal `>=` filter still drops the below-floor
        hits. This pins down the asymmetric case the other tests don't
        cover: top drives the suppression decision, the filter handles
        the rest.
        """
        bm25_hits = [
            _make_bm25_hit("a:0", "a", 5.0),  # top — above floor
            _make_bm25_hit("b:0", "b", 0.8),  # below floor
            _make_bm25_hit("c:0", "c", 0.3),  # below floor
        ]
        h = _make_hybrid(bm25_hits, bm25_threshold=1.0)
        results, timing = h.search("anything", top_k=5)

        # Top above floor → no adaptive suppression
        self.assertFalse(timing["bm25_threshold_suppressed"])
        # The normal filter ran and dropped the 2 below-floor hits
        self.assertEqual(timing["bm25_dropped_threshold"], 2)
        # Only the 5.0 hit survives (the equality assertion implies
        # the below-floor hits are absent — no need for explicit
        # assertNotIn lines).
        result_ids = {r["id"] for r in results}
        self.assertEqual(result_ids, {"a:0"})


if __name__ == "__main__":
    unittest.main()
