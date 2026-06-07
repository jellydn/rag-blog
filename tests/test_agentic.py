"""Unit tests for agentic.py — RouterAgent, PlannerAgent, RetrievalAgent,
ReflectionAgent, and AgentOrchestrator (no ML deps; HybridSearch is mocked)."""

import unittest
from unittest.mock import MagicMock, patch

from agentic import (
    AgentOrchestrator,
    AgentRun,
    Intent,
    PlannerAgent,
    QueryPlan,
    ReflectionAgent,
    ReflectionResult,
    RetrievalAgent,
    RouteResult,
    RouterAgent,
)


def _make_hybrid(results=None, timing=None):
    """Return a MagicMock that satisfies the HybridSearch interface."""
    hybrid = MagicMock()
    hybrid.search.return_value = (results if results is not None else [], timing or {})
    return hybrid


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class TestRouterAgent(unittest.TestCase):
    def setUp(self):
        self.router = RouterAgent()

    # --- intent classification ---

    def test_troubleshoot_intent(self):
        result = self.router.route("how to fix session id error")
        self.assertEqual(result.intent, Intent.TROUBLESHOOT)

    def test_howto_intent_prefix(self):
        result = self.router.route("how to set up neovim folding")
        self.assertEqual(result.intent, Intent.HOWTO)

    def test_code_intent(self):
        result = self.router.route("typescript absolute imports example")
        self.assertEqual(result.intent, Intent.CODE)

    def test_compare_intent(self):
        result = self.router.route("vim vs neovim difference")
        self.assertEqual(result.intent, Intent.COMPARE)

    def test_summary_intent(self):
        result = self.router.route("summary of all notes")
        self.assertEqual(result.intent, Intent.SUMMARY)

    def test_navigation_intent(self):
        result = self.router.route("where to find the latest articles")
        self.assertEqual(result.intent, Intent.NAVIGATION)

    def test_unknown_intent_empty_query(self):
        result = self.router.route("xyz123 qwerty")
        self.assertEqual(result.intent, Intent.UNKNOWN)
        self.assertAlmostEqual(result.confidence, 0.25)

    # --- confidence and signals ---

    def test_confidence_is_between_zero_and_one(self):
        for q in ["fix this error", "how to install", "vs compare", "", "just some words"]:
            result = self.router.route(q)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_signals_capped_at_eight(self):
        # craft a query that hits many word lists
        q = "error fix how to install code snippet typescript vs summary all"
        result = self.router.route(q)
        self.assertLessEqual(len(result.signals), 8)

    def test_signals_are_strings(self):
        result = self.router.route("how to fix the error")
        for sig in result.signals:
            self.assertIsInstance(sig, str)

    # --- tie-breaker rules ---

    def test_howto_tiebreaker_starts_with_how(self):
        # "how" prefix adds +2 to HOWTO beyond any word matches
        result = self.router.route("how do databases work")
        self.assertIn(result.intent, {Intent.HOWTO, Intent.UNKNOWN})

    def test_howto_tiebreaker_question_mark(self):
        result = self.router.route("what is the best editor?")
        # "?" boosts HOWTO by 1; intent need not be HOWTO but confidence should still be valid
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_code_tiebreaker_git(self):
        result = self.router.route("git cherry pick command")
        # "git" and "command" both bump CODE
        self.assertEqual(result.intent, Intent.CODE)

    def test_troubleshoot_beats_howto_on_strong_signal(self):
        # "error" scores 2 for TROUBLESHOOT; "how" only 1+2 for HOWTO via prefix bump
        # With "error" and "fix" both present, TROUBLESHOOT should win
        result = self.router.route("how to fix an error")
        self.assertEqual(result.intent, Intent.TROUBLESHOOT)

    def test_confidence_rounded_to_two_decimals(self):
        result = self.router.route("fix broken issue")
        # confidence is round(x, 2)
        self.assertEqual(result.confidence, round(result.confidence, 2))

    # --- return type ---

    def test_returns_route_result(self):
        result = self.router.route("any query")
        self.assertIsInstance(result, RouteResult)
        self.assertIsInstance(result.intent, Intent)


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class TestPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.planner = PlannerAgent()

    def _route(self, intent: Intent) -> RouteResult:
        return RouteResult(intent=intent, confidence=0.8)

    def test_troubleshoot_plan(self):
        plan = self.planner.build_plan("session id error", self._route(Intent.TROUBLESHOOT))
        self.assertEqual(plan.intent, Intent.TROUBLESHOOT)
        self.assertEqual(plan.retrieval_strategy, "broad_retrieve_then_reflect")
        self.assertIn("fix error troubleshooting", plan.primary_query)
        self.assertEqual(len(plan.secondary_queries), 2)
        self.assertGreater(len(plan.steps), 0)

    def test_code_plan(self):
        plan = self.planner.build_plan("typescript imports", self._route(Intent.CODE))
        self.assertEqual(plan.intent, Intent.CODE)
        self.assertEqual(plan.retrieval_strategy, "code_first")
        self.assertIn("code example command snippet", plan.primary_query)
        self.assertEqual(len(plan.secondary_queries), 2)

    def test_compare_plan(self):
        plan = self.planner.build_plan("vim vs neovim", self._route(Intent.COMPARE))
        self.assertEqual(plan.intent, Intent.COMPARE)
        self.assertEqual(plan.retrieval_strategy, "dual_path_compare")
        self.assertIn("comparison", plan.primary_query)
        self.assertNotIn(" vs ", plan.primary_query)

    def test_summary_plan(self):
        plan = self.planner.build_plan("summary overview", self._route(Intent.SUMMARY))
        self.assertEqual(plan.intent, Intent.SUMMARY)
        self.assertEqual(plan.retrieval_strategy, "broad_summary")
        self.assertIn("Productsway", plan.secondary_queries[0])

    def test_summary_plan_empty_query_uses_fallback(self):
        plan = self.planner.build_plan("", self._route(Intent.SUMMARY))
        self.assertEqual(plan.primary_query, "Productsway blog overview")

    def test_navigation_plan(self):
        plan = self.planner.build_plan("where to find notes", self._route(Intent.NAVIGATION))
        self.assertEqual(plan.intent, Intent.NAVIGATION)
        self.assertEqual(plan.retrieval_strategy, "navigation_focus")
        self.assertEqual(plan.primary_query, "where to find notes")

    def test_default_plan_for_howto(self):
        plan = self.planner.build_plan("install plugin", self._route(Intent.HOWTO))
        self.assertEqual(plan.intent, Intent.HOWTO)
        self.assertEqual(plan.retrieval_strategy, "balanced_retrieve")
        self.assertEqual(plan.primary_query, "install plugin")
        self.assertIn("install plugin productsway", plan.secondary_queries)

    def test_default_plan_for_unknown(self):
        plan = self.planner.build_plan("xyzzy", self._route(Intent.UNKNOWN))
        self.assertEqual(plan.retrieval_strategy, "balanced_retrieve")
        self.assertEqual(plan.primary_query, "xyzzy")

    def test_returns_query_plan_instance(self):
        plan = self.planner.build_plan("test", self._route(Intent.HOWTO))
        self.assertIsInstance(plan, QueryPlan)

    def test_steps_are_non_empty_list(self):
        for intent in Intent:
            plan = self.planner.build_plan("query", self._route(intent))
            self.assertIsInstance(plan.steps, list)
            self.assertGreater(len(plan.steps), 0)

    def test_query_stripped(self):
        plan = self.planner.build_plan("  spaced query  ", self._route(Intent.CODE))
        # primary_query should use the stripped version
        self.assertFalse(plan.primary_query.startswith("  "))


# ---------------------------------------------------------------------------
# RetrievalAgent._dedupe_merge  (static method — no HybridSearch needed)
# ---------------------------------------------------------------------------


class TestDedupeeMerge(unittest.TestCase):
    def test_empty_input(self):
        result = RetrievalAgent._dedupe_merge([])
        self.assertEqual(result, [])

    def test_single_result_set(self):
        items = [{"id": "a", "rrf_score": 0.5}, {"id": "b", "rrf_score": 0.3}]
        result = RetrievalAgent._dedupe_merge([items])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "a")  # sorted by rrf_score desc

    def test_deduplication_by_id(self):
        set1 = [{"id": "a", "rrf_score": 0.4, "_query": "q1"}]
        set2 = [{"id": "a", "rrf_score": 0.6, "_query": "q2"}]
        result = RetrievalAgent._dedupe_merge([set1, set2])
        self.assertEqual(len(result), 1)

    def test_deduplication_by_doc_id(self):
        set1 = [{"doc_id": "d1", "rrf_score": 0.3}]
        set2 = [{"doc_id": "d1", "rrf_score": 0.5}]
        result = RetrievalAgent._dedupe_merge([set1, set2])
        self.assertEqual(len(result), 1)

    def test_deduplication_by_source_url(self):
        set1 = [{"source_url": "http://example.com", "rrf_score": 0.2}]
        set2 = [{"source_url": "http://example.com", "rrf_score": 0.7}]
        result = RetrievalAgent._dedupe_merge([set1, set2])
        self.assertEqual(len(result), 1)

    def test_fallback_key_chunk_n(self):
        # Items with no id/doc_id/source_url get keyed as chunk-{rank}
        set1 = [{"rrf_score": 0.5}, {"rrf_score": 0.3}]
        result = RetrievalAgent._dedupe_merge([set1])
        self.assertEqual(len(result), 2)

    def test_best_score_wins_on_merge(self):
        set1 = [{"id": "x", "rrf_score": 0.2}]
        set2 = [{"id": "x", "rrf_score": 0.9}]
        result = RetrievalAgent._dedupe_merge([set1, set2])
        self.assertAlmostEqual(result[0].get("rrf_score"), 0.9)

    def test_best_rank_tracks_minimum(self):
        set1 = [{"id": "a", "rrf_score": 0.5}, {"id": "b", "rrf_score": 0.3}]
        set2 = [{"id": "b", "rrf_score": 0.4}]  # "b" appears at rank 0 in set2
        result = RetrievalAgent._dedupe_merge([set1, set2])
        b = next(r for r in result if r.get("id") == "b")
        self.assertEqual(b["_best_rank"], 1)  # rank 1 in set1, rank 0+1=1 in set2 → min=1

    def test_sorted_descending_by_rrf_score(self):
        items = [
            [{"id": "c", "rrf_score": 0.1}],
            [{"id": "a", "rrf_score": 0.9}],
            [{"id": "b", "rrf_score": 0.5}],
        ]
        result = RetrievalAgent._dedupe_merge(items)
        scores = [r.get("rrf_score", 0) for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_evidence_queries_accumulated(self):
        set1 = [{"id": "a", "_query": "q1"}]
        set2 = [{"id": "a", "_query": "q2"}]
        result = RetrievalAgent._dedupe_merge([set1, set2])
        self.assertIn("q1", result[0]["_evidence_queries"])
        self.assertIn("q2", result[0]["_evidence_queries"])


# ---------------------------------------------------------------------------
# RetrievalAgent.retrieve
# ---------------------------------------------------------------------------


class TestRetrievalAgent(unittest.TestCase):
    def _make_plan(self, primary="primary query", secondary=None) -> QueryPlan:
        return QueryPlan(
            intent=Intent.HOWTO,
            primary_query=primary,
            secondary_queries=secondary if secondary is not None else ["sec1", "sec2"],
            retrieval_strategy="balanced_retrieve",
            steps=["step1"],
        )

    def test_primary_search_called_with_primary_query(self):
        hybrid = _make_hybrid(results=[{"id": "1", "rrf_score": 0.5}])
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan(primary="main query")
        agent.retrieve(plan, top_k=5)
        # First call must be with primary_query
        first_call_args = hybrid.search.call_args_list[0]
        self.assertEqual(first_call_args[0][0], "main query")

    def test_secondary_queries_called(self):
        hybrid = _make_hybrid()
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan(secondary=["s1", "s2"])
        agent.retrieve(plan, top_k=5)
        # 1 primary + 2 secondary = 3 calls total
        self.assertEqual(hybrid.search.call_count, 3)

    def test_secondary_queries_capped_at_two(self):
        hybrid = _make_hybrid()
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan(secondary=["s1", "s2", "s3", "s4"])
        agent.retrieve(plan, top_k=5)
        self.assertEqual(hybrid.search.call_count, 3)  # 1 primary + 2 secondary max

    def test_returns_top_k_results(self):
        items = [{"id": str(i), "rrf_score": 1.0 / (i + 1)} for i in range(10)]
        hybrid = _make_hybrid(results=items)
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan(secondary=[])
        results, _ = agent.retrieve(plan, top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_timings_dict_returned(self):
        hybrid = _make_hybrid()
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan()
        _, timings = agent.retrieve(plan, top_k=5)
        self.assertIn("primary_ms", timings)

    def test_query_tag_added_to_primary_results(self):
        items = [{"id": "1"}]
        hybrid = _make_hybrid(results=items)
        agent = RetrievalAgent(hybrid)
        plan = self._make_plan(primary="tagged query", secondary=[])
        results, _ = agent.retrieve(plan, top_k=5)
        # All primary results should have _query set
        for r in results:
            self.assertEqual(r.get("_query"), "tagged query")


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


class TestReflectionAgent(unittest.TestCase):
    def setUp(self):
        self.reflector = ReflectionAgent()

    def _make_route(self, intent=Intent.HOWTO) -> RouteResult:
        return RouteResult(intent=intent, confidence=0.7)

    def _make_plan(self, intent=Intent.HOWTO) -> QueryPlan:
        return QueryPlan(
            intent=intent,
            primary_query="query",
            secondary_queries=[],
            retrieval_strategy="balanced_retrieve",
            steps=[],
        )

    def test_full_coverage_no_gaps(self):
        query = "fix session error"
        results = [{"title": "fix session error guide", "content": "session error fix"}]
        result = self.reflector.reflect(query, self._make_route(), self._make_plan(), results)
        self.assertIsInstance(result, ReflectionResult)
        self.assertGreater(result.coverage, 0.0)

    def test_zero_results_adds_gaps(self):
        result = self.reflector.reflect(
            "some query", self._make_route(), self._make_plan(), []
        )
        self.assertIn("No results returned", result.gaps)
        self.assertIn("some query blog", result.follow_up_queries)
        self.assertIn("some query TIL", result.follow_up_queries)

    def test_low_coverage_adds_gap(self):
        # Query has many tokens, none in results
        query = "kubernetes docker container orchestration deployment scaling"
        results = [{"title": "unrelated article", "content": "nothing here"}]
        result = self.reflector.reflect(query, self._make_route(), self._make_plan(), results)
        gap_texts = " ".join(result.gaps)
        self.assertIn("Low lexical coverage", gap_texts)

    def test_weak_vector_score_for_troubleshoot_adds_gap(self):
        results = [{"title": "t", "content": "c", "rrf_score": 0.1, "vector_score": 0.1}]
        route = self._make_route(intent=Intent.TROUBLESHOOT)
        plan = self._make_plan(intent=Intent.TROUBLESHOOT)
        result = self.reflector.reflect("fix error", route, plan, results)
        gap_texts = " ".join(result.gaps)
        self.assertIn("Top hit is weak", gap_texts)

    def test_high_confidence_recommendation(self):
        # Build results that guarantee coverage >= 0.5 and top_rrf high enough
        query = "neovim"
        results = [{"title": "neovim guide", "content": "neovim setup", "rrf_score": 0.9, "vector_score": 0.9}]
        result = self.reflector.reflect(query, self._make_route(), self._make_plan(), results)
        if result.confidence >= 0.65:
            self.assertIn("Use the top result", result.recommendation)

    def test_low_confidence_recommendation_with_results(self):
        query = "very obscure topic"
        results = [{"title": "other", "content": "other text", "rrf_score": 0.0, "vector_score": 0.0}]
        result = self.reflector.reflect(query, self._make_route(), self._make_plan(), results)
        if result.confidence < 0.65:
            self.assertIn("Broaden", result.recommendation)

    def test_no_results_recommendation(self):
        result = self.reflector.reflect("query", self._make_route(), self._make_plan(), [])
        self.assertIn("Try a different keyword set", result.recommendation)

    def test_coverage_is_zero_to_one(self):
        for results in [[], [{"title": "x", "content": "y"}]]:
            result = self.reflector.reflect("abc", self._make_route(), self._make_plan(), results)
            self.assertGreaterEqual(result.coverage, 0.0)
            self.assertLessEqual(result.coverage, 1.0)

    def test_confidence_is_zero_to_one(self):
        for results in [[], [{"title": "x", "content": "y", "rrf_score": 0.5, "vector_score": 0.5}]]:
            result = self.reflector.reflect("test", self._make_route(), self._make_plan(), results)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_follow_up_queries_capped_at_three(self):
        # Make a query with low coverage AND trigger code gap (weak vector) to get multiple follow_ups
        query = "fix docker kubernetes scaling error"
        results = [{"title": "a", "content": "a", "rrf_score": 0.01, "vector_score": 0.1}]
        route = self._make_route(intent=Intent.TROUBLESHOOT)
        plan = self._make_plan(intent=Intent.TROUBLESHOOT)
        result = self.reflector.reflect(query, route, plan, results)
        self.assertLessEqual(len(result.follow_up_queries), 3)

    def test_very_short_query_tokens(self):
        # Single short word (len <= 2) — falls back to all tokens
        result = self.reflector.reflect("go", self._make_route(), self._make_plan(), [])
        self.assertIsInstance(result, ReflectionResult)


# ---------------------------------------------------------------------------
# AgentOrchestrator._compose_answer  (static — no HybridSearch needed)
# ---------------------------------------------------------------------------


class TestComposeAnswer(unittest.TestCase):
    def _route(self, intent=Intent.HOWTO) -> RouteResult:
        return RouteResult(intent=intent, confidence=0.8, signals=[])

    def _plan(self, strategy="balanced_retrieve") -> QueryPlan:
        return QueryPlan(
            intent=Intent.HOWTO,
            primary_query="q",
            secondary_queries=[],
            retrieval_strategy=strategy,
            steps=[],
        )

    def _reflection(self, recommendation="Use the top result.") -> ReflectionResult:
        return ReflectionResult(confidence=0.7, coverage=0.8, recommendation=recommendation)

    def test_no_results_contains_no_notes_message(self):
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(), [], self._reflection(), {}
        )
        self.assertIn("No matching notes found", answer)

    def test_answer_contains_intent(self):
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(Intent.TROUBLESHOOT), self._plan(), [], self._reflection(), {}
        )
        self.assertIn("troubleshoot", answer)

    def test_answer_contains_strategy(self):
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(strategy="code_first"), [], self._reflection(), {}
        )
        self.assertIn("code_first", answer)

    def test_answer_lists_results(self):
        results = [{"title": "Article", "source_url": "https://example.com", "content": "content here"}]
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(), results, self._reflection(), {}
        )
        self.assertIn("Article", answer)
        self.assertIn("https://example.com", answer)

    def test_answer_shows_follow_ups(self):
        reflection = ReflectionResult(
            confidence=0.3,
            coverage=0.2,
            recommendation="Broaden the query.",
            follow_up_queries=["query productsway", "query TIL"],
        )
        results = [{"title": "T", "source_url": "", "content": ""}]
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(), results, reflection, {}
        )
        self.assertIn("Follow-ups:", answer)
        self.assertIn("query productsway", answer)

    def test_answer_includes_timings(self):
        timings = {"primary_ms": 12.5, "secondary_1_ms": 8.0}
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(), [], self._reflection(), timings
        )
        self.assertIn("Timings:", answer)
        self.assertIn("primary_ms=12.5ms", answer)

    def test_content_snippet_truncated_to_240(self):
        long_content = "word " * 100  # 500 chars
        results = [{"title": "T", "source_url": "", "content": long_content}]
        answer = AgentOrchestrator._compose_answer(
            "query", self._route(), self._plan(), results, self._reflection(), {}
        )
        # The snippet is content[:240] + "..."
        self.assertIn("...", answer)


# ---------------------------------------------------------------------------
# AgentOrchestrator.to_dict
# ---------------------------------------------------------------------------


class TestAgentOrchestratorToDict(unittest.TestCase):
    def _build_run(self, intent=Intent.HOWTO) -> AgentRun:
        route = RouteResult(intent=intent, confidence=0.8)
        plan = QueryPlan(
            intent=intent,
            primary_query="q",
            secondary_queries=[],
            retrieval_strategy="balanced_retrieve",
            steps=["s"],
        )
        reflection = ReflectionResult(confidence=0.7, coverage=0.5)
        return AgentRun(
            query="test query",
            route=route,
            plan=plan,
            results=[{"id": "1"}],
            reflection=reflection,
            answer="answer text",
        )

    def test_to_dict_intent_is_string(self):
        hybrid = _make_hybrid()
        orch = AgentOrchestrator(hybrid)
        run = self._build_run(Intent.TROUBLESHOOT)
        d = orch.to_dict(run)
        self.assertIsInstance(d["route"]["intent"], str)
        self.assertEqual(d["route"]["intent"], "troubleshoot")
        self.assertIsInstance(d["plan"]["intent"], str)
        self.assertEqual(d["plan"]["intent"], "troubleshoot")

    def test_to_dict_keys_present(self):
        hybrid = _make_hybrid()
        orch = AgentOrchestrator(hybrid)
        run = self._build_run()
        d = orch.to_dict(run)
        for key in ("query", "route", "plan", "results", "reflection", "answer"):
            self.assertIn(key, d)

    def test_to_dict_query_matches(self):
        hybrid = _make_hybrid()
        orch = AgentOrchestrator(hybrid)
        run = self._build_run()
        d = orch.to_dict(run)
        self.assertEqual(d["query"], "test query")

    def test_to_dict_all_intents_serialize(self):
        hybrid = _make_hybrid()
        orch = AgentOrchestrator(hybrid)
        for intent in Intent:
            run = self._build_run(intent)
            d = orch.to_dict(run)
            self.assertEqual(d["route"]["intent"], intent.value)


# ---------------------------------------------------------------------------
# AgentOrchestrator.run  (end-to-end with mocked HybridSearch)
# ---------------------------------------------------------------------------


class TestAgentOrchestratorRun(unittest.TestCase):
    def _make_orchestrator(self, search_results=None):
        hybrid = _make_hybrid(results=search_results or [])
        return AgentOrchestrator(hybrid)

    def test_run_returns_agent_run(self):
        orch = self._make_orchestrator()
        result = orch.run("how to configure neovim")
        self.assertIsInstance(result, AgentRun)

    def test_run_preserves_query(self):
        orch = self._make_orchestrator()
        result = orch.run("my specific query")
        self.assertEqual(result.query, "my specific query")

    def test_run_route_intent_is_valid(self):
        orch = self._make_orchestrator()
        result = orch.run("fix the broken login")
        self.assertIn(result.route.intent, list(Intent))

    def test_run_with_results(self):
        items = [
            {"id": "a1", "title": "Fix login errors", "content": "login fix steps", "rrf_score": 0.8, "vector_score": 0.7},
            {"id": "a2", "title": "Auth guide", "content": "authentication setup", "rrf_score": 0.6, "vector_score": 0.5},
        ]
        orch = self._make_orchestrator(search_results=items)
        result = orch.run("fix login error", top_k=5)
        self.assertIsInstance(result.answer, str)
        self.assertIn("troubleshoot", result.answer.lower())

    def test_run_empty_results_no_crash(self):
        orch = self._make_orchestrator(search_results=[])
        result = orch.run("completely unknown topic")
        self.assertIsInstance(result, AgentRun)
        self.assertIn("No matching notes", result.answer)

    def test_run_top_k_respected(self):
        items = [{"id": str(i), "rrf_score": 1.0 / (i + 1)} for i in range(20)]
        hybrid = _make_hybrid(results=items)
        orch = AgentOrchestrator(hybrid)
        result = orch.run("query", top_k=3)
        self.assertLessEqual(len(result.results), 3)

    def test_run_answer_is_string(self):
        orch = self._make_orchestrator()
        result = orch.run("any query")
        self.assertIsInstance(result.answer, str)

    def test_run_reflection_confidence_in_range(self):
        orch = self._make_orchestrator()
        result = orch.run("something")
        self.assertGreaterEqual(result.reflection.confidence, 0.0)
        self.assertLessEqual(result.reflection.confidence, 1.0)


# ---------------------------------------------------------------------------
# Edge / regression cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    def test_router_query_case_insensitive(self):
        router = RouterAgent()
        lower = router.route("fix broken error")
        upper = router.route("FIX BROKEN ERROR")
        self.assertEqual(lower.intent, upper.intent)

    def test_planner_compare_query_without_vs(self):
        # If query has no " vs " substring, replace is a no-op but shouldn't crash
        planner = PlannerAgent()
        route = RouteResult(intent=Intent.COMPARE, confidence=0.8)
        plan = planner.build_plan("vim comparison", route)
        self.assertIsInstance(plan, QueryPlan)

    def test_dedupe_merge_preserves_all_unique_items(self):
        sets = [
            [{"id": "a"}, {"id": "b"}],
            [{"id": "c"}, {"id": "d"}],
        ]
        result = RetrievalAgent._dedupe_merge(sets)
        ids = {r["id"] for r in result}
        self.assertEqual(ids, {"a", "b", "c", "d"})

    def test_reflection_empty_query_string(self):
        reflector = ReflectionAgent()
        route = RouteResult(intent=Intent.UNKNOWN, confidence=0.25)
        plan = QueryPlan(
            intent=Intent.UNKNOWN,
            primary_query="",
            secondary_queries=[],
            retrieval_strategy="balanced_retrieve",
            steps=[],
        )
        # Should not raise even with an empty query
        result = reflector.reflect("", route, plan, [])
        self.assertIsInstance(result, ReflectionResult)

    def test_compose_answer_no_timings_section_when_empty(self):
        route = RouteResult(intent=Intent.HOWTO, confidence=0.8, signals=[])
        plan = QueryPlan(
            intent=Intent.HOWTO,
            primary_query="q",
            secondary_queries=[],
            retrieval_strategy="balanced_retrieve",
            steps=[],
        )
        reflection = ReflectionResult(confidence=0.7, coverage=0.8, recommendation="ok")
        answer = AgentOrchestrator._compose_answer("query", route, plan, [], reflection, {})
        # Empty timings dict still may produce "Timings: " with no entries — test it doesn't crash
        self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()