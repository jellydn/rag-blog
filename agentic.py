#!/usr/bin/env python3
"""Advanced agent patterns for the Productsway RAG project.

Day 2 goal: add a lightweight agentic layer on top of retrieval.

This module is intentionally LLM-free so it can run locally and deterministically:
- RouterAgent: classifies query intent
- PlannerAgent: builds a retrieval plan
- RetrievalAgent: executes one or more hybrid searches
- ReflectionAgent: checks coverage/confidence and proposes next actions
- AgentOrchestrator: combines the above into a single response
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from rag_pipeline import HybridSearch


class Intent(str, Enum):
    HOWTO = "howto"
    TROUBLESHOOT = "troubleshoot"
    CODE = "code"
    COMPARE = "compare"
    SUMMARY = "summary"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RouteResult:
    intent: Intent
    confidence: float
    signals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryPlan:
    intent: Intent
    primary_query: str
    secondary_queries: list[str]
    retrieval_strategy: str
    steps: list[str]


@dataclass(slots=True)
class ReflectionResult:
    confidence: float
    coverage: float
    gaps: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass(slots=True)
class AgentRun:
    query: str
    route: RouteResult
    plan: QueryPlan
    results: list[dict[str, Any]]
    reflection: ReflectionResult
    answer: str


class RouterAgent:
    """Heuristic intent classifier for query routing."""

    TROUBLESHOOT_WORDS = {
        "error",
        "fix",
        "broken",
        "issue",
        "problem",
        "fail",
        "failed",
        "cannot",
        "can't",
        "why",
        "bug",
    }
    HOWTO_WORDS = {"how", "install", "setup", "set up", "configure", "usage", "guide"}
    CODE_WORDS = {
        "code",
        "snippet",
        "example",
        "command",
        "cli",
        "api",
        "typescript",
        "javascript",
        "python",
        "bash",
    }
    COMPARE_WORDS = {"vs", "versus", "compare", "difference", "between"}
    SUMMARY_WORDS = {"summarize", "summary", "overview", "all", "what do you know"}
    NAVIGATION_WORDS = {"where", "find", "search", "latest", "list", "show me"}

    def route(self, query: str) -> RouteResult:
        q = query.lower()
        signals: list[str] = []
        scores = {i: 0 for i in Intent}

        def bump(intent: Intent, word: str, value: int = 1):
            scores[intent] += value
            signals.append(f"{intent.value}:{word}")

        for word in self.TROUBLESHOOT_WORDS:
            if word in q:
                bump(Intent.TROUBLESHOOT, word, 2)
        for word in self.HOWTO_WORDS:
            if word in q:
                bump(Intent.HOWTO, word, 1)
        for word in self.CODE_WORDS:
            if word in q:
                bump(Intent.CODE, word, 1)
        for word in self.COMPARE_WORDS:
            if word in q:
                bump(Intent.COMPARE, word, 2)
        for word in self.SUMMARY_WORDS:
            if word in q:
                bump(Intent.SUMMARY, word, 2)
        for word in self.NAVIGATION_WORDS:
            if word in q:
                bump(Intent.NAVIGATION, word, 1)

        # Tie-breakers / defaults
        if q.startswith("how ") or q.startswith("how to"):
            scores[Intent.HOWTO] += 2
        if "?" in query:
            scores[Intent.HOWTO] += 1
        if any(tok in q for tok in ["git", "cli", "command", "terminal"]):
            scores[Intent.CODE] += 1

        best_intent = max(scores.items(), key=lambda item: item[1])[0]
        best_score = scores[best_intent]
        total = max(sum(scores.values()), 1)
        confidence = min(1.0, best_score / max(total, 1) + (0.15 if best_score > 0 else 0.0))
        if best_score == 0:
            best_intent = Intent.UNKNOWN
            confidence = 0.25

        return RouteResult(intent=best_intent, confidence=round(confidence, 2), signals=signals[:8])


class PlannerAgent:
    """Builds a retrieval plan from the routed intent."""

    def build_plan(self, query: str, route: RouteResult) -> QueryPlan:
        q = query.strip()
        secondary: list[str] = []
        steps: list[str] = []

        if route.intent == Intent.TROUBLESHOOT:
            primary = f"{q} fix error troubleshooting"
            secondary = [f"{q} how to fix", f"{q} solution workaround"]
            strategy = "broad_retrieve_then_reflect"
            steps = [
                "Retrieve the most relevant fixes and failure modes",
                "Prefer code blocks, commands, and error explanations",
                "If coverage is low, broaden to adjacent setup articles",
            ]
        elif route.intent == Intent.CODE:
            primary = f"{q} code example command snippet"
            secondary = [f"{q} example", f"{q} command"]
            strategy = "code_first"
            steps = [
                "Retrieve code-heavy chunks and CLI snippets",
                "Prefer exact command examples over prose",
                "Reflect on whether the top hits contain executable instructions",
            ]
        elif route.intent == Intent.COMPARE:
            # Case-insensitive rewrite -- catches " vs ", " VS ", " Vs " etc.
            # so a query like "PostgreSQL VS MySQL" is handled the same as
            # "PostgreSQL vs MySQL".
            primary = re.sub(r" vs ", " comparison ", q, flags=re.IGNORECASE)
            secondary = [
                re.sub(r" vs ", " between ", q, flags=re.IGNORECASE),
                f"{q} difference",
            ]
            strategy = "dual_path_compare"
            steps = [
                "Retrieve both sides of the comparison",
                "Look for contrast in purpose, trade-offs, and implementation",
                "Summarize differences with concrete evidence",
            ]
        elif route.intent == Intent.SUMMARY:
            primary = q if q else "Productsway blog overview"
            secondary = ["Productsway homepage", "Productsway TIL overview"]
            strategy = "broad_summary"
            steps = [
                "Collect broad coverage across the repo",
                "Prioritize homepage and category overview chunks",
                "Find the most representative examples",
            ]
        elif route.intent == Intent.NAVIGATION:
            primary = q
            secondary = [f"latest {q}", f"{q} productsway"]
            strategy = "navigation_focus"
            steps = [
                "Answer with the latest or most direct matching content",
                "Return exact links and titles",
                "Keep context terse and actionable",
            ]
        else:
            primary = q
            secondary = [f"{q} productsway", f"{q} blog note"]
            strategy = "balanced_retrieve"
            steps = [
                "Run balanced retrieval across all notes",
                "Inspect the top results for direct matches",
                "Use reflection to decide whether to broaden the search",
            ]

        return QueryPlan(
            intent=route.intent,
            primary_query=primary,
            secondary_queries=secondary,
            retrieval_strategy=strategy,
            steps=steps,
        )


class RetrievalAgent:
    """Executes one or more retrieval passes and merges results."""

    def __init__(self, hybrid: HybridSearch):
        self.hybrid = hybrid

    @staticmethod
    def _dedupe_merge(results_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for results in results_sets:
            for rank, item in enumerate(results):
                key = str(
                    item.get("id")
                    or item.get("doc_id")
                    or item.get("source_url")
                    or f"chunk-{rank}"
                )
                if key not in merged:
                    merged[key] = dict(item)
                    merged[key]["_evidence_queries"] = []
                    merged[key]["_best_rank"] = rank + 1
                else:
                    # Keep the best scoring metadata and append evidence
                    if item.get("rrf_score", 0) > merged[key].get("rrf_score", 0):
                        merged[key].update(item)
                    merged[key]["_best_rank"] = min(merged[key]["_best_rank"], rank + 1)
                merged[key]["_evidence_queries"].append(item.get("_query", ""))

        out = list(merged.values())
        out.sort(key=lambda r: (r.get("rrf_score", 0), r.get("vector_score", 0)), reverse=True)
        return out

    def retrieve(
        self, plan: QueryPlan, top_k: int = 5
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        timings: dict[str, float] = {}
        result_sets: list[list[dict[str, Any]]] = []

        t0 = time.time()
        primary_results, timing = self.hybrid.search(plan.primary_query, top_k=top_k)
        timings["primary_ms"] = round((time.time() - t0) * 1000, 1)
        timings.update({f"primary_{k}": v for k, v in timing.items()})
        for r in primary_results:
            r["_query"] = plan.primary_query
        result_sets.append(primary_results)

        for idx, q in enumerate(plan.secondary_queries[:2], start=1):
            t1 = time.time()
            secondary_results, _ = self.hybrid.search(q, top_k=max(3, top_k // 2))
            timings[f"secondary_{idx}_ms"] = round((time.time() - t1) * 1000, 1)
            for r in secondary_results:
                r["_query"] = q
            result_sets.append(secondary_results)

        merged = self._dedupe_merge(result_sets)
        return merged[:top_k], timings


class ReflectionAgent:
    """Checks retrieval coverage and proposes follow-up queries."""

    def reflect(
        self, query: str, route: RouteResult, plan: QueryPlan, results: list[dict[str, Any]]
    ) -> ReflectionResult:
        q_tokens = {t for t in query.lower().split() if len(t) > 2}
        if not q_tokens:
            q_tokens = set(query.lower().split())

        matched_tokens = set()
        gaps: list[str] = []
        follow_ups: list[str] = []

        for r in results:
            text = f"{r.get('title', '')} {r.get('content', '')}".lower()
            for tok in q_tokens:
                if tok in text:
                    matched_tokens.add(tok)

        coverage = len(matched_tokens) / max(len(q_tokens), 1)
        top_rrf = results[0].get("rrf_score", 0) if results else 0.0
        top_vector = results[0].get("vector_score", 0) if results else 0.0
        confidence = min(1.0, round(coverage * 0.55 + top_rrf * 3 + top_vector * 0.15, 2))

        if coverage < 0.5:
            gaps.append("Low lexical coverage — consider a broader query or a neighboring topic")
            follow_ups.append(f"{query} productsway")
        if not results:
            gaps.append("No results returned")
            follow_ups.extend([f"{query} blog", f"{query} TIL"])
        elif route.intent in {Intent.TROUBLESHOOT, Intent.CODE} and top_vector < 0.35:
            gaps.append("Top hit is weak on code/command similarity")
            follow_ups.append(f"{query} command example")

        recommendation = (
            "Use the top result directly."
            if confidence >= 0.65
            else (
                "Broaden the query and inspect the second-best hit."
                if results
                else "Try a different keyword set."
            )
        )

        return ReflectionResult(
            confidence=confidence,
            coverage=round(coverage, 2),
            gaps=gaps,
            follow_up_queries=follow_ups[:3],
            recommendation=recommendation,
        )


class AgentOrchestrator:
    """High-level agentic query workflow."""

    def __init__(self, hybrid: HybridSearch):
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent(hybrid)
        self.reflector = ReflectionAgent()

    def _orchestrate(self, query: str, top_k: int):
        """Internal: yields intermediate results as the agent runs.

        Yield format: ``(kind, value)`` where ``kind`` is one of
        ``"route"``, ``"plan"``, ``"result"``, ``"reflection"``, ``"answer"``.

        Single source of truth for the orchestration order. Both
        ``run()`` (sync, returns AgentRun) and ``stream()`` (sync
        generator, yields SSE events) consume this generator so
        they cannot drift apart.
        """
        route = self.router.route(query)
        yield ("route", route)
        plan = self.planner.build_plan(query, route)
        yield ("plan", plan)
        results, timings = self.retriever.retrieve(plan, top_k=top_k)
        for r in results:
            yield ("result", r)
        reflection = self.reflector.reflect(query, route, plan, results)
        yield ("reflection", reflection)
        answer = self._compose_answer(query, route, plan, results, reflection, timings)
        yield ("answer", answer)

    def stream(self, query: str, top_k: int = 5) -> Iterator[dict[str, Any]]:
        """Yield SSE-friendly events as the agent runs.

        Consumes ``_orchestrate()`` and converts each intermediate
        result to a dict ready for ``json.dumps``. The HTTP client
        sees events as they're produced (route, plan, result x N,
        reflection, answer) -- the actual server endpoint
        (``GET /agent/query/stream``) consumes this generator and
        yields SSE messages between each event.

        Each yielded event is ``{"type": <kind>, ...}`` where
        ``<kind>`` matches the orchestrator tuple kind.
        """
        result_idx = 0
        for kind, value in self._orchestrate(query, top_k):
            if kind == "route":
                yield {
                    "type": "route",
                    "route": asdict(value) | {"intent": value.intent.value},
                }
            elif kind == "plan":
                yield {
                    "type": "plan",
                    "plan": asdict(value) | {"intent": value.intent.value},
                }
            elif kind == "result":
                result_idx += 1
                yield {"type": "result", "index": result_idx, **value}
            elif kind == "reflection":
                yield {"type": "reflection", "reflection": asdict(value)}
            elif kind == "answer":
                yield {"type": "answer", "answer": value}

    def run(self, query: str, top_k: int = 5) -> AgentRun:
        """Run the agent synchronously, returning the final AgentRun.

        Consumes ``_orchestrate()`` and assembles the AgentRun from
        the intermediate results. Behavior is identical to the
        pre-refactor inline implementation; the refactor just routes
        the orchestration through the shared generator so ``run()``
        and ``stream()`` cannot drift apart.
        """
        route: RouteResult | None = None
        plan: QueryPlan | None = None
        results: list[dict[str, Any]] = []
        reflection: ReflectionResult | None = None
        answer: str = ""

        for kind, value in self._orchestrate(query, top_k):
            if kind == "route":
                route = value
            elif kind == "plan":
                plan = value
            elif kind == "result":
                results.append(value)
            elif kind == "reflection":
                reflection = value
            elif kind == "answer":
                answer = value

        # The generator always yields all 5 event kinds, so the
        # assertions document the invariant.
        assert route is not None and plan is not None and reflection is not None
        return AgentRun(
            query=query,
            route=route,
            plan=plan,
            results=results,
            reflection=reflection,
            answer=answer,
        )

    @staticmethod
    def _compose_answer(
        query: str,
        route: RouteResult,
        plan: QueryPlan,
        results: list[dict[str, Any]],
        reflection: ReflectionResult,
        timings: dict[str, float],
    ) -> str:
        lines = [
            f"Intent: {route.intent.value} (confidence {route.confidence:.2f})",
            f"Plan: {plan.retrieval_strategy}",
            f"Reflection: {reflection.recommendation}",
        ]
        if timings:
            timing_bits = ", ".join(f"{k}={v}ms" for k, v in timings.items() if k.endswith("ms"))
            lines.append(f"Timings: {timing_bits}")

        if not results:
            lines.append("No matching notes found.")
            return "\n".join(lines)

        lines.append("Top evidence:")
        for idx, r in enumerate(results, start=1):
            title = r.get("title", "Untitled")
            url = r.get("source_url", "")
            snippet = r.get("content", "")[:240].replace("\n", " ")
            lines.append(f"{idx}. {title}")
            lines.append(f"   {url}")
            lines.append(f"   {snippet}...")
        if reflection.follow_up_queries:
            lines.append("Follow-ups:")
            for q in reflection.follow_up_queries:
                lines.append(f"- {q}")
        return "\n".join(lines)

    def to_dict(self, run: AgentRun) -> dict[str, Any]:
        route = asdict(run.route)
        route["intent"] = run.route.intent.value

        plan = asdict(run.plan)
        plan["intent"] = run.plan.intent.value

        return {
            "query": run.query,
            "route": route,
            "plan": plan,
            "results": run.results,
            "reflection": asdict(run.reflection),
            "answer": run.answer,
        }
