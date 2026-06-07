"""Unit tests for the agent endpoints added to server.py in Day 2.

Tests cover:
- GET /agent/query         — returns AgentResult JSON
- GET /agent/query/stream  — returns SSE events in the correct order
- get_agent singleton      — initialised once and reused

HybridSearch and AgentOrchestrator are fully mocked so the test suite
runs without any ML dependencies or on-disk LanceDB data.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RESULT = {
    "id": "article:0",
    "title": "Fix session ID unknown",
    "source_url": "https://productsway.com/til/fix-session-id",
    "content": "Set the session secret before starting the server.",
    "rrf_score": 0.75,
    "vector_score": 0.68,
    "_query": "session id unknown fix error troubleshooting",
    "_best_rank": 1,
    "_evidence_queries": ["session id unknown fix error troubleshooting"],
}

_SAMPLE_ROUTE = {
    "intent": "troubleshoot",
    "confidence": 0.82,
    "signals": ["troubleshoot:error", "troubleshoot:fix"],
}

_SAMPLE_PLAN = {
    "intent": "troubleshoot",
    "primary_query": "session id unknown fix error troubleshooting",
    "secondary_queries": [
        "session id unknown how to fix",
        "session id unknown solution workaround",
    ],
    "retrieval_strategy": "broad_retrieve_then_reflect",
    "steps": ["step 1"],
}

_SAMPLE_REFLECTION = {
    "confidence": 0.71,
    "coverage": 0.80,
    "gaps": [],
    "follow_up_queries": [],
    "recommendation": "Use the top result directly.",
}

_SAMPLE_ANSWER = (
    "Intent: troubleshoot (confidence 0.82)\n"
    "Plan: broad_retrieve_then_reflect\n"
    "Reflection: Use the top result directly.\n"
    "Top evidence:\n"
    "1. Fix session ID unknown\n"
    "   https://productsway.com/til/fix-session-id\n"
    "   Set the session secret before starting the server...."
)


def _make_mock_run():
    """Return a mock AgentRun-like object with the sample data above."""
    run = MagicMock()
    run.query = "how to fix session id unknown with socket.io"
    return run


def _make_mock_agent():
    """Return a mock AgentOrchestrator whose run/to_dict return sample data."""
    agent = MagicMock()
    agent.run.return_value = _make_mock_run()
    agent.to_dict.return_value = {
        "query": "how to fix session id unknown with socket.io",
        "route": _SAMPLE_ROUTE,
        "plan": _SAMPLE_PLAN,
        "results": [_SAMPLE_RESULT],
        "reflection": _SAMPLE_REFLECTION,
        "answer": _SAMPLE_ANSWER,
    }
    return agent


# ---------------------------------------------------------------------------
# GET /agent/query
# ---------------------------------------------------------------------------


class TestAgentQueryEndpoint(unittest.TestCase):
    def setUp(self):
        # Patch get_agent so it returns a mock without touching disk or models
        self.mock_agent = _make_mock_agent()
        self.patcher = patch.object(server, "get_agent", return_value=self.mock_agent)
        self.patcher.start()
        self.client = TestClient(server.app, raise_server_exceptions=True)

    def tearDown(self):
        self.patcher.stop()
        # Reset singleton so other tests start fresh
        server._agent = None

    def test_returns_200(self):
        resp = self.client.get("/agent/query", params={"q": "test query"})
        self.assertEqual(resp.status_code, 200)

    def test_response_has_required_fields(self):
        resp = self.client.get("/agent/query", params={"q": "test query"})
        body = resp.json()
        for field in ("query", "route", "plan", "results", "reflection", "answer", "timing"):
            self.assertIn(field, body, f"missing field: {field}")

    def test_query_field_echoes_input(self):
        resp = self.client.get(
            "/agent/query",
            params={"q": "how to fix session id unknown with socket.io"},
        )
        self.assertEqual(resp.json()["query"], "how to fix session id unknown with socket.io")

    def test_route_intent_is_string(self):
        resp = self.client.get("/agent/query", params={"q": "test"})
        route = resp.json()["route"]
        self.assertIsInstance(route["intent"], str)

    def test_timing_contains_api_ms(self):
        resp = self.client.get("/agent/query", params={"q": "test"})
        timing = resp.json()["timing"]
        self.assertIn("api_ms", timing)
        self.assertIsInstance(timing["api_ms"], float)

    def test_missing_q_returns_422(self):
        resp = self.client.get("/agent/query")
        self.assertEqual(resp.status_code, 422)

    def test_top_k_default_is_five(self):
        resp = self.client.get("/agent/query", params={"q": "test"})
        self.assertEqual(resp.status_code, 200)
        # Verify agent.run was called with top_k=5
        self.mock_agent.run.assert_called_once()
        call_kwargs = self.mock_agent.run.call_args
        self.assertEqual(call_kwargs[1].get("top_k", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else 5), 5)

    def test_custom_top_k(self):
        resp = self.client.get("/agent/query", params={"q": "test", "top_k": 3})
        self.assertEqual(resp.status_code, 200)
        call_args = self.mock_agent.run.call_args
        # top_k=3 should be passed to agent.run
        passed_top_k = call_args[1].get("top_k") if call_args[1] else call_args[0][1]
        self.assertEqual(passed_top_k, 3)

    def test_results_is_a_list(self):
        resp = self.client.get("/agent/query", params={"q": "test"})
        self.assertIsInstance(resp.json()["results"], list)

    def test_agent_run_called_with_query(self):
        q = "unique query text 42"
        self.client.get("/agent/query", params={"q": q})
        self.mock_agent.run.assert_called_once()
        actual_q = self.mock_agent.run.call_args[0][0]
        self.assertEqual(actual_q, q)

    def test_empty_results_still_returns_200(self):
        self.mock_agent.to_dict.return_value = {
            "query": "q",
            "route": _SAMPLE_ROUTE,
            "plan": _SAMPLE_PLAN,
            "results": [],
            "reflection": _SAMPLE_REFLECTION,
            "answer": "No matching notes found.",
        }
        resp = self.client.get("/agent/query", params={"q": "obscure thing"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"], [])


# ---------------------------------------------------------------------------
# GET /agent/query/stream
# ---------------------------------------------------------------------------


class TestAgentQueryStreamEndpoint(unittest.TestCase):
    def setUp(self):
        self.mock_agent = _make_mock_agent()
        self.patcher = patch.object(server, "get_agent", return_value=self.mock_agent)
        self.patcher.start()
        self.client = TestClient(server.app, raise_server_exceptions=True)

    def tearDown(self):
        self.patcher.stop()
        server._agent = None

    def _parse_events(self, text: str) -> list[dict]:
        """Parse SSE data lines into a list of dicts."""
        events = []
        for line in text.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                events.append(json.loads(payload))
        return events

    def test_returns_200_with_sse_content_type(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

    def test_missing_q_returns_422(self):
        resp = self.client.get("/agent/query/stream")
        self.assertEqual(resp.status_code, 422)

    def test_first_event_is_route_type(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["type"], "route")

    def test_route_event_contains_route_and_plan(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        route_event = events[0]
        self.assertIn("route", route_event)
        self.assertIn("plan", route_event)

    def test_result_events_present_for_each_result(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        result_events = [e for e in events if e.get("type") == "result"]
        # There is 1 item in _SAMPLE_RESULT list
        self.assertEqual(len(result_events), 1)

    def test_result_event_has_index(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        result_events = [e for e in events if e.get("type") == "result"]
        self.assertEqual(result_events[0]["index"], 1)

    def test_reflection_event_present(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        types = [e["type"] for e in events]
        self.assertIn("reflection", types)

    def test_answer_event_present(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        types = [e["type"] for e in events]
        self.assertIn("answer", types)

    def test_done_event_is_last(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        self.assertEqual(events[-1]["type"], "done")

    def test_event_order_route_results_reflection_answer_done(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        events = self._parse_events(resp.text)
        types = [e["type"] for e in events]
        # "route" must come before any "result"
        if "result" in types:
            self.assertLess(types.index("route"), types.index("result"))
        # "reflection" must come after all "result" events
        if "result" in types:
            last_result_idx = max(i for i, t in enumerate(types) if t == "result")
            self.assertGreater(types.index("reflection"), last_result_idx)
        # "answer" after "reflection"
        self.assertGreater(types.index("answer"), types.index("reflection"))
        # "done" is last
        self.assertEqual(types[-1], "done")

    def test_empty_results_stream_has_no_result_events(self):
        self.mock_agent.to_dict.return_value = {
            "query": "q",
            "route": _SAMPLE_ROUTE,
            "plan": _SAMPLE_PLAN,
            "results": [],
            "reflection": _SAMPLE_REFLECTION,
            "answer": "No matching notes found.",
        }
        resp = self.client.get("/agent/query/stream", params={"q": "obscure"})
        events = self._parse_events(resp.text)
        result_events = [e for e in events if e.get("type") == "result"]
        self.assertEqual(len(result_events), 0)

    def test_cache_control_header(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")

    def test_x_accel_buffering_header(self):
        resp = self.client.get("/agent/query/stream", params={"q": "test"})
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")


# ---------------------------------------------------------------------------
# get_agent singleton
# ---------------------------------------------------------------------------


class TestGetAgentSingleton(unittest.TestCase):
    def setUp(self):
        # Reset both singletons before each test
        server._agent = None
        server._hybrid = None

    def tearDown(self):
        server._agent = None
        server._hybrid = None

    def test_get_agent_returns_same_instance_on_second_call(self):
        mock_hybrid = MagicMock()
        mock_agent_instance = MagicMock()
        with (
            patch("server.create_hybrid_search", return_value=mock_hybrid),
            patch("server.AgentOrchestrator", return_value=mock_agent_instance) as MockOrch,
        ):
            first = server.get_agent()
            second = server.get_agent()
            self.assertIs(first, second)
            # AgentOrchestrator constructor called only once
            MockOrch.assert_called_once()

    def test_get_agent_initialised_with_hybrid(self):
        mock_hybrid = MagicMock()
        with (
            patch("server.create_hybrid_search", return_value=mock_hybrid),
            patch("server.AgentOrchestrator") as MockOrch,
        ):
            server.get_agent()
            MockOrch.assert_called_once_with(mock_hybrid)

    def test_get_agent_not_none_after_call(self):
        mock_hybrid = MagicMock()
        with (
            patch("server.create_hybrid_search", return_value=mock_hybrid),
            patch("server.AgentOrchestrator", return_value=MagicMock()),
        ):
            server.get_agent()
            self.assertIsNotNone(server._agent)


if __name__ == "__main__":
    unittest.main()
