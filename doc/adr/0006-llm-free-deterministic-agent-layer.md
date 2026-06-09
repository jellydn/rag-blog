# 6. LLM-free deterministic agent layer on hybrid retrieval

Date: 2026-06-07

## Status

Accepted

## Context

Day 2 adds “agentic” behavior—routing, planning, multi-pass retrieval, and reflection—without introducing provider keys, latency, or non-determinism from an LLM. The stack must still run locally on CPU and share the same hybrid engine as `/query` and the CLI (ADR-0004).

## Decision

Implement **`agentic.py`** as a pure-Python orchestration layer over **`HybridSearch`**:

| Component | Role |
|-----------|------|
| **RouterAgent** | Heuristic intent (`howto`, `troubleshoot`, `code`, …) from keyword signals |
| **PlannerAgent** | Primary + secondary query rewrites and a retrieval strategy label |
| **RetrievalAgent** | Multiple hybrid searches; merge/dedupe evidence by chunk id |
| **ReflectionAgent** | Coverage/confidence from scores and lexical overlap; follow-up query suggestions |
| **AgentOrchestrator** | `run()` → structured `AgentRun`; `to_dict()` for API |

Expose via FastAPI:

- `GET /agent/query` — JSON payload (`route`, `plan`, `results`, `reflection`, `answer`)
- `GET /agent/query/stream` — SSE events: `route`, `result`, `reflection`, `answer`, `done`

Lazy singleton **`get_agent()`** in `server.py` wires `AgentOrchestrator(get_hybrid())`.

No LLM calls in this path; “answer” is assembled from retrieved chunks and heuristics.

## Consequences

### Positive

- Debuggable, testable agent flow without API spend.
- Improves recall on ambiguous queries via rewrites and multi-pass retrieval.
- Clear seam to swap Planner/Reflection for LLM-backed implementations later.

### Negative

- Routing and reflection quality capped by keyword heuristics.
- Not a tool-using or autonomous agent; retrieval-orchestration only.
- Stream protocol is custom JSON-in-SSE, not FastAPI `EventSourceResponse` / `ServerSentEvent` (see References).

## References

- Hybrid retrieval: ADR-0001; shared engine: ADR-0004.
- FastAPI streaming: [StreamingResponse](https://github.com/fastapi/fastapi/blob/master/docs/en/docs/advanced/custom-response.md) (`text/event-stream`); optional upgrade path: [Server-Sent Events tutorial](https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/server-sent-events.md) (`EventSourceResponse`).
