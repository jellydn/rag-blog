# Day 2 — Advanced Agent Patterns

## Goal
Add a lightweight agentic layer on top of the RAG engine to explore:
- routing
- planning
- retrieval orchestration
- reflection / self-checking

## What I built

I added `agentic.py`, which implements a deterministic, LLM-free agent pipeline:

1. **RouterAgent** — classifies the query intent
2. **PlannerAgent** — generates a retrieval plan and query rewrites
3. **RetrievalAgent** — runs multiple hybrid retrieval passes and merges the evidence
4. **ReflectionAgent** — checks coverage/confidence and suggests follow-up queries
5. **AgentOrchestrator** — returns a structured answer payload

I also added two new API endpoints:
- `GET /agent/query`
- `GET /agent/query/stream`

## Why this design

- **Deterministic** — easy to debug and test locally
- **No extra model dependency** — builds on the existing hybrid retrieval stack
- **Composable** — routing + planning + reflection can later be swapped for LLM calls
- **Useful now** — even without an LLM, the agent can improve search by running multiple retrieval strategies

## Key trade-offs

- Heuristic routing is less flexible than an LLM planner
- The reflection step is only as good as the lexical overlap and score heuristics
- This is still a retrieval-first agent, not a fully autonomous tool-using agent

## Next steps

- Add a semantic reranker for the final top-k
- Add an LLM-backed planner when provider credentials are available
- Add tests for routing confidence and query rewrite coverage
- Expand the stream endpoint into a UI-friendly event protocol

## Example

```bash
curl "http://localhost:8000/agent/query?q=how+to+fix+session+id+unknown+with+socket.io"
```

Expected behavior:
- route as `troubleshoot`
- expand the query with fix / workaround terms
- retrieve the relevant TIL article(s)
- return a reflection result with confidence and follow-ups
