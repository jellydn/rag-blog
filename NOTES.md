# Working Notes

## User Profile

- **Senior Python engineer**, 15+ years experience
- **No AI/ML background** — terminology like "embeddings", "transformers", "attention" is new
- **Has shipped Day 1 + Day 2 of the rag-blog** — works in production, but built by recipe
- Comfortable reading code, math notation, and reasoning about systems
- Familiar with: FastAPI, Docker, JSON, NumPy basics, async I/O, design patterns, ADRs

## Teaching Preferences

- **Style:** Concept + code walkthrough. Tie every concept to a specific file/line in `rag-blog`.
- **Pacing:** One concept per lesson, ~5–15 minutes. Hands-on exercises must run in the repo.
- **Tone:** Engineer-to-engineer. Skip the "AI is magic" framing. Use real engineering metaphors (data structures, indexes, caching, contracts).
- **Citations:** Every claim should have a link. They learn better when they can verify.
- **No fluff:** No "Introduction to Python" filler. Assume Python and software engineering fluency.

## Zone of Proximal Development (Working Estimate)

| Concept | Status |
| --- | --- |
| Python | Expert |
| Reading JSON, async, HTTP, REST | Expert |
| Embeddings (concept) | **Unknown — start here** |
| Vector math (cosine, dot product) | School-level, rusty |
| BM25 | New |
| RRF | New |
| Transformer architecture | New |
| LLM inference | New |
| Agents & tool use | Concept-aware, implementation-shallow |
| RAG end-to-end | Built one, not deeply understood |

## Open Questions to Revisit

- Are you doing the 7-day track solo, or with a group? (Affects community recommendations)
- Are you interested in a specific vendor (OpenAI / Anthropic / open-source) for the LLM portion?
- Is the goal to stay on `sentence-transformers` + local models, or migrate to a hosted LLM?

## Lessons To Date

- `0001-embeddings-vector-of-meaning.md` — what an embedding is, cosine similarity, encoder intuition, the `all-MiniLM-L6-v2` we use
- `0002-vector-search-at-scale.md` — exact k-NN, HNSW, IVF, PQ, LanceDB indexes, recall benchmarking
- `0003-bm25-keyword-search-still-matters.md` — TF-IDF → BM25, k1 and b knobs, the formula, when to use over embeddings

## Behavioral Notes

- User is hands-on: ran the playground.py from Lesson 0001 without being prompted, then asked for the polysemy extension. Comfortable running scripts and reading outputs.
- User grasps abstract concepts quickly once anchored in code. The "engineer-to-engineer framing" callouts (e.g. "embeddings are a hash function from meaning") landed well — keep using that pattern.
- User accepts 10–15 min lessons with a runnable exercise; longer write-ups would feel like filler.
