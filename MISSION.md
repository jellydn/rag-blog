# Mission


## Why I'm Learning This

I'm a senior Python engineer (15+ years) with no prior background in AI, ML, or LLMs. I have been **building** a RAG engine for my blog ([productsway.com](https://productsway.com)) for two days — it works, I can ship it, the benchmarks are good — but I have been mostly following recipes. I don't really understand *why* it works.

I want to go from **"I can ship a RAG"** to **"I can explain every line of my RAG to another engineer"**, and ultimately to a place where I can design AI systems, evaluate new models, and contribute to architectural decisions — not just consume libraries.

This is a 7-day AI engineer track. Days 1–2 produced working code. Days 3–7 are where I need depth, not just momentum.

## What "Done" Looks Like

By the end of this track I should be able to:

1. **Explain** embeddings, vector search, BM25, RRF, hybrid search, RAG, and agents in my own words, with the math and the trade-offs.
2. **Read** the `rag-blog` source code (`chunking.py`, `rag_pipeline.py`, `agentic.py`) and point to the line where any concept is implemented, and explain why that line exists.
3. **Evaluate** new models, vector DBs, or agent frameworks on first principles, not by vibes.
4. **Write ADRs** that an AI-naïve senior engineer could follow without re-doing my research.

## Scope (What I'm Focusing On)

In scope, in priority order:

- **Embeddings & vector search** — what vectors of meaning are, cosine similarity, encoder models, the `all-MiniLM-L6-v2` model we chose
- **BM25 & keyword search** — TF-IDF → BM25 evolution, when keywords beat vectors
- **Reciprocal Rank Fusion (RRF)** — why and how we fuse two ranked lists
- **Hybrid search composition** — the full picture
- **LLM basics** — what a transformer decoder does, tokenization, temperature, context windows
- **RAG theory** — the retrieval → augmentation → generation pipeline, and why naive RAG fails
- **Agents & tool use** — what an "agent" really is, the deterministic layer we built in `agentic.py`, and when to introduce LLMs into the loop
- **Evaluation** — what "good" means for retrieval and for answers

Out of scope (or much later):

- Training my own models
- Reinforcement learning
- Multimodal / vision / audio
- Building ML infrastructure at scale (CUDA kernels, distributed training)

## Connection to the Project

Every lesson should be tied back to a concrete artifact in the `rag-blog` codebase. Theory without the code is just trivia. Code without theory is copy-paste engineering. The project is my anchor.

## Time Horizon

7 days of structured learning (matching the project). Realistic pace: 1 lesson per concept, ~30–60 minutes each, with a hands-on exercise that I actually run in the repo.
