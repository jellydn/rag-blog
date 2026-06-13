# Resources

Curated reading list. Each entry has a **trust tier**:
- **T1 — Authoritative** (primary docs, model cards, foundational papers)
- **T2 — Trusted practitioner** (engineer blogs, vendor engineering blogs)
- **T3 — Useful but opinionated** (tutorials, comparisons, benchmark write-ups)

Always prefer T1 → T2 → T3. Never trust a single T3 source for a design decision.

---

## Embeddings & Vector Search

| Resource | Tier | Why |
| --- | --- | --- |
| [Vicki Boykis — *What Are Embeddings?*](https://vickiboykis.com/what_are_embeddings/) | T1 | Practitioner-targeted mental model. Maps embeddings to a data engineering problem. The single best starting point for engineers. |
| [Pinecone — *Dense Vector Embeddings*](https://www.pinecone.io/learn/series/nlp/dense-vector-embeddings-nlp/) | T2 | Clear explanation of cosine similarity, normalization, why dense beats sparse for semantics. |
| [Lilian Weng — *Learning Word Embedding*](https://lilianweng.github.io/posts/2017-10-15-word-embedding/) | T1 | Evolution from word2vec → GloVe → context-aware encoders. Math-heavy, deep. |
| [HuggingFace — `all-MiniLM-L6-v2` model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | T1 | **Our model.** Training data, pooling strategy, benchmarks, expected behavior. Must read. |
| [SBERT documentation](https://www.sbert.net/) | T1 | API reference for the library we use. |
| [Lilian Weng — *The Transformer Family v2*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | T1 | Deeper architecture. Read *after* the practical side clicks. |
| [Zilliz — *Sparse and Dense Embeddings*](https://zilliz.com/learn/sparse-and-dense-embeddings) | T2 | Tradeoff comparison. Good for hybrid-search framing. |

## BM25 & Keyword Search

| Resource | Tier | Why |
| --- | --- | --- |
| [Elastic — *Practical BM25 (Part 2)*](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | T1 | Definitive explanation of $k_1$ and $b$. |
| [Zilliz — *Mastering BM25*](https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus) | T2 | Implementation-level walkthrough. Good for our hand-rolled BM25. |
| [OpenSearch — *Hybrid retrieval with sparse + semantic encoders*](https://opensearch.org/blog/improving-document-retrieval-with-sparse-semantic-encoders/) | T2 | Why hybrid search is the production default. |
| Robertson & Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.nowpublishers.com/article/Details/INR-019) | T1 | Academic foundation. Dense but authoritative. |

## Reciprocal Rank Fusion (RRF)

| Resource | Tier | Why |
| --- | --- | --- |
| [Elasticsearch — *RRF reference*](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) | T1 | Original spec, $k=60$ convention. |
| [BigData Boutique — *RRF Explained*](https://bigdataboutique.com/blog/reciprocal-rank-fusion-how-it-works-and-when-to-use-it) | T2 | Intuition-first, explains the smoother. |
| [Weaviate — *Hybrid Search Explained*](https://weaviate.io/blog/hybrid-search-explained) | T2 | RRF vs. relative score fusion. Useful contrast. |

## LLM Fundamentals (For Later Lessons)

| Resource | Tier | Why |
| --- | --- | --- |
| [Andrej Karpathy — *Let's build GPT: from scratch, in code*](https://www.youtube.com/watch?v=kCc8FmEb1nY) | T1 | Best one-shot mental model of a transformer decoder. |
| [Andrej Karpathy — *Intro to Large Language Models*](https://www.youtube.com/watch?v=zjkBMFhNj_g) | T1 | 1-hour overview for engineers. |
| [Lilian Weng — *Prompt Engineering*](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | T1 | Practical prompting patterns. |

## RAG Theory (For Later Lessons)

| Resource | Tier | Why |
| --- | --- | --- |
| [Pinecone — *Retrieval Augmented Generation*](https://www.pinecone.io/learn/series/rag/rag/) | T2 | End-to-end RAG pipeline mental model. |
| [Lilian Weng — *LLM-Powered Autonomous Agents*](https://lilianweng.github.io/posts/2023-06-23-agent/) | T1 | Foundational agent post. |

## Evaluation (For Later Lessons)

| Resource | Tier | Why |
| --- | --- | --- |
| [Pinecone — *RAG Evaluation*](https://www.pinecone.io/learn/series/vector-search-quality/) | T2 | Practical metrics for retrieval and generation. |

---

## Reading Order

1. Vicki Boykis embeddings post → everything else on embeddings makes more sense
2. HuggingFace model card for `all-MiniLM-L6-v2` → understand our specific encoder
3. Elastic BM25 post → understand our keyword index
4. RRF reference → understand the fusion
5. Karpathy videos → LLMs (later lessons)

## Vector Indexes & ANN Search

| Resource | Tier | Why |
| --- | --- | --- |
| [LanceDB Vector Index documentation](https://docs.lancedb.com/indexing/vector-index) | T1 | Canonical reference for the index types we'll use in code. |
| [Malkov & Yashunin, 2016 — HNSW paper](https://arxiv.org/abs/1603.09320) | T1 | The original HNSW paper. Read the abstract + figures; the rest is math. |
| [Pinecone — HNSW explained](https://www.pinecone.io/learn/series/vector-search/hnsw/) | T2 | Interactive walkthrough of the graph algorithm. |
| [Pinecone — Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) | T2 | The "magic triangle" framing for recall/speed/memory. |
| [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/Getting-started) | T1 | Reference implementation; great for understanding operational constraints. |
| [Milvus — IVF_PQ docs](https://milvus.io/docs/ivf-pq.md) | T2 | Parameter guidance from a competing vector DB. |
