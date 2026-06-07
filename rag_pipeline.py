#!/usr/bin/env python3
"""Ingestion, vector store, BM25, and hybrid search for Productsway RAG."""

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import Document, MarkdownChunker, parse_chunk_id
from config import (
    BM25_JSON,
    BM25_WEIGHT,
    CHUNKS_DIR,
    CONTENT_DIR,
    DB_DIR,
    MODEL_NAME,
    RRF_K,
    VECTOR_WEIGHT,
    ensure_data_dirs,
)

ensure_data_dirs()


def chunk_record_to_meta(c: dict) -> dict:
    return {
        "doc_id": c["doc_id"],
        "title": c["title"],
        "source_url": c["source_url"],
        "category": c["category"],
        "chunk_index": c["chunk_index"],
        "total_chunks": c["total_chunks"],
    }


def build_chunk_meta(chunks: List[dict]) -> Dict[str, dict]:
    return {c["id"]: chunk_record_to_meta(c) for c in chunks}


def hit_from_chunk_record(r: dict) -> dict:
    return {
        "id": r["id"],
        "doc_id": r["doc_id"],
        "title": r["title"],
        "content": r["content"],
        "source_url": r["source_url"],
        "category": r["category"],
        "chunk_index": r["chunk_index"],
        "total_chunks": r["total_chunks"],
    }


def hit_from_bm25_only(
    chunk_key: str, content: str, meta: dict, bm25_rank: int, rrf_score: float
) -> dict:
    doc_id = meta.get("doc_id") or parse_chunk_id(chunk_key)[0]
    return {
        **hit_from_chunk_record(
            {
                "id": chunk_key,
                "doc_id": doc_id,
                "title": meta.get("title", doc_id.replace("-", " ").title()),
                "content": content,
                "source_url": meta.get(
                    "source_url",
                    "https://productsway.com/"
                    if doc_id == "homepage"
                    else f"https://productsway.com/notes/{doc_id}",
                ),
                "category": meta.get("category", "TIL"),
                "chunk_index": meta.get("chunk_index", parse_chunk_id(chunk_key)[1]),
                "total_chunks": meta.get("total_chunks", 1),
            }
        ),
        "rrf_score": rrf_score,
        "vector_rank": None,
        "bm25_rank": bm25_rank,
        "vector_score": 0.0,
    }


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Model loaded in {time.time() - t0:.1f}s (dim={self.dimension})")

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def embed_one(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_freqs: dict = {}
        self.chunk_meta: Dict[str, dict] = {}
        self.doc_lengths: List[int] = []
        self.doc_texts: List[str] = []
        self.doc_ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self.avg_doc_length: float = 0
        self.total_docs: int = 0
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z0-9#]{1,50}\b", text.lower())

    def add_documents(self, doc_ids: List[str], texts: List[str]):
        self.doc_ids = doc_ids
        self.doc_texts = texts
        self._doc_tokens = [self._tokenize(t) for t in texts]
        self.doc_lengths = [len(toks) for toks in self._doc_tokens]
        self.total_docs = len(texts)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

        term_to_df: dict = {}
        for toks in self._doc_tokens:
            for token in set(toks):
                term_to_df[token] = term_to_df.get(token, 0) + 1
        self.doc_freqs = term_to_df
        self._built = True
        print(f"  BM25 index built: {self.total_docs} docs, {len(term_to_df)} unique terms")

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n = self.total_docs
        scores = np.zeros(n)
        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for i in range(n):
                toks = self._doc_tokens[i]
                tf = toks.count(token)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_length)
                if denom > 0:
                    scores[i] += idf * (tf * (self.k1 + 1) / denom + self.delta)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "doc_id": self.doc_ids[idx],
                "content": self.doc_texts[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
            if scores[idx] > 0
        ]

    def to_json_dict(self) -> dict:
        return {
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_lengths": self.doc_lengths,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_freqs": self.doc_freqs,
            "chunk_meta": self.chunk_meta,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "BM25Index":
        bm25 = cls()
        bm25.doc_ids = data["doc_ids"]
        bm25.doc_texts = data["doc_texts"]
        bm25.doc_lengths = data["doc_lengths"]
        bm25.total_docs = data["total_docs"]
        bm25.avg_doc_length = data["avg_doc_length"]
        bm25.doc_freqs = data["doc_freqs"]
        bm25.chunk_meta = data.get("chunk_meta", {})
        bm25._doc_tokens = [bm25._tokenize(t) for t in bm25.doc_texts]
        bm25._built = True
        return bm25


class VectorStore:
    TABLE_NAME = "rag_chunks"

    def __init__(self, db_path: str, dimension: int = 384):
        self.db = lancedb.connect(str(db_path))
        self.dimension = dimension
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.table = self.db.open_table(self.TABLE_NAME)
            print(f"  Opened existing table: {self.table.count_rows()} rows")
        except Exception:
            self.table = None
            print("  New table will be created on first ingest")

    def _create_table(self):
        import pyarrow as pa

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("category", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("total_chunks", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), self.dimension)),
        ])
        self.table = self.db.create_table(self.TABLE_NAME, schema=schema, mode="overwrite")

    def ingest(self, chunks: List[dict], embedder: Embedder):
        if not chunks:
            print("  No chunks to ingest")
            return

        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)
        if self.table is None:
            self._create_table()

        import pyarrow as pa

        batch = pa.table({
            "id": [c["id"] for c in chunks],
            "doc_id": [c["doc_id"] for c in chunks],
            "title": [c["title"] for c in chunks],
            "content": [c["content"] for c in chunks],
            "source_url": [c["source_url"] for c in chunks],
            "category": [c["category"] for c in chunks],
            "chunk_index": [c["chunk_index"] for c in chunks],
            "total_chunks": [c["total_chunks"] for c in chunks],
            "vector": [emb.tolist() for emb in embeddings],
        })
        self.table.add(batch)
        print(f"  Ingested {len(chunks)} chunks → {self.table.count_rows()} total")

    def vector_search(self, query_vector: List[float], top_k: int = 10) -> List[dict]:
        if self.table is None:
            return []
        results = (
            self.table.search(query_vector).metric("cosine").limit(top_k).to_list()
        )
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "content": r["content"],
                "source_url": r["source_url"],
                "category": r["category"],
                "chunk_index": r.get("chunk_index", 0),
                "total_chunks": r.get("total_chunks", 1),
                "score": r["_distance"],
            }
            for r in results
        ]

    def count(self) -> int:
        return 0 if self.table is None else self.table.count_rows()


def load_bm25_index(db_dir: Path | None = None) -> BM25Index:
    path = (db_dir or DB_DIR) / "bm25_data.json"
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at {path}. Run: python rag_pipeline.py")
    with open(path, encoding="utf-8") as f:
        return BM25Index.from_json_dict(json.load(f))


def save_bm25_index(bm25: BM25Index, path: Path | None = None) -> None:
    out = path or BM25_JSON
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bm25.to_json_dict(), f, indent=2)


class HybridSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        rrf_k: int = RRF_K,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 5) -> tuple[List[dict], dict]:
        t0 = time.time()
        query_vec = self.embedder.embed_one(query)
        vec_results = self.vector_store.vector_search(query_vec, top_k=top_k * 2)
        t_vec = time.time() - t0

        t1 = time.time()
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        t_bm25 = time.time() - t1

        scores: Dict[str, dict] = {}
        for rank, r in enumerate(vec_results):
            cid = r["id"]
            scores[cid] = {
                **hit_from_chunk_record(r),
                "rrf_score": self.vector_weight / (self.rrf_k + rank + 1),
                "vector_rank": rank + 1,
                "bm25_rank": None,
                "vector_score": float(1 - r.get("score", 0)),
            }

        for rank, r in enumerate(bm25_results):
            cid = r["doc_id"]
            boost = self.bm25_weight / (self.rrf_k + rank + 1)
            if cid in scores:
                scores[cid]["rrf_score"] += boost
                scores[cid]["bm25_rank"] = rank + 1
            else:
                scores[cid] = hit_from_bm25_only(
                    cid,
                    r["content"],
                    self.bm25.chunk_meta.get(cid, {}),
                    rank + 1,
                    boost,
                )

        results = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[
            :top_k
        ]
        timing = {
            "vector_search_ms": round(t_vec * 1000, 1),
            "bm25_search_ms": round(t_bm25 * 1000, 1),
            "total_ms": round((t_vec + t_bm25) * 1000, 1),
        }
        return results, timing


def create_hybrid_search() -> HybridSearch:
    embedder = Embedder(MODEL_NAME)
    vector_store = VectorStore(str(DB_DIR), embedder.dimension)
    bm25 = load_bm25_index()
    return HybridSearch(vector_store, bm25, embedder)


def load_documents() -> List[Document]:
    docs = []
    for filepath in sorted(CONTENT_DIR.glob("*.md")):
        content = filepath.read_text(encoding="utf-8").strip()
        slug = filepath.stem
        if slug == "homepage":
            category = "Homepage"
            source_url = "https://productsway.com/"
        elif slug.startswith("til-") or slug.startswith("til "):
            category = "TIL"
            source_url = f"https://productsway.com/notes/{slug}"
        else:
            category = "Guide"
            source_url = f"https://productsway.com/notes/{slug}"
        title = content.split("\n")[0].lstrip("#").strip() if content else slug
        docs.append(
            Document(
                id=slug,
                title=title,
                content=content,
                source_url=source_url,
                category=category,
            )
        )
    print(f"Loaded {len(docs)} documents from {CONTENT_DIR}")
    return docs


def run_ingestion():
    print("=" * 60)
    print("RAG Pipeline — Ingestion")
    print("=" * 60)

    docs = load_documents()
    chunker = MarkdownChunker()
    all_chunks: List[dict] = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc))
    print(f"Generated {len(all_chunks)} chunks from {len(docs)} documents")

    chunks_meta = [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "content_length": len(c["content"]),
            "source_url": c["source_url"],
            "category": c["category"],
        }
        for c in all_chunks
    ]
    with open(CHUNKS_DIR / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, indent=2)

    embedder = Embedder()
    vector_store = VectorStore(str(DB_DIR), embedder.dimension)
    vector_store.ingest(all_chunks, embedder)

    bm25 = BM25Index()
    bm25.add_documents(
        doc_ids=[c["id"] for c in all_chunks],
        texts=[c["content"] for c in all_chunks],
    )
    bm25.chunk_meta = build_chunk_meta(all_chunks)
    save_bm25_index(bm25)

    print("\n✅ Ingestion complete!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Vector dim: {embedder.dimension}")
    print(f"   DB location: {DB_DIR}")
    return vector_store, bm25, embedder


if __name__ == "__main__":
    run_ingestion()