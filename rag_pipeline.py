#!/usr/bin/env python3
"""
Production RAG Pipeline for Productsway.com

Features:
- Markdown-aware semantic chunking
- Hybrid search (vector + BM25 full-text + re-ranking)
- LanceDB vector store (file-based, no server)
- Streaming FastAPI server
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path("/opt/data/rag-blog/data")
CONTENT_DIR = DATA_DIR / "content"
CHUNKS_DIR = DATA_DIR / "chunks"
DB_DIR = DATA_DIR / "lancedb"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, ~80MB, fast on CPU
CHUNK_SIZE = 512  # target chars per chunk
CHUNK_OVERLAP = 64  # overlap between chunks

os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A raw document scraped from the site."""
    id: str
    title: str
    content: str
    source_url: str
    category: str  # "TIL", "Guide", "Homepage", "Post"
    tags: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A chunk of a document with embedding."""
    id: str
    doc_id: str
    title: str
    content: str
    embedding: List[float]  # 384-dim vector
    source_url: str
    category: str
    chunk_index: int
    total_chunks: int


# ─── Markdown-Aware Chunker ──────────────────────────────────────────────────

class MarkdownChunker:
    """
    Smart chunker that respects markdown structure.
    - Keeps headings with their content
    - Splits on ## or ### boundaries when content is too long
    - Never splits mid-code-block
    - Preserves context overlap
    """

    # These headings indicate a new logical section
    SECTION_HEADING_RE = re.compile(r'^##+\s+', re.MULTILINE)
    CODE_BLOCK_RE = re.compile(r'^```', re.MULTILINE)

    def chunk(self, doc: Document) -> List[dict]:
        """Split a Document into chunks, returning list of dicts."""
        raw = doc.content
        chunks_raw = self._split_by_headings(raw)
        
        result = []
        for i, chunk_text in enumerate(chunks_raw):
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < 20:
                continue
            
            # Determine a sub-title from the chunk's first heading line
            first_line = chunk_text.split('\n')[0]
            sub_title = first_line.lstrip('#').strip() if first_line.startswith('#') else doc.title
            
            result.append({
                "title": sub_title,
                "content": chunk_text,
                "doc_id": doc.id,
                "source_url": doc.source_url,
                "category": doc.category,
                "chunk_index": i,
                "total_chunks": len([c for c in chunks_raw if len(c.strip()) >= 20]),
            })
        
        return result

    def _split_by_headings(self, text: str) -> List[str]:
        """Split markdown text on ## or ### headings."""
        # First pass: split on section headings
        heading_matches = list(self.SECTION_HEADING_RE.finditer(text))
        
        if not heading_matches:
            # No headings — split by char count with overlap
            return self._split_by_size(text)
        
        sections = []
        for i, match in enumerate(heading_matches):
            start = match.start()
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
            section = text[start:end].strip()
            if section:
                # If section is still too long, further split
                if len(section) > CHUNK_SIZE * 1.5:
                    sections.extend(self._split_by_size(section))
                else:
                    sections.append(section)
        
        # Also grab content before first heading
        if heading_matches and heading_matches[0].start() > 0:
            pre = text[:heading_matches[0].start()].strip()
            if pre:
                sections.insert(0, pre)
        
        return sections if sections else [text]

    def _split_by_size(self, text: str) -> List[str]:
        """Split text into chunks of target size, respecting code blocks."""
        chunks = []
        lines = text.split('\n')
        current = []
        current_len = 0
        in_code = False
        
        for line in lines:
            # Track code blocks
            if self.CODE_BLOCK_RE.match(line.strip()):
                in_code = not in_code
            
            current.append(line)
            current_len += len(line) + 1  # +1 for newline
            
            if current_len >= CHUNK_SIZE and not in_code:
                chunks.append('\n'.join(current))
                
                # Keep overlap lines
                overlap_chars = 0
                overlap_lines = []
                for l in reversed(current):
                    if overlap_chars >= CHUNK_OVERLAP:
                        break
                    overlap_lines.insert(0, l)
                    overlap_chars += len(l) + 1
                
                current = overlap_lines
                current_len = overlap_chars
                in_code = False  # Reset code tracking at boundary
        
        if current:
            remaining = '\n'.join(current).strip()
            if remaining:
                chunks.append(remaining)
        
        return chunks


# ─── Embedding ───────────────────────────────────────────────────────────────

class Embedder:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Model loaded in {time.time()-t0:.1f}s (dim={self.dimension})")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def embed_one(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


# ─── BM25 Full-Text Search ──────────────────────────────────────────────────

class BM25Index:
    """
    Simple in-memory BM25 search for hybrid ranking.
    Implements BM25+ ranking function.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_freqs: dict = {}  # term -> # docs containing term
        self.doc_lengths: List[int] = []
        self.doc_texts: List[str] = []
        self.doc_ids: List[str] = []
        self.avg_doc_length: float = 0
        self.total_docs: int = 0
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alpha, remove short tokens."""
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9#]{1,50}\b', text.lower())
        return tokens

    def add_documents(self, doc_ids: List[str], texts: List[str]):
        """Add documents to the index."""
        self.doc_ids = doc_ids
        self.doc_texts = texts
        self.doc_lengths = [len(self._tokenize(t)) for t in texts]
        self.total_docs = len(texts)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

        # Build term frequency maps
        term_to_df = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                term_to_df[token] = term_to_df.get(token, 0) + 1
        self.doc_freqs = term_to_df
        self._built = True
        print(f"  BM25 index built: {self.total_docs} docs, {len(term_to_df)} unique terms")

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Search with BM25+ ranking."""
        if not self._built:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        N = self.total_docs
        scores = np.zeros(N)
        
        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            if df == 0:
                continue
            
            for i in range(N):
                doc_tokens = self._tokenize(self.doc_texts[i])
                tf = doc_tokens.count(token)
                dl = self.doc_lengths[i]
                
                # BM25+ scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_length)
                score = idf * (numerator / denominator + self.delta) if denominator > 0 else 0
                scores[i] += score
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "content": self.doc_texts[idx],
                    "score": float(scores[idx]),
                })
        
        return results


# ─── Vector Store (LanceDB) ──────────────────────────────────────────────────

class VectorStore:
    """LanceDB-backed vector store with hybrid search capability."""

    TABLE_NAME = "rag_chunks"

    def __init__(self, db_path: str, dimension: int = 384):
        self.db = lancedb.connect(str(db_path))
        self.dimension = dimension
        self._ensure_table()

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        try:
            self.table = self.db.open_table(self.TABLE_NAME)
            print(f"  Opened existing table: {self.table.count_rows()} rows")
        except Exception:
            self.table = None
            print("  New table will be created on first ingest")

    def _create_table(self, first_chunk: dict, embedding: List[float]):
        """Create the table with the first batch of data."""
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
        """Ingest chunks into the vector store."""
        if not chunks:
            print("  No chunks to ingest")
            return

        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)
        
        # Create or append to table
        if self.table is None:
            self._create_table(chunks[0], embeddings[0].tolist())
        
        import pyarrow as pa
        
        batch = pa.table({
            "id": [c["doc_id"] + f":{c['chunk_index']}" for c in chunks],
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
        """Search by vector similarity."""
        if self.table is None:
            return []
        
        results = (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "content": r["content"],
                "source_url": r["source_url"],
                "category": r["category"],
                "score": r["_distance"],
            }
            for r in results
        ]

    def count(self) -> int:
        if self.table is None:
            return 0
        return self.table.count_rows()


# ─── Hybrid Search ──────────────────────────────────────────────────────────

class HybridSearch:
    """
    Combines vector search + BM25 with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        rrf_k: int = 60,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Hybrid search with RRF fusion."""
        # 1. Vector search
        query_vec = self.embedder.embed_one(query)
        vec_results = self.vector_store.vector_search(query_vec, top_k=top_k * 2)
        
        # 2. BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # 3. Reciprocal Rank Fusion
        scores = {}
        
        for rank, r in enumerate(vec_results):
            doc_id = r["id"]
            score = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))
            if doc_id not in scores:
                scores[doc_id] = {**r, "rrf_score": 0, "vector_rank": rank + 1, "bm25_rank": None}
            scores[doc_id]["rrf_score"] += score
        
        for rank, r in enumerate(bm25_results):
            doc_id = r["doc_id"]
            # BM25 results use doc_id as key (no chunk ID)
            score = self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
            
            # Map BM25 doc_id to matching vector results
            matched = False
            for vid, vdata in scores.items():
                if vdata["doc_id"] == doc_id:
                    vdata["rrf_score"] += score
                    vdata["bm25_rank"] = rank + 1
                    matched = True
                    break
            
            if not matched:
                # BM25-only result not in vector results
                key = f"bm25:{doc_id}"
                scores[key] = {
                    "id": key,
                    "doc_id": doc_id,
                    "content": r["content"],
                    "title": doc_id.replace("-", " ").title(),
                    "source_url": f"https://productsway.com/notes/{doc_id}",
                    "category": "TIL",
                    "rrf_score": score,
                    "vector_rank": None,
                    "bm25_rank": rank + 1,
                }
        
        # Sort by RRF score descending
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]
        
        return sorted_results


# ─── Ingestion Pipeline ──────────────────────────────────────────────────────

def load_documents() -> List[Document]:
    """Load all scraped markdown files as Documents."""
    docs = []
    
    for filepath in sorted(CONTENT_DIR.glob("*.md")):
        content = filepath.read_text().strip()
        slug = filepath.stem
        
        # Determine category from filename
        if slug == "homepage":
            category = "Homepage"
        elif slug.startswith("til-") or slug.startswith("til "):
            category = "TIL"
        else:
            category = "Guide"
        
        # Determine title from first heading
        title = content.split('\n')[0].lstrip('#').strip() if content else slug
        source_url = f"https://productsway.com/notes/{slug}" if slug != "homepage" else "https://productsway.com/"
        
        doc = Document(
            id=slug,
            title=title,
            content=content,
            source_url=source_url,
            category=category,
        )
        docs.append(doc)
    
    print(f"Loaded {len(docs)} documents from {CONTENT_DIR}")
    return docs


def run_ingestion():
    """Full ingestion pipeline: load → chunk → embed → store."""
    print("=" * 60)
    print("RAG Pipeline — Ingestion")
    print("=" * 60)
    
    # 1. Load documents
    docs = load_documents()
    
    # 2. Chunk documents
    chunker = MarkdownChunker()
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} chunks from {len(docs)} documents")
    
    # Save chunks metadata for later inspection
    chunks_meta = []
    for c in all_chunks:
        chunks_meta.append({
            "doc_id": c["doc_id"],
            "title": c["title"],
            "content_length": len(c["content"]),
            "source_url": c["source_url"],
            "category": c["category"],
        })
    with open(CHUNKS_DIR / "chunks_meta.json", "w") as f:
        json.dump(chunks_meta, f, indent=2)
    
    # 3. Embed
    embedder = Embedder()
    
    # 4. Store in LanceDB
    vector_store = VectorStore(str(DB_DIR), embedder.dimension)
    vector_store.ingest(all_chunks, embedder)
    
    # 5. Build BM25 index
    bm25 = BM25Index()
    bm25.add_documents(
        doc_ids=[c["doc_id"] for c in all_chunks],
        texts=[c["content"] for c in all_chunks],
    )
    
    # Save BM25 index data for the server
    import pickle
    with open(DB_DIR / "bm25_data.pkl", "wb") as f:
        pickle.dump({
            "doc_ids": bm25.doc_ids,
            "doc_texts": bm25.doc_texts,
            "doc_lengths": bm25.doc_lengths,
            "total_docs": bm25.total_docs,
            "avg_doc_length": bm25.avg_doc_length,
            "doc_freqs": bm25.doc_freqs,
        }, f)
    
    print(f"\n✅ Ingestion complete!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Vector dim: {embedder.dimension}")
    print(f"   DB location: {DB_DIR}")
    
    return vector_store, bm25, embedder


if __name__ == "__main__":
    run_ingestion()
