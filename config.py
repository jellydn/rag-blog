"""Paths and constants (stdlib only — safe for scrape without ML deps)."""

import os
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
# Drop vector hits with cosine similarity below this floor. 0.0 = off
# (every hit passes). 0.3 is a reasonable production default that filters
# out chunks that the encoder considered essentially unrelated to the query.
COSINE_THRESHOLD = 0.0
# Drop BM25 hits with raw score below this floor. 0.0 = off (every hit
# passes). 1.0 is a reasonable production default that filters out chunks
# that scored on a single high-IDF token (a "single-token incidental
# match" — e.g. ".set({...}" matching the query word "set" via the JS
# method call) or on a pile of stopword-grade terms that barely nudge the
# score. The current BM25+ implementation uses delta=1.0, so a chunk
# that hits on *any* query term typically scores >= 2.0 on that term
# alone; a 1.0 floor still lets legitimate single-token matches through.
BM25_THRESHOLD = 1.0


def data_dir() -> Path:
    env = os.environ.get("RAG_BLOG_DATA")
    if env:
        return Path(env)
    opt = Path("/opt/data/rag-blog/data")
    if opt.exists():
        return opt
    return Path(__file__).resolve().parent / "data"


def ensure_data_dirs(base: Path | None = None) -> Path:
    root = base or data_dir()
    for sub in ("content", "chunks", "lancedb"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


DATA_DIR = data_dir()
CONTENT_DIR = DATA_DIR / "content"
CHUNKS_DIR = DATA_DIR / "chunks"
DB_DIR = DATA_DIR / "lancedb"
BM25_JSON = DB_DIR / "bm25_data.json"
