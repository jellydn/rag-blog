"""Paths and constants (stdlib only — safe for scrape without ML deps)."""

import os
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


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