# 3. Chunk identity and markdown-aware chunking

Date: 2026-06-06

## Status

Accepted

## Context

RRF fusion must merge vector and BM25 hits for the **same passage**. Using document slug only as the BM25 key caused overwrites when one note produced multiple chunks and incorrect boosts during fusion. Chunking must respect markdown structure and not split inside fenced code blocks.

## Decision

- **Chunking** (`chunking.py`): Split on `##` / `###` outside code fences; fall back to size-based splits (512 chars, 64 overlap) inside long sections; minimum chunk length 20 characters.
- **Identity:** Every chunk has stable id **`{doc_id}:{chunk_index}`** on the chunk record (`chunking.chunk_id`), used as LanceDB `id` and BM25 `doc_ids` entry.
- **Metadata:** At ingest, persist `chunk_meta` in `bm25_data.json` keyed by chunk id (title, source_url, category, indices) so BM25-only RRF results expose the same fields as vector hits.

## Consequences

### Positive

- RRF merges on a single key; no URL/slug guessing in fusion.
- Code blocks and in-block `##` comments stay intact in chunks.
- BM25-only results remain presentable in API/CLI output.

### Negative

- Re-ingest mandatory when slug list or chunking rules change.
- `chunk_index` is positional per ingest, not content-hash stable across full rebuilds.
