# 5. Curated static note catalog for ingestion

Date: 2026-06-07

## Status

Accepted

## Context

The corpus is fixed to public notes and guides on [productsway.com](https://productsway.com), not the entire site or dynamic feeds. We need repeatable ingest without a headless browser, sitemap crawler, or CMS API on Day 1. Scraping must stay importable from `config` only (stdlib path setup) so it never pulls ML dependencies.

## Decision

- **`scrape_content.py`** maintains an explicit **`NOTE_SLUGS`** list: `(slug, category)` pairs (TIL vs Guide) aligned with the live notes index.
- Fetch each page with **`urllib.request`** and a desktop **User-Agent**; parse HTML with **`html.parser.HTMLParser`** (`ContentExtractor`) — no BeautifulSoup or Playwright.
- Strip chrome via tag skip list (`nav`, `header`, `footer`, `script`, `style`); preserve headings and fenced-style code from `<pre>` / `<code>`.
- Write one markdown file per slug under `data/content/` (via `CONTENT_DIR` / `ensure_data_dirs()`), then hand off to `rag_pipeline.py` for chunk → embed → store.

## Consequences

### Positive

- Deterministic corpus: same slugs every run; easy to audit what is indexed.
- Zero extra scrape dependencies beyond the Python stdlib.
- Fits VPS and Docker ingest profile without browser stacks.

### Negative

- New posts require a manual slug entry and re-scrape; no automatic discovery.
- Fragile to site HTML redesign (extractor is not schema-driven).
- Category labels are metadata at scrape time only unless propagated into chunk metadata at ingest.

## References

- Ingest pipeline and data dirs: ADR-0002, ADR-0004.
- Operator flow: `mise run pipeline`, Docker Compose `ingest` profile (`README.md`).
