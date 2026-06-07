# rag-blog — https://just.systems/man/en/

set positional-arguments := true

# Show recipes
default:
    @just --list

# Python dependencies (API + ingest)
install:
    pip install -r requirements.txt

# Chunker tests (no sentence-transformers required)
test:
    python3 -m unittest discover -s tests -v

# Lint
lint:
    ruff check .

# Format (write)
fmt:
    ruff format .

# Check format without writing
fmt-check:
    ruff format --check .

# Lint + format check + tests
check: lint fmt-check test

# Install git hooks from prek.toml
prek-install:
    prek install

# Run all prek hooks on the repo
prek *args:
    prek run --all-files {{args}}

# Scrape → ingest → optional query (needs network + ML deps for ingest)
pipeline:
    python3 scrape_content.py
    python3 rag_pipeline.py

# API server
serve:
    python3 server.py

# ── Docker ───────────────────────────────────────────────────────────────

# Build image
docker-build:
    docker compose build

# Scrape + ingest into compose volume (first run)
docker-ingest:
    docker compose --profile ingest run --rm ingest

# Run API in foreground
docker-up:
    docker compose up

# Run API detached
docker-up-d:
    docker compose up -d

# Stop API
docker-down:
    docker compose down

# Tail API logs
docker-logs:
    docker compose logs -f api
