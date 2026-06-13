# rag-blog — https://just.systems/man/en/

set positional-arguments := true

export PATH := justfile_directory() + "/.venv/bin:" + env_var_or_default("PATH", "")

# Show recipes
default:
    @just --list

# Create .venv and install deps + dev tools (ruff, ty)
install:
    uv sync

# Sync locked deps only
sync:
    uv sync --frozen

# Chunker tests (stdlib; runs in uv venv)
test:
    uv run python -m unittest discover -s tests -v

# CSS lint (balanced @media blocks, body.index print present,
# no inline <style>/style= in source HTML). Stdlib-only.
css-lint:
    python scripts/css_lint.py

# Ruff lint
lint:
    uv run ruff check .

# Ruff format (write)
fmt:
    uv run ruff format .

# Ruff format check
fmt-check:
    uv run ruff format --check .

# Astral ty type checker
typecheck:
    uv run ty check

# Lint + format + types + tests + CSS lint
check: lint fmt-check typecheck test css-lint

# Install git hooks from prek.toml
prek-install:
    prek install

# Run all prek hooks on the repo
prek *args:
    prek run --all-files {{args}}

# Scrape → ingest
pipeline:
    uv run python scrape_content.py
    uv run python rag_pipeline.py

# API server
serve:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Lockfile from pyproject.toml
lock:
    uv lock

# ── Docker ───────────────────────────────────────────────────────────────

docker-build:
    docker compose build

docker-ingest:
    docker compose --profile ingest run --rm ingest

docker-up:
    docker compose up

docker-up-d:
    docker compose up -d

docker-down:
    docker compose down

docker-logs:
    docker compose logs -f api
