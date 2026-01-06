# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python 3.12 OCR-to-vector pipeline with a layered layout. Core domain entities live in `domain/`, ingestion/parsing logic in `ingestion/`, embedding generation in `embedding/`, retrieval/search in `retrieval/`, and database access in `storage/`. The API/CLI entry points are in `api/cli/`. Shared utilities and configuration are in `shared/`. The legacy pipeline and scripts remain under `app/` (use only when you need the older workflow). Architecture constraints are documented in `docs/ARCHITECTURE.md`, `docs/DOMAIN_RULES.md`, and `docs/PACKAGE_RULES.md`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Start pgvector DB: `docker-compose up -d`
- Build app image (includes OCR deps): `docker build -t ocr-vector-db .`
- Ingest documents (new CLI): `python -m api.cli.ingest docs/*.md pdf_data/*.pdf`
- Search: `python -m api.cli.search "query text" --view text --top-k 5`
- Legacy ingest (if needed): `python app/tools/ingest.py "pdf_data/*.pdf" --dry-run`

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python style (PEP 8). Prefer type hints for public APIs and dataclasses for config/data objects (see `shared/config.py`). Use `snake_case` for functions/variables, `PascalCase` for classes, and keep module names lowercase. Avoid cross-layer imports that violate `docs/PACKAGE_RULES.md`.

## Testing Guidelines
There is no pytest suite; tests are runnable scripts. Use:
- `python test_integration.py` for end-to-end import and validation checks.
- `python test_korean_validator.py` for eligibility validation rules.
Add new tests as standalone scripts or extend existing ones, and document the command in your PR.

## Commit & Pull Request Guidelines
Recent history uses a bracketed tag prefix, e.g. `[Refact] ...` or `[Fix] ...`. Follow that pattern with a short, specific summary. PRs should include a clear description, list of commands run (if any), and note config or schema changes (e.g., `.env` variables, DB migrations, new indexes).

## Configuration & Security Notes
Configuration is via `.env`. At minimum set `PG_CONN`, `COLLECTION_NAME`, and one embedding provider key (`VOYAGE_API_KEY` or `GOOGLE_API_KEY`). For scanned PDFs, set `ENABLE_AUTO_OCR=true` and install `ocrmypdf`. To always OCR PDFs regardless of text density, set `FORCE_OCR=true`. Default OCR languages are `kor+eng` (override with `OCR_LANGS`). The `Dockerfile` installs `tesseract-ocr`, `ghostscript`, and `poppler-utils` for OCR support. Never commit real API keys; use example values in docs or local env files only.


## Constraints
your answer should be korean
