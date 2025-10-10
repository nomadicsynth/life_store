# Copilot instructions for this repo

Goal: help AI agents be productive immediately in this codebase by encoding architecture, workflows, and conventions that are not obvious from a single file.

## Big picture

- This repo is a collection of tiny, independent tools. Main pieces:
  - `inventory_photo_capture_app.py` - a Gradio UI that saves photos of items into a simple folder layout. No DB; filesystem is the source of truth.
  - `validate_items.py` - a CLI validator for `item.json` files using `jsonschema` (Draft-07) with optional file-existence and SHA-256 checks.
  - `priority_engine.py` - intelligent task prioritization system that learns what matters from embeddings and user behavior. Uses PCA/eigendecomposition to discover latent task dimensions and PDV (Preference Direction Vector) to learn preferences. No predefined properties.
  - `chat_log_analyzer.py` - analyzes AI conversation logs to extract passive task prioritization feedback (the "journaling sneak-attack").
  - `task_manager.py` / `task_manager_app.py` - task management with Gradio UI (work in progress, will integrate priority_engine).
- Data model is file-first: items live under `data/inventory/Box_<BOX_ID>/Item_<ITEM_ID>/`. Photos are `photo_<YYYYMMDD>_<HHMMSS>_<index>.jpg`. Metadata (when present) is `item.json` defined by `docs/item.schema.json`.

## Daily workflows

- Environment
  - Create venv and install deps:
    - `python -m venv .venv && source .venv/bin/activate`
    - `pip install -r requirements.txt`
- Run the photo app
  - HTTPS:
    - `python inventory_photo_capture_app.py --host 0.0.0.0 --port 8443 --cert cert.pem --key key.pem`
  - HTTP (local):
    - `python inventory_photo_capture_app.py --host 127.0.0.1 --port 8443`
  - Env overrides: `LIFESTORE_HOST`, `LIFESTORE_PORT`, `LIFESTORE_SSL_CERT`, `LIFESTORE_SSL_KEY` map to the corresponding CLI flags.
  - TLS rules: both `--cert` and `--key` must exist or the app exits; providing only one is a fatal error.
- Validate metadata
  - `python validate_items.py --inventory data/inventory --schema docs/item.schema.json --check-files --check-hashes`
  - Exit codes: `0` all valid, `1` validation/check failures, `2` usage/unexpected error.
- Priority engine & task manager
  - Test priority engine: `python test_priority_engine.py`
  - Example integration: `python example_integration.py`
  - Requires Ollama running with `granite-embedding:278m` model for embeddings
  - See `docs/priority_engine.md` for architecture details
- Tests
  - `pytest tests/`
  - Pattern: tests call CLIs via a `main(argv)` function rather than spawning processes. Follow this for new CLIs.

## Architecture & patterns

- Filesystem layout
  - Root: `data/inventory/`
  - Boxes: `Box_<BOX_ID>/` (IDs are free-form but assumed `[A-Za-z0-9_-]`).
  - Items: `Item_<ITEM_ID>/` containing photos and, optionally, `item.json`.
- UI app behavior
  - Box IDs are created on demand. Save requires at least one photo and a selected box.
  - Item IDs can be provided or auto-generated (8-char UUID slice). After a successful save, UI resets for the next item.
  - Images are saved as JPEG with timestamped filenames.
- Schema & validation
  - Canonical schema lives in `docs/item.schema.json` (see `docs/item_json_schema.md` for narrative docs and examples).
  - `validate_items.py` uses `jsonschema.Draft7Validator`; optional checks ensure referenced files exist and hashes match.

## Conventions

- Keep dependencies minimal; update `requirements.txt` if you add a widely used package.
- Writing style in docs: use hyphens instead of an em-dash (see `AGENTS.md`).
- Prefer small, composable scripts. For CLIs, expose a `main(argv)` callable to ease testing.

## Key references

- Inventory app UI: `inventory_photo_capture_app.py`
- Validator: `validate_items.py`
- Priority engine: `priority_engine.py`, `chat_log_analyzer.py`
- Tests: `tests/test_validate_items.py`, `test_priority_engine.py`
- Schema docs: `docs/item_json_schema.md`
- JSON Schemas: `docs/item.schema.json`, `docs/task.schema.json`
- Priority engine docs: `docs/priority_engine.md`, `docs/priority_engine_summary.md`
- Getting started: `AGENTS.md`, `docs/inventory_app.md`
- Reference implementation: `references/deep_research_tool.py` (dimension learning inspiration)

## Notes for future changes

- If you implement writing `item.json` from the photo app, follow the schema and reuse the SHA-256 helper from `validate_items.py` to compute `hash_sha256` values.
- When introducing new tools, mirror the test approach in `tests/test_validate_items.py` and keep the file-first architecture unless there is a strong reason otherwise.
