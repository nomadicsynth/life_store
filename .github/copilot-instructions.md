# Copilot instructions for this repo

Goal: help AI agents be productive immediately in this codebase by encoding architecture, workflows, and conventions that are not obvious from a single file.

## Big picture

- This repo is a collection of tiny, independent tools. Today there are two main pieces:
  - `inventory_photo_capture_app.py` - a Gradio UI that saves photos of items into a simple folder layout. No DB; filesystem is the source of truth.
  - `validate_items.py` - a CLI validator for `item.json` files using `jsonschema` (Draft-07) with optional file-existence and SHA-256 checks.
- Data model is file-first: items live under `inventory/Box_<BOX_ID>/Item_<ITEM_ID>/`. Photos are `photo_<YYYYMMDD>_<HHMMSS>_<index>.jpg`. Metadata (when present) is `item.json` defined by `docs/item.schema.json`.

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
  - Env overrides: `HOST`, `PORT`, `SSL_CERT`, `SSL_KEY` map to the corresponding CLI flags.
  - TLS rules: both `--cert` and `--key` must exist or the app exits; providing only one is a fatal error.
- Validate metadata
  - `python validate_items.py --inventory inventory --schema docs/item.schema.json --check-files --check-hashes`
  - Exit codes: `0` all valid, `1` validation/check failures, `2` usage/unexpected error.
- Tests
  - `pytest tests/`
  - Pattern: tests call CLIs via a `main(argv)` function rather than spawning processes. Follow this for new CLIs.

## Architecture & patterns

- Filesystem layout
  - Root: `inventory/`
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

- App UI: `inventory_photo_capture_app.py`
- Validator: `validate_items.py`
- Tests: `tests/test_validate_items.py`
- Schema docs: `docs/item_json_schema.md`
- JSON Schema: `docs/item.schema.json`
- Getting started: `AGENTS.md`, `docs/inventory_app.md` (remember the filename mismatch noted above)

## Notes for future changes

- If you implement writing `item.json` from the photo app, follow the schema and reuse the SHA-256 helper from `validate_items.py` to compute `hash_sha256` values.
- When introducing new tools, mirror the test approach in `tests/test_validate_items.py` and keep the file-first architecture unless there is a strong reason otherwise.
