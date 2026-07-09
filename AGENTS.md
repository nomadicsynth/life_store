# AGENTS.md

## Setup commands

- Install dependencies with `uv`:
  - `uv sync`
  - This creates/updates the virtual environment and installs all dependencies from `pyproject.toml`.
- Activate the virtual environment (if needed):
  - `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- Start `inventory_app`: `python inventory_photo_capture_app.py`
- Run tests: `pytest tests/`

## Package management

- This project uses `uv` for package management.
- **Never manually edit dependencies in `pyproject.toml`**. Use `uv add <package>` to add dependencies and `uv remove <package>` to remove them.
- Run `uv sync` after any dependency changes to update the virtual environment.

## Writing style for general text

- Use hyphens (-) instead of an em-dashes (—).
