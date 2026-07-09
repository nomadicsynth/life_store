# AGENTS.md

AI agent guide for the `life_store` codebase — a collection of independent, minimal Python tools.

## Setup

```bash
uv sync
```

Always use `uv` for dependency and venv management. Never edit `pyproject.toml` dependencies manually — use `uv add <pkg>`, `uv remove <pkg>`, etc.

Python 3.13+ required (see `.python-version`).

## Commands

| Task | Command |
|------|---------|
| Run tests | `uv run pytest tests/` |
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Remove dependency | `uv remove <package>` |
| Priority engine demo | `uv run python test_priority_engine.py` (requires Ollama) |

All scripts are run directly as `python <script>.py` — no console entry points in `pyproject.toml`.

## Architecture

Five independent domains, each with its own scripts and optional docs:

| Domain | Files | Docs |
|--------|-------|------|
| **Inventory** | `inventory_photo_capture_app.py`, `inventory_db.py`, `inventory_cli.py`, `validate_items.py`, `migrate_to_db.py` | [`docs/inventory_app.md`](docs/inventory_app.md), [`docs/item_json_schema.md`](docs/item_json_schema.md) |
| **Priority Engine** | `priority_engine.py`, `chat_log_analyzer.py`, `test_priority_engine.py`, `example_integration.py` | [`docs/priority_engine.md`](docs/priority_engine.md), [`docs/priority_engine_summary.md`](docs/priority_engine_summary.md) |
| **Task Manager** | `task_manager.py`, `task_manager_app.py` | [`docs/task_manager_integration.md`](docs/task_manager_integration.md) |
| **Cannabis Logistics** | `cannabis_logistics.py`, `budgeting.py` | [`docs/cannabis_logistics.md`](docs/cannabis_logistics.md) |
| **Validation** | `validate_items.py` | — |

### Data model

- **Inventory**: hybrid — photos on filesystem (`data/inventory/Box_<id>/Item_<id>/photo_*.jpg`), metadata in SQLite (`data/inventory/inventory.db`). Legacy `item.json` files validated by `validate_items.py` using [`docs/item.schema.json`](docs/item.schema.json).
- **Task Manager**: SQLite only (`tasks.db`).
- **Cannabis Logistics**: SQLite only (`data/therapeutics/cannabis/cannabis_logistics.db`).
- Base data path is `data/`, overridable via `LIFESTORE_DATA_PATH`.

### Key components

- `priority_engine.py`: ML-based task prioritization using PCA on embeddings + PDV (Preference Direction Vector) for learning preferences. No predefined task properties — dimensions are learned.
- `chat_log_analyzer.py`: extracts passive prioritization signals from AI conversation logs.
- Both require Ollama with `granite-embedding:278m` (768-dim embeddings).

## Conventions

- **CLI pattern**: every CLI exposes `main(argv)` for testability. Tests call `module.main([...])` directly — never spawn subprocesses.
- **Exit codes**: `0` success, `1` validation/usage error, `2` unexpected error.
- **Environment**: `.env` loaded via `python-dotenv`. Namespaced vars (e.g., `LIFESTORE_INVENTORY_HOST`, `LIFESTORE_CANNABIS_DB`).
- **Dependencies**: minimal. Use `uv add` / `uv remove` — never edit `pyproject.toml` manually.
- **Writing style**: hyphens instead of em-dashes in docs.
- **Scripts**: small and composable. Shell wrappers (`start_*.sh`) activate venv + load `.env`.

## Testing

- Suite: `uv run pytest tests/`
- Pattern: use `tmp_path` or `tempfile` for isolated fixtures, call `main([...])` directly.
- See `tests/test_validate_items.py`, `tests/test_inventory_db.py`, `tests/test_cannabis_logistics.py` for examples.

## References

- JSON schemas: [`docs/item.schema.json`](docs/item.schema.json), [`docs/task.schema.json`](docs/task.schema.json)
- Shell starters: `start_inv_capture.sh`, `start_task_manager.sh`
- Env template: `.env-template`
