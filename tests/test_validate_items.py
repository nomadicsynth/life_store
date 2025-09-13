import json
import os
from pathlib import Path
import hashlib

import validate_items


def make_item_dir(tmp_path: Path, box_id: str, item_id: str) -> Path:
    d = tmp_path / "inventory" / f"Box_{box_id}" / f"Item_{item_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_photo(dir_path: Path, name: str, content: bytes) -> str:
    fpath = dir_path / name
    fpath.write_bytes(content)
    return name


def write_item_json(dir_path: Path, box_id: str, item_id: str, photos):
    # photos: list of dicts with keys: file, caption, hash_sha256
    data = {
        "schema_version": 1,
        "box_id": box_id,
        "item_id": item_id,
        "title": "Test Item",
        "description": "Smoke test",
        "tags": ["test"],
        "created_at": "2025-09-13T00:00:00Z",
        "updated_at": "2025-09-13T00:00:00Z",
        "photos": photos,
    }
    (dir_path / "item.json").write_text(json.dumps(data), encoding="utf-8")


def test_validator_passes_on_valid_item(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "docs" / "item.schema.json"

    item_dir = make_item_dir(tmp_path, "B001", "ab12cd34")
    fname = "photo_20250101_000000_1.jpg"
    content = b"sample image content"
    write_photo(item_dir, fname, content)
    h = hashlib.sha256(content).hexdigest()
    write_item_json(
        item_dir, "B001", "ab12cd34", photos=[{"file": fname, "caption": "", "hash_sha256": h}]
    )

    rc = validate_items.main(
        [
            "--inventory",
            str(tmp_path / "inventory"),
            "--schema",
            str(schema_path),
            "--check-files",
            "--check-hashes",
        ]
    )
    assert rc == 0


def test_validator_fails_on_missing_file(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "docs" / "item.schema.json"

    item_dir = make_item_dir(tmp_path, "B002", "deadbeef")
    # Do NOT create the photo file
    fname = "photo_20250101_000000_1.jpg"
    write_item_json(
        item_dir,
        "B002",
        "deadbeef",
        photos=[{"file": fname, "caption": "", "hash_sha256": "0" * 64}],
    )

    rc = validate_items.main(
        [
            "--inventory",
            str(tmp_path / "inventory"),
            "--schema",
            str(schema_path),
            "--check-files",
        ]
    )
    assert rc == 1
