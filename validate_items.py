#!/usr/bin/env python3
"""
Walk `inventory/` and validate each `item.json` against `docs/item.schema.json`.

Features:
- JSON Schema validation (draft-07).
- Optional checks:
  * Verify all referenced photo files exist.
  * Verify SHA-256 hashes of photos match `hash_sha256`.
- Clear per-item results and a summary with a proper exit code.

Usage:
  python validate_items.py [--inventory INVENTORY_DIR] [--schema SCHEMA_PATH] [--check-files] [--check-hashes]

Exit codes:
  0: All valid
  1: Validation or checks failed
  2: CLI usage or unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from typing import List, Tuple

try:
    from jsonschema import Draft7Validator
except Exception as e:  # pragma: no cover
    print("Missing dependency: jsonschema. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(2)


DEFAULT_INVENTORY = os.environ.get("INVENTORY_ROOT", "inventory")
DEFAULT_SCHEMA = os.path.join("docs", "item.schema.json")


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_schema(schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_item_jsons(root: str) -> List[str]:
    results: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "item.json" in filenames:
            results.append(os.path.join(dirpath, "item.json"))
    return sorted(results)


def validate_item(path: str, validator: Draft7Validator, check_files: bool, check_hashes: bool) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"{path}: Failed to parse JSON: {e}"]

    schema_errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if schema_errors:
        for err in schema_errors:
            loc = "/".join([str(p) for p in err.path]) or "<root>"
            errors.append(f"{path}: Schema error at {loc}: {err.message}")

    # Optional file existence check
    item_dir = os.path.dirname(path)
    if check_files:
        photos = data.get("photos", []) if isinstance(data, dict) else []
        for idx, p in enumerate(photos):
            if not isinstance(p, dict):
                errors.append(f"{path}: photos[{idx}] is not an object")
                continue
            fname = p.get("file")
            if not isinstance(fname, str):
                errors.append(f"{path}: photos[{idx}].file missing or not a string")
                continue
            fpath = os.path.join(item_dir, fname)
            if not os.path.isfile(fpath):
                errors.append(f"{path}: Missing photo file: {fname}")

    # Optional hash check
    if check_hashes and not errors:
        # Only compute hashes if basic checks passed to avoid noise
        photos = data.get("photos", [])
        for idx, p in enumerate(photos):
            fname = p.get("file")
            expected = p.get("hash_sha256")
            if not isinstance(fname, str) or not isinstance(expected, str):
                errors.append(f"{path}: photos[{idx}] missing file or hash_sha256")
                continue
            fpath = os.path.join(item_dir, fname)
            if not os.path.isfile(fpath):
                errors.append(f"{path}: Missing photo file for hashing: {fname}")
                continue
            try:
                actual = sha256_of_file(fpath)
            except Exception as e:
                errors.append(f"{path}: Failed to hash {fname}: {e}")
                continue
            if actual.lower() != expected.lower():
                errors.append(
                    f"{path}: Hash mismatch for {fname}: expected {expected}, got {actual}"
                )

    return (len(errors) == 0), errors


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Validate inventory item.json files")
    ap.add_argument("--inventory", default=DEFAULT_INVENTORY, help="Inventory root directory (default: inventory)")
    ap.add_argument("--schema", default=DEFAULT_SCHEMA, help="Path to JSON Schema file (default: docs/item.schema.json)")
    ap.add_argument("--check-files", action="store_true", help="Check that referenced photo files exist")
    ap.add_argument("--check-hashes", action="store_true", help="Verify SHA-256 hashes of photos (implies --check-files)")
    args = ap.parse_args(argv)

    if args.check_hashes:
        args.check_files = True

    inv_root = args.inventory
    schema_path = args.schema

    if not os.path.isdir(inv_root):
        print(f"Inventory directory not found: {inv_root}", file=sys.stderr)
        return 1
    if not os.path.isfile(schema_path):
        print(f"Schema file not found: {schema_path}", file=sys.stderr)
        return 1

    try:
        schema = load_schema(schema_path)
        validator = Draft7Validator(schema)
    except Exception as e:
        print(f"Failed to load schema: {e}", file=sys.stderr)
        return 2

    item_paths = find_item_jsons(inv_root)
    if not item_paths:
        print("No item.json files found.")
        return 1

    total = len(item_paths)
    failures = 0
    for item_json in item_paths:
        ok, errs = validate_item(item_json, validator, args.check_files, args.check_hashes)
        if ok:
            print(f"OK  - {item_json}")
        else:
            failures += 1
            print(f"FAIL- {item_json}")
            for msg in errs:
                print(f"  - {msg}")

    print("")
    print(f"Validated: {total} | Passed: {total - failures} | Failed: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(2)
