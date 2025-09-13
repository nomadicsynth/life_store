# Item metadata schema (`item.json`)

This document defines the on-disk schema for per-item metadata files that live alongside photos. Each `Item_<id>/` directory contains an `item.json` describing the item and its photos.

## Location and naming

- Root: `inventory/Box_<box_id>/Item_<item_id>/`
- Metadata file: `inventory/Box_<box_id>/Item_<item_id>/item.json`
- Photo files: `photo_<YYYYMMDD>_<HHMMSS>_<index>.jpg`

IDs are sanitized for filesystem safety. Allowed characters: `[A-Za-z0-9_-]`.

## Example `item.json`

```json
{
  "schema_version": 1,
  "box_id": "B001",
  "item_id": "ab12cd34",
  "title": "Winter coats",
  "description": "Down and wool coats, sizes M-L",
  "tags": ["clothing", "winter"],
  "created_at": "2025-09-13T12:10:45Z",
  "updated_at": "2025-09-13T12:15:02Z",
  "photos": [
    {
      "file": "photo_20250913_121045_1.jpg",
      "caption": "Black wool coat",
      "hash_sha256": "4f8d..."
    },
    {
      "file": "photo_20250913_121045_2.jpg",
      "caption": "Blue down coat",
      "hash_sha256": "9a1c..."
    }
  ]
}
```

## Field reference

Top-level object fields:

- `schema_version` (integer, required)
  - Purpose: version the document for forward compatibility.
  - Current value: `1`.
- `box_id` (string, required)
  - The logical ID of the box containing this item. Sanitized to `[A-Za-z0-9_-]`.
- `item_id` (string, required)
  - The logical ID of the item. Often auto-generated. Sanitized to `[A-Za-z0-9_-]`.
- `title` (string, optional)
  - Short human-friendly name.
- `description` (string, optional)
  - Longer freeform notes.
- `tags` (array of string, optional)
  - Zero or more labels for search or grouping.
- `created_at` (string, required)
  - ISO 8601 UTC timestamp: `YYYY-MM-DDTHH:MM:SSZ`.
- `updated_at` (string, required)
  - ISO 8601 UTC timestamp. Update this on changes.
- `photos` (array, required)
  - Ordered list of photo objects, earliest to latest.

Photo object fields:

- `file` (string, required)
  - Filename of the image relative to the item folder, for example `photo_20250913_121045_1.jpg`.
- `caption` (string, optional)
  - Per-photo note.
- `hash_sha256` (string, required)
  - Hex-encoded SHA-256 of the image file contents for integrity checks.

## Conventions and constraints

- Filenames are case-sensitive on most Linux filesystems. Use consistent casing for IDs and file names.
- Item and box directories must exist before saving photos. The app creates them automatically.
- `photos[*].file` must point to an existing file within the same `Item_<id>/` directory.
- Avoid editing `item.json` while the app is writing to it. The app uses atomic writes and a lock file to reduce races.
- Schema evolution is additive whenever possible. Do not remove or repurpose fields within the same `schema_version`.

## JSON Schema (Draft 2020-12)

You can validate `item.json` files with this schema. Save it as `docs/item.schema.json` or embed in tooling as needed.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schemas/item.schema.json",
  "title": "Inventory Item",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "box_id",
    "item_id",
    "created_at",
    "updated_at",
    "photos"
  ],
  "properties": {
    "schema_version": { "type": "integer", "const": 1 },
    "box_id": { "type": "string", "pattern": "^[A-Za-z0-9_-]+$" },
    "item_id": { "type": "string", "pattern": "^[A-Za-z0-9_-]+$" },
    "title": { "type": "string" },
    "description": { "type": "string" },
    "tags": {
      "type": "array",
      "items": { "type": "string" }
    },
    "created_at": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$"
    },
    "updated_at": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$"
    },
    "photos": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["file", "hash_sha256"],
        "properties": {
          "file": { "type": "string" },
          "caption": { "type": "string" },
          "hash_sha256": { "type": "string", "pattern": "^[A-Fa-f0-9]{64}$" }
        }
      }
    }
  }
}
```

## Validation tips

Using Python:

```bash
python - <<'PY'
import json, sys
from jsonschema import validate

schema = json.load(open('docs/item.schema.json'))
data = json.load(open(sys.argv[1]))
validate(instance=data, schema=schema)
print('OK')
PY
```

Or install `jsonschema` and run a small checker:

```bash
pip install jsonschema
python -c "import json,sys;from jsonschema import validate;schema=json.load(open('docs/item.schema.json'));data=json.load(open(sys.argv[1]));validate(instance=data,schema=schema);print('OK')" inventory/Box_B001/Item_ab12cd34/item.json
```

## Versioning and migration

- Keep `schema_version` as an integer that increments on breaking changes.
- Readers should accept unknown optional fields to allow forward compatibility.
- When bumping the schema, provide a small migration script that rewrites old files to the new shape.

## Integrity checks

- `hash_sha256` lets you detect accidental edits or sync corruption. You can recompute and compare hashes to verify.
- If a photo is missing or the hash mismatches, flag the item for review.

## Backups and recovery

- The filesystem is the source of truth. You can rebuild any derived indexes by walking `inventory/` and reading `item.json` files.
- Use atomic writes and a lock file when updating `item.json` to avoid partial writes and races.
