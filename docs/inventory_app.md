# Inventory app - Photo capture and cataloging

A tiny Gradio web app to quickly catalog physical items into boxes with photos. Photos are saved to a simple, portable folder structure, with metadata stored in SQLite.

- Capture photos from your webcam/phone camera
- Group photos into Boxes (`Box_<id>`) and Items (`Item_<id>`)
- Auto-generate item IDs or enter your own
- Add titles, descriptions, and locations
- Photos saved as JPEGs, metadata in SQLite database

Tech: Python + Gradio + SQLite.

## Quick start

### Requirements

- Python 3.9+ (3.10/3.11 recommended)
- `pip`

### Setup

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (HTTPS)

If you have a certificate/private key pair (PEM files):

```bash
python inventory_photo_capture_app.py \
  --host 0.0.0.0 \
  --port 8443 \
  --cert cert.pem \
  --key key.pem
```

Then open: `https://<your-host>:8443`

### Run (HTTP, local)

You can also run without TLS for local use:

```bash
python inventory_photo_capture_app.py --host 127.0.0.1 --port 8443
```

Open: `http://127.0.0.1:8443`

### Env vars (optional)

You can override defaults via namespaced environment variables instead of CLI flags. These overrides
the whole project:

- `LIFESTORE_HOST` (default: `127.0.0.1`)
- `LIFESTORE_PORT` (default: `8443`)
- `LIFESTORE_SSL_CERT` (path to cert PEM)
- `LIFESTORE_SSL_KEY` (path to key PEM)

There are additional environment variables for inventory management:

- `LIFESTORE_INVENTORY_HOST` (default: `LIFESTORE_HOST`)
- `LIFESTORE_INVENTORY_PORT` (default: `LIFESTORE_PORT`)
- `LIFESTORE_INVENTORY_SSL_CERT` (default: `LIFESTORE_SSL_CERT`)
- `LIFESTORE_INVENTORY_SSL_KEY` (default: `LIFESTORE_SSL_KEY`)

Storage location:

- `LIFESTORE_DATA_PATH` (default: `data`) - Base path for all inventory data. Set in `.env` file.
  - Database: `{LIFESTORE_DATA_PATH}/inventory/inventory.db`
  - Photos: `{LIFESTORE_DATA_PATH}/inventory/Box_*/Item_*/photo_*.jpg`

Example `.env` file:

```bash
LIFESTORE_DATA_PATH=/mnt/embiggen/life_store_data
```

Example CLI:

```bash
export LIFESTORE_HOST=0.0.0.0
export LIFESTORE_PORT=8443
export LIFESTORE_SSL_CERT=cert.pem
export LIFESTORE_SSL_KEY=key.pem
python inventory_photo_capture_app.py
```

## Usage

### Web UI

1. Add a Box
   - Enter a Box ID (e.g., `B001`) and optional location (e.g., "Garage shelf 2").
   - Click "Add Box".
   - Select the box from the "Select Box" dropdown.
2. Capture photos
   - Use the camera input to frame an item, click "Add Photo".
   - Repeat to add multiple photos.
   - "Clear" removes all photos in the current batch.
3. Save
   - Optionally enter an Item ID or leave "Auto-generate Item ID" checked.
   - Add title and description (optional).
   - Click "Save All" to write images to disk and metadata to database.
4. Rinse and repeat for the next item.

### Command-line interface

Query and manage inventory from the terminal:

```bash
# List all boxes
python inventory_cli.py list-boxes

# List all items (or filter by box)
python inventory_cli.py list-items
python inventory_cli.py list-items --box B001

# Show detailed item info
python inventory_cli.py show-item <item_id>

# Update box metadata
python inventory_cli.py update-box B001 --location "Attic" --description "Winter storage"

# Update item metadata
python inventory_cli.py update-item <item_id> --title "Winter coats" --description "Sizes M-L"

# Search by tag
python inventory_cli.py search-tag clothing

# Export entire database to JSON
python inventory_cli.py export inventory_backup.json
```

### Migration tool

Import existing filesystem inventory into the database:

```bash
# Dry run (show what would be done)
python migrate_to_db.py --dry-run

# Actually migrate
python migrate_to_db.py
```

Notes

- Images are saved as JPEGs with timestamped filenames.
- Auto-generated Item IDs are short (8 chars) UUID slices.
- Metadata (titles, descriptions, locations) stored in SQLite.
- Photos remain on filesystem - database only tracks paths.

## Data layout

The app writes under `{LIFESTORE_DATA_PATH}/inventory/` using a predictable, portable structure:

```text
{LIFESTORE_DATA_PATH}/inventory/
  inventory.db              # SQLite database with metadata
  Box_<BOX_ID>/
    Item_<ITEM_ID>/
      photo_<YYYYMMDD>_<HHMMSS>_1.jpg
      photo_<YYYYMMDD>_<HHMMSS>_2.jpg
      ...
```

**Database schema:**

- `boxes` - Box metadata (box_id, location, description, timestamps)
- `items` - Item metadata (item_id, box_id, title, description, timestamps)
- `photos` - Photo records (photo_id, item_id, file_path, caption, hash_sha256, created_at)
- `tags` - Item tags for categorization and search

**Benefits:**

- Photos stay portable - just files on disk
- Metadata queryable via SQL
- Easy backup/restore - copy entire `{LIFESTORE_DATA_PATH}/inventory/` folder
- Can rebuild database from filesystem using migration tool

## HTTPS notes

- For HTTPS, both `--cert` and `--key` must be provided and must exist.
- If only one is provided, the app will abort with a clear error.
- For local experiments, you can generate a self-signed certificate. Example (Linux/macOS):

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=localhost"
  ```

- For remote access, prefer a proper certificate (for example, via a reverse proxy like Caddy or Nginx terminating TLS) and keep the app bound to `127.0.0.1`.
