# Inventory app - Photo capture and cataloging

A tiny Gradio web app to quickly catalog physical items into boxes with photos. It saves images into a simple, portable folder structure without a database.

- Capture photos from your webcam/phone camera
- Group photos into Boxes (`Box_<id>`) and Items (`Item_<id>`)
- Auto-generate item IDs or enter your own
- Photos saved as JPEGs under `inventory/`

Tech: Python + Gradio.

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
python inventory_app.py \
  --host 0.0.0.0 \
  --port 8443 \
  --cert cert.pem \
  --key key.pem
```

Then open: `https://<your-host>:8443`

### Run (HTTP, local)

You can also run without TLS for local use:

```bash
python inventory_app.py --host 127.0.0.1 --port 8443
```

Open: `http://127.0.0.1:8443`

### Env vars (optional)

You can override defaults via environment variables instead of CLI flags:

- `HOST` (default: `127.0.0.1`)
- `PORT` (default: `8443`)
- `SSL_CERT` (path to cert PEM)
- `SSL_KEY` (path to key PEM)

Example:

```bash
export HOST=0.0.0.0
export PORT=8443
export SSL_CERT=cert.pem
export SSL_KEY=key.pem
python inventory_app.py
```

## Usage

1. Add a Box
   - Enter a Box ID (e.g., `B001`) and click "Add Box".
   - Select the box from the "Select Box" dropdown.
2. Capture photos
   - Use the camera input to frame an item, click "Add Photo".
   - Repeat to add multiple photos.
   - "Clear" removes all photos in the current batch.
3. Save
   - Optionally enter an Item ID or leave "Auto-generate Item ID" checked.
   - Click "Save All" to write images to disk.
4. Rinse and repeat for the next item.

Notes

- Images are saved as JPEGs with timestamped filenames.
- Auto-generated Item IDs are short (8 chars) UUID slices.

## Data layout

The app writes under `inventory/` using a predictable, portable structure:

```text
inventory/
  Box_<BOX_ID>/
    Item_<ITEM_ID>/
      photo_<YYYYMMDD>_<HHMMSS>_1.jpg
      photo_<YYYYMMDD>_<HHMMSS>_2.jpg
      ...
```

This makes backup/restore trivial. You can also search your filesystem for items or copy folders around without any special tooling.

## HTTPS notes

- For HTTPS, both `--cert` and `--key` must be provided and must exist.
- If only one is provided, the app will abort with a clear error.
- For local experiments, you can generate a self-signed certificate. Example (Linux/macOS):

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=localhost"
  ```

- For remote access, prefer a proper certificate (for example, via a reverse proxy like Caddy or Nginx terminating TLS) and keep the app bound to `127.0.0.1`.
