# Life Store

A collection of minimal, independent apps for life stuff that will grow over time.

## Contents

- Overview
- Current apps
  - Inventory Photo Capture
- Ideas, contributions?

## Overview

This repo hosts independent, minimal apps. Each app tries to be:

- Simple to run locally
- Minimal dependencies
- Easy to back up (plain files where possible)

## Current apps

### Inventory Photo Capture (`inventory_photo_capture_app.py`)

A tiny Gradio web app to quickly catalog physical items into “boxes” with photos.

- Capture photos from your webcam/phone camera
- Group photos into Boxes (`Box_<id>`) and Items (`Item_<id>`)
- Auto-generate item IDs or enter your own
- Photos are saved as JPEGs into a simple, portable folder structure
- Full guide: see `docs/inventory_app.md` for setup, running, usage, data layout, and HTTPS notes.

Env overrides

- You can configure the app using these environment variables:
  - `LIFESTORE_HOST`, `LIFESTORE_PORT`, `LIFESTORE_SSL_CERT`, `LIFESTORE_SSL_KEY`
  - They map directly to the `--host`, `--port`, `--cert`, and `--key` CLI flags.

## Ideas, contributions?

If you stumble onto this and it’s useful to you too, drop me a line or star the repo. Ideas are welcome! If you find a rough edge while using an app, you're welcome to submit a PR.
