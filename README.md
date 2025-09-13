# Life Store

This repo is the story of me getting my life in order - one tiny tool at a time. I’m building an ecosystem of apps as I need them, and letting the repo double as the logbook of what I tried, what I learned, and how far I’ve come.

The first chapter is simple in theory: catalog everything I own and sell what I can to fund the next steps of stabilizing life. But one does not simply sell their stuff. I have AuDHD, which means even “take a few photos and list it” turns into a full-on rabbit hole.

So here we are: I’m building an inventory system that will use a VLM (or whatever model makes sense) to label items from photos, automatically research what they’re worth, write up descriptions, generate marketplace blurbs, and probably do three other helpful-but-distracting things I’ll think of instead of actually listing the items. The point isn’t just to avoid the task - it’s to build a workflow that makes it doable, repeatable, and eventually effortless.

This repo will grow as a collection of small, focused tools. The first one helps me capture photos and organize them into boxes and items. As I go, I’ll bolt on the smarts - models, automations, and glue - to turn chaos into traction.

## Contents

- Overview
- Current apps
  - Inventory Photo Capture (`inventory_photo_capture_app.py`)
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
