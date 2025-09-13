import os
import uuid
import gradio as gr
from datetime import datetime
import argparse

# Root folder where everything is stored
INVENTORY_ROOT = "inventory"

LAN_IP = "127.0.0.1"
PORT = 8443
SSL_CERT = "cert.pem"
SSL_KEY = "key.pem"


def list_boxes():
    """Return a sorted list of existing box IDs (without 'Box_' prefix)."""
    if not os.path.isdir(INVENTORY_ROOT):
        return []
    boxes = []
    for name in os.listdir(INVENTORY_ROOT):
        if name.startswith("Box_") and os.path.isdir(os.path.join(INVENTORY_ROOT, name)):
            boxes.append(name[4:])
    return sorted(boxes)


def create_box(new_box_id):
    """Create a new box folder if it does not already exist and return updated choices + status."""
    new_box_id = (new_box_id or "").strip()
    if not new_box_id:
        return gr.update(), "⚠️ Enter a Box ID to add."
    box_folder = f"Box_{new_box_id}"
    path = os.path.join(INVENTORY_ROOT, box_folder)
    os.makedirs(path, exist_ok=True)
    choices = list_boxes()
    return gr.update(choices=choices, value=new_box_id), f"📦 Box '{new_box_id}' ready."

def add_photo(current_image, photos_state):
    """Append the currently captured image (PIL) to the list in state."""
    if current_image is None:
        return photos_state, gr.update(value=None), "⚠️ No image to add."
    photos_state = photos_state or []
    photos_state.append(current_image)
    return photos_state, gr.update(value=None), f"🟢 Added photo (total: {len(photos_state)})"


def clear_photos(_btn, photos_state):
    """Clear all captured photos."""
    return [], "🧹 Cleared photos." if photos_state else "No photos to clear." 


def save_photos(photos_state, selected_box_id, item_id, auto_generate):
    """Save list of PIL images captured via webcam."""
    if not photos_state or not selected_box_id:
        return "❌ Capture at least one photo and select a box."

    box_folder = f"Box_{selected_box_id.strip()}"
    box_path = os.path.join(INVENTORY_ROOT, box_folder)
    os.makedirs(box_path, exist_ok=True)

    if auto_generate or not item_id.strip():
        item_id = str(uuid.uuid4())[:8]
    item_folder = f"Item_{item_id.strip()}"
    item_path = os.path.join(box_path, item_folder)
    os.makedirs(item_path, exist_ok=True)

    ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx, img in enumerate(photos_state, start=1):
        filename = f"photo_{ts_prefix}_{idx}.jpg"
        file_path = os.path.join(item_path, filename)
        try:
            img.save(file_path)
        except Exception as e:
            return f"⚠️ Error saving image {idx}: {e}"

    return f"✅ Saved {len(photos_state)} photo(s) to {os.path.relpath(item_path, INVENTORY_ROOT)} (Item ID: {item_id})"

def post_save_reset(status_text):
    """If last status indicates success, produce component updates to reset inputs for next item.
    Returns tuple matching outputs: photos_state, gallery, cam, item_id textbox, save_status (unchanged)"""
    if not status_text.startswith("✅"):
        # No reset; pass through (None signals no change for component updates we won't include)
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), status_text
    # Reset states/components
    return [], gr.update(value=None), gr.update(value=None), gr.update(value=""), status_text

with gr.Blocks() as demo:
    gr.Markdown("# 📔 Inventory Photo Capture")
    photos_state = gr.State([])

    with gr.Row():
        with gr.Column():
            with gr.Row():
                webcamoptions = gr.WebcamOptions(mirror=False, constraints={"video": {"width": 512, "height": 512, "facingMode": "environment"}})
                cam = gr.Image(type="pil", label="Capture Photo", sources=["webcam"], webcam_options=webcamoptions)  # camera capture
            with gr.Row():
                add_btn = gr.Button("📸 Add Photo")
        with gr.Column():
            with gr.Row():
                gallery = gr.Gallery(label="Captured Photos", columns=4)
            with gr.Row():
                clear_btn = gr.Button("❌ Clear")

    gr.Markdown("### Box Management")
    with gr.Row():
        new_box_input = gr.Textbox(label="New Box ID", placeholder="e.g. B001")
    with gr.Row():
        add_box_btn = gr.Button("📦 Add Box")
    box_dropdown = gr.Dropdown(label="Select Box", choices=list_boxes(), value=None, interactive=True, filterable=False)
    box_status = gr.Markdown("")

    item_id = gr.Textbox(label="Item ID (optional, leave blank to auto-generate)")
    auto_generate = gr.Checkbox(label="Auto-generate Item ID", value=True)
    save_status = gr.Textbox(label="Status")

    save_btn = gr.Button("💾 Save All")

    # Interactions
    add_btn.click(fn=add_photo, inputs=[cam, photos_state], outputs=[photos_state, cam, save_status]).then(
        fn=lambda photos: photos, inputs=photos_state, outputs=gallery
    )
    clear_btn.click(fn=clear_photos, inputs=[clear_btn, photos_state], outputs=[photos_state, save_status]).then(
        fn=lambda photos: photos, inputs=photos_state, outputs=gallery
    )
    add_box_btn.click(fn=create_box, inputs=new_box_input, outputs=[box_dropdown, box_status])
    # Chain save -> reset
    save_btn.click(fn=save_photos, inputs=[photos_state, box_dropdown, item_id, auto_generate], outputs=save_status).then(
        fn=post_save_reset, inputs=save_status, outputs=[photos_state, gallery, cam, item_id, save_status]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory photo capture app")
    parser.add_argument("--host", default=os.environ.get("HOST", LAN_IP), help="Host/IP to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", PORT)), help="Port to bind")
    parser.add_argument("--cert", default=os.environ.get("SSL_CERT", SSL_CERT), help="Path to SSL certificate (PEM)")
    parser.add_argument("--key", default=os.environ.get("SSL_KEY", SSL_KEY), help="Path to SSL private key (PEM)")
    args = parser.parse_args()

    os.makedirs(INVENTORY_ROOT, exist_ok=True)

    cert_path = args.cert.strip()
    key_path = args.key.strip()
    use_https = False
    launch_kwargs = dict(server_name=args.host, server_port=args.port)
    if cert_path and key_path:
        if os.path.isfile(cert_path) and os.path.isfile(key_path):
            launch_kwargs.update(dict(ssl_certfile=cert_path, ssl_keyfile=key_path, ssl_verify=False))
            use_https = True
        else:
            raise SystemExit(f"[FATAL] Cert or key file missing: cert={cert_path}, key={key_path}. Aborting.")
    elif cert_path or key_path:
        raise SystemExit("[FATAL] Both --cert and --key are required for HTTPS. Aborting.")

    demo.launch(**launch_kwargs)
