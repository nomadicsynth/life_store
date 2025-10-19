source .venv/bin/activate

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

python inventory_photo_capture_app.py
