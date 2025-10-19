#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Start the intelligent task manager app
python task_manager_app.py --host ${TASK_MANAGER_HOST} --port ${TASK_MANAGER_PORT} --db ${TASK_MANAGER_DB}
