#!/bin/bash

# Start a separate terminal for the worker
gnome-terminal --title="Celery CLI Test Worker" -- bash -c "
source .venv_env/bin/activate
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
echo 'Starting CLI test worker...'
python test_cli_celery.py worker
read -p 'Press any key to close this window...'
"

# Wait for worker to start
sleep 3

# Run the test client
source .venv_env/bin/activate
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
echo 'Running CLI test client...'
python test_cli_celery.py run