#!/bin/bash

# This script runs the full Paper2Code webapp with Celery in foreground mode

echo "Starting Paper2Code with Celery in foreground mode..."

# Create a terminal for Celery worker
gnome-terminal --title="Celery Worker" -- bash -c "
source .venv_env/bin/activate
cd webapp

# Set environment variables
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0 

# Load .env file if it exists
if [ -f ../.env ]; then
  echo 'Loading environment variables from .env file...'
  export \$(grep -v '^#' ../.env | xargs)
fi

# Kill any existing Celery workers
pkill -f 'celery worker' || true
sleep 2

# Start Celery worker in foreground mode
echo 'Starting Celery worker...'
celery -A app:celery worker --loglevel=info -n worker1@%h
"

# Wait for Celery to start
sleep 3

# Run the Flask app in the main terminal
source .venv_env/bin/activate
cd webapp

# Set environment variables
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0 
export FLASK_APP=app.py
export FLASK_ENV=development

# Load .env file if it exists
if [ -f ../.env ]; then
  echo 'Loading environment variables from .env file...'
  export $(grep -v '^#' ../.env | xargs)
fi

# Start the Flask app
echo 'Starting Flask app...'
python app.py