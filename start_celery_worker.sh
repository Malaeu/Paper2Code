#!/bin/bash

# Script to start the Celery worker for Paper2Code

echo "Starting Celery worker for Paper2Code..."

# Activate virtual environment
source .venv_env/bin/activate

# Set environment variables
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0 

# Load environment variables from .env file
if [ -f .env ]; then
  echo 'Loading environment variables from .env file...'
  export $(grep -v '^#' .env | xargs)
fi

# Kill any existing Celery workers
pkill -f 'celery worker' || true
sleep 2

# Go to webapp directory
cd webapp

# Make sure the app is importable
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Start Celery worker
echo "Starting Celery worker..."
celery -A app:celery worker --loglevel=info -n worker1@%h

# Note: If this fails, try the minimal app instead:
# celery -A minimal_app:celery worker --loglevel=info -n worker1@%h