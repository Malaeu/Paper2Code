#!/bin/bash

# Start Celery Flower monitoring tool

echo "Starting Celery Flower monitoring tool..."

# Activate virtual environment
source .venv_env/bin/activate

# Set Redis broker URL
export CELERY_BROKER_URL=redis://localhost:6379/0

# Load .env file if it exists
if [ -f .env ]; then
  echo 'Loading environment variables from .env file...'
  export $(grep -v '^#' .env | xargs)
fi

# Start Flower
cd webapp
celery -A app:celery flower --port=5555 --broker=$CELERY_BROKER_URL

echo "Monitor available at http://localhost:5555"