#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs uploads/papers uploads/datasets uploads/outputs

# Set environment variables
export FLASK_APP=run.py
export FLASK_ENV=development

echo "Setup complete. To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To start the Flask development server, run:"
echo "flask run"