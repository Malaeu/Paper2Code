#!/usr/bin/env python3
"""
Paper2Code Web Application (Minimal Version)

This is a simplified version of the Flask application without complex dependencies.
It provides a minimal web interface for the Paper2Code system.
"""

import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from celery import Celery

# Configure Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Add context processor for templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'papers'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'outputs'), exist_ok=True)

# Configure Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Make sure Celery tasks are properly registered with this app instance
@celery.task(bind=True)
def analyze_dataset_task(self, session_id):
    """Celery task to analyze the uploaded dataset."""
    try:
        # Load session info
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', session_id)
        with open(os.path.join(session_folder, 'session_info.json'), 'r') as f:
            session_info = json.load(f)
        
        # Update status
        session_info['status'] = 'analyzing'
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        # Analyze dataset using simple function
        dataset_analysis = simple_analyze_dataset(
            session_info['dataset_path'], 
            session_info['dataset_format']
        )
        
        # Save analysis result
        with open(os.path.join(session_folder, 'dataset_analysis.json'), 'w') as f:
            json.dump(dataset_analysis, f, indent=2)
        
        # Update status
        session_info['status'] = 'analyzed'
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        return {'status': 'success', 'session_id': session_id}
    
    except Exception as e:
        # Update status with error
        session_info['status'] = 'error'
        session_info['error'] = str(e)
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        return {'status': 'error', 'message': str(e), 'session_id': session_id}

# Allowed file extensions
ALLOWED_PAPER_EXTENSIONS = {'pdf', 'json', 'md'}
ALLOWED_DATASET_EXTENSIONS = {'csv', 'parquet', 'xlsx', 'xls', 'json'}

def allowed_file(filename, allowed_extensions):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Simple dataset analysis function (placeholder)
def simple_analyze_dataset(dataset_path, dataset_format):
    """
    Simple dataset analysis without pandas dependency.
    Returns basic info about the dataset file.
    """
    file_size = os.path.getsize(dataset_path)
    file_info = {
        "filename": os.path.basename(dataset_path),
        "format": dataset_format,
        "size_bytes": file_size,
        "size_readable": f"{file_size / (1024*1024):.2f} MB",
        "last_modified": datetime.fromtimestamp(os.path.getmtime(dataset_path)).isoformat(),
        "columns": ["This is a placeholder. Full analysis requires pandas library."],
    }
    return file_info

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads for papers and datasets."""
    if request.method == 'POST':
        # Check if the post request has the file parts
        if 'paper' not in request.files or 'dataset' not in request.files:
            flash('Both paper and dataset files are required')
            return redirect(request.url)
        
        paper_file = request.files['paper']
        dataset_file = request.files['dataset']
        
        # If user does not select files, browser submits empty file without filename
        if paper_file.filename == '' or dataset_file.filename == '':
            flash('Both paper and dataset files are required')
            return redirect(request.url)
        
        # Create a unique session ID for this adaptation
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Process paper file
        if paper_file and allowed_file(paper_file.filename, ALLOWED_PAPER_EXTENSIONS):
            paper_filename = secure_filename(paper_file.filename)
            paper_path = os.path.join(session_folder, paper_filename)
            paper_file.save(paper_path)
        else:
            flash('Invalid paper file format. Allowed formats: PDF, JSON, Markdown')
            return redirect(request.url)
        
        # Process dataset file
        if dataset_file and allowed_file(dataset_file.filename, ALLOWED_DATASET_EXTENSIONS):
            dataset_filename = secure_filename(dataset_file.filename)
            dataset_path = os.path.join(session_folder, dataset_filename)
            dataset_file.save(dataset_path)
            
            # Determine dataset format
            dataset_format = dataset_filename.rsplit('.', 1)[1].lower()
            if dataset_format in ('xlsx', 'xls'):
                dataset_format = 'excel'
        else:
            flash('Invalid dataset file format. Allowed formats: CSV, Parquet, Excel, JSON')
            return redirect(request.url)
        
        # Store session information
        session_info = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'paper_path': paper_path,
            'dataset_path': dataset_path,
            'dataset_format': dataset_format,
            'status': 'uploaded',
            'repo_name': request.form.get('repo_name', 'AdaptedModel')
        }
        
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        # Start dataset analysis as a background task
        analyze_dataset_task.delay(session_id)
        
        return redirect(url_for('configure', session_id=session_id))
    
    return render_template('upload.html')

@app.route('/configure/<session_id>', methods=['GET', 'POST'])
def configure(session_id):
    """Configure the adaptation parameters."""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', session_id)
    
    # Check if session exists
    if not os.path.exists(session_folder):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load session info
    with open(os.path.join(session_folder, 'session_info.json'), 'r') as f:
        session_info = json.load(f)
    
    # Check if dataset analysis is complete
    analysis_path = os.path.join(session_folder, 'dataset_analysis.json')
    if not os.path.exists(analysis_path):
        return render_template('waiting.html', session_id=session_id, status=session_info['status'])
    
    # Load dataset analysis
    with open(analysis_path, 'r') as f:
        dataset_analysis = json.load(f)
    
    if request.method == 'POST':
        # Extract variable mappings from form
        variable_mapping = {}
        for key, value in request.form.items():
            if key.startswith('map_') and value:
                original_var = key.replace('map_', '')
                variable_mapping[original_var] = value
        
        # Create configuration
        config = {
            'paper': {
                'title': request.form.get('paper_title', ''),
                'authors': request.form.get('paper_authors', ''),
                'year': int(request.form.get('paper_year', 2023)),
                'methodology': request.form.get('paper_methodology', ''),
                'json_path': session_info['paper_path']
            },
            'dataset': {
                'path': session_info['dataset_path'],
                'format': session_info['dataset_format']
            },
            'variable_mapping': {
                'original_to_adapted': variable_mapping
            },
            'methodology': {
                'maintain_landmark_analysis': request.form.get('maintain_landmark') == 'on',
                'maintain_monte_carlo_cv': request.form.get('maintain_monte_carlo') == 'on',
                'iterations': int(request.form.get('iterations', 1000)),
                'models_to_include': request.form.getlist('models')
            },
            'output': {
                'repo_name': request.form.get('repo_name', session_info['repo_name']),
                'output_dir': os.path.join(session_folder, 'repo'),
                'include_tests': request.form.get('include_tests') == 'on',
                'include_documentation': request.form.get('include_documentation') == 'on'
            },
            'advanced': {
                'use_direct_api': True
            }
        }
        
        # Generate dataset description
        dataset_description = f"""# Dataset Description

## Overview
This dataset contains variables that will be mapped to the methodology from the paper.

## Dataset Information
- Format: {session_info['dataset_format']}
- Columns: {', '.join(dataset_analysis.get('columns', []))}

## Variable Mapping
The following variables from the original paper will be mapped to dataset variables:
"""
        for orig, adapted in variable_mapping.items():
            dataset_description += f"- {orig} -> {adapted}\n"
        
        # Save configuration and description
        config_path = os.path.join(session_folder, 'adapt_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        description_path = os.path.join(session_folder, 'dataset_description.md')
        with open(description_path, 'w') as f:
            f.write(dataset_description)
        
        # Update session info
        session_info['status'] = 'configured'
        session_info['config_path'] = config_path
        session_info['description_path'] = description_path
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        # Normally we would start plan generation as a background task here
        # But in this minimal version, we'll just update the status
        session_info['status'] = 'plan_generated'
        session_info['plan_path'] = 'Placeholder - full functionality requires additional libraries'
        with open(os.path.join(session_folder, 'session_info.json'), 'w') as f:
            json.dump(session_info, f, indent=2)
        
        return redirect(url_for('result', session_id=session_id))
    
    return render_template(
        'configure.html', 
        session_id=session_id, 
        session_info=session_info,
        dataset_analysis=dataset_analysis
    )

@app.route('/result/<session_id>')
def result(session_id):
    """Show the results of the adaptation process."""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', session_id)
    
    # Check if session exists
    if not os.path.exists(session_folder):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load session info
    with open(os.path.join(session_folder, 'session_info.json'), 'r') as f:
        session_info = json.load(f)
    
    return render_template(
        'result.html',
        session_id=session_id,
        session_info=session_info
    )

@app.route('/status/<session_id>')
def status(session_id):
    """API endpoint to check the status of a session."""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', session_id)
    
    # Check if session exists
    if not os.path.exists(session_folder):
        return jsonify({'status': 'not_found'})
    
    # Load session info
    with open(os.path.join(session_folder, 'session_info.json'), 'r') as f:
        session_info = json.load(f)
    
    return jsonify({
        'status': session_info['status'],
        'error': session_info.get('error', None)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))