import os
from flask import jsonify, request, current_app
from flask_login import login_required, current_user

from app.models.projects import Project
from app.api import api_bp


@api_bp.route('/projects/<int:project_id>/status', methods=['GET'])
@login_required
def get_project_status(project_id):
    """API endpoint to get project status and progress."""
    project = Project.query.get(project_id)
    
    if not project:
        return jsonify({
            'status': 'error',
            'message': 'Project not found'
        }), 404
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to view this project'
        }), 403
    
    # Return project status
    return jsonify({
        'status': 'success',
        'data': {
            'project_id': project.id,
            'status': project.status.value,
            'status_color': project.get_status_color(),
            'progress': project.progress,
            'current_task': project.current_task,
            'error_message': project.error_message
        }
    })


@api_bp.route('/projects/<int:project_id>/logs', methods=['GET'])
@login_required
def get_project_logs(project_id):
    """API endpoint to get project logs."""
    project = Project.query.get(project_id)
    
    if not project:
        return jsonify({
            'status': 'error',
            'message': 'Project not found'
        }), 404
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to view this project'
        }), 403
    
    # Check if log file exists
    if not project.log_path or not os.path.exists(project.log_path):
        return jsonify({
            'status': 'success',
            'data': {
                'logs': 'No logs available'
            }
        })
    
    # Read the last N lines of the log file
    max_lines = int(request.args.get('max_lines', 50))
    logs = []
    
    try:
        with open(project.log_path, 'r') as f:
            # Read the entire file and get the last N lines
            lines = f.readlines()
            logs = lines[-max_lines:] if len(lines) > max_lines else lines
            logs = [line.strip() for line in logs]
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error reading log file: {str(e)}'
        }), 500
    
    # Return logs
    return jsonify({
        'status': 'success',
        'data': {
            'logs': logs
        }
    })