import os
from flask import jsonify, request, current_app, send_file
from flask_login import login_required, current_user
import mimetypes

from app.models.projects import Project
from app.api import api_bp
from app.services.export_service import ProjectExportService


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


@api_bp.route('/projects/<int:project_id>/export', methods=['POST'])
@login_required
def export_project(project_id):
    """API endpoint to export a project."""
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
            'message': 'You do not have permission to export this project'
        }), 403
    
    # Check if project is completed or has output
    if project.status.value not in ['completed', 'failed', 'cancelled']:
        if not project.output_path:
            return jsonify({
                'status': 'error',
                'message': 'This project cannot be exported as it has no output files'
            }), 400
    
    # Get export options from request
    include_logs = request.json.get('include_logs', True)
    
    # Start export
    success, message, export_path = ProjectExportService.export_project(
        project_id=project_id,
        include_logs=include_logs
    )
    
    if not success:
        return jsonify({
            'status': 'error',
            'message': message
        }), 400
    
    # Return success with download URL
    return jsonify({
        'status': 'success',
        'message': message,
        'data': {
            'export_path': export_path,
            'download_url': f"/api/projects/{project_id}/download",
            'export_size': ProjectExportService.format_size(os.path.getsize(export_path)),
            'export_date': project.last_exported_at.isoformat() if project.last_exported_at else None
        }
    })


@api_bp.route('/projects/<int:project_id>/download', methods=['GET'])
@login_required
def download_project(project_id):
    """API endpoint to download an exported project."""
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
            'message': 'You do not have permission to download this project'
        }), 403
    
    # Check if project has been exported
    export_path = ProjectExportService.get_export_filepath(project_id)
    if not export_path:
        return jsonify({
            'status': 'error',
            'message': 'This project has not been exported or the export file is missing'
        }), 404
    
    # Determine the filename for download
    filename = os.path.basename(export_path)
    
    # Set content type
    content_type, _ = mimetypes.guess_type(export_path)
    if not content_type:
        content_type = 'application/zip'
    
    # Return the file as an attachment
    return send_file(
        export_path,
        mimetype=content_type,
        as_attachment=True,
        download_name=filename
    )