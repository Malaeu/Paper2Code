"""
API endpoints for directory management.
"""

from flask import request, jsonify
from flask_login import login_required

from app.utils.path_utils import sanitize_path, check_directory_permissions
from . import api_bp


@api_bp.route('/directories/validate', methods=['POST'])
@login_required
def validate_directory():
    """
    Validate a directory path.
    
    Request body:
        {
            "path": "directory_path"
        }
        
    Returns:
        {
            "status": "success" | "error",
            "data": {
                "exists": true | false,
                "is_directory": true | false,
                "readable": true | false,
                "writable": true | false,
                "executable": true | false
            },
            "message": "Error message if status is error"
        }
    """
    path = request.json.get('path')
    if not path:
        return jsonify({
            "status": "error", 
            "message": "No path provided"
        }), 400
    
    try:
        # Sanitize the path first
        sanitized_path = sanitize_path(path)
        
        # Check directory permissions
        permissions = check_directory_permissions(sanitized_path)
        
        return jsonify({
            "status": "success", 
            "path": sanitized_path,
            "data": permissions
        })
    except ValueError as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"An error occurred: {str(e)}"
        }), 500