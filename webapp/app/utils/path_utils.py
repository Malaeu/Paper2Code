"""
Path utility functions for secure path handling in Paper2Code.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from flask import current_app


def get_allowed_base_paths() -> List[str]:
    """
    Get a list of allowed base paths for directory configurations.
    
    Returns:
        List of allowed base paths
    """
    try:
        project_root = current_app.config.get('PROJECT_ROOT', '')
        app_root = current_app.root_path
        upload_folder = os.path.join(app_root, 'uploads')
        
        # Define allowed base paths
        allowed_paths = [
            project_root,  # Project root directory
            app_root,      # Application root
            upload_folder,  # Upload folder
            '/tmp',        # Temporary directory
            '/var/tmp'     # Another temporary directory
        ]
        
        # Filter out empty paths and normalize them
        return [os.path.realpath(p) for p in allowed_paths if p]
    except:
        # Default fallback if we can't get the app config
        return ['/tmp']


def sanitize_path(path: str) -> str:
    """
    Sanitize a path to prevent directory traversal attacks.
    
    Args:
        path: The path to sanitize
        
    Returns:
        Sanitized path
        
    Raises:
        ValueError: If the path is not within allowed directories
    """
    if not path:
        raise ValueError("Path cannot be empty")
        
    # Normalize the path to resolve '..' and '.'
    normalized_path = os.path.normpath(path)
    
    # Get the real path (resolves symbolic links)
    real_path = os.path.realpath(normalized_path)
    
    # Check if the path is within allowed directories
    allowed_base_paths = get_allowed_base_paths()
    for base_path in allowed_base_paths:
        if real_path.startswith(base_path):
            return real_path
    
    # If we get here, the path is not allowed
    raise ValueError(f"Path {path} is not within allowed directories")


def check_directory_permissions(path: str) -> Dict[str, bool]:
    """
    Check if a directory has proper permissions.
    
    Args:
        path: The path to check
        
    Returns:
        Dictionary with permission info
    """
    try:
        real_path = sanitize_path(path)
        
        # Initialize with all permissions false
        permissions = {
            "exists": False,
            "readable": False,
            "writable": False,
            "executable": False,
            "is_directory": False
        }
        
        # Check if the path exists
        if not os.path.exists(real_path):
            return permissions
            
        permissions["exists"] = True
        permissions["is_directory"] = os.path.isdir(real_path)
        
        # Check permissions
        permissions["readable"] = os.access(real_path, os.R_OK)
        permissions["writable"] = os.access(real_path, os.W_OK)
        permissions["executable"] = os.access(real_path, os.X_OK)
        
        return permissions
    except ValueError:
        # Return all false if the path is not valid
        return {
            "exists": False,
            "readable": False,
            "writable": False,
            "executable": False,
            "is_directory": False
        }


def get_directory_usage(path: str, max_depth: int = 5, timeout: int = 5) -> Dict[str, Any]:
    """
    Get usage information for a directory with depth limit and timeout.
    
    Args:
        path: Directory path
        max_depth: Maximum depth to scan
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with usage information
    """
    import signal
    from datetime import datetime
    
    # Default response if something goes wrong
    default_response = {
        'file_count': 0,
        'total_size': 0,
        'used_space': '0 B',
        'free_space': '0 B',
        'usage_percent': 0,
        'error': None
    }
    
    # Define timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Directory scan took too long")
    
    try:
        # Sanitize the path first
        real_path = sanitize_path(path)
        
        # Check if path exists and is a directory
        if not os.path.exists(real_path) or not os.path.isdir(real_path):
            default_response['error'] = 'Directory does not exist'
            return default_response
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Get file count and total size
        file_count = 0
        total_size = 0
        current_depth = 0
        
        for root, dirs, files in os.walk(real_path):
            # Check depth
            rel_path = os.path.relpath(root, real_path)
            if rel_path == '.':
                current_depth = 0
            else:
                current_depth = rel_path.count(os.sep) + 1
            
            # Skip if we've exceeded max depth
            if current_depth > max_depth:
                del dirs[:]
                continue
                
            file_count += len(files)
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp) and os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        
        # Get disk usage
        statvfs = os.statvfs(real_path)
        free_space = statvfs.f_frsize * statvfs.f_bavail
        total_space = statvfs.f_frsize * statvfs.f_blocks
        used_space = total_space - free_space
        
        # Calculate usage percentage
        usage_percent = (used_space / total_space) * 100 if total_space > 0 else 0
        
        # Format sizes
        def format_size(size_bytes):
            """Format bytes to human-readable size."""
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} PB"
        
        # Reset the alarm
        signal.alarm(0)
        
        return {
            'file_count': file_count,
            'total_size': total_size,
            'used_space': format_size(total_size),
            'free_space': format_size(free_space),
            'usage_percent': round(usage_percent, 2),
            'error': None
        }
    except TimeoutError:
        # Handle timeout
        default_response['error'] = 'Directory scan timed out'
        return default_response
    except ValueError as e:
        # Handle invalid path
        default_response['error'] = str(e)
        return default_response
    except Exception as e:
        # Handle other exceptions
        default_response['error'] = f'Error scanning directory: {str(e)}'
        return default_response
    finally:
        # Always reset the alarm to avoid issues
        signal.alarm(0)