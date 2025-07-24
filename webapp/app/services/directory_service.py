"""Service layer for directory configuration operations."""

import os
import shutil
from typing import Dict, List, Optional, Any, Tuple
from flask import current_app

from app.extensions import db
from app.models.config.model_config import DirectoryConfig
from app.utils.path_utils import sanitize_path, check_directory_permissions, get_directory_usage


class DirectoryService:
    """Service for managing directory configurations."""
    
    @staticmethod
    def get_all_directories() -> List[Dict[str, Any]]:
        """
        Get all directory configurations.
        
        Returns:
            List of directory configurations as dictionaries
        """
        directories = DirectoryConfig.query.order_by(DirectoryConfig.is_default.desc(), DirectoryConfig.name).all()
        
        # Add permission and usage properties
        result = []
        
        for directory in directories:
            # Check directory permissions
            permissions = check_directory_permissions(directory.path)
            
            # Get basic usage stats (only if directory exists)
            if permissions['exists'] and permissions['is_directory']:
                usage_info = get_directory_usage(directory.path)
            else:
                usage_info = {
                    'file_count': 0,
                    'total_size': 0,
                    'used_space': '0 B',
                    'free_space': '0 B',
                    'usage_percent': 0,
                    'error': 'Directory does not exist'
                }
            
            # Create result dictionary
            result.append({
                'id': directory.id,
                'name': directory.name,
                'path': directory.path,
                'description': directory.description,
                'is_default': directory.is_default,
                'created_at': directory.created_at.isoformat(),
                'updated_at': directory.updated_at.isoformat(),
                'exists': permissions['exists'] and permissions['is_directory'],
                'writable': permissions['writable'],
                'readable': permissions['readable'],
                'executable': permissions['executable'],
                'usage_info': usage_info
            })
            
        return result
    
    @staticmethod
    def get_directory_by_id(directory_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a directory configuration by ID.
        
        Args:
            directory_id: The database ID of the directory
            
        Returns:
            Directory configuration as a dictionary or None if not found
        """
        directory = DirectoryConfig.query.get(directory_id)
        
        if not directory:
            return None
            
        # Check directory permissions
        permissions = check_directory_permissions(directory.path)
        
        # Get basic usage stats (only if directory exists)
        if permissions['exists'] and permissions['is_directory']:
            usage_info = get_directory_usage(directory.path)
        else:
            usage_info = {
                'file_count': 0,
                'total_size': 0,
                'used_space': '0 B',
                'free_space': '0 B',
                'usage_percent': 0,
                'error': 'Directory does not exist'
            }
        
        # Create result dictionary
        result = {
            'id': directory.id,
            'name': directory.name,
            'path': directory.path,
            'description': directory.description,
            'is_default': directory.is_default,
            'created_at': directory.created_at.isoformat(),
            'updated_at': directory.updated_at.isoformat(),
            'exists': permissions['exists'] and permissions['is_directory'],
            'writable': permissions['writable'],
            'readable': permissions['readable'],
            'executable': permissions['executable'],
            'usage_info': usage_info
        }
        
        return result
    
    @staticmethod
    def get_directory_by_name(name: str) -> Optional[Dict[str, Any]]:
        """
        Get a directory configuration by name.
        
        Args:
            name: The name of the directory
            
        Returns:
            Directory configuration as a dictionary or None if not found
        """
        directory = DirectoryConfig.query.filter_by(name=name).first()
        
        if not directory:
            return None
            
        return DirectoryService.get_directory_by_id(directory.id)
    
    @staticmethod
    def get_default_directory() -> Optional[Dict[str, Any]]:
        """
        Get the default directory configuration.
        
        Returns:
            Default directory configuration as a dictionary or None if not found
        """
        directory = DirectoryConfig.query.filter_by(is_default=True).first()
        
        if not directory:
            return None
            
        return DirectoryService.get_directory_by_id(directory.id)
    
    @staticmethod
    def create_directory(
        name: str,
        path: str,
        description: Optional[str] = None,
        is_default: bool = False,
        create_if_missing: bool = True
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Create a new directory configuration.
        
        Args:
            name: Directory name
            path: Directory path
            description: Directory description
            is_default: Whether this is the default directory
            create_if_missing: Create the directory if it doesn't exist
            
        Returns:
            Tuple of (success, message, directory_dict)
        """
        try:
            # Check if name already exists
            if DirectoryConfig.query.filter_by(name=name).first():
                return False, f"Directory name '{name}' already exists", None
            
            # Sanitize the path
            try:
                sanitized_path = sanitize_path(path)
            except ValueError as e:
                return False, f"Invalid path: {str(e)}", None
            
            # Create directory config
            directory = DirectoryConfig(
                name=name,
                path=sanitized_path,
                description=description,
                is_default=is_default
            )
            
            db.session.add(directory)
            
            # If this is set as default, clear other defaults
            if is_default:
                other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
                for other_dir in other_dirs:
                    other_dir.is_default = False
            
            # Create the directory if it doesn't exist
            if create_if_missing and not os.path.exists(sanitized_path):
                try:
                    os.makedirs(sanitized_path, exist_ok=True)
                except Exception as e:
                    db.session.rollback()
                    return False, f"Error creating directory: {str(e)}", None
            
            db.session.commit()
            
            return True, f"Directory '{name}' created successfully", DirectoryService.get_directory_by_id(directory.id)
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error creating directory configuration: {str(e)}")
            return False, f"Error creating directory configuration: {str(e)}", None
    
    @staticmethod
    def update_directory(
        directory_id: int,
        name: str,
        path: str,
        description: Optional[str] = None,
        is_default: bool = False,
        create_if_missing: bool = True,
        move_files: bool = False
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Update an existing directory configuration.
        
        Args:
            directory_id: The database ID of the directory to update
            name: Directory name
            path: Directory path
            description: Directory description
            is_default: Whether this is the default directory
            create_if_missing: Create the directory if it doesn't exist
            move_files: Move files from old directory to new directory if path changed
            
        Returns:
            Tuple of (success, message, directory_dict)
        """
        try:
            # Get the directory
            directory = DirectoryConfig.query.get(directory_id)
            
            if not directory:
                return False, f"Directory with ID {directory_id} not found", None
            
            # Check if name already exists (if changed)
            if name != directory.name and DirectoryConfig.query.filter_by(name=name).first():
                return False, f"Directory name '{name}' already exists", None
            
            # Sanitize the path
            try:
                sanitized_path = sanitize_path(path)
            except ValueError as e:
                return False, f"Invalid path: {str(e)}", None
            
            # Save old path before update
            old_path = directory.path
            
            # Update directory config
            directory.name = name
            directory.path = sanitized_path
            directory.description = description
            directory.is_default = is_default
            
            # If this is set as default, clear other defaults
            if is_default:
                other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
                for other_dir in other_dirs:
                    other_dir.is_default = False
            
            # Create the directory if it doesn't exist
            if create_if_missing and not os.path.exists(sanitized_path):
                try:
                    os.makedirs(sanitized_path, exist_ok=True)
                except Exception as e:
                    return False, f"Error creating directory: {str(e)}", None
            
            # If path changed and move_files is True, move files from old directory to new directory
            if old_path != sanitized_path and move_files and os.path.exists(old_path) and os.path.isdir(old_path):
                try:
                    # Create destination directory if it doesn't exist
                    if not os.path.exists(sanitized_path):
                        os.makedirs(sanitized_path, exist_ok=True)
                    
                    # Copy files from old directory to new directory with limits
                    for item in os.listdir(old_path):
                        source = os.path.join(old_path, item)
                        destination = os.path.join(sanitized_path, item)
                        
                        # Skip files larger than 100MB for safety
                        if os.path.isfile(source) and os.path.getsize(source) > 100 * 1024 * 1024:
                            current_app.logger.warning(f"Skipped large file: {item}")
                            continue
                            
                        if os.path.isdir(source):
                            shutil.copytree(source, destination, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source, destination)
                except Exception as e:
                    current_app.logger.error(f"Error moving files: {str(e)}")
                    # Continue with update even if moving files fails
            
            db.session.commit()
            
            return True, f"Directory '{name}' updated successfully", DirectoryService.get_directory_by_id(directory.id)
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error updating directory configuration: {str(e)}")
            return False, f"Error updating directory configuration: {str(e)}", None
    
    @staticmethod
    def delete_directory(directory_id: int) -> Tuple[bool, str]:
        """
        Delete a directory configuration.
        
        Args:
            directory_id: The database ID of the directory to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the directory
            directory = DirectoryConfig.query.get(directory_id)
            
            if not directory:
                return False, f"Directory with ID {directory_id} not found"
            
            # Check if this is the only directory
            if DirectoryConfig.query.count() <= 1:
                return False, "Cannot delete the only directory configuration"
            
            # Remember if it was default
            was_default = directory.is_default
            
            # Get the name for the message
            directory_name = directory.name
            
            # Delete the directory
            db.session.delete(directory)
            
            # If we deleted the default directory, set a new one
            if was_default:
                # Select a new default directory
                new_default = DirectoryConfig.query.first()
                if new_default:
                    new_default.is_default = True
            
            db.session.commit()
            
            return True, f"Directory '{directory_name}' deleted successfully"
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error deleting directory configuration: {str(e)}")
            return False, f"Error deleting directory configuration: {str(e)}"
    
    @staticmethod
    def set_default_directory(directory_id: int) -> Tuple[bool, str]:
        """
        Set a directory as the default.
        
        Args:
            directory_id: The database ID of the directory to set as default
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the directory
            directory = DirectoryConfig.query.get(directory_id)
            
            if not directory:
                return False, f"Directory with ID {directory_id} not found"
            
            # Verify the directory exists
            permissions = check_directory_permissions(directory.path)
            if not permissions['exists'] or not permissions['is_directory']:
                return False, f"Cannot set non-existent directory '{directory.name}' as default"
            
            # Set this directory as default
            directory.is_default = True
            
            # Clear other defaults
            other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
            for other_dir in other_dirs:
                other_dir.is_default = False
            
            db.session.commit()
            
            return True, f"Directory '{directory.name}' set as the default directory"
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error setting default directory: {str(e)}")
            return False, f"Error setting default directory: {str(e)}"
    
    @staticmethod
    def get_directory_path_by_name(name: str, default_path: Optional[str] = None) -> str:
        """
        Get the path of a directory by name.
        
        Args:
            name: The name of the directory
            default_path: Path to return if the directory is not found
            
        Returns:
            Directory path or default_path if not found
        """
        directory = DirectoryConfig.query.filter_by(name=name).first()
        
        if directory:
            return directory.path
            
        return default_path if default_path else ""
    
    @staticmethod
    def update_all_project_paths() -> Tuple[int, int, List[str]]:
        """
        Update all projects' paths to match the current directory configuration.
        This is useful after directory configurations have changed.
        
        Returns:
            Tuple of (total_projects, updated_projects, error_messages)
        """
        from app.models.projects import Project
        
        total_projects = 0
        updated_projects = 0
        errors = []
        
        try:
            # Get all projects
            projects = Project.query.all()
            total_projects = len(projects)
            
            # Update each project's paths
            for project in projects:
                try:
                    if project.update_paths_to_config():
                        updated_projects += 1
                except Exception as e:
                    errors.append(f"Error updating project {project.id}: {str(e)}")
            
            return total_projects, updated_projects, errors
            
        except Exception as e:
            errors.append(f"Error fetching projects: {str(e)}")
            return total_projects, updated_projects, errors