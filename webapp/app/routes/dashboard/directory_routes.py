from flask import render_template, flash, redirect, url_for, request, jsonify
from flask_login import login_required
import os
import shutil
from datetime import datetime

from app.extensions import db
from app.models.config import DirectoryConfig
from app.utils.path_utils import sanitize_path, check_directory_permissions, get_directory_usage
from .forms import DirectoryConfigForm
from . import dashboard_bp


@dashboard_bp.route('/settings/directories')
@login_required
def settings_directories():
    """Directory settings page."""
    # Get all directories
    directories = DirectoryConfig.query.all()
    
    # Add permission and usage properties
    for directory in directories:
        # Check directory permissions
        permissions = check_directory_permissions(directory.path)
        directory.exists = permissions['exists'] and permissions['is_directory']
        directory.writable = permissions['writable']
        directory.readable = permissions['readable']
        
        # Get basic usage stats (only if directory exists)
        if directory.exists:
            directory.usage_info = get_directory_usage(directory.path)
        else:
            directory.usage_info = {
                'file_count': 0,
                'total_size': 0,
                'used_space': '0 B',
                'free_space': '0 B',
                'usage_percent': 0,
                'error': 'Directory does not exist'
            }
    
    return render_template(
        'dashboard/settings_directories.html',
        title='Directory Settings',
        directories=directories
    )


@dashboard_bp.route('/settings/directories/new', methods=['GET', 'POST'])
@login_required
def create_directory():
    """Create a new directory configuration."""
    form = DirectoryConfigForm()
    
    if form.validate_on_submit():
        try:
            # Sanitize the path
            sanitized_path = sanitize_path(form.path.data)
            
            # Create directory config
            directory = DirectoryConfig(
                name=form.name.data,
                path=sanitized_path,
                description=form.description.data,
                is_default=form.is_default.data
            )
            
            db.session.add(directory)
            
            # If this is set as default, clear other defaults
            if form.is_default.data:
                other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
                for other_dir in other_dirs:
                    other_dir.is_default = False
            
            # Create the directory if it doesn't exist
            if not os.path.exists(sanitized_path):
                try:
                    os.makedirs(sanitized_path, exist_ok=True)
                    flash(f'Directory "{sanitized_path}" created successfully', 'success')
                except Exception as e:
                    flash(f'Error creating directory: {str(e)}', 'error')
            
            db.session.commit()
            
            flash(f'Directory "{form.name.data}" configuration created successfully', 'success')
            return redirect(url_for('dashboard.settings_directories'))
            
        except ValueError as e:
            # Handle path validation error
            flash(f'Invalid directory path: {str(e)}', 'error')
    
    # Display form errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_directory.html',
        title='Add New Directory',
        form=form,
        directory=None
    )


@dashboard_bp.route('/settings/directories/edit/<int:directory_id>', methods=['GET', 'POST'])
@login_required
def edit_directory(directory_id):
    """Edit an existing directory configuration."""
    directory = DirectoryConfig.query.get_or_404(directory_id)
    
    # Add runtime properties using our permission checking utility
    permissions = check_directory_permissions(directory.path)
    directory.exists = permissions['exists'] and permissions['is_directory']
    directory.writable = permissions['writable']
    directory.readable = permissions['readable']
    
    if request.method == 'GET':
        # Populate form with directory data
        form = DirectoryConfigForm(obj=directory)
    else:
        form = DirectoryConfigForm()
        
        if form.validate_on_submit():
            try:
                # Sanitize the path
                sanitized_path = sanitize_path(form.path.data)
                old_path = directory.path
                
                # Update directory config
                directory.name = form.name.data
                directory.path = sanitized_path
                directory.description = form.description.data
                directory.is_default = form.is_default.data
                
                # If this is set as default, clear other defaults
                if form.is_default.data:
                    other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
                    for other_dir in other_dirs:
                        other_dir.is_default = False
                
                # Create the directory if it doesn't exist
                if not os.path.exists(sanitized_path):
                    try:
                        os.makedirs(sanitized_path, exist_ok=True)
                        flash(f'Directory "{sanitized_path}" created successfully', 'success')
                    except Exception as e:
                        flash(f'Error creating directory: {str(e)}', 'error')
                
                # If path changed and old directory exists, offer to move files
                if old_path != sanitized_path and os.path.exists(old_path) and request.form.get('move_files') == 'yes':
                    try:
                        # Verify both paths are valid
                        old_path_sanitized = sanitize_path(old_path)
                        
                        # Copy files from old directory to new directory with limits
                        for item in os.listdir(old_path_sanitized):
                            source = os.path.join(old_path_sanitized, item)
                            destination = os.path.join(sanitized_path, item)
                            
                            # Skip files larger than 100MB for safety
                            if os.path.isfile(source) and os.path.getsize(source) > 100 * 1024 * 1024:
                                flash(f'Skipped large file: {item}', 'warning')
                                continue
                                
                            if os.path.isdir(source):
                                shutil.copytree(source, destination, dirs_exist_ok=True)
                            else:
                                shutil.copy2(source, destination)
                                
                        flash(f'Files moved from "{old_path_sanitized}" to "{sanitized_path}"', 'success')
                    except Exception as e:
                        flash(f'Error moving files: {str(e)}', 'error')
                
                db.session.commit()
                directory.updated_at = datetime.utcnow()
                db.session.commit()
                
                flash(f'Directory "{form.name.data}" updated successfully', 'success')
                return redirect(url_for('dashboard.settings_directories'))
                
            except ValueError as e:
                # Handle path validation error
                flash(f'Invalid directory path: {str(e)}', 'error')
        
        # Display form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_directory.html',
        title=f'Edit Directory: {directory.name}',
        form=form,
        directory=directory
    )


@dashboard_bp.route('/settings/directories/set-default/<int:directory_id>')
@login_required
def set_default_directory(directory_id):
    """Set a directory as the default."""
    # Get the directory
    directory = DirectoryConfig.query.get_or_404(directory_id)
    
    # Verify the directory exists
    permissions = check_directory_permissions(directory.path)
    if not permissions['exists'] or not permissions['is_directory']:
        flash(f'Cannot set non-existent directory "{directory.name}" as default', 'error')
        return redirect(url_for('dashboard.settings_directories'))
    
    # Set this directory as default
    directory.is_default = True
    
    # Clear other defaults
    other_dirs = DirectoryConfig.query.filter(DirectoryConfig.id != directory.id).all()
    for other_dir in other_dirs:
        other_dir.is_default = False
    
    db.session.commit()
    
    flash(f'Directory "{directory.name}" set as the default directory', 'success')
    return redirect(url_for('dashboard.settings_directories'))


@dashboard_bp.route('/settings/directories/delete/<int:directory_id>')
@login_required
def delete_directory(directory_id):
    """Delete a directory configuration."""
    # Get the directory
    directory = DirectoryConfig.query.get_or_404(directory_id)
    
    # Check if this is the only directory
    if DirectoryConfig.query.count() <= 1:
        flash('Cannot delete the only directory configuration', 'error')
        return redirect(url_for('dashboard.settings_directories'))
    
    # Check if this is the default directory
    was_default = directory.is_default
    
    # Delete the directory config
    db.session.delete(directory)
    
    # If we deleted the default directory, set a new one
    if was_default:
        # Select a new default directory
        new_default = DirectoryConfig.query.first()
        if new_default:
            new_default.is_default = True
            flash(f'Directory "{new_default.name}" was set as the new default', 'info')
    
    db.session.commit()
    
    flash(f'Directory "{directory.name}" deleted successfully', 'success')
    return redirect(url_for('dashboard.settings_directories'))


