from flask import render_template, flash, redirect, url_for, request, jsonify
from flask_login import login_required
from datetime import datetime

from app.services.directory_service import DirectoryService
from app.utils.path_utils import check_directory_permissions
from .forms import DirectoryConfigForm
from . import dashboard_bp


@dashboard_bp.route('/settings/directories')
@login_required
def settings_directories():
    """Directory settings page."""
    # Get all directories with enhanced information
    directories = DirectoryService.get_all_directories()
    
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
        # Create directory using service
        success, message, directory = DirectoryService.create_directory(
            name=form.name.data,
            path=form.path.data,
            description=form.description.data,
            is_default=form.is_default.data,
            create_if_missing=True
        )
        
        if success:
            flash(message, 'success')
            
            # Update all project paths when a new directory is created
            total, updated, errors = DirectoryService.update_all_project_paths()
            if updated > 0:
                flash(f"Updated paths for {updated} out of {total} projects to use the new directory configuration.", 'success')
            if errors:
                flash(f"Encountered {len(errors)} errors while updating project paths. Check the logs for details.", 'warning')
                
            return redirect(url_for('dashboard.settings_directories'))
        else:
            flash(message, 'error')
    
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
    # Get directory data from service
    directory_data = DirectoryService.get_directory_by_id(directory_id)
    
    if not directory_data:
        flash("Directory not found", "error")
        return redirect(url_for('dashboard.settings_directories'))
    
    if request.method == 'GET':
        # Create a form with data from database
        form = DirectoryConfigForm()
        form.name.data = directory_data['name']
        form.path.data = directory_data['path']
        form.description.data = directory_data['description']
        form.is_default.data = directory_data['is_default']
    else:
        # Process form submission
        form = DirectoryConfigForm()
        
        if form.validate_on_submit():
            # Update directory using service
            success, message, updated_directory = DirectoryService.update_directory(
                directory_id=directory_id,
                name=form.name.data,
                path=form.path.data,
                description=form.description.data,
                is_default=form.is_default.data,
                create_if_missing=True,
                move_files=form.move_files.data
            )
            
            if success:
                flash(message, 'success')
                
                # Update all project paths when a directory is updated
                total, updated, errors = DirectoryService.update_all_project_paths()
                if updated > 0:
                    flash(f"Updated paths for {updated} out of {total} projects to use the new directory configuration.", 'success')
                if errors:
                    flash(f"Encountered {len(errors)} errors while updating project paths. Check the logs for details.", 'warning')
                    
                return redirect(url_for('dashboard.settings_directories'))
            else:
                flash(message, 'error')
        
        # Display form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_directory.html',
        title=f'Edit Directory: {directory_data["name"]}',
        form=form,
        directory=directory_data
    )


@dashboard_bp.route('/settings/directories/set-default/<int:directory_id>')
@login_required
def set_default_directory(directory_id):
    """Set a directory as the default."""
    # Set default using service
    success, message = DirectoryService.set_default_directory(directory_id)
    
    if success:
        flash(message, 'success')
        
        # Update all project paths when the default directory is changed
        total, updated, errors = DirectoryService.update_all_project_paths()
        if updated > 0:
            flash(f"Updated paths for {updated} out of {total} projects to use the new directory configuration.", 'success')
        if errors:
            flash(f"Encountered {len(errors)} errors while updating project paths. Check the logs for details.", 'warning')
    else:
        flash(message, 'error')
        
    return redirect(url_for('dashboard.settings_directories'))


@dashboard_bp.route('/settings/directories/delete/<int:directory_id>')
@login_required
def delete_directory(directory_id):
    """Delete a directory configuration."""
    # Delete using service
    success, message = DirectoryService.delete_directory(directory_id)
    
    if success:
        flash(message, 'success')
        
        # Update all project paths when a directory is deleted
        total, updated, errors = DirectoryService.update_all_project_paths()
        if updated > 0:
            flash(f"Updated paths for {updated} out of {total} projects to use the new directory configuration.", 'success')
        if errors:
            flash(f"Encountered {len(errors)} errors while updating project paths. Check the logs for details.", 'warning')
    else:
        flash(message, 'error')
        
    return redirect(url_for('dashboard.settings_directories'))


@dashboard_bp.route('/api/directories')
@login_required
def api_get_directories():
    """API endpoint to get all directories."""
    directories = DirectoryService.get_all_directories()
    return jsonify({'success': True, 'data': directories})


@dashboard_bp.route('/api/directories/<int:directory_id>')
@login_required
def api_get_directory(directory_id):
    """API endpoint to get a single directory."""
    directory = DirectoryService.get_directory_by_id(directory_id)
    
    if directory:
        return jsonify({'success': True, 'data': directory})
    else:
        return jsonify({'success': False, 'error': 'Directory not found'}), 404


@dashboard_bp.route('/api/directories/check-path')
@login_required
def api_check_path():
    """API endpoint to check if a path exists and get permissions."""
    path = request.args.get('path', '')
    
    if not path:
        return jsonify({'success': False, 'error': 'No path provided'})
    
    try:
        # Sanitize the path first
        from app.utils.path_utils import sanitize_path, check_directory_permissions, get_directory_usage
        sanitized_path = sanitize_path(path)
        
        # Check permissions
        permissions = check_directory_permissions(sanitized_path)
        
        # Only get usage info if the directory exists
        usage_info = None
        if permissions['exists'] and permissions['is_directory']:
            usage_info = get_directory_usage(sanitized_path)
        
        # Return the results
        return jsonify({
            'success': True,
            'path': sanitized_path,
            'exists': permissions['exists'],
            'is_directory': permissions['is_directory'],
            'readable': permissions['readable'],
            'writable': permissions['writable'],
            'executable': permissions['executable'],
            'usage_info': usage_info
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error checking path: {str(e)}'})


@dashboard_bp.route('/settings/directories/update-project-paths')
@login_required
def update_project_paths():
    """Update all project paths to use the current directory configuration."""
    # Update all project paths
    total, updated, errors = DirectoryService.update_all_project_paths()
    
    if errors:
        flash(f"Encountered {len(errors)} errors while updating project paths. Check the logs for details.", 'warning')
    elif updated == 0:
        flash(f"No projects needed path updates. All {total} projects are already using the current directory configuration.", 'info')
    else:
        flash(f"Successfully updated paths for {updated} out of {total} projects.", 'success')
        
    return redirect(url_for('dashboard.settings_directories'))


