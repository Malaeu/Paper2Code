from flask import render_template, flash, redirect, url_for, request, current_app
from flask_login import login_required, current_user
import os

from app.extensions import db
from app.models.projects import Project, ProjectStatus, ProjectType
from app.models.config import (
    ModelConfig, ModelProvider, ModelCostInfo, 
    DirectoryConfig, ProjectSettings, get_available_models
)
from app.services.pipeline import PipelineService
from .forms import ProjectForm, ModelConfigForm, DirectoryConfigForm, ApiKeyForm
from . import dashboard_bp


@dashboard_bp.route('/')
@login_required
def index():
    """Dashboard home page showing user's projects and stats."""
    # Get user's projects
    projects = Project.query.filter_by(user_id=current_user.id).order_by(
        Project.created_at.desc()).all()
    
    # Import services here to avoid circular imports
    from app.services.usage_service import UsageService
    
    # Get usage statistics
    usage_stats = UsageService.get_user_usage_stats(current_user.id)
    
    # Calculate statistics
    completed_projects = sum(1 for p in projects if p.status == ProjectStatus.COMPLETED)
    
    stats = {
        'total_projects': len(projects),
        'papers_processed': len(projects),  # For now, this equals total projects
        'code_repositories': completed_projects,
        'api_calls': current_user.total_api_calls,
        'total_tokens': usage_stats['total_tokens'],
        'total_cost': usage_stats['total_cost'],
    }
    
    return render_template(
        'dashboard/index.html',
        title='Dashboard',
        projects=projects,
        stats=stats,
        usage_stats=usage_stats
    )


@dashboard_bp.route('/projects/new', methods=['GET', 'POST'])
@login_required
def create_project():
    """Create a new project."""
    form = ProjectForm()
    
    if form.validate_on_submit():
        # Create new project
        project = Project(
            name=form.name.data,
            user_id=current_user.id,
            description=form.description.data,
            project_type=ProjectType(form.project_type.data)
        )
        
        # Set programming language if provided
        if form.programming_language.data:
            project.programming_language = form.programming_language.data
            
        # Save to database to get an ID
        db.session.add(project)
        db.session.commit()
        
        # Handle paper upload
        if form.paper.data:
            if project.save_paper(form.paper.data):
                flash('Project created successfully!', 'success')
                return redirect(url_for('dashboard.view_project', project_id=project.id))
            else:
                db.session.delete(project)
                db.session.commit()
                flash('Error saving paper file', 'error')
        else:
            flash('No paper file provided', 'error')
    
    # Display any form errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/create_project.html',
        title='New Project',
        form=form
    )


@dashboard_bp.route('/projects/<int:project_id>')
@login_required
def view_project(project_id):
    """View project details."""
    project = Project.query.get_or_404(project_id)
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        flash('You do not have permission to view this project', 'error')
        return redirect(url_for('dashboard.index'))
    
    return render_template(
        'dashboard/view_project.html',
        title=project.name,
        project=project
    )


@dashboard_bp.route('/projects/<int:project_id>/process', methods=['POST'])
@login_required
def process_project(project_id):
    """Start processing a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        flash('You do not have permission to process this project', 'error')
        return redirect(url_for('dashboard.index'))
    
    # Check if the project is ready to be processed
    if project.status not in [ProjectStatus.CREATED, ProjectStatus.FAILED]:
        flash('This project is already being processed or has been completed', 'error')
        return redirect(url_for('dashboard.view_project', project_id=project_id))
    
    # Start the pipeline
    if PipelineService.start_pipeline(project_id):
        flash('Project processing started', 'success')
    else:
        flash('Failed to start project processing', 'error')
    
    return redirect(url_for('dashboard.view_project', project_id=project_id))


@dashboard_bp.route('/projects/<int:project_id>/cancel', methods=['POST'])
@login_required
def cancel_project(project_id):
    """Cancel processing a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        flash('You do not have permission to cancel this project', 'error')
        return redirect(url_for('dashboard.index'))
    
    # Check if the project is currently being processed
    if project.status not in [ProjectStatus.PROCESSING, ProjectStatus.PLANNING, 
                             ProjectStatus.ANALYZING, ProjectStatus.CODING]:
        flash('This project is not currently being processed', 'error')
        return redirect(url_for('dashboard.view_project', project_id=project_id))
    
    # Update project status to cancelled
    project.update_status(ProjectStatus.CANCELLED)
    flash('Project processing cancelled', 'success')
    
    return redirect(url_for('dashboard.view_project', project_id=project_id))


@dashboard_bp.route('/projects/<int:project_id>/retry', methods=['POST'])
@login_required
def retry_project(project_id):
    """Retry processing a failed project."""
    project = Project.query.get_or_404(project_id)
    
    # Check if the user owns this project
    if project.user_id != current_user.id:
        flash('You do not have permission to retry this project', 'error')
        return redirect(url_for('dashboard.index'))
    
    # Check if the project has failed
    if project.status not in [ProjectStatus.FAILED, ProjectStatus.CANCELLED]:
        flash('Only failed or cancelled projects can be retried', 'error')
        return redirect(url_for('dashboard.view_project', project_id=project_id))
    
    # Start the pipeline
    if PipelineService.start_pipeline(project_id):
        flash('Project processing restarted', 'success')
    else:
        flash('Failed to restart project processing', 'error')
    
    return redirect(url_for('dashboard.view_project', project_id=project_id))


@dashboard_bp.route('/settings')
@login_required
def settings():
    """Dashboard settings page."""
    return render_template(
        'dashboard/settings.html',
        title='Dashboard Settings'
    )


@dashboard_bp.route('/usage')
@login_required
def usage_dashboard():
    """Usage statistics dashboard."""
    # Import here to avoid circular imports
    from app.services.usage_service import UsageService
    
    # Get usage statistics
    stats = UsageService.get_user_usage_stats(current_user.id)
    
    # Calculate summary stats for the dashboard
    summary = {
        'total_api_calls': stats['total_api_calls'],
        'total_tokens': stats['total_tokens'],
        'total_cost': stats['total_cost'],
        'active_keys': sum(1 for key in stats['api_keys'] if key.get('calls_count', 0) > 0),
        'model_count': len(stats['models']),
    }
    
    return render_template(
        'dashboard/usage.html',
        title='Usage Dashboard',
        stats=stats,
        summary=summary
    )


@dashboard_bp.route('/settings/models')
@login_required
def settings_models():
    """Model settings page."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    # Get all models
    models = ModelService.get_all_models()
    
    # Get saved API keys from settings
    api_keys = ModelService.get_api_keys()
    
    # Calculate usage statistics
    total_cost = sum(model.get('total_cost', 0) for model in models)
    total_tokens = sum(model.get('total_tokens_used', 0) for model in models)
    total_api_calls = current_user.total_api_calls if current_user else 0
    
    # Get providers for filtering
    providers = [
        {'id': provider.value, 'name': provider.display_name} 
        for provider in ModelProvider
    ]
    
    return render_template(
        'dashboard/settings_models.html',
        title='AI Model Settings',
        models=models,
        api_keys=api_keys,
        total_cost=total_cost,
        total_tokens=total_tokens,
        total_api_calls=total_api_calls,
        providers=providers
    )


@dashboard_bp.route('/settings/models/new', methods=['GET', 'POST'])
@login_required
def create_model():
    """Create a new model configuration."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    form = ModelConfigForm()
    
    if form.validate_on_submit():
        # Prepare model data from form
        model_data = {
            'model_id': form.model_id.data,
            'display_name': form.display_name.data,
            'provider': form.provider.data,
            'description': form.description.data,
            'context_length': form.context_length.data,
            'supports_vision': form.supports_vision.data,
            'supports_function_calling': form.supports_function_calling.data,
            'requires_api_key': form.requires_api_key.data,
            'command_args': form.command_args.data,
            'gpt_version': form.gpt_version.data,
            'is_active': form.is_active.data,
            'is_default': form.is_default.data,
            'input_cost_per_1k_tokens': form.input_cost_per_1k_tokens.data or 0.0,
            'output_cost_per_1k_tokens': form.output_cost_per_1k_tokens.data or 0.0
        }
        
        # Create model through service
        success, message, model = ModelService.create_model(model_data)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('dashboard.settings_models'))
        else:
            flash(message, 'error')
    
    # Display form errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_model.html',
        title='Add New Model',
        form=form,
        model=None
    )


@dashboard_bp.route('/settings/models/edit/<int:model_id>', methods=['GET', 'POST'])
@login_required
def edit_model(model_id):
    """Edit an existing model configuration."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    # Get model data
    model_dict = ModelService.get_model_by_id(model_id)
    if not model_dict:
        flash(f'Model with ID {model_id} not found', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    if request.method == 'GET':
        # Create form and populate with model data
        form = ModelConfigForm()
        
        # Set form values from model data
        form.model_id.data = model_dict.get('model_id')
        form.display_name.data = model_dict.get('display_name')
        form.provider.data = model_dict.get('provider')
        form.description.data = model_dict.get('description')
        form.context_length.data = model_dict.get('context_length')
        form.supports_vision.data = model_dict.get('supports_vision')
        form.supports_function_calling.data = model_dict.get('supports_function_calling')
        form.requires_api_key.data = model_dict.get('requires_api_key')
        form.command_args.data = model_dict.get('command_args')
        form.gpt_version.data = model_dict.get('gpt_version')
        form.is_active.data = model_dict.get('is_active')
        form.is_default.data = model_dict.get('is_default')
        
        # Set cost info values if available
        form.input_cost_per_1k_tokens.data = model_dict.get('input_cost_per_1k_tokens', 0.0)
        form.output_cost_per_1k_tokens.data = model_dict.get('output_cost_per_1k_tokens', 0.0)
    else:
        form = ModelConfigForm()
        
        if form.validate_on_submit():
            # Prepare model data from form
            model_data = {
                'model_id': form.model_id.data,
                'display_name': form.display_name.data,
                'provider': form.provider.data,
                'description': form.description.data,
                'context_length': form.context_length.data,
                'supports_vision': form.supports_vision.data,
                'supports_function_calling': form.supports_function_calling.data,
                'requires_api_key': form.requires_api_key.data,
                'command_args': form.command_args.data,
                'gpt_version': form.gpt_version.data,
                'is_active': form.is_active.data,
                'is_default': form.is_default.data,
                'input_cost_per_1k_tokens': form.input_cost_per_1k_tokens.data or 0.0,
                'output_cost_per_1k_tokens': form.output_cost_per_1k_tokens.data or 0.0
            }
            
            # Update model through service
            success, message, updated_model = ModelService.update_model(model_id, model_data)
            
            if success:
                flash(message, 'success')
                return redirect(url_for('dashboard.settings_models'))
            else:
                flash(message, 'error')
        
        # Display form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_model.html',
        title=f'Edit Model: {model_dict.get("display_name")}',
        form=form,
        model=model_dict
    )


@dashboard_bp.route('/settings/models/set-default', methods=['POST'])
@login_required
def set_default_model():
    """Set a model as the default."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    model_id = request.form.get('model_id', type=int)
    
    if not model_id:
        flash('Model ID is required', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Set model as default through service
    success, message = ModelService.set_default_model(model_id)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
        
    return redirect(url_for('dashboard.settings_models'))


@dashboard_bp.route('/settings/models/toggle-status', methods=['POST'])
@login_required
def toggle_model_status():
    """Activate or deactivate a model."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    model_id = request.form.get('model_id', type=int)
    action = request.form.get('action')
    
    if not model_id or action not in ['activate', 'deactivate']:
        flash('Invalid request', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Toggle model status through service
    success, message = ModelService.toggle_model_status(model_id, action == 'activate')
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
        
    return redirect(url_for('dashboard.settings_models'))


@dashboard_bp.route('/settings/api-keys', methods=['POST'])
@login_required
def save_api_key():
    """Save an API key for a provider."""
    # Import here to avoid circular imports
    from app.services.model_service import ModelService
    
    provider = request.form.get('provider')
    api_key = request.form.get('api_key')
    
    if not provider or not api_key:
        flash('Provider and API key are required', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Save API key through service
    success, message = ModelService.save_api_key(provider, api_key)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
        
    return redirect(url_for('dashboard.settings_models'))