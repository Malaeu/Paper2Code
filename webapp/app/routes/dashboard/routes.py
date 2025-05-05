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
    
    # Calculate statistics
    completed_projects = sum(1 for p in projects if p.status == ProjectStatus.COMPLETED)
    
    stats = {
        'total_projects': len(projects),
        'papers_processed': len(projects),  # For now, this equals total projects
        'code_repositories': completed_projects,
        'api_calls': current_user.total_api_calls
    }
    
    return render_template(
        'dashboard/index.html',
        title='Dashboard',
        projects=projects,
        stats=stats
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


@dashboard_bp.route('/settings/models')
@login_required
def settings_models():
    """Model settings page."""
    # Get all models
    models = get_available_models()
    
    # Get saved API keys from settings
    api_keys = {
        'openai': ProjectSettings.get('openai_api_key', ''),
        'anthropic': ProjectSettings.get('anthropic_api_key', ''),
        'huggingface': ProjectSettings.get('huggingface_api_key', '')
    }
    
    # Calculate usage statistics
    total_cost = sum(model.get('total_cost', 0) for model in models)
    total_tokens = sum(model.get('total_tokens_used', 0) for model in models)
    total_api_calls = current_user.total_api_calls if current_user else 0
    
    return render_template(
        'dashboard/settings_models.html',
        title='AI Model Settings',
        models=models,
        api_keys=api_keys,
        total_cost=total_cost,
        total_tokens=total_tokens,
        total_api_calls=total_api_calls
    )


@dashboard_bp.route('/settings/models/new', methods=['GET', 'POST'])
@login_required
def create_model():
    """Create a new model configuration."""
    form = ModelConfigForm()
    
    if form.validate_on_submit():
        # Create cost info
        cost_info = ModelCostInfo(
            model_id=form.model_id.data,
            input_cost_per_1k_tokens=form.input_cost_per_1k_tokens.data or 0.0,
            output_cost_per_1k_tokens=form.output_cost_per_1k_tokens.data or 0.0
        )
        db.session.add(cost_info)
        db.session.flush()  # To get the ID
        
        # Create model config
        model_config = ModelConfig(
            model_id=form.model_id.data,
            display_name=form.display_name.data,
            provider=ModelProvider(form.provider.data),
            description=form.description.data,
            context_length=form.context_length.data,
            supports_vision=form.supports_vision.data,
            supports_function_calling=form.supports_function_calling.data,
            requires_api_key=form.requires_api_key.data,
            command_args=form.command_args.data,
            gpt_version=form.gpt_version.data,
            is_active=form.is_active.data,
            is_default=form.is_default.data,
            cost_info_id=cost_info.id
        )
        
        db.session.add(model_config)
        
        # If this is set as default, clear other defaults
        if form.is_default.data:
            other_models = ModelConfig.query.filter(ModelConfig.id != model_config.id).all()
            for model in other_models:
                model.is_default = False
        
        db.session.commit()
        
        flash(f'Model "{form.display_name.data}" created successfully', 'success')
        return redirect(url_for('dashboard.settings_models'))
    
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
    model = ModelConfig.query.get_or_404(model_id)
    
    if request.method == 'GET':
        # Populate form with model data
        form = ModelConfigForm(obj=model)
        
        # Set cost info values if available
        if model.cost_info:
            form.input_cost_per_1k_tokens.data = model.cost_info.input_cost_per_1k_tokens
            form.output_cost_per_1k_tokens.data = model.cost_info.output_cost_per_1k_tokens
    else:
        form = ModelConfigForm()
        
        if form.validate_on_submit():
            # Update model config
            model.model_id = form.model_id.data
            model.display_name = form.display_name.data
            model.provider = ModelProvider(form.provider.data)
            model.description = form.description.data
            model.context_length = form.context_length.data
            model.supports_vision = form.supports_vision.data
            model.supports_function_calling = form.supports_function_calling.data
            model.requires_api_key = form.requires_api_key.data
            model.command_args = form.command_args.data
            model.gpt_version = form.gpt_version.data
            model.is_active = form.is_active.data
            model.is_default = form.is_default.data
            
            # Update cost info
            if model.cost_info:
                model.cost_info.input_cost_per_1k_tokens = form.input_cost_per_1k_tokens.data or 0.0
                model.cost_info.output_cost_per_1k_tokens = form.output_cost_per_1k_tokens.data or 0.0
            else:
                cost_info = ModelCostInfo(
                    model_id=model.model_id,
                    input_cost_per_1k_tokens=form.input_cost_per_1k_tokens.data or 0.0,
                    output_cost_per_1k_tokens=form.output_cost_per_1k_tokens.data or 0.0
                )
                db.session.add(cost_info)
                db.session.flush()  # To get the ID
                model.cost_info_id = cost_info.id
            
            # If this is set as default, clear other defaults
            if form.is_default.data:
                other_models = ModelConfig.query.filter(ModelConfig.id != model.id).all()
                for other_model in other_models:
                    other_model.is_default = False
            
            db.session.commit()
            
            flash(f'Model "{form.display_name.data}" updated successfully', 'success')
            return redirect(url_for('dashboard.settings_models'))
        
        # Display form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')
    
    return render_template(
        'dashboard/edit_model.html',
        title=f'Edit Model: {model.display_name}',
        form=form,
        model=model
    )


@dashboard_bp.route('/settings/models/set-default', methods=['POST'])
@login_required
def set_default_model():
    """Set a model as the default."""
    model_id = request.form.get('model_id', type=int)
    
    if not model_id:
        flash('Model ID is required', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Get the model
    model = ModelConfig.query.get_or_404(model_id)
    
    # Set this model as default
    model.is_default = True
    
    # Clear other defaults
    other_models = ModelConfig.query.filter(ModelConfig.id != model.id).all()
    for other_model in other_models:
        other_model.is_default = False
    
    db.session.commit()
    
    flash(f'Model "{model.display_name}" set as the default model', 'success')
    return redirect(url_for('dashboard.settings_models'))


@dashboard_bp.route('/settings/models/toggle-status', methods=['POST'])
@login_required
def toggle_model_status():
    """Activate or deactivate a model."""
    model_id = request.form.get('model_id', type=int)
    action = request.form.get('action')
    
    if not model_id or action not in ['activate', 'deactivate']:
        flash('Invalid request', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Get the model
    model = ModelConfig.query.get_or_404(model_id)
    
    # Update model status
    if action == 'activate':
        model.is_active = True
        message = f'Model "{model.display_name}" activated'
    else:
        model.is_active = False
        message = f'Model "{model.display_name}" deactivated'
    
    db.session.commit()
    
    flash(message, 'success')
    return redirect(url_for('dashboard.settings_models'))


@dashboard_bp.route('/settings/api-keys', methods=['POST'])
@login_required
def save_api_key():
    """Save an API key for a provider."""
    provider = request.form.get('provider')
    api_key = request.form.get('api_key')
    
    if not provider or not api_key:
        flash('Provider and API key are required', 'error')
        return redirect(url_for('dashboard.settings_models'))
    
    # Save the API key to settings
    setting_key = f'{provider}_api_key'
    ProjectSettings.set(setting_key, api_key, f'API key for {provider}')
    
    # Also set in environment for current session
    os.environ[f'{provider.upper()}_API_KEY'] = api_key
    
    flash(f'API key for {provider.capitalize()} saved successfully', 'success')
    return redirect(url_for('dashboard.settings_models'))