"""API routes for configuration management."""

from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user

from app.models.config import ModelConfig, ModelProvider
from app.services.model_service import ModelService

config_api_bp = Blueprint('config_api', __name__)


@config_api_bp.route('/models', methods=['GET'])
@login_required
def get_models():
    """Get all model configurations."""
    active_only = request.args.get('active_only', 'false').lower() == 'true'
    models = ModelService.get_all_models(active_only=active_only)
    return jsonify({'models': models})


@config_api_bp.route('/models/<int:model_id>', methods=['GET'])
@login_required
def get_model(model_id):
    """Get a model configuration by ID."""
    model = ModelService.get_model_by_id(model_id)
    
    if not model:
        return jsonify({'error': f'Model with ID {model_id} not found'}), 404
        
    return jsonify({'model': model})


@config_api_bp.route('/models/default', methods=['GET'])
@login_required
def get_default_model():
    """Get the default model configuration."""
    model = ModelService.get_default_model()
    
    if not model:
        return jsonify({'error': 'No default model found'}), 404
        
    return jsonify({'model': model})


@config_api_bp.route('/models', methods=['POST'])
@login_required
def create_model():
    """Create a new model configuration."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    success, message, model = ModelService.create_model(data)
    
    if not success:
        return jsonify({'error': message}), 400
        
    return jsonify({'message': message, 'model': model}), 201


@config_api_bp.route('/models/<int:model_id>', methods=['PUT'])
@login_required
def update_model(model_id):
    """Update a model configuration."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    success, message, model = ModelService.update_model(model_id, data)
    
    if not success:
        return jsonify({'error': message}), 400 if 'not found' not in message else 404
        
    return jsonify({'message': message, 'model': model})


@config_api_bp.route('/models/<int:model_id>', methods=['DELETE'])
@login_required
def delete_model(model_id):
    """Delete a model configuration."""
    success, message = ModelService.delete_model(model_id)
    
    if not success:
        return jsonify({'error': message}), 400 if 'not found' not in message else 404
        
    return jsonify({'message': message})


@config_api_bp.route('/models/<int:model_id>/default', methods=['PUT'])
@login_required
def set_default_model(model_id):
    """Set a model as the default."""
    success, message = ModelService.set_default_model(model_id)
    
    if not success:
        return jsonify({'error': message}), 400 if 'not found' not in message else 404
        
    return jsonify({'message': message})


@config_api_bp.route('/models/<int:model_id>/status', methods=['PUT'])
@login_required
def toggle_model_status(model_id):
    """Activate or deactivate a model."""
    data = request.json
    
    if not data or 'activate' not in data:
        return jsonify({'error': 'No data provided or missing activate parameter'}), 400
        
    success, message = ModelService.toggle_model_status(model_id, data['activate'])
    
    if not success:
        return jsonify({'error': message}), 400 if 'not found' not in message else 404
        
    return jsonify({'message': message})


@config_api_bp.route('/api-keys', methods=['GET'])
@login_required
def get_api_keys():
    """Get all saved API keys (masked)."""
    api_keys = ModelService.get_api_keys()
    
    # Mask API keys for security
    masked_keys = {}
    for provider, key in api_keys.items():
        if key:
            # Mask all but the first and last 4 characters
            if len(key) > 8:
                masked_keys[provider] = key[:4] + '*' * (len(key) - 8) + key[-4:]
            else:
                masked_keys[provider] = '********'
        else:
            masked_keys[provider] = ''
            
    return jsonify({'api_keys': masked_keys})


@config_api_bp.route('/api-keys', methods=['POST'])
@login_required
def save_api_key():
    """Save an API key for a provider."""
    data = request.json
    
    if not data or 'provider' not in data or 'api_key' not in data:
        return jsonify({'error': 'Missing provider or api_key parameters'}), 400
        
    success, message = ModelService.save_api_key(data['provider'], data['api_key'])
    
    if not success:
        return jsonify({'error': message}), 400
        
    return jsonify({'message': message})


@config_api_bp.route('/providers', methods=['GET'])
@login_required
def get_providers():
    """Get all model providers."""
    providers = []
    
    for provider in ModelProvider:
        providers.append({
            'id': provider.value,
            'name': provider.display_name,
            'description': provider.description,
            'website': provider.website
        })
        
    return jsonify({'providers': providers})