"""Service layer for model configuration operations."""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from flask import current_app

from app.extensions import db
from app.models.config.model_config import (
    ModelConfig, ModelCostInfo, ModelProvider, ProjectSettings
)


class ModelService:
    """Service for managing AI model configurations."""
    
    @staticmethod
    def get_all_models(active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all model configurations.
        
        Args:
            active_only: If True, return only active models
            
        Returns:
            List of model configurations as dictionaries
        """
        query = ModelConfig.query
        
        if active_only:
            query = query.filter_by(is_active=True)
            
        models = query.order_by(ModelConfig.is_default.desc(), ModelConfig.display_name).all()
        return [model.to_dict() for model in models]
    
    @staticmethod
    def get_model_by_id(model_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a model configuration by ID.
        
        Args:
            model_id: The database ID of the model
            
        Returns:
            Model configuration as a dictionary or None if not found
        """
        model = ModelConfig.query.get(model_id)
        return model.to_dict() if model else None
    
    @staticmethod
    def get_model_by_identifier(identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get a model configuration by its model identifier.
        
        Args:
            identifier: The model identifier (e.g., 'gpt-4o', 'claude-3-opus')
            
        Returns:
            Model configuration as a dictionary or None if not found
        """
        model = ModelConfig.query.filter_by(model_id=identifier).first()
        return model.to_dict() if model else None
    
    @staticmethod
    def get_default_model() -> Optional[Dict[str, Any]]:
        """
        Get the default model configuration.
        
        Returns:
            The default model configuration as a dictionary or None if not found
        """
        model = ModelConfig.query.filter_by(is_default=True).first()
        
        # If no default is set, get the first active model
        if not model:
            model = ModelConfig.query.filter_by(is_active=True).first()
            
        return model.to_dict() if model else None
    
    @staticmethod
    def create_model(model_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Create a new model configuration.
        
        Args:
            model_data: Dictionary containing model configuration data
            
        Returns:
            Tuple of (success, message, model_dict)
        """
        try:
            # Check if model ID already exists
            if ModelConfig.query.filter_by(model_id=model_data.get('model_id')).first():
                return False, f"Model ID '{model_data.get('model_id')}' already exists", None
            
            # Extract cost information
            input_cost = model_data.pop('input_cost_per_1k_tokens', 0.0) or 0.0
            output_cost = model_data.pop('output_cost_per_1k_tokens', 0.0) or 0.0
            
            # Get provider enum
            provider_value = model_data.pop('provider', 'other')
            provider = ModelProvider(provider_value)
            
            # Create cost info
            cost_info = ModelCostInfo(
                model_id=model_data.get('model_id'),
                input_cost_per_1k_tokens=input_cost,
                output_cost_per_1k_tokens=output_cost
            )
            db.session.add(cost_info)
            db.session.flush()  # To get the ID
            
            # Create model config
            model_config = ModelConfig(
                provider=provider,
                cost_info_id=cost_info.id,
                **model_data
            )
            
            db.session.add(model_config)
            
            # If this is set as default, clear other defaults
            if model_data.get('is_default', False):
                ModelService.clear_other_defaults(model_config.id)
                
            db.session.commit()
            
            return True, f"Model '{model_data.get('display_name')}' created successfully", model_config.to_dict()
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error creating model: {str(e)}")
            return False, f"Error creating model: {str(e)}", None
    
    @staticmethod
    def update_model(model_id: int, model_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Update an existing model configuration.
        
        Args:
            model_id: The database ID of the model to update
            model_data: Dictionary containing updated model configuration data
            
        Returns:
            Tuple of (success, message, model_dict)
        """
        try:
            # Get the model
            model = ModelConfig.query.get(model_id)
            
            if not model:
                return False, f"Model with ID {model_id} not found", None
            
            # Check if model ID already exists (if changed)
            if (model_data.get('model_id') != model.model_id and 
                ModelConfig.query.filter_by(model_id=model_data.get('model_id')).first()):
                return False, f"Model ID '{model_data.get('model_id')}' already exists", None
            
            # Extract cost information
            input_cost = model_data.pop('input_cost_per_1k_tokens', 0.0) or 0.0
            output_cost = model_data.pop('output_cost_per_1k_tokens', 0.0) or 0.0
            
            # Get provider enum
            provider_value = model_data.pop('provider', model.provider.value)
            provider = ModelProvider(provider_value)
            
            # Update model config
            for key, value in model_data.items():
                setattr(model, key, value)
                
            model.provider = provider
            
            # Update cost info
            if model.cost_info:
                model.cost_info.input_cost_per_1k_tokens = input_cost
                model.cost_info.output_cost_per_1k_tokens = output_cost
                
                # Update model ID in cost info if changed
                if model_data.get('model_id') and model.cost_info.model_id != model_data.get('model_id'):
                    model.cost_info.model_id = model_data.get('model_id')
            else:
                # Create cost info if it doesn't exist
                cost_info = ModelCostInfo(
                    model_id=model.model_id,
                    input_cost_per_1k_tokens=input_cost,
                    output_cost_per_1k_tokens=output_cost
                )
                db.session.add(cost_info)
                db.session.flush()  # To get the ID
                model.cost_info_id = cost_info.id
            
            # If this is set as default, clear other defaults
            if model_data.get('is_default', False):
                ModelService.clear_other_defaults(model.id)
                
            db.session.commit()
            
            return True, f"Model '{model.display_name}' updated successfully", model.to_dict()
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error updating model: {str(e)}")
            return False, f"Error updating model: {str(e)}", None
    
    @staticmethod
    def delete_model(model_id: int) -> Tuple[bool, str]:
        """
        Delete a model configuration.
        
        Args:
            model_id: The database ID of the model to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the model
            model = ModelConfig.query.get(model_id)
            
            if not model:
                return False, f"Model with ID {model_id} not found"
            
            # Check if it's the only model
            if ModelConfig.query.count() <= 1:
                return False, "Cannot delete the only model configuration"
            
            # Check if it's the default model
            if model.is_default:
                # Find another model to set as default
                other_model = ModelConfig.query.filter(ModelConfig.id != model_id).first()
                if other_model:
                    other_model.is_default = True
                    
            # Delete cost info
            if model.cost_info:
                db.session.delete(model.cost_info)
                
            # Delete model
            db.session.delete(model)
            db.session.commit()
            
            return True, f"Model '{model.display_name}' deleted successfully"
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error deleting model: {str(e)}")
            return False, f"Error deleting model: {str(e)}"
    
    @staticmethod
    def set_default_model(model_id: int) -> Tuple[bool, str]:
        """
        Set a model as the default.
        
        Args:
            model_id: The database ID of the model to set as default
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the model
            model = ModelConfig.query.get(model_id)
            
            if not model:
                return False, f"Model with ID {model_id} not found"
            
            # Make sure the model is active
            if not model.is_active:
                return False, "Cannot set an inactive model as default"
            
            # Set this model as default
            model.is_default = True
            
            # Clear other defaults
            ModelService.clear_other_defaults(model_id)
            
            db.session.commit()
            
            return True, f"Model '{model.display_name}' set as the default model"
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error setting default model: {str(e)}")
            return False, f"Error setting default model: {str(e)}"
    
    @staticmethod
    def toggle_model_status(model_id: int, activate: bool) -> Tuple[bool, str]:
        """
        Activate or deactivate a model.
        
        Args:
            model_id: The database ID of the model to update
            activate: True to activate, False to deactivate
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the model
            model = ModelConfig.query.get(model_id)
            
            if not model:
                return False, f"Model with ID {model_id} not found"
            
            # Update model status
            model.is_active = activate
            
            # If deactivating a default model, find a new default
            if not activate and model.is_default:
                # Find another active model to set as default
                other_model = ModelConfig.query.filter(
                    ModelConfig.id != model_id, 
                    ModelConfig.is_active == True
                ).first()
                
                if other_model:
                    model.is_default = False
                    other_model.is_default = True
                else:
                    return False, "Cannot deactivate the only active model"
            
            db.session.commit()
            
            action = "activated" if activate else "deactivated"
            return True, f"Model '{model.display_name}' {action} successfully"
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error toggling model status: {str(e)}")
            return False, f"Error toggling model status: {str(e)}"
    
    @staticmethod
    def save_api_key(provider: str, api_key: str) -> Tuple[bool, str]:
        """
        Save an API key for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            api_key: The API key to save
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Save the API key to settings
            setting_key = f'{provider}_api_key'
            ProjectSettings.set(setting_key, api_key, f'API key for {provider}')
            
            # Also set in environment for current session
            if api_key:
                os.environ[f'{provider.upper()}_API_KEY'] = api_key
            elif f'{provider.upper()}_API_KEY' in os.environ:
                del os.environ[f'{provider.upper()}_API_KEY']
            
            return True, f"API key for {provider.capitalize()} saved successfully"
        
        except Exception as e:
            current_app.logger.error(f"Error saving API key: {str(e)}")
            return False, f"Error saving API key: {str(e)}"
    
    @staticmethod
    def get_api_keys() -> Dict[str, str]:
        """
        Get saved API keys for all providers.
        
        Returns:
            Dictionary mapping provider names to API keys
        """
        return {
            'openai': ProjectSettings.get('openai_api_key', ''),
            'anthropic': ProjectSettings.get('anthropic_api_key', ''),
            'huggingface': ProjectSettings.get('huggingface_api_key', ''),
            'deepseek': ProjectSettings.get('deepseek_api_key', '')
        }
    
    @staticmethod
    def update_usage_stats(model_id: str, input_tokens: int, output_tokens: int) -> float:
        """
        Update usage statistics for a model.
        
        Args:
            model_id: The model identifier (e.g., 'gpt-4o', 'claude-3-opus')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            
        Returns:
            Cost of this operation
        """
        try:
            # Get model cost info
            cost_info = ModelCostInfo.query.filter_by(model_id=model_id).first()
            
            if not cost_info:
                return 0.0
                
            # Update usage stats
            cost = cost_info.update_usage(input_tokens, output_tokens)
            db.session.commit()
            
            return cost
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error updating usage stats: {str(e)}")
            return 0.0
    
    @staticmethod
    def clear_other_defaults(model_id: int) -> None:
        """
        Clear default flag from all models except the specified one.
        
        Args:
            model_id: The database ID of the model to keep as default
        """
        other_models = ModelConfig.query.filter(ModelConfig.id != model_id).all()
        for other_model in other_models:
            other_model.is_default = False