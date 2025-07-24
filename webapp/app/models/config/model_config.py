import datetime
import enum
import json
import os
from flask import current_app
from typing import Dict, List, Optional, Set, Tuple, Any

from app.extensions import db


class ModelProvider(enum.Enum):
    """Enum for model providers."""
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    LOCAL = 'local'
    DEEPSEEK = 'deepseek'
    HUGGINGFACE = 'huggingface'
    OTHER = 'other'
    
    @property
    def display_name(self):
        """Return human-readable display name."""
        names = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'local': 'Local Models',
            'deepseek': 'DeepSeek AI',
            'huggingface': 'Hugging Face',
            'other': 'Other Providers'
        }
        return names.get(self.value, self.value)
    
    @property
    def description(self):
        """Return provider description."""
        descriptions = {
            'openai': 'OpenAI provides powerful AI models like GPT-4, GPT-3.5, etc.',
            'anthropic': 'Anthropic offers Claude models with strong reasoning capabilities.',
            'local': 'Locally hosted models that run on your own hardware.',
            'deepseek': 'DeepSeek AI provides specialized coding models.',
            'huggingface': 'Hugging Face offers a variety of open-source models.',
            'other': 'Other model providers supported by the system.'
        }
        return descriptions.get(self.value, '')
    
    @property
    def website(self):
        """Return provider website."""
        websites = {
            'openai': 'https://openai.com',
            'anthropic': 'https://anthropic.com',
            'deepseek': 'https://deepseek.ai',
            'huggingface': 'https://huggingface.co',
            'local': '#',
            'other': '#'
        }
        return websites.get(self.value, '#')


class ModelCostInfo(db.Model):
    """Model for tracking cost information for different models."""
    
    __tablename__ = 'model_cost_info'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(100), unique=True, nullable=False)
    
    # Cost per token (in USD)
    input_cost_per_1k_tokens = db.Column(db.Float, default=0.0, nullable=False)
    output_cost_per_1k_tokens = db.Column(db.Float, default=0.0, nullable=False)
    
    # Usage metrics
    total_tokens_used = db.Column(db.Integer, default=0, nullable=False)
    total_cost = db.Column(db.Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    
    def update_usage(self, input_tokens: int, output_tokens: int) -> float:
        """
        Update usage metrics and return the cost of this operation.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            
        Returns:
            float: Cost of this operation
        """
        # Calculate cost
        input_cost = input_tokens * self.input_cost_per_1k_tokens / 1000
        output_cost = output_tokens * self.output_cost_per_1k_tokens / 1000
        total_cost = input_cost + output_cost
        
        # Update metrics
        self.total_tokens_used += (input_tokens + output_tokens)
        self.total_cost += total_cost
        
        return total_cost


class ModelConfig(db.Model):
    """Model for storing configuration of AI models."""
    
    __tablename__ = 'model_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(100), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    provider = db.Column(db.Enum(ModelProvider), default=ModelProvider.OTHER, nullable=False)
    
    # Model characteristics
    description = db.Column(db.Text, nullable=True)
    context_length = db.Column(db.Integer, default=8192, nullable=False)
    supports_vision = db.Column(db.Boolean, default=False, nullable=False)
    supports_function_calling = db.Column(db.Boolean, default=False, nullable=False)
    
    # Implementation details
    requires_api_key = db.Column(db.Boolean, default=True, nullable=False)
    command_args = db.Column(db.String(255), nullable=True)  # CLI args for local models
    gpt_version = db.Column(db.String(50), nullable=True)    # For OpenAI models
    
    # Cost reference (nullable if free/local)
    cost_info_id = db.Column(db.Integer, db.ForeignKey('model_cost_info.id'), nullable=True)
    cost_info = db.relationship('ModelCostInfo', backref='model', uselist=False)
    
    # Status fields
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_default = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    
    def __init__(self, **kwargs):
        """Initialize a model configuration."""
        super(ModelConfig, self).__init__(**kwargs)
        
        # Set provider-specific defaults
        if self.provider == ModelProvider.OPENAI:
            self.requires_api_key = True
        elif self.provider == ModelProvider.ANTHROPIC:
            self.requires_api_key = True
        elif self.provider == ModelProvider.LOCAL:
            self.requires_api_key = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model config to dictionary."""
        result = {
            'id': self.id,
            'model_id': self.model_id,
            'display_name': self.display_name,
            'provider': self.provider.value,
            'provider_display': self.provider.display_name,
            'description': self.description,
            'context_length': self.context_length,
            'supports_vision': self.supports_vision,
            'supports_function_calling': self.supports_function_calling,
            'requires_api_key': self.requires_api_key,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'command_args': self.command_args,
            'gpt_version': self.gpt_version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        # Add cost information if available
        if self.cost_info:
            result.update({
                'input_cost_per_1k_tokens': self.cost_info.input_cost_per_1k_tokens,
                'output_cost_per_1k_tokens': self.cost_info.output_cost_per_1k_tokens,
                'total_tokens_used': self.cost_info.total_tokens_used,
                'total_cost': self.cost_info.total_cost
            })
            
        return result
        
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the model configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Required fields
        if not self.model_id:
            errors.append("Model ID is required")
        if not self.display_name:
            errors.append("Display name is required")
        if self.context_length <= 0:
            errors.append("Context length must be greater than 0")
            
        # Provider-specific validation
        if self.provider == ModelProvider.OPENAI and not self.gpt_version:
            errors.append("GPT version is required for OpenAI models")
        
        if self.provider == ModelProvider.LOCAL and not self.command_args:
            errors.append("Command arguments are required for local models")
            
        return len(errors) == 0, errors
        
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of running this model.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        if not self.cost_info:
            return 0.0
            
        input_cost = input_tokens * self.cost_info.input_cost_per_1k_tokens / 1000
        output_cost = output_tokens * self.cost_info.output_cost_per_1k_tokens / 1000
        
        return input_cost + output_cost
        
    def get_api_key_env_var(self) -> str:
        """
        Get the environment variable name for this model's API key.
        
        Returns:
            Environment variable name
        """
        if self.provider == ModelProvider.OPENAI:
            return 'OPENAI_API_KEY'
        elif self.provider == ModelProvider.ANTHROPIC:
            return 'ANTHROPIC_API_KEY'
        elif self.provider == ModelProvider.HUGGINGFACE:
            return 'HUGGINGFACE_API_KEY'
        elif self.provider == ModelProvider.DEEPSEEK:
            return 'DEEPSEEK_API_KEY'
        else:
            return f'{self.provider.value.upper()}_API_KEY'
            
    def needs_api_key(self) -> bool:
        """
        Check if this model requires an API key.
        
        Returns:
            True if an API key is required, False otherwise
        """
        return self.requires_api_key
        
    def has_api_key(self) -> bool:
        """
        Check if an API key is available for this model.
        
        Returns:
            True if an API key is available, False otherwise
        """
        if not self.requires_api_key:
            return True
            
        env_var = self.get_api_key_env_var()
        return env_var in os.environ and bool(os.environ[env_var])
        
    def __repr__(self):
        return f'<ModelConfig {self.display_name} ({self.model_id})>'


class DirectoryConfig(db.Model):
    """Model for storing directory configurations."""
    
    __tablename__ = 'directory_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    path = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    is_default = db.Column(db.Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)


class ProjectSettings(db.Model):
    """Model for storing global project settings."""
    
    __tablename__ = 'project_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        setting = cls.query.filter_by(key=key).first()
        if setting and setting.value:
            try:
                return json.loads(setting.value)
            except:
                return setting.value
        return default
    
    @classmethod
    def set(cls, key: str, value: Any, description: Optional[str] = None) -> None:
        """Set a setting value."""
        setting = cls.query.filter_by(key=key).first()
        
        if isinstance(value, (dict, list, tuple, set)):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
            
        if setting:
            setting.value = value_str
            if description:
                setting.description = description
        else:
            setting = cls(key=key, value=value_str, description=description)
            db.session.add(setting)
            
        db.session.commit()


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of all available models with their configurations.
    
    Returns:
        List of model configurations as dictionaries
    """
    # Define default models if none are configured
    default_models = [
        {
            'model_id': 'gpt-4o-mini',
            'display_name': 'GPT-4o Mini',
            'provider': ModelProvider.OPENAI,
            'description': 'Compact variant of GPT-4o with improved reasoning, context understanding, and instruction following.',
            'context_length': 128000,
            'supports_vision': True,
            'supports_function_calling': True,
            'requires_api_key': True,
            'gpt_version': 'o4-mini',
            'is_active': True,
            'is_default': True,
            'input_cost_per_1k_tokens': 0.15,
            'output_cost_per_1k_tokens': 0.60
        },
        {
            'model_id': 'gpt-3.5-turbo',
            'display_name': 'GPT-3.5 Turbo',
            'provider': ModelProvider.OPENAI,
            'description': 'Good balance of capabilities and cost-effectiveness.',
            'context_length': 16385,
            'supports_vision': True,
            'supports_function_calling': True,
            'requires_api_key': True,
            'gpt_version': 'o35-turbo',
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.0005,
            'output_cost_per_1k_tokens': 0.0015
        },
        {
            'model_id': 'claude-3-opus',
            'display_name': 'Claude 3 Opus',
            'provider': ModelProvider.ANTHROPIC,
            'description': 'Most powerful Claude model, excellent for complex reasoning and understanding.',
            'context_length': 200000,
            'supports_vision': True,
            'supports_function_calling': True,
            'requires_api_key': True,
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.015,
            'output_cost_per_1k_tokens': 0.075
        },
        {
            'model_id': 'claude-3-sonnet',
            'display_name': 'Claude 3 Sonnet',
            'provider': ModelProvider.ANTHROPIC,
            'description': 'Great balance of intelligence and speed for a wide range of tasks.',
            'context_length': 200000,
            'supports_vision': True,
            'supports_function_calling': True,
            'requires_api_key': True,
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.003,
            'output_cost_per_1k_tokens': 0.015
        },
        {
            'model_id': 'claude-3-haiku',
            'display_name': 'Claude 3 Haiku',
            'provider': ModelProvider.ANTHROPIC,
            'description': 'Fast and cost-effective Claude model, good for simpler tasks.',
            'context_length': 200000,
            'supports_vision': True,
            'supports_function_calling': True,
            'requires_api_key': True,
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.00025,
            'output_cost_per_1k_tokens': 0.00125
        },
        {
            'model_id': 'deepseek-coder-v2-lite',
            'display_name': 'DeepSeek Coder V2 Lite',
            'provider': ModelProvider.DEEPSEEK,
            'description': 'Specialized coding model from DeepSeek AI.',
            'context_length': 16384,
            'supports_vision': False,
            'supports_function_calling': True,
            'requires_api_key': False,
            'command_args': '--model_name "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" --tp_size 2',
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.0,
            'output_cost_per_1k_tokens': 0.0
        },
        {
            'model_id': 'llama-3-70b',
            'display_name': 'Llama 3 70B',
            'provider': ModelProvider.LOCAL,
            'description': 'Meta\'s Llama 3 70B model for local inference.',
            'context_length': 8192,
            'supports_vision': False,
            'supports_function_calling': False,
            'requires_api_key': False,
            'command_args': '--model_name "meta-llama/Llama-3-70b-chat-hf"',
            'is_active': True,
            'is_default': False,
            'input_cost_per_1k_tokens': 0.0,
            'output_cost_per_1k_tokens': 0.0
        }
    ]
    
    # Get configured models from database
    models = ModelConfig.query.filter_by(is_active=True).all()
    
    # If no models, initialize with defaults
    if not models:
        for model_data in default_models:
            provider = model_data.pop('provider')
            input_cost = model_data.pop('input_cost_per_1k_tokens', 0)
            output_cost = model_data.pop('output_cost_per_1k_tokens', 0)
            
            # Create cost info
            cost_info = ModelCostInfo(
                model_id=model_data['model_id'],
                input_cost_per_1k_tokens=input_cost,
                output_cost_per_1k_tokens=output_cost
            )
            db.session.add(cost_info)
            db.session.flush()  # To get the cost_info ID
            
            # Create model config
            model_config = ModelConfig(
                provider=provider,
                cost_info_id=cost_info.id,
                **model_data
            )
            db.session.add(model_config)
            
        db.session.commit()
        models = ModelConfig.query.filter_by(is_active=True).all()
    
    # Convert to list of dictionaries
    return [model.to_dict() for model in models]


def get_default_directories() -> List[Dict[str, Any]]:
    """
    Get default directory configurations.
    
    Returns:
        List of directory configurations as dictionaries
    """
    project_root = current_app.config['PROJECT_ROOT']
    
    return [
        {
            'name': 'examples',
            'path': os.path.join(project_root, 'examples'),
            'description': 'Example papers and datasets',
            'is_default': True
        },
        {
            'name': 'data',
            'path': os.path.join(project_root, 'data'),
            'description': 'Data files and datasets',
            'is_default': False
        },
        {
            'name': 'outputs',
            'path': os.path.join(project_root, 'outputs'),
            'description': 'Generated output files',
            'is_default': False
        },
        {
            'name': 'uploads',
            'path': os.path.join(current_app.root_path, 'uploads'),
            'description': 'User uploaded files',
            'is_default': False
        }
    ]


def initialize_directories() -> None:
    """Initialize directory configurations if they don't exist."""
    # Check if any directories are configured
    if DirectoryConfig.query.count() > 0:
        return
        
    # Get default directories
    default_dirs = get_default_directories()
    
    # Create directory configurations
    for dir_data in default_dirs:
        dir_config = DirectoryConfig(**dir_data)
        db.session.add(dir_config)
        
    db.session.commit()