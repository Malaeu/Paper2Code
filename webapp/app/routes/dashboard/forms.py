from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import (
    StringField, TextAreaField, SelectField, SubmitField, 
    BooleanField, FloatField, IntegerField, HiddenField
)
from wtforms.validators import DataRequired, Length, Optional, NumberRange

from app.models.projects import ProjectType
from app.models.config import ModelProvider


class ProjectForm(FlaskForm):
    """Form for creating or editing a project."""
    
    name = StringField('Project Name', validators=[
        DataRequired(),
        Length(min=3, max=255)
    ])
    
    description = TextAreaField('Description', validators=[
        Optional(),
        Length(max=1000)
    ])
    
    project_type = SelectField('Project Type', validators=[DataRequired()], coerce=str)
    
    paper = FileField('Scientific Paper (PDF)', validators=[
        FileRequired(),
        FileAllowed(['pdf'], 'PDF files only!')
    ])
    
    programming_language = SelectField('Programming Language', validators=[Optional()], coerce=str)
    
    # Model selection for the project
    model_config_id = SelectField('AI Model', validators=[DataRequired()], coerce=int)
    
    submit = SubmitField('Create Project')
    
    def __init__(self, *args, **kwargs):
        super(ProjectForm, self).__init__(*args, **kwargs)
        
        # Populate project type choices
        self.project_type.choices = [
            (pt.value, pt.display_name) for pt in ProjectType
        ]
        
        # Populate programming language choices
        self.programming_language.choices = [
            ('python', 'Python'),
            ('java', 'Java'),
            ('javascript', 'JavaScript'),
            ('c++', 'C++'),
            ('go', 'Go'),
            ('rust', 'Rust')
        ]
        
        # Populate model choices (we'll set these in the route)
        # Will be set in route: 
        # form.model_config_id.choices = [(m.id, m.display_name) for m in ModelConfig.query.filter_by(is_active=True).all()]


class ModelConfigForm(FlaskForm):
    """Form for creating or editing a model configuration."""
    
    model_id = StringField('Model ID', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    
    display_name = StringField('Display Name', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    
    provider = SelectField('Provider', validators=[DataRequired()], coerce=str)
    
    description = TextAreaField('Description', validators=[
        Optional(),
        Length(max=1000)
    ])
    
    context_length = IntegerField('Context Length', validators=[
        DataRequired(),
        NumberRange(min=1, message='Context length must be positive')
    ])
    
    supports_vision = BooleanField('Supports Vision', default=False)
    supports_function_calling = BooleanField('Supports Function Calling', default=False)
    requires_api_key = BooleanField('Requires API Key', default=True)
    
    command_args = StringField('Command Arguments', validators=[
        Optional(),
        Length(max=255)
    ])
    
    gpt_version = StringField('GPT Version', validators=[
        Optional(),
        Length(max=50)
    ])
    
    is_active = BooleanField('Active', default=True)
    is_default = BooleanField('Default Model', default=False)
    
    # Cost information
    input_cost_per_1k_tokens = FloatField('Input Cost per 1K Tokens', validators=[
        Optional(),
        NumberRange(min=0, message='Cost must be non-negative')
    ])
    
    output_cost_per_1k_tokens = FloatField('Output Cost per 1K Tokens', validators=[
        Optional(),
        NumberRange(min=0, message='Cost must be non-negative')
    ])
    
    submit = SubmitField('Save Model')
    
    def __init__(self, *args, **kwargs):
        super(ModelConfigForm, self).__init__(*args, **kwargs)
        
        # Populate provider choices
        self.provider.choices = [
            (p.value, p.display_name) for p in ModelProvider
        ]


class DirectoryConfigForm(FlaskForm):
    """Form for creating or editing a directory configuration."""
    
    name = StringField('Name', validators=[
        DataRequired(),
        Length(min=1, max=100)
    ])
    
    path = StringField('Path', validators=[
        DataRequired(),
        Length(min=1, max=255)
    ])
    
    description = TextAreaField('Description', validators=[
        Optional(),
        Length(max=1000)
    ])
    
    is_default = BooleanField('Default Directory', default=False)
    
    move_files = BooleanField('Move existing files to new location', default=False)
    
    submit = SubmitField('Save Directory')
    
    def validate_path(self, field):
        """Validate that the path is in a proper format."""
        # Import here to avoid circular imports
        from app.utils.path_utils import sanitize_path
        
        try:
            # Just try to sanitize the path - will raise ValueError if invalid
            sanitize_path(field.data)
        except ValueError as e:
            raise ValidationError(str(e))


class ApiKeyForm(FlaskForm):
    """Form for entering API keys."""
    
    provider = HiddenField('Provider', validators=[DataRequired()])
    api_key = StringField('API Key', validators=[DataRequired()])
    submit = SubmitField('Save API Key')