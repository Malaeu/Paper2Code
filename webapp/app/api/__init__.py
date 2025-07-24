from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import and register api modules
from .auth import auth_routes  # Important: keep this import after api_bp creation
from .projects import project_routes
from .config import config_api_bp
from .usage import usage_api_bp
from .directories import *  # Import all directory API endpoints

# Register blueprints
api_bp.register_blueprint(auth_routes.auth_api_bp, url_prefix='/auth')
api_bp.register_blueprint(project_routes.project_api_bp, url_prefix='/projects')
api_bp.register_blueprint(config_api_bp, url_prefix='/config')
api_bp.register_blueprint(usage_api_bp, url_prefix='/usage')