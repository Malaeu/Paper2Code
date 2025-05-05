from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import and register api modules
from .auth import auth_routes  # Important: keep this import after api_bp creation
from .projects import project_routes
from .directories import *  # Import all directory API endpoints