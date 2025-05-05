import os
from flask import Flask
from logging.config import dictConfig

def create_app(config_name=None):
    # Configure logging
    dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            }
        },
        'handlers': {
            'wsgi': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/paper2code.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'formatter': 'default'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi', 'file']
        }
    })
    
    # Create Flask app instance
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    
    # Load appropriate config
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
        
    app.config.from_object(f'config.{config_name.capitalize()}Config')
    
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Initialize extensions
    from app.extensions import db, migrate, celery, csrf, login_manager, mail
    
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    celery.conf.update(app.config)
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    
    # Initialize default directories
    with app.app_context():
        from app.models.config.model_config import initialize_directories
        initialize_directories()
    
    return app