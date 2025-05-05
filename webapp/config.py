import os
from datetime import timedelta

class Config:
    """Base config class"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-production')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size
    
    # Application
    SITE_URL = os.environ.get('SITE_URL', 'http://localhost:5000')
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Celery
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    REMEMBER_COOKIE_SECURE = os.environ.get('REMEMBER_COOKIE_SECURE', 'False').lower() == 'true'
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = 'Lax'
    
    # Email configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@paper2code.example.com')
    MAIL_MAX_EMAILS = 5
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    
    @staticmethod
    def init_app(app):
        # Create upload directory if it doesn't exist
        os.makedirs(os.path.join(app.root_path, 'uploads'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'uploads', 'papers'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'uploads', 'datasets'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'uploads', 'outputs'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'uploads', 'temp'), exist_ok=True)
        
        # Create logs directory for projects
        os.makedirs(os.path.join(app.root_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'logs', 'projects'), exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL', 'sqlite:///dev.db')
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test.db')
    WTF_CSRF_ENABLED = False
    

class ProductionConfig(Config):
    """Production configuration"""
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///prod.db')
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SITE_URL = os.environ.get('SITE_URL', 'https://paper2code.example.com')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific logging
        import logging
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler('logs/paper2code.log', maxBytes=10485760, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # Set production log level
        app.logger.setLevel(logging.INFO)
        app.logger.info('Paper2Code startup')