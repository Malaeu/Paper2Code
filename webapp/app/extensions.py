from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager
from flask_mail import Mail
from celery import Celery

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
login_manager = LoginManager()
mail = Mail()
celery = Celery('paper2code')

# Configure Celery
class FlaskCelery(Celery):
    def __init__(self, *args, **kwargs):
        super(FlaskCelery, self).__init__(*args, **kwargs)
        self.flask_app = None
        
        # Configure Celery to return JSON serializable results
        self.conf.update(
            accept_content=['json', 'pickle'],
            task_serializer='json',
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
        )
        
    def init_app(self, app):
        self.flask_app = app
        self.conf.update(app.config)
        
celery = FlaskCelery('paper2code')