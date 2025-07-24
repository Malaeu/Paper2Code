from flask_migrate import Migrate, init
from app import create_app
from app.extensions import db

app = create_app('development')
with app.app_context():
    migrate = Migrate(app, db)
    init('migrations')