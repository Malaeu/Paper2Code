from flask import render_template, current_app, redirect, url_for
from flask_login import current_user
from app.routes.main import main_bp

@main_bp.route('/')
def index():
    # Redirect to dashboard if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    return render_template('index.html', title='Paper2Code')

@main_bp.route('/health')
def health():
    return {'status': 'healthy', 'version': '1.0.0'}