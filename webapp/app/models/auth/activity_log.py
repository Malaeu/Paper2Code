"""User activity log model definition."""

import datetime
import enum
from app.extensions import db


class ActivityType(enum.Enum):
    """Enum for types of user activities."""
    
    LOGIN = 'login'
    LOGOUT = 'logout'
    PROFILE_UPDATE = 'profile_update'
    PASSWORD_CHANGE = 'password_change'
    EMAIL_CHANGE = 'email_change' 
    EMAIL_VERIFY = 'email_verify'
    API_KEY_CREATE = 'api_key_create'
    API_KEY_DELETE = 'api_key_delete'
    API_KEY_TOGGLE = 'api_key_toggle'
    PASSWORD_RESET_REQUEST = 'password_reset_request'
    PASSWORD_RESET = 'password_reset'
    API_CALL = 'api_call'
    EMAIL_SENT = 'email_sent'
    PROJECT_CREATE = 'project_create'
    PROJECT_UPDATE = 'project_update'
    PROJECT_DELETE = 'project_delete'
    CONFIG_UPDATE = 'config_update'


class UserActivityLog(db.Model):
    """Model for tracking user activity."""
    
    __tablename__ = 'user_activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    activity_type = db.Column(db.Enum(ActivityType), nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    meta_data = db.Column(db.Text, nullable=True)  # JSON data for additional information
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('activity_logs', lazy='dynamic'))
    
    def __init__(self, user_id=None, activity_type=None, ip_address=None, 
                 user_agent=None, description=None, meta_data=None):
        self.user_id = user_id
        self.activity_type = activity_type
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.description = description
        self.meta_data = meta_data
    
    def to_dict(self):
        """Convert activity log to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'activity_type': self.activity_type.value,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'description': self.description,
            'meta_data': self.meta_data,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<UserActivityLog {self.activity_type.value} by user {self.user_id}>'