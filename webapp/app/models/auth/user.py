import datetime
import secrets
import enum
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db


class UserRole(enum.Enum):
    USER = 'user'
    ADMIN = 'admin'


class UserStatus(enum.Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    SUSPENDED = 'suspended'


class User(UserMixin, db.Model):
    """User model for authentication and authorization."""
    
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True, nullable=False)
    email = db.Column(db.String(120), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.Enum(UserRole), default=UserRole.USER, nullable=False)
    status = db.Column(db.Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    
    # Account verification
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_token = db.Column(db.String(100), unique=True, nullable=True)
    verification_token_expiry = db.Column(db.DateTime, nullable=True)
    pending_email = db.Column(db.String(120), nullable=True)
    
    # Two-factor authentication
    two_factor_enabled = db.Column(db.Boolean, default=False, nullable=False)
    two_factor_secret = db.Column(db.String(32), nullable=True)
    
    # Authentication history
    last_login = db.Column(db.DateTime, nullable=True)
    last_login_ip = db.Column(db.String(45), nullable=True)
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)
    
    # API keys relation
    api_keys = db.relationship('ApiKey', back_populates='user', cascade='all, delete-orphan')
    
    # Usage metrics
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    total_api_calls = db.Column(db.Integer, default=0, nullable=False)
    
    def __init__(self, username, email, password, role=UserRole.USER):
        self.username = username
        self.email = email.lower()
        self.set_password(password)
        self.role = role
        self.generate_verification_token()
    
    def set_password(self, password):
        """Hash password and store the hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)
    
    def generate_verification_token(self):
        """Generate a token for email verification."""
        self.verification_token = secrets.token_urlsafe(64)
        self.verification_token_expiry = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        return self.verification_token
    
    def verify_email(self, token):
        """Verify user's email using the provided token."""
        if (self.verification_token and 
                self.verification_token == token and 
                self.verification_token_expiry > datetime.datetime.utcnow()):
            self.email_verified = True
            self.verification_token = None
            self.verification_token_expiry = None
            self.status = UserStatus.ACTIVE
            return True
        return False
        
    def verify_email_change(self, token):
        """Verify email change using the provided token."""
        if (self.verification_token and 
                self.verification_token == token and 
                self.verification_token_expiry > datetime.datetime.utcnow() and
                self.pending_email):
            old_email = self.email
            self.email = self.pending_email
            self.pending_email = None
            self.verification_token = None
            self.verification_token_expiry = None
            return old_email
        return None
    
    def update_login_info(self, ip_address):
        """Update login information after successful login."""
        self.last_login = datetime.datetime.utcnow()
        self.last_login_ip = ip_address
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def increment_failed_login(self):
        """Increment failed login attempts and lock account if necessary."""
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            # Lock for 30 minutes
            self.locked_until = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    
    def is_account_locked(self):
        """Check if account is temporarily locked due to failed login attempts."""
        if self.locked_until and self.locked_until > datetime.datetime.utcnow():
            return True
        return False
    
    def has_role(self, role):
        """Check if user has the specified role."""
        if isinstance(role, str):
            return self.role.value == role
        return self.role == role
    
    def is_admin(self):
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN
    
    def is_active_user(self):
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE and self.email_verified
    
    @property
    def is_active(self):
        """Required by Flask-Login."""
        return self.status == UserStatus.ACTIVE
    
    def increment_api_usage(self, count=1):
        """Increment API usage count."""
        self.total_api_calls += count
    
    def to_dict(self):
        """Convert user to dictionary (for API responses)."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'status': self.status.value,
            'email_verified': self.email_verified,
            'two_factor_enabled': self.two_factor_enabled,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'total_api_calls': self.total_api_calls
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class ApiKey(db.Model):
    """API Key model for storing user API keys."""
    
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    service = db.Column(db.String(50), nullable=False)  # e.g., 'openai', 'anthropic', 'vllm'
    key_name = db.Column(db.String(64), nullable=False)
    key_prefix = db.Column(db.String(10), nullable=False)  # Store prefix for identification
    key_hash = db.Column(db.String(256), nullable=False)  # Store hashed key for security
    
    # Usage tracking
    last_used = db.Column(db.DateTime, nullable=True)
    calls_count = db.Column(db.Integer, default=0, nullable=False)
    tokens_used = db.Column(db.Integer, default=0, nullable=False)
    estimated_cost = db.Column(db.Float, default=0.0, nullable=False)
    
    # Quota management
    has_quota = db.Column(db.Boolean, default=False, nullable=False)
    daily_quota = db.Column(db.Integer, nullable=True)
    monthly_quota = db.Column(db.Integer, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    
    # Relationship
    user = db.relationship('User', back_populates='api_keys')
    
    def __init__(self, user_id, service, key_name, key, has_quota=False, 
                 daily_quota=None, monthly_quota=None):
        self.user_id = user_id
        self.service = service
        self.key_name = key_name
        self.key_prefix = key[:6]  # Store first 6 chars as prefix
        self.key_hash = generate_password_hash(key)  # Hash the key
        self.has_quota = has_quota
        self.daily_quota = daily_quota
        self.monthly_quota = monthly_quota
    
    def verify_key(self, key):
        """Verify if the provided key matches the stored hash."""
        return check_password_hash(self.key_hash, key)
    
    def update_usage(self, tokens_used=0, cost=0.0):
        """Update usage statistics for this key."""
        self.last_used = datetime.datetime.utcnow()
        self.calls_count += 1
        self.tokens_used += tokens_used
        self.estimated_cost += cost
    
    def check_quota(self):
        """Check if key is within quota limits."""
        if not self.has_quota:
            return True
            
        # Check daily quota
        if self.daily_quota:
            today = datetime.datetime.utcnow().date()
            today_start = datetime.datetime.combine(today, datetime.time.min)
            today_end = datetime.datetime.combine(today, datetime.time.max)
            
            # Count tokens used today
            from sqlalchemy import func
            from app.extensions import db
            
            daily_usage = db.session.query(func.sum(ApiKeyUsage.tokens_used)).filter(
                ApiKeyUsage.api_key_id == self.id,
                ApiKeyUsage.timestamp.between(today_start, today_end)
            ).scalar() or 0
            
            if daily_usage >= self.daily_quota:
                return False
        
        # Check monthly quota
        if self.monthly_quota:
            today = datetime.datetime.utcnow().date()
            month_start = datetime.datetime(today.year, today.month, 1)
            next_month = today.month + 1 if today.month < 12 else 1
            next_year = today.year if today.month < 12 else today.year + 1
            month_end = datetime.datetime(next_year, next_month, 1) - datetime.timedelta(seconds=1)
            
            # Count tokens used this month
            from sqlalchemy import func
            from app.extensions import db
            
            monthly_usage = db.session.query(func.sum(ApiKeyUsage.tokens_used)).filter(
                ApiKeyUsage.api_key_id == self.id,
                ApiKeyUsage.timestamp.between(month_start, month_end)
            ).scalar() or 0
            
            if monthly_usage >= self.monthly_quota:
                return False
                
        return True
    
    def get_masked_key(self):
        """Return a masked version of the key (for display)."""
        return f"{self.key_prefix}{'*' * 20}"
    
    def to_dict(self):
        """Convert API key to dictionary (for API responses)."""
        return {
            'id': self.id,
            'service': self.service,
            'key_name': self.key_name,
            'key_prefix': self.key_prefix,
            'is_active': self.is_active,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'calls_count': self.calls_count,
            'tokens_used': self.tokens_used,
            'estimated_cost': self.estimated_cost,
            'has_quota': self.has_quota,
            'daily_quota': self.daily_quota,
            'monthly_quota': self.monthly_quota,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<ApiKey {self.key_name} ({self.service})>'


class ApiKeyUsage(db.Model):
    """Model for tracking detailed API key usage."""
    
    __tablename__ = 'api_key_usage'
    
    id = db.Column(db.Integer, primary_key=True)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_keys.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    endpoint = db.Column(db.String(100), nullable=False)  # API endpoint used
    tokens_used = db.Column(db.Integer, default=0, nullable=False)  # Tokens consumed
    cost = db.Column(db.Float, default=0.0, nullable=False)  # Cost of this request
    
    # Optional contextual data
    task_type = db.Column(db.String(50), nullable=True)  # e.g., 'paper_processing', 'code_generation'
    user_request_id = db.Column(db.String(36), nullable=True)  # For tracking user requests
    
    def __init__(self, api_key_id, endpoint, tokens_used=0, cost=0.0, 
                 task_type=None, user_request_id=None):
        self.api_key_id = api_key_id
        self.endpoint = endpoint
        self.tokens_used = tokens_used
        self.cost = cost
        self.task_type = task_type
        self.user_request_id = user_request_id
    
    def to_dict(self):
        """Convert usage record to dictionary."""
        return {
            'id': self.id,
            'api_key_id': self.api_key_id,
            'timestamp': self.timestamp.isoformat(),
            'endpoint': self.endpoint,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'task_type': self.task_type,
            'user_request_id': self.user_request_id
        }
    
    def __repr__(self):
        return f'<ApiKeyUsage {self.endpoint} ({self.tokens_used} tokens)>'