from datetime import datetime, timedelta
import functools
import jwt
from flask import request, jsonify, current_app, make_response
from werkzeug.security import check_password_hash

from app.extensions import db
from app.models.auth import User, ApiKey, UserStatus
from app.utils import PasswordValidator
from app.services import EmailService
from app.api import api_bp


def token_required(f):
    """Decorator for requiring a valid JWT token for API endpoints."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # Check if token is in cookies
        if not token:
            token = request.cookies.get('token')
            
        if not token:
            return jsonify({
                'status': 'error',
                'message': 'Authentication token is missing',
                'code': 'token_missing'
            }), 401
            
        try:
            # Decode token
            data = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            
            # Get user from token data
            current_user = User.query.get(data['user_id'])
            
            if not current_user:
                return jsonify({
                    'status': 'error',
                    'message': 'User not found',
                    'code': 'user_not_found'
                }), 401
                
            # Check if user is active
            if current_user.status != UserStatus.ACTIVE:
                return jsonify({
                    'status': 'error',
                    'message': 'Account is not active',
                    'code': 'account_inactive'
                }), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({
                'status': 'error',
                'message': 'Token has expired',
                'code': 'token_expired'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid token',
                'code': 'token_invalid'
            }), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated


def api_key_required(f):
    """Decorator for requiring a valid API key for API endpoints."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = None
        
        # Check if API key is in X-API-Key header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'API key is missing',
                'code': 'api_key_missing'
            }), 401
            
        # Find the API key in the database
        api_key_obj = None
        for key in ApiKey.query.filter_by(is_active=True).all():
            if check_password_hash(key.key_hash, api_key):
                api_key_obj = key
                break
                
        if not api_key_obj:
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key',
                'code': 'api_key_invalid'
            }), 401
            
        # Check if the API key is within quota limits
        if not api_key_obj.check_quota():
            return jsonify({
                'status': 'error',
                'message': 'API key quota exceeded',
                'code': 'quota_exceeded'
            }), 429
            
        # Get the user associated with this API key
        current_user = User.query.get(api_key_obj.user_id)
        
        if not current_user:
            return jsonify({
                'status': 'error',
                'message': 'User not found',
                'code': 'user_not_found'
            }), 401
            
        # Check if user is active
        if current_user.status != UserStatus.ACTIVE:
            return jsonify({
                'status': 'error',
                'message': 'Account is not active',
                'code': 'account_inactive'
            }), 401
            
        # Update API key usage
        api_key_obj.update_usage()
        current_user.increment_api_usage()
        db.session.commit()
            
        return f(current_user, api_key_obj, *args, **kwargs)
    
    return decorated


@api_bp.route('/auth/login', methods=['POST'])
def login():
    """API endpoint for user login."""
    data = request.get_json()
    
    # Validate required fields
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({
            'status': 'error',
            'message': 'Email and password are required',
            'code': 'missing_fields'
        }), 400
        
    user = User.query.filter_by(email=data.get('email').lower()).first()
    
    # Check if user exists and password is correct
    if not user or not user.check_password(data.get('password')):
        # Record failed login attempt if user exists
        if user:
            user.increment_failed_login()
            db.session.commit()
            
        return jsonify({
            'status': 'error',
            'message': 'Invalid email or password',
            'code': 'invalid_credentials'
        }), 401
        
    # Check if account is locked
    if user.is_account_locked():
        return jsonify({
            'status': 'error',
            'message': 'Account is temporarily locked due to too many failed attempts',
            'code': 'account_locked'
        }), 403
        
    # Check if account is active
    if user.status != UserStatus.ACTIVE:
        if user.status == UserStatus.PENDING:
            return jsonify({
                'status': 'error',
                'message': 'Account is not yet activated. Please check your email for verification link',
                'code': 'account_pending'
            }), 403
        else:
            return jsonify({
                'status': 'error',
                'message': 'Account is not active',
                'code': 'account_inactive'
            }), 403
    
    # Update login info
    user.update_login_info(request.remote_addr)
    db.session.commit()
    
    # Generate JWT token
    token_expiry = datetime.utcnow() + timedelta(days=1)  # Token valid for 1 day
    token = jwt.encode(
        {
            'user_id': user.id,
            'exp': token_expiry
        },
        current_app.config['SECRET_KEY'],
        algorithm='HS256'
    )
    
    # Create response
    response = make_response(jsonify({
        'status': 'success',
        'message': 'Login successful',
        'data': {
            'user': user.to_dict(),
            'token': token,
            'expires_at': token_expiry.isoformat()
        }
    }))
    
    # Set token cookie if requested
    if data.get('remember', False):
        response.set_cookie(
            'token',
            token,
            httponly=True,
            secure=current_app.config.get('SESSION_COOKIE_SECURE', False),
            samesite='Lax',
            max_age=86400  # 1 day in seconds
        )
    
    return response


@api_bp.route('/auth/register', methods=['POST'])
def register():
    """API endpoint for user registration."""
    data = request.get_json()
    
    # Validate required fields
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({
            'status': 'error',
            'message': 'Username, email, and password are required',
            'code': 'missing_fields'
        }), 400
        
    # Check if username or email already exists
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({
            'status': 'error',
            'message': 'Username already in use',
            'code': 'username_exists'
        }), 409
        
    if User.query.filter_by(email=data.get('email').lower()).first():
        return jsonify({
            'status': 'error',
            'message': 'Email already registered',
            'code': 'email_exists'
        }), 409
        
    # Validate password strength
    is_valid, password_errors = PasswordValidator.validate(data.get('password'))
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Password does not meet security requirements',
            'details': password_errors,
            'code': 'password_weak'
        }), 400
        
    # Check if password is a common one
    if PasswordValidator.is_common_password(data.get('password')):
        return jsonify({
            'status': 'error',
            'message': 'This password is too common and easily guessable',
            'code': 'password_common'
        }), 400
        
    # Create new user
    user = User(
        username=data.get('username'),
        email=data.get('email'),
        password=data.get('password')
    )
    
    db.session.add(user)
    db.session.commit()
    
    # Send verification email
    EmailService.send_verification_email(user)
    
    return jsonify({
        'status': 'success',
        'message': 'Registration successful! Please check your email to verify your account.',
        'data': {'user_id': user.id}
    }), 201


@api_bp.route('/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """API endpoint for user logout."""
    # Create response
    response = make_response(jsonify({
        'status': 'success',
        'message': 'Logout successful'
    }))
    
    # Clear token cookie
    response.set_cookie('token', '', expires=0)
    
    return response


@api_bp.route('/auth/verify/<token>', methods=['GET'])
def verify_email(token):
    """API endpoint for email verification."""
    # Find user with this verification token
    user = User.query.filter_by(verification_token=token).first()
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'Invalid verification link',
            'code': 'invalid_token'
        }), 400
    
    # Check if token is expired
    if user.verification_token_expiry < datetime.utcnow():
        return jsonify({
            'status': 'error',
            'message': 'Verification link has expired',
            'code': 'token_expired'
        }), 400
    
    # Verify the email
    user.verify_email(token)
    db.session.commit()
    
    # Send welcome email
    EmailService.send_welcome_email(user)
    
    return jsonify({
        'status': 'success',
        'message': 'Email verified successfully'
    })


@api_bp.route('/auth/password/reset', methods=['POST'])
def password_reset_request():
    """API endpoint for password reset request."""
    data = request.get_json()
    
    # Validate required fields
    if not data or not data.get('email'):
        return jsonify({
            'status': 'error',
            'message': 'Email is required',
            'code': 'missing_fields'
        }), 400
        
    # Find user by email
    user = User.query.filter_by(email=data.get('email').lower()).first()
    
    # Generate reset token
    if user:
        # Set new token and expiry (24 hours)
        user.verification_token = user.generate_verification_token()
        user.verification_token_expiry = datetime.utcnow() + timedelta(hours=24)
        db.session.commit()
        
        # Send password reset email
        EmailService.send_password_reset_email(user)
    
    # Always show success, even if email not found (security best practice)
    return jsonify({
        'status': 'success',
        'message': 'If your email exists in our system, you will receive a password reset link shortly'
    })


@api_bp.route('/auth/password/reset/<token>', methods=['POST'])
def password_reset(token):
    """API endpoint for password reset with token."""
    data = request.get_json()
    
    # Validate required fields
    if not data or not data.get('password'):
        return jsonify({
            'status': 'error',
            'message': 'New password is required',
            'code': 'missing_fields'
        }), 400
        
    # Find user with this reset token
    user = User.query.filter_by(verification_token=token).first()
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'Invalid reset link',
            'code': 'invalid_token'
        }), 400
    
    # Check if token is expired
    if user.verification_token_expiry < datetime.utcnow():
        return jsonify({
            'status': 'error',
            'message': 'Reset link has expired',
            'code': 'token_expired'
        }), 400
        
    # Validate password strength
    is_valid, password_errors = PasswordValidator.validate(data.get('password'))
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Password does not meet security requirements',
            'details': password_errors,
            'code': 'password_weak'
        }), 400
        
    # Check if password is a common one
    if PasswordValidator.is_common_password(data.get('password')):
        return jsonify({
            'status': 'error',
            'message': 'This password is too common and easily guessable',
            'code': 'password_common'
        }), 400
    
    # Update password
    user.set_password(data.get('password'))
    user.verification_token = None
    user.verification_token_expiry = None
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Password has been reset successfully'
    })


@api_bp.route('/auth/user', methods=['GET'])
@token_required
def get_user(current_user):
    """API endpoint for getting current user info."""
    return jsonify({
        'status': 'success',
        'data': {'user': current_user.to_dict()}
    })


@api_bp.route('/auth/password/change', methods=['POST'])
@token_required
def change_password(current_user):
    """API endpoint for changing user password."""
    data = request.get_json()
    
    # Validate required fields
    if not data or not data.get('current_password') or not data.get('new_password'):
        return jsonify({
            'status': 'error',
            'message': 'Current password and new password are required',
            'code': 'missing_fields'
        }), 400
        
    # Verify current password
    if not current_user.check_password(data.get('current_password')):
        return jsonify({
            'status': 'error',
            'message': 'Current password is incorrect',
            'code': 'invalid_password'
        }), 401
        
    # Validate password strength
    is_valid, password_errors = PasswordValidator.validate(data.get('new_password'))
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Password does not meet security requirements',
            'details': password_errors,
            'code': 'password_weak'
        }), 400
        
    # Check if password is a common one
    if PasswordValidator.is_common_password(data.get('new_password')):
        return jsonify({
            'status': 'error',
            'message': 'This password is too common and easily guessable',
            'code': 'password_common'
        }), 400
    
    # Update password
    current_user.set_password(data.get('new_password'))
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Password updated successfully'
    })