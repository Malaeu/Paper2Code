from flask import (
    render_template, redirect, url_for, flash, request, 
    current_app, session, abort
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.urls import url_parse
from datetime import datetime, timedelta
import json

from app.extensions import db, login_manager
from app.models.auth import User, UserStatus
from app.models.auth.activity_log import ActivityType
from app.services.email_service import EmailService
from app.services.activity_service import ActivityService
from app.utils import PasswordValidator
from . import auth_bp

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    """Load a user from the database using the user_id."""
    return User.query.get(int(user_id))

# Configure the login manager
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
login_manager.refresh_view = 'auth.login'
login_manager.needs_refresh_message = 'Please reauthenticate to access this page.'
login_manager.needs_refresh_message_category = 'info'

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    # If user is already authenticated, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        # Get form data
        email = request.form.get('email')
        password = request.form.get('password')
        remember_me = request.form.get('remember_me', 'off') == 'on'
        
        # Validate form data
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('auth/login.html')
        
        # Look up user by email
        user = User.query.filter_by(email=email.lower()).first()
        
        # Check if user exists and password is correct
        if user is None or not user.check_password(password):
            # Record failed login attempt if user exists
            if user:
                user.increment_failed_login()
                db.session.commit()
            
            flash('Invalid email or password.', 'error')
            return render_template('auth/login.html')
        
        # Check if account is locked
        if user.is_account_locked():
            unlock_time = user.locked_until.strftime('%H:%M:%S')
            flash(f'Account temporarily locked due to too many failed attempts. '
                  f'Try again after {unlock_time}.', 'error')
            return render_template('auth/login.html')
        
        # Check if account is active
        if user.status != UserStatus.ACTIVE:
            if user.status == UserStatus.PENDING:
                flash('Your account is not yet activated. Please check your email for '
                      'the verification link.', 'warning')
            else:
                flash('Your account is not active. Please contact support.', 'error')
            return render_template('auth/login.html')
        
        # Update login info
        user.update_login_info(request.remote_addr)
        db.session.commit()
        
        # Log in the user
        login_user(user, remember=remember_me)
        
        # Log the login activity
        ActivityService.log_activity(
            user_id=user.id,
            activity_type=ActivityType.LOGIN,
            description=f"User logged in from {request.remote_addr}"
        )
        
        # Redirect to the page the user was trying to access
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.index')
            
        return redirect(next_page)
    
    # GET request - show login form
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    user_id = current_user.id
    
    # Log logout activity before actually logging out
    ActivityService.log_activity(
        user_id=user_id,
        activity_type=ActivityType.LOGOUT,
        description="User logged out"
    )
    
    logout_user()
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration."""
    # If user is already authenticated, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('auth/register.html')
        
        if password != password_confirm:
            flash('Passwords do not match.', 'error')
            return render_template('auth/register.html')
            
        # Validate password strength
        is_valid, password_errors = PasswordValidator.validate(password)
        if not is_valid:
            for error in password_errors:
                flash(error, 'error')
            return render_template('auth/register.html')
            
        # Check if password is a common one
        if PasswordValidator.is_common_password(password):
            flash('This password is too common and easily guessable. Please choose a stronger password.', 'error')
            return render_template('auth/register.html')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already in use.', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email.lower()).first():
            flash('Email already registered.', 'error')
            return render_template('auth/register.html')
        
        # Create new user
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        
        # Send verification email
        EmailService.send_verification_email(user)
        
        flash('Registration successful! Please check your email to verify your account.', 'success')
        return redirect(url_for('auth.login'))
    
    # GET request - show registration form
    return render_template('auth/register.html')

@auth_bp.route('/verify/<token>')
def verify_email(token):
    """Handle email verification."""
    # Find user with this verification token
    user = User.query.filter_by(verification_token=token).first()
    
    if not user:
        flash('Invalid verification link.', 'error')
        return redirect(url_for('auth.login'))
    
    # Check if token is expired
    if user.verification_token_expiry < datetime.utcnow():
        flash('Verification link has expired. Please request a new one.', 'error')
        return redirect(url_for('auth.login'))
    
    # Verify the email
    user.verify_email(token)
    db.session.commit()
    
    # Send welcome email
    EmailService.send_welcome_email(user)
    
    flash('Email verified! You can now log in.', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/password/reset', methods=['GET', 'POST'])
def password_reset_request():
    """Handle password reset request."""
    # If user is already authenticated, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Email is required.', 'error')
            return render_template('auth/password_reset_request.html')
        
        # Find user by email
        user = User.query.filter_by(email=email.lower()).first()
        
        # Generate reset token
        if user:
            # Set new token and expiry (24 hours)
            user.verification_token = user.generate_verification_token()
            user.verification_token_expiry = datetime.utcnow() + timedelta(hours=24)
            db.session.commit()
            
            # Send password reset email
            EmailService.send_password_reset_email(user)
        
        # Always show success, even if email not found (security best practice)
        flash('If your email exists in our system, you will receive a password reset link shortly.', 'info')
        return redirect(url_for('auth.login'))
    
    # GET request - show password reset request form
    return render_template('auth/password_reset_request.html')

@auth_bp.route('/password/reset/<token>', methods=['GET', 'POST'])
def password_reset(token):
    """Handle password reset with token."""
    # If user is already authenticated, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    # Find user with this reset token
    user = User.query.filter_by(verification_token=token).first()
    
    if not user:
        flash('Invalid reset link.', 'error')
        return redirect(url_for('auth.login'))
    
    # Check if token is expired
    if user.verification_token_expiry < datetime.utcnow():
        flash('Reset link has expired. Please request a new one.', 'error')
        return redirect(url_for('auth.password_reset_request'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        if not password or not password_confirm:
            flash('Both password fields are required.', 'error')
            return render_template('auth/password_reset.html', token=token)
        
        if password != password_confirm:
            flash('Passwords do not match.', 'error')
            return render_template('auth/password_reset.html', token=token)
            
        # Validate password strength
        is_valid, password_errors = PasswordValidator.validate(password)
        if not is_valid:
            for error in password_errors:
                flash(error, 'error')
            return render_template('auth/password_reset.html', token=token)
            
        # Check if password is a common one
        if PasswordValidator.is_common_password(password):
            flash('This password is too common and easily guessable. Please choose a stronger password.', 'error')
            return render_template('auth/password_reset.html', token=token)
        
        # Update password
        user.set_password(password)
        user.verification_token = None
        user.verification_token_expiry = None
        user.failed_login_attempts = 0
        user.locked_until = None
        db.session.commit()
        
        flash('Password has been reset. You can now log in with your new password.', 'success')
        return redirect(url_for('auth.login'))
    
    # GET request - show password reset form
    return render_template('auth/password_reset.html', token=token)

@auth_bp.route('/profile')
@login_required
def profile():
    """Display user profile."""
    return render_template('auth/profile.html')

@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile."""
    if request.method == 'POST':
        username = request.form.get('username')
        
        # Check if username is already taken
        if username != current_user.username and User.query.filter_by(username=username).first():
            flash('Username already in use.', 'error')
            return render_template('auth/edit_profile.html')
        
        # Get old username for logging
        old_username = current_user.username
        
        # Update user profile
        current_user.username = username
        db.session.commit()
        
        # Log profile update activity
        ActivityService.log_activity(
            user_id=current_user.id,
            activity_type=ActivityType.PROFILE_UPDATE,
            description="Username changed",
            meta_data={
                "old_username": old_username,
                "new_username": username
            }
        )
        
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('auth.profile'))
    
    # GET request - show profile edit form
    return render_template('auth/edit_profile.html')

@auth_bp.route('/profile/password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password."""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form data
        if not current_password or not new_password or not confirm_password:
            flash('All fields are required.', 'error')
            return render_template('auth/change_password.html')
        
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'error')
            return render_template('auth/change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('auth/change_password.html')
            
        # Validate password strength
        is_valid, password_errors = PasswordValidator.validate(new_password)
        if not is_valid:
            for error in password_errors:
                flash(error, 'error')
            return render_template('auth/change_password.html')
            
        # Check if password is a common one
        if PasswordValidator.is_common_password(new_password):
            flash('This password is too common and easily guessable. Please choose a stronger password.', 'error')
            return render_template('auth/change_password.html')
        
        # Update password
        current_user.set_password(new_password)
        db.session.commit()
        
        # Log password change activity
        ActivityService.log_activity(
            user_id=current_user.id,
            activity_type=ActivityType.PASSWORD_CHANGE,
            description="Password changed"
        )
        
        flash('Password updated successfully.', 'success')
        return redirect(url_for('auth.profile'))
    
    # GET request - show change password form
    return render_template('auth/change_password.html')


@auth_bp.route('/profile/email/change', methods=['POST'])
@login_required
def change_email():
    """Handle email change request with verification."""
    new_email = request.form.get('new_email')
    password = request.form.get('password')
    
    # Validate form data
    if not new_email or not password:
        flash('All fields are required.', 'error')
        return redirect(url_for('auth.edit_profile'))
    
    # Verify password
    if not current_user.check_password(password):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('auth.edit_profile'))
    
    # Check if email already exists
    if User.query.filter_by(email=new_email.lower()).first():
        flash('Email already registered to another account.', 'error')
        return redirect(url_for('auth.edit_profile'))
    
    # Don't allow changing to the same email
    if new_email.lower() == current_user.email.lower():
        flash('New email must be different from current email.', 'error')
        return redirect(url_for('auth.edit_profile'))
    
    # Store the new email in a pending state
    current_user.pending_email = new_email.lower()
    current_user.verification_token = current_user.generate_verification_token()
    current_user.verification_token_expiry = datetime.utcnow() + timedelta(hours=24)
    db.session.commit()
    
    # Send verification email to the new address
    EmailService.send_email_change_verification(current_user, new_email)
    
    # Log email change request activity
    ActivityService.log_activity(
        user_id=current_user.id,
        activity_type=ActivityType.EMAIL_CHANGE,
        description="Email change requested",
        meta_data={
            "current_email": current_user.email,
            "new_email": new_email
        }
    )
    
    flash('A verification email has been sent to your new email address. '
          'Please check your inbox and click the verification link to complete the email change.', 
          'info')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/email/verify/<token>')
@login_required
def verify_email_change(token):
    """Handle email change verification."""
    # Verify the token matches the current user's token
    if not current_user.verification_token or current_user.verification_token != token:
        flash('Invalid verification link.', 'error')
        return redirect(url_for('auth.profile'))
    
    # Check if token is expired
    if current_user.verification_token_expiry < datetime.utcnow():
        flash('Verification link has expired. Please request a new email change.', 'error')
        return redirect(url_for('auth.edit_profile'))
    
    # Check if there's a pending email change
    if not current_user.pending_email:
        flash('No pending email change found.', 'error')
        return redirect(url_for('auth.profile'))
    
    # Verify the email change
    old_email = current_user.verify_email_change(token)
    if old_email:
        db.session.commit()
        
        # Send confirmation emails to both old and new addresses
        EmailService.send_email(
            subject='Paper2Code - Your Email Has Been Changed',
            recipients=[old_email],
            html_body=render_template('email/email_change_old.html', 
                                      user=current_user, 
                                      old_email=old_email),
            text_body=render_template('email/email_change_old.txt', 
                                      user=current_user, 
                                      old_email=old_email)
        )
        
        EmailService.send_email(
            subject='Paper2Code - Your Email Change is Complete',
            recipients=[current_user.email],
            html_body=render_template('email/email_change_confirmed.html', 
                                      user=current_user),
            text_body=render_template('email/email_change_confirmed.txt', 
                                      user=current_user)
        )
        
        # Log email verification activity
        ActivityService.log_activity(
            user_id=current_user.id,
            activity_type=ActivityType.EMAIL_VERIFY,
            description="Email address verified and changed",
            meta_data={
                "old_email": old_email,
                "new_email": current_user.email
            }
        )
        
        flash('Your email address has been successfully updated.', 'success')
    else:
        flash('Failed to verify email change. Please try again.', 'error')
    
    return redirect(url_for('auth.profile'))


@auth_bp.route('/profile/delete', methods=['POST'])
@login_required
def delete_account():
    """Handle account deletion."""
    password = request.form.get('password')
    delete_confirmation = request.form.get('delete_confirmation')
    
    # Validate form data
    if not password or not delete_confirmation:
        flash('All fields are required.', 'error')
        return redirect(url_for('auth.profile'))
    
    # Verify delete confirmation
    if delete_confirmation != 'DELETE':
        flash('Please type DELETE exactly to confirm account deletion.', 'error')
        return redirect(url_for('auth.profile'))
    
    # Verify password
    if not current_user.check_password(password):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('auth.profile'))
    
    try:
        # Get user data for confirmation email
        user_email = current_user.email
        user_username = current_user.username
        
        # Delete related data (API keys and usage will be deleted via cascade)
        # Note: For other related data, you would delete those relations here
        
        # Log account deletion activity before deleting the user
        ActivityService.log_activity(
            user_id=current_user.id,
            activity_type=ActivityType.PROFILE_UPDATE,
            description="Account deleted",
            meta_data={
                "username": current_user.username,
                "email": current_user.email
            }
        )
        
        # Set status to inactive first (soft delete) in case we need to recover
        current_user.status = UserStatus.INACTIVE
        db.session.commit()
        
        # Hard delete the user
        db.session.delete(current_user)
        db.session.commit()
        
        # Log out the user
        logout_user()
        
        # Send confirmation email
        EmailService.send_email(
            subject='Paper2Code - Account Deletion Confirmation',
            recipients=[user_email],
            html_body=render_template('email/account_deletion.html', username=user_username),
            text_body=render_template('email/account_deletion.txt', username=user_username)
        )
        
        flash('Your account has been permanently deleted. We are sorry to see you go.', 'info')
        return redirect(url_for('main.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error deleting account: {str(e)}")
        db.session.rollback()
        flash('An error occurred while trying to delete your account. Please try again.', 'error')
        return redirect(url_for('auth.profile'))


@auth_bp.route('/profile/activity')
@login_required
def activity():
    """Display user activity history."""
    # Get user's recent activities
    activities = ActivityService.get_user_activities(current_user.id, limit=50)
    
    return render_template('auth/activity.html', activities=activities)