import os
import threading
from flask import current_app, render_template
from flask_mail import Message
from app.extensions import mail

class EmailService:
    """Service for sending various types of emails."""
    
    @staticmethod
    def send_email_async(app, msg):
        """Send email asynchronously."""
        with app.app_context():
            mail.send(msg)
    
    @staticmethod
    def send_email(subject, recipients, html_body, text_body=None, sender=None, attachments=None):
        """
        Send an email.
        
        Args:
            subject (str): Email subject
            recipients (list): List of recipient email addresses
            html_body (str): HTML content of the email
            text_body (str, optional): Plain text content of the email. Defaults to None.
            sender (str, optional): Sender email address. Defaults to the configured default sender.
            attachments (list, optional): List of attachment tuples (filename, content_type, data). Defaults to None.
        """
        app = current_app._get_current_object()
        
        # Use configured default sender if not provided
        if sender is None:
            sender = app.config['MAIL_DEFAULT_SENDER']
        
        msg = Message(subject, sender=sender, recipients=recipients)
        msg.html = html_body
        
        if text_body:
            msg.body = text_body
        
        if attachments:
            for attachment in attachments:
                msg.attach(*attachment)
        
        # Send email in a separate thread to avoid blocking
        threading.Thread(target=EmailService.send_email_async,
                          args=(app, msg)).start()
    
    @staticmethod
    def send_verification_email(user):
        """
        Send verification email to a newly registered user.
        
        Args:
            user: User object containing email and verification token
        """
        verification_url = f"{current_app.config.get('SITE_URL', 'http://localhost:5000')}/auth/verify/{user.verification_token}"
        
        subject = "Paper2Code - Verify Your Email Address"
        recipients = [user.email]
        
        html = render_template('email/verify_email.html',
                              user=user,
                              verification_url=verification_url)
        
        text = render_template('email/verify_email.txt',
                              user=user,
                              verification_url=verification_url)
        
        EmailService.send_email(subject, recipients, html, text)
    
    @staticmethod
    def send_password_reset_email(user):
        """
        Send password reset email to a user.
        
        Args:
            user: User object containing email and verification token
        """
        reset_url = f"{current_app.config.get('SITE_URL', 'http://localhost:5000')}/auth/password/reset/{user.verification_token}"
        
        subject = "Paper2Code - Password Reset Request"
        recipients = [user.email]
        
        html = render_template('email/reset_password.html',
                              user=user,
                              reset_url=reset_url)
        
        text = render_template('email/reset_password.txt',
                              user=user,
                              reset_url=reset_url)
        
        EmailService.send_email(subject, recipients, html, text)
    
    @staticmethod
    def send_welcome_email(user):
        """
        Send welcome email to a newly verified user.
        
        Args:
            user: User object containing username and email
        """
        subject = "Welcome to Paper2Code!"
        recipients = [user.email]
        
        html = render_template('email/welcome.html', user=user)
        text = render_template('email/welcome.txt', user=user)
        
        EmailService.send_email(subject, recipients, html, text)
        
    @staticmethod
    def send_email_change_verification(user, new_email):
        """
        Send verification email for email address change.
        
        Args:
            user: User object containing verification token
            new_email: The new email address to verify
        """
        verification_url = f"{current_app.config.get('SITE_URL', 'http://localhost:5000')}/auth/email/verify/{user.verification_token}"
        
        subject = "Paper2Code - Verify Your New Email Address"
        recipients = [new_email]
        
        html = render_template('email/verify_email_change.html',
                              user=user,
                              new_email=new_email,
                              verification_url=verification_url)
        
        text = render_template('email/verify_email_change.txt',
                              user=user,
                              new_email=new_email,
                              verification_url=verification_url)
        
        EmailService.send_email(subject, recipients, html, text)