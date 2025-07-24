"""Service for logging user activities."""

import json
from flask import request, current_app
from app.extensions import db
from app.models.auth.activity_log import UserActivityLog, ActivityType


class ActivityService:
    """Service for logging user activities."""
    
    @staticmethod
    def log_activity(user_id, activity_type, description=None, meta_data=None):
        """
        Log a user activity.
        
        Args:
            user_id (int): The ID of the user performing the activity
            activity_type (ActivityType): The type of activity
            description (str, optional): A description of the activity
            meta_data (dict, optional): Additional data about the activity
        
        Returns:
            UserActivityLog: The created activity log
        """
        try:
            # Get request information if available
            ip_address = request.remote_addr if request else None
            user_agent = request.headers.get('User-Agent') if request else None
            
            # Convert meta_data to JSON string if provided
            meta_data_json = json.dumps(meta_data) if meta_data else None
            
            # Create the activity log
            activity_log = UserActivityLog(
                user_id=user_id,
                activity_type=activity_type,
                ip_address=ip_address,
                user_agent=user_agent,
                description=description,
                meta_data=meta_data_json
            )
            
            db.session.add(activity_log)
            db.session.commit()
            
            return activity_log
        except Exception as e:
            current_app.logger.error(f"Error logging activity: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_user_activities(user_id, limit=50, activity_type=None):
        """
        Get recent activities for a user.
        
        Args:
            user_id (int): The ID of the user
            limit (int, optional): Maximum number of activities to return
            activity_type (ActivityType, optional): Filter by activity type
        
        Returns:
            list: List of UserActivityLog objects
        """
        query = UserActivityLog.query.filter_by(user_id=user_id)
        
        if activity_type:
            query = query.filter_by(activity_type=activity_type)
        
        return query.order_by(UserActivityLog.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_admin_activity_feed(limit=100, user_id=None):
        """
        Get recent activities for admin dashboard.
        
        Args:
            limit (int, optional): Maximum number of activities to return
            user_id (int, optional): Filter by user ID
        
        Returns:
            list: List of UserActivityLog objects
        """
        query = UserActivityLog.query
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        return query.order_by(UserActivityLog.created_at.desc()).limit(limit).all()