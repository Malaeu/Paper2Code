"""API routes for usage tracking and statistics."""

import datetime
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user

from app.services.usage_service import UsageService
from app.utils.api import api_error, api_result

usage_api_bp = Blueprint('usage_api', __name__)


@usage_api_bp.route('/stats', methods=['GET'])
@login_required
def get_usage_stats():
    """Get usage statistics for the current user."""
    try:
        stats = UsageService.get_user_usage_stats(current_user.id)
        return api_result(stats)
    except Exception as e:
        current_app.logger.error(f"Error getting usage stats: {str(e)}")
        return api_error("Failed to retrieve usage statistics")


@usage_api_bp.route('/history', methods=['GET'])
@login_required
def get_usage_history():
    """Get detailed usage history for the current user."""
    try:
        # Parse query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        service = request.args.get('service')
        limit = request.args.get('limit', 100, type=int)
        
        # Convert date strings to datetime objects if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                return api_error("Invalid start_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
                # Set time to end of day if just a date is provided
                if end_datetime.hour == 0 and end_datetime.minute == 0 and end_datetime.second == 0:
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            except ValueError:
                return api_error("Invalid end_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        # Get history
        history = UsageService.get_detailed_usage_history(
            current_user.id,
            start_date=start_datetime,
            end_date=end_datetime,
            service=service,
            limit=limit
        )
        
        return api_result({"history": history})
    except Exception as e:
        current_app.logger.error(f"Error getting usage history: {str(e)}")
        return api_error("Failed to retrieve usage history")


@usage_api_bp.route('/daily-summary', methods=['GET'])
@login_required
def get_daily_summary():
    """Get usage summary grouped by day."""
    try:
        # Parse query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        service = request.args.get('service')
        
        # Convert date strings to datetime objects if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                return api_error("Invalid start_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
                # Set time to end of day if just a date is provided
                if end_datetime.hour == 0 and end_datetime.minute == 0 and end_datetime.second == 0:
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            except ValueError:
                return api_error("Invalid end_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        # Get summary
        summary = UsageService.get_usage_summary_by_day(
            current_user.id,
            start_date=start_datetime,
            end_date=end_datetime,
            service=service
        )
        
        return api_result({"daily_summary": summary})
    except Exception as e:
        current_app.logger.error(f"Error getting daily usage summary: {str(e)}")
        return api_error("Failed to retrieve daily usage summary")


@usage_api_bp.route('/model-breakdown', methods=['GET'])
@login_required
def get_model_breakdown():
    """Get cost breakdown by model."""
    try:
        # Parse query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert date strings to datetime objects if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                return api_error("Invalid start_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
                # Set time to end of day if just a date is provided
                if end_datetime.hour == 0 and end_datetime.minute == 0 and end_datetime.second == 0:
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            except ValueError:
                return api_error("Invalid end_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        # Get breakdown
        breakdown = UsageService.get_cost_breakdown_by_model(
            current_user.id,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        return api_result(breakdown)
    except Exception as e:
        current_app.logger.error(f"Error getting model cost breakdown: {str(e)}")
        return api_error("Failed to retrieve model cost breakdown")


@usage_api_bp.route('/estimate-cost', methods=['POST'])
@login_required
def estimate_cost():
    """Estimate cost for a model usage."""
    try:
        data = request.get_json()
        if not data:
            return api_error("Missing request body")
        
        model_id = data.get('model_id')
        input_tokens = data.get('input_tokens', 0)
        output_tokens = data.get('output_tokens', 0)
        
        if not model_id:
            return api_error("Missing model_id parameter")
        
        try:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
        except (ValueError, TypeError):
            return api_error("Input and output tokens must be integers")
        
        # Get cost estimate
        estimate = UsageService.estimate_cost(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        if 'error' in estimate and not any(
            key in estimate for key in ['model_id', 'display_name']
        ):
            return api_error(estimate['error'])
        
        return api_result(estimate)
    except Exception as e:
        current_app.logger.error(f"Error estimating cost: {str(e)}")
        return api_error("Failed to estimate cost")


@usage_api_bp.route('/track', methods=['POST'])
@login_required
def track_usage():
    """Track API usage manually (primarily for testing)."""
    try:
        data = request.get_json()
        if not data:
            return api_error("Missing request body")
        
        model_id = data.get('model_id')
        input_tokens = data.get('input_tokens', 0)
        output_tokens = data.get('output_tokens', 0)
        endpoint = data.get('endpoint', 'manual')
        task_type = data.get('task_type')
        request_id = data.get('request_id')
        
        if not model_id:
            return api_error("Missing model_id parameter")
        
        try:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
        except (ValueError, TypeError):
            return api_error("Input and output tokens must be integers")
        
        # Track usage
        success, cost = UsageService.track_api_usage(
            user_id=current_user.id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            endpoint=endpoint,
            task_type=task_type,
            request_id=request_id
        )
        
        if not success:
            return api_error("Failed to track API usage")
        
        return api_result({
            "success": True,
            "cost": cost,
            "total_tokens": input_tokens + output_tokens
        })
    except Exception as e:
        current_app.logger.error(f"Error tracking API usage: {str(e)}")
        return api_error("Failed to track API usage")