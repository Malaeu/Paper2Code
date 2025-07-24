"""Service for tracking and managing cost and usage data."""

import datetime
import json
from typing import Dict, List, Optional, Any, Tuple
from flask import current_app

from app.extensions import db
from app.models.auth import User
from app.models.auth.user import ApiKey, ApiKeyUsage
from app.models.config.model_config import ModelConfig, ModelCostInfo
from app.services.model_service import ModelService


class UsageService:
    """Service for tracking and managing usage and costs."""
    
    @staticmethod
    def track_api_usage(
        user_id: int,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        endpoint: str,
        task_type: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Track API usage for a user.
        
        Args:
            user_id: User ID
            model_id: Model identifier
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            endpoint: API endpoint used
            task_type: Type of task (e.g., 'paper_processing', 'code_generation')
            request_id: Unique identifier for this request
            
        Returns:
            Tuple of (success, cost)
        """
        try:
            # Get user
            user = User.query.get(user_id)
            if not user:
                current_app.logger.error(f"User with ID {user_id} not found")
                return False, 0.0
            
            # Get model config
            model_config = ModelConfig.query.filter_by(model_id=model_id).first()
            
            # Update model usage statistics
            cost = 0.0
            if model_config and model_config.cost_info:
                cost = model_config.cost_info.update_usage(input_tokens, output_tokens)
            
            # Update user's API call count
            user.increment_api_usage()
            
            # Look for relevant API key to track usage
            api_key = None
            if model_config:
                # Get API key for this provider
                api_key = ApiKey.query.filter_by(
                    user_id=user_id,
                    service=model_config.provider.value,
                    is_active=True
                ).first()
            
            # If API key found, update its usage
            if api_key:
                api_key.update_usage(tokens_used=input_tokens + output_tokens, cost=cost)
                
                # Create detailed usage record
                usage_record = ApiKeyUsage(
                    api_key_id=api_key.id,
                    endpoint=endpoint,
                    tokens_used=input_tokens + output_tokens,
                    cost=cost,
                    task_type=task_type,
                    user_request_id=request_id
                )
                db.session.add(usage_record)
            
            db.session.commit()
            return True, cost
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error tracking API usage: {str(e)}")
            return False, 0.0
    
    @staticmethod
    def get_user_usage_stats(user_id: int) -> Dict[str, Any]:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary containing usage statistics
        """
        try:
            # Get user
            user = User.query.get(user_id)
            if not user:
                return {
                    'total_api_calls': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'api_keys': [],
                    'models': []
                }
            
            # Get API keys
            api_keys = ApiKey.query.filter_by(user_id=user_id).all()
            
            # Calculate totals
            total_tokens = sum(key.tokens_used for key in api_keys)
            total_cost = sum(key.estimated_cost for key in api_keys)
            
            # Get usage by model (from ModelCostInfo)
            model_usage = []
            models = ModelConfig.query.filter(ModelConfig.cost_info_id.isnot(None)).all()
            
            for model in models:
                if model.cost_info and model.cost_info.total_tokens_used > 0:
                    model_usage.append({
                        'model_id': model.model_id,
                        'display_name': model.display_name,
                        'provider': model.provider.value,
                        'provider_display': model.provider.display_name,
                        'tokens_used': model.cost_info.total_tokens_used,
                        'cost': model.cost_info.total_cost
                    })
            
            # Get usage by API key
            api_key_usage = []
            for key in api_keys:
                api_key_usage.append({
                    'id': key.id,
                    'service': key.service,
                    'key_name': key.key_name,
                    'key_prefix': key.key_prefix,
                    'calls_count': key.calls_count,
                    'tokens_used': key.tokens_used,
                    'estimated_cost': key.estimated_cost,
                    'last_used': key.last_used.isoformat() if key.last_used else None
                })
            
            # Return statistics
            return {
                'total_api_calls': user.total_api_calls,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'api_keys': api_key_usage,
                'models': model_usage
            }
            
        except Exception as e:
            current_app.logger.error(f"Error getting user usage stats: {str(e)}")
            return {
                'total_api_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_keys': [],
                'models': []
            }
    
    @staticmethod
    def get_detailed_usage_history(
        user_id: int,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        service: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get detailed usage history for a user.
        
        Args:
            user_id: User ID
            start_date: Start date for filtering
            end_date: End date for filtering
            service: Service/provider name for filtering
            limit: Maximum number of records to return
            
        Returns:
            List of usage records
        """
        try:
            # Get API keys for this user
            api_keys = ApiKey.query.filter_by(user_id=user_id).all()
            
            if not api_keys:
                return []
                
            # Build query for ApiKeyUsage
            query = ApiKeyUsage.query.filter(ApiKeyUsage.api_key_id.in_([key.id for key in api_keys]))
            
            # Apply filters
            if start_date:
                query = query.filter(ApiKeyUsage.timestamp >= start_date)
            
            if end_date:
                query = query.filter(ApiKeyUsage.timestamp <= end_date)
            
            if service:
                # Get API keys for this service
                service_key_ids = [key.id for key in api_keys if key.service == service]
                if service_key_ids:
                    query = query.filter(ApiKeyUsage.api_key_id.in_(service_key_ids))
                else:
                    return []
            
            # Get results
            usage_records = query.order_by(ApiKeyUsage.timestamp.desc()).limit(limit).all()
            
            # Build response
            key_map = {key.id: key for key in api_keys}
            results = []
            
            for record in usage_records:
                api_key = key_map.get(record.api_key_id)
                if api_key:
                    results.append({
                        'id': record.id,
                        'timestamp': record.timestamp.isoformat(),
                        'service': api_key.service,
                        'key_name': api_key.key_name,
                        'endpoint': record.endpoint,
                        'tokens_used': record.tokens_used,
                        'cost': record.cost,
                        'task_type': record.task_type
                    })
            
            return results
            
        except Exception as e:
            current_app.logger.error(f"Error getting usage history: {str(e)}")
            return []
    
    @staticmethod
    def get_usage_summary_by_day(
        user_id: int,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        service: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get usage summary grouped by day.
        
        Args:
            user_id: User ID
            start_date: Start date for filtering
            end_date: End date for filtering
            service: Service/provider name for filtering
            
        Returns:
            List of daily usage summaries
        """
        try:
            # Default date range if not provided (last 30 days)
            if not end_date:
                end_date = datetime.datetime.utcnow()
            if not start_date:
                start_date = end_date - datetime.timedelta(days=30)
            
            # Get API keys for this user
            api_keys = ApiKey.query.filter_by(user_id=user_id).all()
            
            if not api_keys:
                return []
                
            # Filter API keys by service if specified
            if service:
                api_keys = [key for key in api_keys if key.service == service]
                if not api_keys:
                    return []
            
            api_key_ids = [key.id for key in api_keys]
            
            # Build query using SQLAlchemy Core for aggregation
            from sqlalchemy import func, cast, Date
            from app.models.auth.user import ApiKeyUsage
            
            # Group by day and aggregate
            query = db.session.query(
                cast(ApiKeyUsage.timestamp, Date).label('day'),
                func.sum(ApiKeyUsage.tokens_used).label('tokens'),
                func.sum(ApiKeyUsage.cost).label('cost'),
                func.count(ApiKeyUsage.id).label('count')
            ).filter(
                ApiKeyUsage.api_key_id.in_(api_key_ids),
                ApiKeyUsage.timestamp.between(start_date, end_date)
            ).group_by(
                cast(ApiKeyUsage.timestamp, Date)
            ).order_by(
                cast(ApiKeyUsage.timestamp, Date)
            )
            
            # Execute query
            results = query.all()
            
            # Format results
            summary = []
            for day, tokens, cost, count in results:
                summary.append({
                    'day': day.isoformat(),
                    'tokens_used': int(tokens),
                    'cost': float(cost),
                    'request_count': int(count)
                })
            
            return summary
            
        except Exception as e:
            current_app.logger.error(f"Error getting usage summary: {str(e)}")
            return []
    
    @staticmethod
    def get_cost_breakdown_by_model(
        user_id: int,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get cost breakdown by model.
        
        Args:
            user_id: User ID
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with cost breakdown by model and provider
        """
        try:
            # This is a bit tricky since we don't directly link ApiKeyUsage to specific models
            # For now, we'll estimate by service and return all models for each service
            
            # Get user's usage stats
            usage_stats = UsageService.get_user_usage_stats(user_id)
            
            # Group models by provider
            provider_models = {}
            for model in usage_stats['models']:
                provider = model['provider']
                if provider not in provider_models:
                    provider_models[provider] = []
                provider_models[provider].append(model)
            
            # Group API key usage by service
            service_usage = {}
            for key in usage_stats['api_keys']:
                service = key['service']
                if service not in service_usage:
                    service_usage[service] = {
                        'tokens_used': 0,
                        'estimated_cost': 0.0
                    }
                service_usage[service]['tokens_used'] += key['tokens_used']
                service_usage[service]['estimated_cost'] += key['estimated_cost']
            
            # Combine the data
            result = {
                'by_provider': [],
                'by_model': usage_stats['models']
            }
            
            for provider, models in provider_models.items():
                provider_data = {
                    'provider': provider,
                    'provider_display': models[0]['provider_display'] if models else provider.title(),
                    'tokens_used': service_usage.get(provider, {}).get('tokens_used', 0),
                    'cost': service_usage.get(provider, {}).get('estimated_cost', 0.0),
                    'model_count': len(models)
                }
                result['by_provider'].append(provider_data)
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"Error getting cost breakdown: {str(e)}")
            return {'by_provider': [], 'by_model': []}
    
    @staticmethod
    def estimate_cost(
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, Any]:
        """
        Estimate the cost of using a model.
        
        Args:
            model_id: Model identifier or ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost estimates
        """
        try:
            # Try to find the model by ID first
            try:
                model_id_int = int(model_id)
                model_dict = ModelService.get_model_by_id(model_id_int)
            except (ValueError, TypeError):
                # Not an integer ID, try to find by model identifier
                model_dict = ModelService.get_model_by_identifier(model_id)
            
            if not model_dict:
                return {
                    'error': f"Model '{model_id}' not found",
                    'input_cost': 0.0,
                    'output_cost': 0.0,
                    'total_cost': 0.0
                }
            
            # Get cost info
            input_cost_per_1k = model_dict.get('input_cost_per_1k_tokens', 0.0)
            output_cost_per_1k = model_dict.get('output_cost_per_1k_tokens', 0.0)
            
            # Calculate costs
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            return {
                'model_id': model_dict.get('model_id'),
                'display_name': model_dict.get('display_name'),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
            
        except Exception as e:
            current_app.logger.error(f"Error estimating cost: {str(e)}")
            return {
                'error': str(e),
                'input_cost': 0.0,
                'output_cost': 0.0,
                'total_cost': 0.0
            }