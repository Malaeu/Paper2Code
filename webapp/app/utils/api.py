"""Utility functions for API responses."""

from flask import jsonify
from typing import Any, Dict, List, Union


def api_result(data: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """
    Create a standardized successful API response.
    
    Args:
        data: The data to return in the response
    
    Returns:
        JSON response with status and data
    """
    return jsonify({
        "success": True,
        "data": data
    })


def api_error(message: str, status_code: int = 400) -> Dict[str, Any]:
    """
    Create a standardized error API response.
    
    Args:
        message: Error message
        status_code: HTTP status code
    
    Returns:
        JSON response with error details
    """
    response = jsonify({
        "success": False,
        "error": message
    })
    response.status_code = status_code
    return response