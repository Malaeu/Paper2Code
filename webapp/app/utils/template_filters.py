"""Template filters for Jinja2."""

import locale
from flask import Blueprint


# Create a template_filters blueprint
template_filters = Blueprint('filters', __name__)


@template_filters.app_template_filter('number_format')
def number_format(value):
    """
    Format a number with thousands separators.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted number as string
    """
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value