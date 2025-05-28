"""Utility calculation functions for LLM evaluation."""

def calculate_utility(model, weights):
    """
    Calculate utility score for an LLM model.
    
    Formula: Utility = α * Quality + β * Speed + γ * Stability − δ * CostPenalty
    
    Args:
        model: dict with keys 'quality', 'speed', 'stability', 'cost_penalty'
        weights: dict with keys 'alpha', 'beta', 'gamma', 'delta'
    
    Returns:
        float: utility score
    """
    return (
        weights["alpha"] * model["quality"] +
        weights["beta"] * model["speed"] +
        weights["gamma"] * model["stability"] -
        weights["delta"] * model["cost_penalty"]
    )

def filter_by_constraints(models, constraints=None):
    """
    Filter models by constraints.
    
    Args:
        models: list of model dicts
        constraints: dict with optional keys like 'min_quality', 'max_cost_penalty'
    
    Returns:
        list: filtered models
    """
    if not constraints:
        return models
    
    filtered = models
    
    if "min_quality" in constraints:
        filtered = [m for m in filtered if m["quality"] >= constraints["min_quality"]]
    
    if "max_cost_penalty" in constraints:
        filtered = [m for m in filtered if m["cost_penalty"] <= constraints["max_cost_penalty"]]
    
    if "min_stability" in constraints:
        filtered = [m for m in filtered if m["stability"] >= constraints["min_stability"]]
    
    return filtered