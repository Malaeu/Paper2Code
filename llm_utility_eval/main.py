#!/usr/bin/env python3
"""LLM Utility Evaluation Tool - Multi-factor model selection."""

import yaml
import argparse
import json
from pathlib import Path
from utils import calculate_utility, filter_by_constraints


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(models_path="models.yaml"):
    """Load model data from YAML file."""
    with open(models_path, "r") as f:
        return yaml.safe_load(f)["models"]


def rank_models(models, weights, constraints=None):
    """Calculate utility scores and rank models."""
    # Filter by constraints if provided
    if constraints:
        models = filter_by_constraints(models, constraints)
    
    # Calculate utility for each model
    scored_models = []
    for model in models:
        utility = calculate_utility(model, weights)
        model["utility"] = round(utility, 4)
        scored_models.append(model)
    
    # Sort by utility score (descending)
    return sorted(scored_models, key=lambda x: x["utility"], reverse=True)


def print_results(ranked_models, top_n=None):
    """Print ranking results."""
    print("\n📊 Ranked Models by Utility Score:")
    print("=" * 60)
    
    models_to_show = ranked_models[:top_n] if top_n else ranked_models
    
    for i, model in enumerate(models_to_show, 1):
        print(f"{i:2d}. {model['name']:<25} | Score: {model['utility']:>7.4f}")
        print(f"    Quality: {model['quality']:.2f} | "
              f"Speed: {model['speed']:.2f} | "
              f"Stability: {model['stability']:.2f} | "
              f"Cost: {model['cost_penalty']:.2f}")
        print()


def export_results(ranked_models, output_path):
    """Export results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(ranked_models, f, indent=2)
    print(f"✅ Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Utility Evaluation - Multi-factor model selection"
    )
    parser.add_argument(
        "--models", 
        default="models.yaml",
        help="Path to models YAML file"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--profile",
        choices=["quality_first", "speed_optimized", "budget_conscious", "balanced"],
        help="Use predefined weight profile"
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show only top N models"
    )
    parser.add_argument(
        "--export",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        help="Filter by minimum quality score"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Filter by maximum cost penalty"
    )
    
    args = parser.parse_args()
    
    # Load data
    config = load_config(args.config)
    models = load_models(args.models)
    
    # Select weights
    if args.profile:
        weights = config["profiles"][args.profile]
        print(f"🎯 Using profile: {args.profile}")
    else:
        weights = config["weights"]
        print("🎯 Using default weights")
    
    # Build constraints from CLI args
    constraints = {}
    if args.min_quality:
        constraints["min_quality"] = args.min_quality
    if args.max_cost:
        constraints["max_cost_penalty"] = args.max_cost
    
    # Add constraints from config if not overridden
    if "constraints" in config and config["constraints"] is not None:
        for key, value in config["constraints"].items():
            if key not in constraints and value is not None:
                constraints[key] = value
    
    # Rank models
    ranked_models = rank_models(models, weights, constraints)
    
    # Display results
    print(f"\n⚖️  Weights: α={weights['alpha']}, β={weights['beta']}, "
          f"γ={weights['gamma']}, δ={weights['delta']}")
    
    if constraints:
        print(f"🔍 Applied constraints: {constraints}")
        print(f"📋 Models after filtering: {len(ranked_models)}/{len(models)}")
    
    print_results(ranked_models, args.top)
    
    # Export if requested
    if args.export:
        export_results(ranked_models, args.export)


if __name__ == "__main__":
    main()