#!/usr/bin/env python3
"""Generate comprehensive statistics for all LLM categories."""

import yaml
from tabulate import tabulate
from utils import calculate_utility, filter_by_constraints


def load_data():
    """Load models and config data."""
    with open("models.yaml", "r") as f:
        models = yaml.safe_load(f)["models"]
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    return models, config


def rank_by_profile(models, weights):
    """Rank models using specific weight profile."""
    scored = []
    for model in models:
        score = calculate_utility(model, weights)
        model_copy = model.copy()
        model_copy["score"] = round(score, 4)
        scored.append(model_copy)
    
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def generate_category_stats(models, config):
    """Generate statistics for each category."""
    categories = {
        "Default (Balanced)": config["weights"],
        "Quality First": config["profiles"]["quality_first"],
        "Speed Optimized": config["profiles"]["speed_optimized"],
        "Budget Conscious": config["profiles"]["budget_conscious"],
    }
    
    results = {}
    
    for category_name, weights in categories.items():
        ranked = rank_by_profile(models, weights)
        results[category_name] = {
            "weights": weights,
            "top_10": ranked[:10]
        }
    
    return results


def print_category_table(category_name, data):
    """Print formatted table for a category."""
    print(f"\n{'='*80}")
    print(f"🏆 {category_name}")
    print(f"⚖️  Weights: α={data['weights']['alpha']}, β={data['weights']['beta']}, "
          f"γ={data['weights']['gamma']}, δ={data['weights']['delta']}")
    print(f"{'='*80}")
    
    table_data = []
    for i, model in enumerate(data['top_10'], 1):
        table_data.append([
            i,
            model['name'],
            f"{model['score']:.4f}",
            f"{model['quality']:.2f}",
            f"{model['speed']:.2f}",
            f"{model['stability']:.2f}",
            f"{model['cost_penalty']:.2f}"
        ])
    
    headers = ["Rank", "Model", "Score", "Quality", "Speed", "Stability", "Cost"]
    from tabulate import tabulate as tab
    print(tab(table_data, headers=headers, tablefmt="grid"))


def generate_comparative_table(results):
    """Generate comparative table showing top 3 in each category."""
    print("\n" + "="*100)
    print("📊 COMPARATIVE SUMMARY - Top 3 Models by Category")
    print("="*100)
    
    table_data = []
    for category, data in results.items():
        top_3 = data['top_10'][:3]
        for i, model in enumerate(top_3, 1):
            table_data.append([
                category,
                i,
                model['name'],
                f"{model['score']:.4f}",
                f"Q:{model['quality']:.2f} S:{model['speed']:.2f} St:{model['stability']:.2f} C:{model['cost_penalty']:.2f}"
            ])
        table_data.append(["---"] * 5)  # Separator
    
    headers = ["Category", "Rank", "Model", "Score", "Metrics (Q/S/St/C)"]
    from tabulate import tabulate as tab
    print(tab(table_data[:-1], headers=headers, tablefmt="grid"))  # Remove last separator


def generate_insights(results):
    """Generate insights from the analysis."""
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    # Find models that appear in top 3 across multiple categories
    top_models = {}
    for category, data in results.items():
        for model in data['top_10'][:3]:
            name = model['name']
            if name not in top_models:
                top_models[name] = []
            top_models[name].append(category)
    
    print("\n🌟 Most Versatile Models (appearing in top 3 across multiple categories):")
    versatile = sorted([(name, cats) for name, cats in top_models.items() 
                       if len(cats) > 1], key=lambda x: len(x[1]), reverse=True)
    
    for name, categories in versatile:
        print(f"  • {name}: {', '.join(categories)}")
    
    # Best in each metric
    all_models = results["Default (Balanced)"]["top_10"][:10]
    
    print("\n🏅 Best in Individual Metrics:")
    best_quality = max(all_models, key=lambda x: x['quality'])
    best_speed = max(all_models, key=lambda x: x['speed'])
    best_stability = max(all_models, key=lambda x: x['stability'])
    best_cost = min(all_models, key=lambda x: x['cost_penalty'])
    
    print(f"  • Highest Quality: {best_quality['name']} ({best_quality['quality']:.2f})")
    print(f"  • Fastest Speed: {best_speed['name']} ({best_speed['speed']:.2f})")
    print(f"  • Most Stable: {best_stability['name']} ({best_stability['stability']:.2f})")
    print(f"  • Most Affordable: {best_cost['name']} (cost penalty: {best_cost['cost_penalty']:.2f})")


def main():
    print("🤖 LLM Utility Evaluation - Comprehensive Statistics")
    print("="*80)
    
    # Load data
    models, config = load_data()
    
    # Generate statistics for each category
    results = generate_category_stats(models, config)
    
    # Print detailed tables for each category
    for category_name, data in results.items():
        print_category_table(category_name, data)
    
    # Generate comparative summary
    generate_comparative_table(results)
    
    # Generate insights
    generate_insights(results)
    
    # Explanation of categories
    print("\n" + "="*80)
    print("📖 CATEGORY EXPLANATIONS")
    print("="*80)
    print("""
• Default (Balanced): Equal consideration of all factors
  → Best for: General-purpose use when you need a well-rounded model

• Quality First: Prioritizes output quality (α=1.5) over other factors
  → Best for: Critical tasks requiring highest accuracy and reasoning

• Speed Optimized: Prioritizes inference speed (β=1.5) for real-time apps
  → Best for: Chat applications, live coding assistants, interactive tools

• Budget Conscious: Heavily penalizes cost (δ=2.0) while maintaining quality
  → Best for: High-volume applications, startups, cost-sensitive projects
""")


if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        import tabulate
    
    main()