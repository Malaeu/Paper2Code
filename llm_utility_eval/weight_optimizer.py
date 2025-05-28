#!/usr/bin/env python3
"""Interactive weight optimizer for finding optimal LLM utility weights."""

import yaml
from utils import calculate_utility
import numpy as np


def load_models():
    """Load model data."""
    with open("models.yaml", "r") as f:
        return yaml.safe_load(f)["models"]


def interactive_weight_selection():
    """Guide user through weight selection process."""
    print("🎯 LLM Weight Optimizer - Interactive Mode")
    print("="*50)
    
    # Ask key questions
    print("\n📋 Answer these questions (1-5 scale):\n")
    
    quality_need = float(input("1. How critical is output quality/accuracy? (1=not important, 5=critical): "))
    speed_need = float(input("2. How important is fast response time? (1=can wait, 5=real-time needed): "))
    stability_need = float(input("3. How critical is reliability/uptime? (1=can retry, 5=must work always): "))
    budget_limit = float(input("4. How constrained is your budget? (1=unlimited, 5=very limited): "))
    
    # Convert to weights
    weights = {
        "alpha": 0.3 + (quality_need / 5) * 1.2,      # 0.3 to 1.5
        "beta": 0.3 + (speed_need / 5) * 1.2,         # 0.3 to 1.5  
        "gamma": 0.4 + (stability_need / 5) * 1.0,    # 0.4 to 1.4
        "delta": 0.2 + (budget_limit / 5) * 1.8       # 0.2 to 2.0
    }
    
    print(f"\n🔧 Calculated weights:")
    print(f"  α (Quality): {weights['alpha']:.2f}")
    print(f"  β (Speed): {weights['beta']:.2f}")
    print(f"  γ (Stability): {weights['gamma']:.2f}")
    print(f"  δ (Cost): {weights['delta']:.2f}")
    
    return weights


def sensitivity_analysis(models, base_weights):
    """Analyze how sensitive the ranking is to weight changes."""
    print("\n📊 Sensitivity Analysis")
    print("="*50)
    
    # Get baseline ranking
    baseline = []
    for model in models:
        score = calculate_utility(model, base_weights)
        baseline.append((model['name'], score))
    baseline.sort(key=lambda x: x[1], reverse=True)
    baseline_top3 = [x[0] for x in baseline[:3]]
    
    print(f"\nBaseline top 3: {', '.join(baseline_top3)}")
    
    # Test each weight
    for param in ['alpha', 'beta', 'gamma', 'delta']:
        print(f"\n🔄 Testing {param}:")
        
        for delta in [-0.3, -0.1, 0.1, 0.3]:
            test_weights = base_weights.copy()
            test_weights[param] += delta
            
            # Ensure weights stay positive
            if test_weights[param] < 0.1:
                test_weights[param] = 0.1
            
            # Get new ranking
            new_ranking = []
            for model in models:
                score = calculate_utility(model, test_weights)
                new_ranking.append((model['name'], score))
            new_ranking.sort(key=lambda x: x[1], reverse=True)
            new_top3 = [x[0] for x in new_ranking[:3]]
            
            # Check if top 3 changed
            changed = new_top3 != baseline_top3
            change_indicator = "⚠️ CHANGED" if changed else "✅ stable"
            
            print(f"  {param} = {test_weights[param]:.2f} ({delta:+.1f}): "
                  f"{new_top3[0]} {change_indicator}")


def find_pareto_optimal(models):
    """Find Pareto-optimal models (not dominated by others)."""
    print("\n🏆 Pareto-Optimal Models")
    print("="*50)
    
    pareto_models = []
    
    for i, model1 in enumerate(models):
        is_dominated = False
        
        for j, model2 in enumerate(models):
            if i == j:
                continue
            
            # Check if model2 dominates model1
            dominates = (
                model2['quality'] >= model1['quality'] and
                model2['speed'] >= model1['speed'] and
                model2['stability'] >= model1['stability'] and
                model2['cost_penalty'] <= model1['cost_penalty']
            )
            
            # At least one strict inequality
            strictly_better = (
                model2['quality'] > model1['quality'] or
                model2['speed'] > model1['speed'] or
                model2['stability'] > model1['stability'] or
                model2['cost_penalty'] < model1['cost_penalty']
            )
            
            if dominates and strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_models.append(model1)
    
    print(f"\nFound {len(pareto_models)} Pareto-optimal models:")
    for model in sorted(pareto_models, key=lambda x: x['quality'], reverse=True):
        print(f"  • {model['name']}: Q={model['quality']:.2f}, "
              f"S={model['speed']:.2f}, St={model['stability']:.2f}, "
              f"C={model['cost_penalty']:.3f}")


def suggest_weights_for_usecase():
    """Suggest weights based on use case."""
    print("\n🎯 Use Case Weight Suggestions")
    print("="*50)
    
    use_cases = {
        "1": ("IDE Code Assistant", {"alpha": 1.2, "beta": 1.3, "gamma": 0.9, "delta": 0.8}),
        "2": ("Customer Support Chat", {"alpha": 0.9, "beta": 1.4, "gamma": 1.2, "delta": 1.5}),
        "3": ("Research & Analysis", {"alpha": 2.0, "beta": 0.3, "gamma": 0.8, "delta": 0.5}),
        "4": ("Content Generation", {"alpha": 1.3, "beta": 0.8, "gamma": 1.0, "delta": 1.0}),
        "5": ("Real-time Translation", {"alpha": 1.1, "beta": 1.6, "gamma": 1.1, "delta": 1.2}),
        "6": ("Educational Tutor", {"alpha": 1.4, "beta": 0.9, "gamma": 1.2, "delta": 0.9}),
        "7": ("API Backend Service", {"alpha": 1.0, "beta": 1.2, "gamma": 1.4, "delta": 1.1}),
    }
    
    print("\nSelect your use case:")
    for key, (name, _) in use_cases.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter number (1-7): ")
    
    if choice in use_cases:
        name, weights = use_cases[choice]
        print(f"\n✅ Suggested weights for {name}:")
        print(f"  α (Quality): {weights['alpha']}")
        print(f"  β (Speed): {weights['beta']}")
        print(f"  γ (Stability): {weights['gamma']}")
        print(f"  δ (Cost): {weights['delta']}")
        return weights
    
    return None


def main():
    """Main interactive flow."""
    models = load_models()
    
    print("\n🤖 LLM Weight Optimization Tool")
    print("="*50)
    print("\nChoose mode:")
    print("  1. Interactive weight selection")
    print("  2. Use case based suggestions") 
    print("  3. Sensitivity analysis")
    print("  4. Find Pareto-optimal models")
    
    mode = input("\nSelect mode (1-4): ")
    
    if mode == "1":
        weights = interactive_weight_selection()
    elif mode == "2":
        weights = suggest_weights_for_usecase()
        if not weights:
            return
    elif mode == "3":
        # Use default weights for analysis
        with open("config.yaml", "r") as f:
            weights = yaml.safe_load(f)["weights"]
        sensitivity_analysis(models, weights)
        return
    elif mode == "4":
        find_pareto_optimal(models)
        return
    else:
        print("Invalid choice")
        return
    
    # Show top 5 with selected weights
    print("\n📊 Top 5 models with your weights:")
    print("="*50)
    
    scored = []
    for model in models:
        score = calculate_utility(model, weights)
        scored.append((model['name'], score, model))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score, model) in enumerate(scored[:5], 1):
        print(f"{i}. {name}: {score:.4f}")
        print(f"   Q={model['quality']:.2f}, S={model['speed']:.2f}, "
              f"St={model['stability']:.2f}, C=${model['cost_penalty']:.3f}")
    
    # Save option
    save = input("\n💾 Save these weights to config? (y/n): ")
    if save.lower() == 'y':
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        config['profiles']['custom_optimized'] = weights
        
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Saved as 'custom_optimized' profile!")
        print("   Use: python main.py --profile custom_optimized")


if __name__ == "__main__":
    main()