#!/usr/bin/env python3
"""Generate comprehensive Markdown report with all LLM statistics."""

import yaml
from datetime import datetime
from utils import calculate_utility


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


def estimate_cost(cost_penalty):
    """Convert cost penalty to estimated price per 1M tokens."""
    # Rough mapping based on real prices
    if cost_penalty <= 0.01:
        return "$0.10-0.50"
    elif cost_penalty <= 0.05:
        return "$0.50-2.00"
    elif cost_penalty <= 0.10:
        return "$3.00-5.00"
    elif cost_penalty <= 0.20:
        return "$5.00-10.00"
    elif cost_penalty <= 0.40:
        return "$10.00-20.00"
    elif cost_penalty <= 0.60:
        return "$20.00-30.00"
    else:
        return "$30.00+"


def generate_markdown_report(models, config):
    """Generate complete Markdown report."""
    report = []
    report.append("# 🤖 LLM Utility Evaluation Report")
    report.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Formula explanation
    report.append("## 📐 Utility Formula")
    report.append("```")
    report.append("Utility = α × Quality + β × Speed + γ × Stability − δ × CostPenalty")
    report.append("```\n")
    
    # All models overview
    report.append("## 📊 All 19 Models Overview")
    report.append("\n| Model | Quality | Speed | Stability | Cost Penalty | Est. Price/1M tokens |")
    report.append("|-------|---------|-------|-----------|--------------|---------------------|")
    
    for model in sorted(models, key=lambda x: x['name']):
        report.append(f"| {model['name']} | {model['quality']:.2f} | {model['speed']:.2f} | "
                     f"{model['stability']:.2f} | {model['cost_penalty']:.3f} | {estimate_cost(model['cost_penalty'])} |")
    
    # Categories explanation
    report.append("\n## 🎯 Category Explanations")
    report.append("\n### Weight Profiles:")
    report.append("- **Default (Balanced)**: α=1.0, β=0.8, γ=1.0, δ=1.2")
    report.append("  - Best for: General-purpose use")
    report.append("- **Quality First**: α=1.5, β=0.5, γ=1.0, δ=0.8")
    report.append("  - Best for: Critical tasks requiring highest accuracy")
    report.append("- **Speed Optimized**: α=0.7, β=1.5, γ=0.8, δ=1.0")
    report.append("  - Best for: Real-time applications, chat, coding assistants")
    report.append("- **Budget Conscious**: α=0.9, β=0.9, γ=0.9, δ=2.0")
    report.append("  - Best for: High-volume applications, startups\n")
    
    # Category rankings
    categories = {
        "Default (Balanced)": config["weights"],
        "Quality First": config["profiles"]["quality_first"],
        "Speed Optimized": config["profiles"]["speed_optimized"],
        "Budget Conscious": config["profiles"]["budget_conscious"],
    }
    
    report.append("## 🏆 Top 10 Rankings by Category\n")
    
    for category_name, weights in categories.items():
        ranked = rank_by_profile(models, weights)[:10]
        
        report.append(f"### {category_name}")
        report.append(f"*Weights: α={weights['alpha']}, β={weights['beta']}, γ={weights['gamma']}, δ={weights['delta']}*\n")
        report.append("| Rank | Model | Score | Quality | Speed | Stability | Cost | Est. Price |")
        report.append("|------|-------|-------|---------|-------|-----------|------|------------|")
        
        for i, model in enumerate(ranked, 1):
            report.append(f"| {i} | **{model['name']}** | {model['score']:.4f} | "
                         f"{model['quality']:.2f} | {model['speed']:.2f} | "
                         f"{model['stability']:.2f} | {model['cost_penalty']:.3f} | "
                         f"{estimate_cost(model['cost_penalty'])} |")
        report.append("")
    
    # Summary insights
    report.append("## 💡 Key Insights\n")
    
    # Find most versatile models
    top_models = {}
    for category_name, weights in categories.items():
        ranked = rank_by_profile(models, weights)[:3]
        for model in ranked:
            name = model['name']
            if name not in top_models:
                top_models[name] = []
            top_models[name].append(category_name)
    
    report.append("### 🌟 Most Versatile Models (Top 3 in multiple categories):")
    versatile = sorted([(name, cats) for name, cats in top_models.items() 
                       if len(cats) > 1], key=lambda x: len(x[1]), reverse=True)
    
    for name, cats in versatile[:5]:
        report.append(f"- **{name}**: {', '.join(cats)}")
    
    # Best in metrics
    report.append("\n### 🏅 Champions by Individual Metrics:")
    best_quality = max(models, key=lambda x: x['quality'])
    best_speed = max(models, key=lambda x: x['speed'])
    best_stability = max(models, key=lambda x: x['stability'])
    best_cost = min(models, key=lambda x: x['cost_penalty'])
    
    report.append(f"- **Highest Quality**: {best_quality['name']} ({best_quality['quality']:.2f})")
    report.append(f"- **Fastest Speed**: {best_speed['name']} ({best_speed['speed']:.2f} - ~{int(best_speed['speed']*400)} tokens/s)")
    report.append(f"- **Most Stable**: {best_stability['name']} ({best_stability['stability']:.2f})")
    report.append(f"- **Most Affordable**: {best_cost['name']} ({estimate_cost(best_cost['cost_penalty'])})")
    
    # Usage recommendations
    report.append("\n## 🚀 Usage Recommendations\n")
    report.append("### By Use Case:")
    report.append("- **For Coding**: Claude 3.5 Sonnet, GPT-4o, DeepSeek-Coder-V2")
    report.append("- **For Chat/Support**: Gemini 2.0 Flash, Claude 3.5 Haiku, GPT-4o-mini")
    report.append("- **For Research/Analysis**: Claude 4 Opus, GPT-4o, Claude 3.5 Sonnet")
    report.append("- **For Budget Projects**: DeepSeek-V3, GPT-4o-mini, Gemini 1.5 Flash")
    report.append("- **For Enterprise**: Claude 4 Sonnet, GPT-4o, Gemini 1.5 Pro")
    
    report.append("\n---")
    report.append("\n*Note: Prices are estimates based on cost penalty values. Check provider websites for exact pricing.*")
    
    return "\n".join(report)


def main():
    # Load data
    models, config = load_data()
    
    # Generate report
    report = generate_markdown_report(models, config)
    
    # Save to file
    with open("LLM_EVALUATION_REPORT.md", "w") as f:
        f.write(report)
    
    print("✅ Report saved to: LLM_EVALUATION_REPORT.md")
    print("📄 You can view it with any Markdown viewer or on GitHub")
    
    # Also save as HTML for easy viewing
    try:
        import markdown
        html_content = markdown.markdown(report, extensions=['tables'])
        with open("LLM_EVALUATION_REPORT.html", "w") as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>""")
        print("✅ HTML report saved to: LLM_EVALUATION_REPORT.html")
    except ImportError:
        print("ℹ️  Install 'markdown' package to generate HTML report: pip install markdown")


if __name__ == "__main__":
    main()