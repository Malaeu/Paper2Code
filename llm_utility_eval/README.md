# 🤖 LLM Utility Evaluation Framework

Multi-factor utility function for LLM selection based on quality, speed, stability, and cost.

## 🧮 Formula

```
Utility = α × Quality + β × Speed + γ × Stability − δ × CostPenalty
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install pyyaml

# Run with default weights
python main.py

# Show only top 5 models
python main.py --top 5

# Use predefined profile
python main.py --profile budget_conscious

# Apply constraints
python main.py --min-quality 0.85 --max-cost 0.10

# Export results
python main.py --export results.json
```

## 📊 Predefined Profiles

- **quality_first**: Prioritizes model quality (α=1.5)
- **speed_optimized**: Prioritizes inference speed (β=1.5)
- **budget_conscious**: Minimizes cost (δ=2.0)
- **balanced**: Equal weights for all factors

## 📁 Files

- `models.yaml` - LLM specifications (quality, speed, stability, cost)
- `config.yaml` - Weights and constraints configuration
- `main.py` - Main evaluation script
- `utils.py` - Utility calculation functions

## 🎯 Example Output

```
📊 Ranked Models by Utility Score:
============================================================
 1. Gemini 2.0 Flash            | Score:  2.7300
    Quality: 0.85 | Speed: 0.98 | Stability: 0.97 | Cost: 0.05

 2. Claude 3.5 Sonnet           | Score:  2.6640
    Quality: 0.88 | Speed: 0.95 | Stability: 0.98 | Cost: 0.10

 3. DeepSeek-V3                 | Score:  2.6280
    Quality: 0.86 | Speed: 0.85 | Stability: 0.92 | Cost: 0.01
```

## 🔧 Customization

### Adding New Models

Edit `models.yaml`:

```yaml
models:
  - name: "New Model"
    quality: 0.90      # 0-1 scale
    speed: 0.85        # 0-1 scale
    stability: 0.95    # 0-1 scale
    cost_penalty: 0.20 # 0-1 scale (higher = more expensive)
```

### Adjusting Weights

Edit `config.yaml`:

```yaml
weights:
  alpha: 1.2   # Increase quality importance
  beta: 0.6    # Decrease speed importance
  gamma: 1.0   # Keep stability neutral
  delta: 1.5   # Increase cost sensitivity
```

## 🛠️ Advanced Usage

### Programmatic Use

```python
from utils import calculate_utility
from main import load_models, rank_models

models = load_models("models.yaml")
weights = {"alpha": 1.0, "beta": 0.8, "gamma": 1.0, "delta": 1.2}
ranked = rank_models(models, weights)

print(f"Best model: {ranked[0]['name']}")
```

### CI Integration

```yaml
# .github/workflows/llm-eval.yml
- name: Evaluate LLMs
  run: |
    python llm_utility_eval/main.py \
      --profile ${{ inputs.profile }} \
      --export evaluation-results.json
```

## 📈 Future Extensions

- [ ] Auto-update from HuggingFace leaderboards
- [ ] Streamlit/Gradio web interface
- [ ] Plotly visualizations
- [ ] A/B testing framework
- [ ] Custom metric plugins