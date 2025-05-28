# Integrated LLM Evaluation Report


## Quality First Rankings

| Model | Utility | RAPI | Quality | Speed | Cost Penalty | Context |
|-------|---------|------|---------|-------|--------------|---------|
| Gemini 2.5 Flash | 2.734 | 4153.0 | 0.85 | 0.99 | 0.007 | 1M |
| Claude 3.5 Sonnet | 2.705 | 95.0 | 0.92 | 0.85 | 0.100 | 128K |
| Gemini 2.0 Flash | 2.700 | Multi-mode | 0.85 | 0.99 | 0.050 | 128K |
| DeepSeek-V3 | 2.697 | Multi-mode | 0.86 | 0.99 | 0.010 | 128K |
| Claude 4 Sonnet | 2.695 | Multi-mode | 0.92 | 0.93 | 0.150 | 128K |
| GPT-4o-mini | 2.674 | Multi-mode | 0.83 | 0.97 | 0.007 | 128K |
| o4-mini | 2.666 | Multi-mode | 0.88 | 0.90 | 0.055 | 128K |
| GPT-4.1 | 2.660 | 292.0 | 0.90 | 0.88 | 0.100 | 1M |
| Gemini 1.5 Flash | 2.616 | Multi-mode | 0.82 | 0.96 | 0.030 | 128K |
| DeepSeek-Coder-V2 | 2.602 | Multi-mode | 0.84 | 0.88 | 0.010 | 128K |

## Speed Optimized Rankings

| Model | Utility | RAPI | Quality | Speed | Cost Penalty | Context |
|-------|---------|------|---------|-------|--------------|---------|
| Gemini 2.5 Flash | 2.849 | 4153.0 | 0.85 | 0.99 | 0.007 | 1M |
| DeepSeek-V3 | 2.813 | Multi-mode | 0.86 | 0.99 | 0.010 | 128K |
| Gemini 2.0 Flash | 2.806 | Multi-mode | 0.85 | 0.99 | 0.050 | 128K |
| GPT-4o-mini | 2.789 | Multi-mode | 0.83 | 0.97 | 0.007 | 128K |
| Claude 3.5 Haiku | 2.766 | Multi-mode | 0.75 | 0.99 | 0.020 | 128K |
| Gemini 1.5 Flash | 2.728 | Multi-mode | 0.82 | 0.96 | 0.030 | 128K |
| Claude 4 Sonnet | 2.665 | Multi-mode | 0.92 | 0.93 | 0.150 | 128K |
| o4-mini | 2.663 | Multi-mode | 0.88 | 0.90 | 0.055 | 128K |
| DeepSeek-Coder-V2 | 2.626 | Multi-mode | 0.84 | 0.88 | 0.010 | 128K |
| GPT-4.1 | 2.610 | 292.0 | 0.90 | 0.88 | 0.100 | 1M |

## Budget Conscious Rankings

| Model | Utility | RAPI | Quality | Speed | Cost Penalty | Context |
|-------|---------|------|---------|-------|--------------|---------|
| Gemini 2.5 Flash | 2.514 | 4153.0 | 0.85 | 0.99 | 0.007 | 1M |
| DeepSeek-V3 | 2.473 | Multi-mode | 0.86 | 0.99 | 0.010 | 128K |
| GPT-4o-mini | 2.461 | Multi-mode | 0.83 | 0.97 | 0.007 | 128K |
| Gemini 2.0 Flash | 2.429 | Multi-mode | 0.85 | 0.99 | 0.050 | 128K |
| Claude 3.5 Haiku | 2.399 | Multi-mode | 0.75 | 0.99 | 0.020 | 128K |
| Gemini 1.5 Flash | 2.379 | Multi-mode | 0.82 | 0.96 | 0.030 | 128K |
| DeepSeek-Coder-V2 | 2.347 | Multi-mode | 0.84 | 0.88 | 0.010 | 128K |
| o4-mini | 2.338 | Multi-mode | 0.88 | 0.90 | 0.055 | 128K |
| Claude 3.5 Sonnet | 2.275 | 95.0 | 0.92 | 0.85 | 0.100 | 128K |
| GPT-4.1 | 2.257 | 292.0 | 0.90 | 0.88 | 0.100 | 1M |

## Default Rankings

| Model | Utility | RAPI | Quality | Speed | Cost Penalty | Context |
|-------|---------|------|---------|-------|--------------|---------|
| Gemini 2.5 Flash | 2.603 | 4153.0 | 0.85 | 0.99 | 0.007 | 1M |
| DeepSeek-V3 | 2.560 | Multi-mode | 0.86 | 0.99 | 0.010 | 128K |
| Gemini 2.0 Flash | 2.552 | Multi-mode | 0.85 | 0.99 | 0.050 | 128K |
| GPT-4o-mini | 2.548 | Multi-mode | 0.83 | 0.97 | 0.007 | 128K |
| Claude 3.5 Haiku | 2.488 | Multi-mode | 0.75 | 0.99 | 0.020 | 128K |
| Gemini 1.5 Flash | 2.482 | Multi-mode | 0.82 | 0.96 | 0.030 | 128K |
| o4-mini | 2.474 | Multi-mode | 0.88 | 0.90 | 0.055 | 128K |
| Claude 3.5 Sonnet | 2.460 | 95.0 | 0.92 | 0.85 | 0.100 | 128K |
| Claude 4 Sonnet | 2.454 | Multi-mode | 0.92 | 0.93 | 0.150 | 128K |
| DeepSeek-Coder-V2 | 2.442 | Multi-mode | 0.84 | 0.88 | 0.010 | 128K |

## Key Insights

1. **Utility vs RAPI**: Utility scoring considers user-defined weights, while RAPI provides absolute performance/cost ratio
2. **Context Window Impact**: Models with 1M+ context (GPT-4.1, Gemini 2.5) get significant RAPI boost
3. **Reasoning Modes**: o4-mini offers flexible quality/latency trade-offs not captured in simple utility scores
4. **Cost Leaders**: Gemini 2.5 Flash dominates on cost-efficiency across all metrics