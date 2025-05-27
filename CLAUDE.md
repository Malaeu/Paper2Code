# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Paper2Code is a multi-agent LLM system that transforms scientific papers into code repositories. It follows a three-stage pipeline:
1. Planning - Creates a plan and architecture design
2. Analysis - Analyzes paper content in detail
3. Code Generation - Generates implementation code

## Commands

### Environment Setup

```bash
# Install OpenAI API dependencies
pip install openai tiktoken

# Install vLLM for open-source models
pip install vllm 

# Install Claude CLI with latest models support
npm i -g @anthropic-ai/claude-code@latest

# Install all dependencies
pip install -r requirements.txt
```

### Claude 4 Upgrade

The pipeline has been upgraded to use Claude 4 models:
- **Sonnet 4** for vision, chat, and literature review tasks
- **Opus 4** with Claude Code for code generation and patching
- See `UPGRADE_CLAUDE4.md` for detailed upgrade information

### Running Paper2Code

#### Using OpenAI API (PDF-based JSON)
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="<OPENAI_API_KEY>"

# Run the pipeline on the example Transformer paper
cd scripts
bash run.sh
```

#### Using OpenAI API (LaTeX source)
```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
cd scripts
bash run_latex.sh
```

#### Using Open Source Models (vLLM)
```bash
# Default model is deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
cd scripts
bash run_llm.sh  # PDF-based JSON
bash run_latex_llm.sh  # LaTeX source
```

### Model-based Evaluation

#### Reference-free Evaluation
```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --eval_result_dir ../results \
    --eval_type ref_free \
    --generated_n 8 \
    --papercoder
```

#### Reference-based Evaluation
```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --gold_repo_dir ../examples/Transformer_gold_repo \
    --eval_result_dir ../results \
    --eval_type ref_based \
    --generated_n 8 \
    --papercoder
```

## Architecture

The system follows a pipeline architecture with three main stages:

1. **Planning Stage** (1_planning.py/1_planning_llm.py)
   - Creates a detailed plan for implementing the paper
   - Designs architecture and file structure
   - Extracts configurations (1.1_extract_config.py)

2. **Analysis Stage** (2_analyzing.py/2_analyzing_llm.py)
   - Analyzes paper content in detail
   - Extracts key components, methods, and experiments

3. **Code Generation Stage** (3_coding.py/3_coding_llm.py)
   - Generates implementation code based on planning and analysis
   - Creates the final repository

Each stage can use either OpenAI models (default files) or open-source models via vLLM (files with _llm suffix).

## Working with the Code

- Script files in `/scripts` directory control the pipeline execution
- Core implementation is in the `/codes` directory
- The system can process papers in two formats:
  - JSON format (converted from PDF using s2orc-doc2json)
  - LaTeX source format
- Output repositories are generated in the `/outputs` directory
- Evaluation metrics assess the quality of generated repositories

## Benchmark Dataset

The Paper2Code benchmark dataset contains 90 papers from ICML, NeurIPS, and ICLR 2024, with each paper having a reference GitHub repository. The dataset can be accessed via Hugging Face: https://huggingface.co/datasets/iaminju/paper2code.