#!/bin/bash

# Two-Phase Adaptation Script
# This script implements the two-phase approach for methodology adaptation:
# 1. Uses a pre-generated adaptation plan
# 2. Applies the plan to generate fully adapted code

# Default values
CONFIG_PATH="custom_adapt/adapt_config.yaml"
API_MODEL="o3-mini-2025-04-16"
OUTPUT_DIR="outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      API_MODEL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config path/to/config.yaml] [--model model-name] [--output output-directory]"
      exit 1
      ;;
  esac
done

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Configuration file not found at $CONFIG_PATH"
  exit 1
fi

# Extract values from YAML config
echo "Parsing configuration file..."
DATASET_PATH=$(grep -o 'path: "[^"]*"' "$CONFIG_PATH" | head -1 | cut -d'"' -f2)
PLAN_PATH=$(grep -o 'adaptation_plan_path: "[^"]*"' "$CONFIG_PATH" | head -1 | cut -d'"' -f2)
REPO_NAME=$(grep -o 'repo_name: "[^"]*"' "$CONFIG_PATH" | head -1 | cut -d'"' -f2)

# Check if adaptation plan exists
if [ ! -f "$PLAN_PATH" ]; then
  echo "Error: Adaptation plan not found at $PLAN_PATH"
  echo "Use run_direct_adapt.sh first to generate a plan, or provide a valid plan path"
  exit 1
fi

echo "==================================================="
echo "🧠 Paper2Code Two-Phase Adaptation"
echo "==================================================="
echo "Configuration file: $CONFIG_PATH"
echo "Dataset path: $DATASET_PATH"
echo "Adaptation plan: $PLAN_PATH"
echo "Output repository: $REPO_NAME"
echo "Using model: $API_MODEL"
echo "==================================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/$REPO_NAME"

# Run the analysis phase with the pre-generated plan
echo "📊 Running analysis phase with pre-generated plan..."
python ../codes/adapt_analyzing_with_plan.py \
  --config "$CONFIG_PATH" \
  --plan_path "$PLAN_PATH" \
  --output_dir "$OUTPUT_DIR/$REPO_NAME/analyzing_artifacts" \
  --gpt_version "$API_MODEL"

# Check if analysis was successful
if [ $? -ne 0 ]; then
  echo "❌ Analysis phase failed"
  exit 1
fi

echo "✅ Analysis phase completed successfully"

# Run the coding phase
echo "💻 Generating adapted code..."
python ../codes/adapt_coding.py \
  --config "$CONFIG_PATH" \
  --analysis_dir "$OUTPUT_DIR/$REPO_NAME/analyzing_artifacts" \
  --output_dir "$OUTPUT_DIR/$REPO_NAME/coding_artifacts" \
  --output_repo_dir "$OUTPUT_DIR/${REPO_NAME}_repo" \
  --gpt_version "$API_MODEL"

# Check if coding was successful
if [ $? -ne 0 ]; then
  echo "❌ Code generation failed"
  exit 1
fi

echo "==================================================="
echo "✅ Adaptation completed successfully!"
echo "==================================================="
echo "Generated repository: $OUTPUT_DIR/${REPO_NAME}_repo"
echo ""
echo "To use the generated code:"
echo "  cd $OUTPUT_DIR/${REPO_NAME}_repo"
echo "  pip install -r requirements.txt"
echo "  python main.py"
echo "==================================================="