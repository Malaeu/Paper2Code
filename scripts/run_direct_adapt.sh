#!/bin/bash

# Direct Adaptation Script
# This script implements the first phase of the two-phase approach,
# generating an adaptation plan from a paper and dataset description.

# Default values
CONFIG_PATH="custom_adapt/adapt_config.yaml"
API_MODEL="o3-mini-2025-04-16"
OUTPUT_DIR="outputs"
UPDATE_CONFIG=true

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
    --no-update-config)
      UPDATE_CONFIG=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config path/to/config.yaml] [--model model-name] [--output output-directory] [--no-update-config]"
      exit 1
      ;;
  esac
done

# Check for config directory and file
CONFIG_DIR=$(dirname "$CONFIG_PATH")
if [ ! -d "$CONFIG_DIR" ]; then
  echo "Creating config directory: $CONFIG_DIR"
  mkdir -p "$CONFIG_DIR"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found. Creating a template at $CONFIG_PATH"
  cat > "$CONFIG_PATH" << EOL
# Adaptation Configuration

# Original Paper Information
paper:
  title: "Example Paper Title"
  authors: "Author et al."
  year: 2023
  methodology: "Survival analysis with machine learning"
  json_path: "examples/paper.json"  # Path to paper JSON file

# Dataset Configuration
dataset:
  path: "data/dataset.csv"
  format: "csv"  # csv, parquet, excel, json
  description_path: "data/dataset_description.md"  # optional path to markdown description file

# Variable Mapping
variable_mapping:
  original_to_adapted:
    # Primary stratification variable
    "race": "gender"
    # Primary categories
    "Black": "female"
    "White": "male"
    # Other variables as needed
    "natriuretic_peptide": "blood_marker_a"
    "troponin": "blood_marker_b"

# Methodology Adjustments
methodology:
  maintain_landmark_analysis: true
  maintain_monte_carlo_cv: true
  iterations: 1000
  models_to_include:
    - "Cox"
    - "RSF"
    - "oRSF"

# Output Configuration
output:
  repo_name: "AdaptedModel"
  output_dir: "outputs"
  include_tests: true
  include_documentation: true

# Advanced Settings
advanced:
  use_direct_api: true
  adaptation_plan_path: ""  # Will be filled automatically
  custom_prompt_path: ""  # Optional path to custom prompt file
EOL
  echo "Please edit the template config file before running the script again."
  echo "At minimum, you should specify the paper.json_path and dataset.path values."
  exit 0
fi

echo "==================================================="
echo "🧠 Paper2Code Direct Adaptation - Phase 1"
echo "==================================================="
echo "Configuration file: $CONFIG_PATH"
echo "Using model: $API_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "==================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract repo name from config for output directory
REPO_NAME=$(grep -o 'repo_name: "[^"]*"' "$CONFIG_PATH" | head -1 | cut -d'"' -f2)
if [ -z "$REPO_NAME" ]; then
  REPO_NAME="AdaptedModel"
fi

# Set update config flag
UPDATE_FLAG=""
if [ "$UPDATE_CONFIG" = true ]; then
  UPDATE_FLAG="--update_config"
fi

# Run the direct adapt script
echo "📝 Generating adaptation plan..."
python direct_adapt.py \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --paper_name "$REPO_NAME" \
  --gpt_version "$API_MODEL" \
  $UPDATE_FLAG

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "❌ Failed to generate adaptation plan"
  exit 1
fi

echo "==================================================="
echo "✅ Adaptation plan generated successfully!"
echo "==================================================="
echo ""
echo "Next steps:"
echo "1. Review the generated plan in $OUTPUT_DIR/$REPO_NAME/"
echo "2. Edit the plan if necessary"
echo "3. Run phase 2 to generate code:"
echo "   ./run_with_plan.sh --config $CONFIG_PATH"
echo "==================================================="