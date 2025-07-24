#!/bin/bash

# Test the complete Paper2Code pipeline in CLI mode without GUI
# This script runs through the entire pipeline process:
# 1. Processing PDF/paper
# 2. Planning
# 3. Analyzing
# 4. Coding
# 5. Output generation

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}     PAPER2CODE PIPELINE CLI TEST        ${NC}"
echo -e "${BLUE}==========================================${NC}"

# Activate virtual environment
source .venv_env/bin/activate
export PYTHONPATH=$(pwd)

# Load environment variables
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
fi

# Check for test paper
TEST_PAPER="examples/Transformer.pdf"
if [ ! -f "$TEST_PAPER" ]; then
    echo -e "${RED}Test paper not found: $TEST_PAPER${NC}"
    echo "Please ensure the test paper exists"
    exit 1
fi

# Create a temporary output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="cli_test_output_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"

# Step 1: PDF Processing
echo -e "\n${YELLOW}[1/5] PROCESSING PDF${NC}"
echo -e "Processing $TEST_PAPER..."

python codes/0_pdf_process.py \
    --input_file "$TEST_PAPER" \
    --output_file "$OUTPUT_DIR/paper.json" \
    --gpt_version o3-mini-2025-04-16

if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/paper.json" ]; then
    echo -e "${GREEN}✓ PDF processing completed successfully${NC}"
    echo -e "  Output: $OUTPUT_DIR/paper.json ($(du -h "$OUTPUT_DIR/paper.json" | cut -f1))"
else
    echo -e "${RED}✗ PDF processing failed${NC}"
    exit 1
fi

# Step 2: Planning
echo -e "\n${YELLOW}[2/5] PLANNING${NC}"
echo -e "Generating plan from processed paper..."

# First try using the LLM version (newer)
python codes/1_planning_llm.py \
    --pdf_json_path "$OUTPUT_DIR/paper.json" \
    --output_json_path "$OUTPUT_DIR/plan.json" \
    --output_dir "$OUTPUT_DIR" \
    --gpt_version o3-mini-2025-04-16

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}LLM planning failed, trying standard planning...${NC}"
    python codes/1_planning.py \
        --pdf_json_path "$OUTPUT_DIR/paper.json" \
        --output_json_path "$OUTPUT_DIR/plan.json" \
        --output_dir "$OUTPUT_DIR" \
        --gpt_version o3-mini-2025-04-16
fi

if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/plan.json" ]; then
    echo -e "${GREEN}✓ Planning completed successfully${NC}"
    echo -e "  Output: $OUTPUT_DIR/plan.json ($(du -h "$OUTPUT_DIR/plan.json" | cut -f1))"
else
    echo -e "${RED}✗ Planning failed${NC}"
    exit 1
fi

# Step 3: Analyzing
echo -e "\n${YELLOW}[3/5] ANALYZING${NC}"
echo -e "Analyzing paper and planning results..."

# First try using the LLM version
python codes/2_analyzing_llm.py \
    --pdf_json_path "$OUTPUT_DIR/paper.json" \
    --plan_json_path "$OUTPUT_DIR/plan.json" \
    --output_dir "$OUTPUT_DIR/analysis" \
    --gpt_version o3-mini-2025-04-16

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}LLM analyzing failed, trying standard analyzing...${NC}"
    python codes/2_analyzing.py \
        --pdf_json_path "$OUTPUT_DIR/paper.json" \
        --plan_json_path "$OUTPUT_DIR/plan.json" \
        --output_dir "$OUTPUT_DIR/analysis" \
        --gpt_version o3-mini-2025-04-16
fi

if [ $? -eq 0 ] && [ -d "$OUTPUT_DIR/analysis" ]; then
    echo -e "${GREEN}✓ Analysis completed successfully${NC}"
    echo -e "  Output directory: $OUTPUT_DIR/analysis/"
    echo -e "  Files generated: $(find "$OUTPUT_DIR/analysis" -type f | wc -l)"
else
    echo -e "${RED}✗ Analysis failed${NC}"
    exit 1
fi

# Step 4: Coding
echo -e "\n${YELLOW}[4/5] CODING${NC}"
echo -e "Generating code from analysis..."

python codes/3_coding.py \
    --pdf_json_path "$OUTPUT_DIR/paper.json" \
    --plan_json_path "$OUTPUT_DIR/plan.json" \
    --analysis_dir "$OUTPUT_DIR/analysis" \
    --output_dir "$OUTPUT_DIR/coding" \
    --repo_dir "$OUTPUT_DIR/repo" \
    --gpt_version o3-mini-2025-04-16

if [ $? -eq 0 ] && [ -d "$OUTPUT_DIR/repo" ]; then
    echo -e "${GREEN}✓ Code generation completed successfully${NC}"
    echo -e "  Output repository: $OUTPUT_DIR/repo/"
    echo -e "  Files generated: $(find "$OUTPUT_DIR/repo" -type f | wc -l)"
else
    echo -e "${RED}✗ Code generation failed${NC}"
    exit 1
fi

# Step 5: Package repository
echo -e "\n${YELLOW}[5/5] PACKAGING RESULT${NC}"
echo -e "Creating final output package..."

# Get repository name from plan if possible
REPO_NAME=$(grep -o '"repo_name":[^,}]*' "$OUTPUT_DIR/plan.json" | cut -d'"' -f4 2>/dev/null)
if [ -z "$REPO_NAME" ]; then
    REPO_NAME="transformer_implementation"
fi

# Create zip archive
cd "$OUTPUT_DIR"
zip -r "${REPO_NAME}.zip" repo/ &>/dev/null

if [ $? -eq 0 ] && [ -f "${REPO_NAME}.zip" ]; then
    echo -e "${GREEN}✓ Repository packaged successfully${NC}"
    echo -e "  Output package: $OUTPUT_DIR/${REPO_NAME}.zip ($(du -h "${REPO_NAME}.zip" | cut -f1))"
else
    echo -e "${RED}✗ Repository packaging failed${NC}"
    exit 1
fi

# Final summary
cd ..
echo -e "\n${BLUE}==========================================${NC}"
echo -e "${BLUE}     PIPELINE EXECUTION COMPLETE          ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "Results summary:"
echo -e "  1. PDF Processing:  ${GREEN}DONE${NC}"
echo -e "  2. Planning:        ${GREEN}DONE${NC}"
echo -e "  3. Analysis:        ${GREEN}DONE${NC}"
echo -e "  4. Code Generation: ${GREEN}DONE${NC}"
echo -e "  5. Packaging:       ${GREEN}DONE${NC}"
echo
echo -e "All outputs saved to: ${YELLOW}$OUTPUT_DIR/${NC}"
echo -e "Final repository: ${YELLOW}$OUTPUT_DIR/repo/${NC}"
echo -e "Packaged ZIP: ${YELLOW}$OUTPUT_DIR/${REPO_NAME}.zip${NC}"
echo
echo -e "To explore the generated code, run:"
echo -e "  ${BLUE}ls -la $OUTPUT_DIR/repo/${NC}"
echo -e "To view a specific file from the repository:"
echo -e "  ${BLUE}cat $OUTPUT_DIR/repo/[filename]${NC}"
echo
echo -e "${GREEN}Pipeline test completed successfully!${NC}"