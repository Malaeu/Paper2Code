#!/bin/bash

# Paper2Code Pipeline with Claude 4 (Sonnet/Opus)
# Make sure ANTHROPIC_API_KEY is set in your environment

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable not set"
    echo "Please run: export ANTHROPIC_API_KEY='your_api_key'"
    exit 1
fi

PAPER_NAME="Transformer"
PDF_PATH="../examples/Transformer.pdf" # .pdf
PDF_JSON_PATH="../examples/Transformer.json" # .json
PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Transformer"
OUTPUT_REPO_DIR="../outputs/Transformer_repo"

# Claude model settings
PLANNING_MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # For planning stage
ANALYSIS_MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"   # For analysis stage  
CODING_MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"     # For coding stage

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \

echo "------- PaperCoder with Claude LLM -------"

# Planning stage with LLM
python ../codes/1_planning_llm.py \
    --paper_name $PAPER_NAME \
    --model_name ${PLANNING_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

# Extract configuration
python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

# Analysis stage with LLM
python ../codes/2_analyzing_llm.py \
    --paper_name $PAPER_NAME \
    --model_name ${ANALYSIS_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

# Coding stage with LLM
python ../codes/3_coding_llm.py  \
    --paper_name $PAPER_NAME \
    --model_name ${CODING_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \

# MCP Task Management System with Claude 4
echo "------- MCP Task Management System with Claude 4 -------"

# Ensure tasks directory exists
mkdir -p ../tasks

# Check if a JSON workflow file was provided
if [ -n "$WORKFLOW_JSON" ] && [ -f "$WORKFLOW_JSON" ]; then
    echo "Converting workflow JSON to task files: $WORKFLOW_JSON"
    python3 ./mcp_to_md.py -i "$WORKFLOW_JSON" -o ../tasks
else
    # If no workflow provided, use the example workflow
    echo "Using example workflow from tasks/example_workflow.json"
    python3 ./mcp_to_md.py -i ../tasks/example_workflow.json -o ../tasks
fi

# Start task watchers in the background
echo "Starting Claude 4 task watchers..."
python3 ./watcher_claude.py > ../claude_watcher.log 2>&1 &
CLAUDE_PID=$!
echo "Claude watcher (Sonnet) started with PID $CLAUDE_PID"

python3 ./watcher_codex.py > ../claude_code_watcher.log 2>&1 &
CODEX_PID=$!
echo "Claude Code watcher (Opus) started with PID $CODEX_PID"

echo "Task watchers are running with Claude 4 models. Check logs for progress."
echo "- claude_watcher.log: Sonnet for vision/chat/lit_review"
echo "- claude_code_watcher.log: Opus for code_patch/code_auto"
echo "To stop watchers, run: kill $CLAUDE_PID $CODEX_PID"

# Register trap to kill background processes on script exit
trap "kill $CLAUDE_PID $CODEX_PID 2>/dev/null" EXIT

# Keep script running to monitor watchers
echo "Pipeline started. Press Ctrl+C to stop all watchers."
wait