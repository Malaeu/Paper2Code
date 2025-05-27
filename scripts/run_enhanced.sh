#!/bin/bash
# Enhanced workflow script that includes PDF processing, figure extraction, and enriched JSON generation

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure all required directories exist
mkdir -p ../tasks
mkdir -p ../outputs/Transformer/snippets

# Check if a workflow file was provided
if [ $# -eq 1 ]; then
    WORKFLOW_JSON="$1"
    if [ ! -f "$WORKFLOW_JSON" ]; then
        echo "Error: Workflow file not found: $WORKFLOW_JSON"
        exit 1
    fi
else
    # Use the enhanced workflow
    WORKFLOW_JSON="../tasks/enhanced_workflow.json"
    echo "No workflow file provided, using enhanced workflow: $WORKFLOW_JSON"
fi

# Convert workflow JSON to task files
echo "Converting workflow JSON to task files"
python3 ./mcp_to_md.py -i "$WORKFLOW_JSON" -o ../tasks

# Start task watchers in the background
echo "Starting task watchers..."

# Start Claude watcher
python3 ./watcher_claude.py > ../claude_watcher.log 2>&1 &
CLAUDE_PID=$!
echo "Claude watcher started with PID $CLAUDE_PID"

# Start Codex watcher
python3 ./watcher_codex.py > ../codex_watcher.log 2>&1 &
CODEX_PID=$!
echo "Codex watcher started with PID $CODEX_PID"

echo ""
echo "Enhanced workflow is running with PDF processing and figure extraction."
echo "Task watchers are running. Press Ctrl+C to stop."
echo "Monitor progress in log files:"
echo "  - ../claude_watcher.log"
echo "  - ../codex_watcher.log"
echo ""
echo "Output files will be generated in ../outputs/Transformer/"
echo "  - vision.json: Table/figure detection results"
echo "  - snippets/*.py: Generated matplotlib code"
echo "  - passages.json: Semantic links between text and figures"
echo "  - enriched.json: Combined paper2data format"

# Function to clean up background processes
cleanup() {
    echo ""
    echo "Stopping task watchers..."
    kill $CLAUDE_PID $CODEX_PID 2>/dev/null
    wait $CLAUDE_PID $CODEX_PID 2>/dev/null
    echo "Task watchers stopped."
    exit 0
}

# Register cleanup function on script termination
trap cleanup INT TERM

# Wait until Ctrl+C is pressed
while true; do
    sleep 1
done