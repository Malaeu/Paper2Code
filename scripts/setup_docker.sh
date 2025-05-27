#!/bin/bash
# Script to pre-pull Docker images needed by Codex CLI

echo "Pre-pulling Docker images for Codex CLI to avoid hanging during tasks..."

# Pull the seatbelt container image used for sandboxed execution
docker pull ghcr.io/openai/seatbelt:latest

# Check if the pull was successful
if [ $? -eq 0 ]; then
    echo "✅ Successfully pulled seatbelt image"
else
    echo "❌ Failed to pull seatbelt image. Make sure Docker is running and you have internet access."
    exit 1
fi

echo ""
echo "Codex CLI setup is complete. You should no longer experience delays from pulling container images."
echo "For best performance:"
echo "1. Limit parallel Codex tasks to 6 (already configured in watcher_codex.py)"
echo "2. Use timeouts and retries in task YAML front-matter:"
echo "   ---"
echo "   timeout: 600  # 10 minutes"
echo "   retries: 2"
echo "   ---"
echo "3. Update Codex CLI to the latest version:"
echo "   npm i -g @openai/codex@latest"
echo ""