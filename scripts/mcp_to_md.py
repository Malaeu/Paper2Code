#!/usr/bin/env python3
"""
Convert MCP JSON tasks into Markdown files for the Paper2Code task management system.
This script converts JSON task specifications to Markdown files with YAML frontmatter
for agent coordination between vision, lit_review, chat, code_patch, and code_auto agents.
"""

import os
import yaml
import tempfile
import unicodedata
import re
import json
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_to_md")

# List of supported agent types
SUPPORTED_AGENTS = ('vision', 'lit_review', 'chat', 'code_patch', 'code_auto')

def slug(text):
    """Normalize text to an ASCII slug suitable for filenames."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()

def validate_task(task):
    """Validate that a task has all required fields."""
    if not task.get("id"):
        raise ValueError("Task must have an 'id' field")
    
    agent = task.get("agent")
    if not agent:
        raise ValueError(f"Task {task.get('id')} must have an 'agent' field")
    
    if agent not in SUPPORTED_AGENTS:
        logger.warning(f"Agent '{agent}' for task {task.get('id')} is not in supported agents list: {SUPPORTED_AGENTS}")
    
    return True

def mcp_node_to_md(node, out_dir):
    """Convert one MCP task node to a Markdown file with YAML front-matter."""
    # Validate the task
    validate_task(node)
    
    # Extract basic task information
    tid = str(node.get("id"))
    agent = node.get("agent")
    
    # Create frontmatter
    fm = {
        "id": tid,
        "agent": agent,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "parallel_ok": bool(node.get("parallel_ok", False)),
        "timeout": int(node.get("timeout", 1800)),
        "depends": node.get("depends", []),
        "priority": node.get("priority", "normal"),
    }
    
    # Add agent-specific fields
    if agent in ("code_patch", "code_auto"):
        fm["files"] = node.get("files", [])
    
    # Prepare the body content
    body = ""
    
    # Add description/prompt
    prompt = node.get("prompt", node.get("description", ""))
    if prompt:
        body += "# PROMPT\n" + prompt + "\n\n"
    
    # Add context if available
    context = node.get("context")
    if context:
        body += "# CONTEXT\n"
        if isinstance(context, dict) or isinstance(context, list):
            body += "```json\n" + json.dumps(context, indent=2) + "\n```\n\n"
        else:
            body += str(context) + "\n\n"
    
    # Add command if specified
    if node.get("command"):
        body += "# COMMAND\n"
        body += f"```bash\n{node['command']}\n```\n\n"
    
    # Add result placeholder
    body += "# RESULT\n"
    body += "Task has not been completed yet.\n"

    # Generate the filename
    filename = os.path.join(out_dir, f"{slug(tid)}-{agent}.md")
    
    # Write the file atomically to avoid races
    fd, tmp_path = tempfile.mkstemp(dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write("---\n")
            yaml.safe_dump(fm, tmp, sort_keys=False, allow_unicode=True)
            tmp.write("---\n")
            tmp.write(body)
        os.replace(tmp_path, filename)
        logger.info(f"Saved task markdown: {filename}")
        return filename
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    parser = argparse.ArgumentParser(
        description="Convert MCP JSON tasks to Markdown task files for Paper2Code."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to input MCP JSON file or directory containing JSON files.")
    parser.add_argument("-o", "--output_dir", default="tasks",
                        help="Directory to write Markdown tasks (default: tasks).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate tasks but don't write files.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_path = args.input
    
    # Check if input is a directory
    if os.path.isdir(input_path):
        logger.info(f"Processing all JSON files in directory: {input_path}")
        json_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.endswith('.json')]
        
        if not json_files:
            logger.warning(f"No JSON files found in {input_path}")
            return
        
        for json_file in json_files:
            process_json_file(json_file, args.output_dir, args.dry_run)
    
    # Otherwise assume it's a single file
    elif os.path.isfile(input_path):
        if not input_path.endswith('.json'):
            logger.warning(f"Input file does not have .json extension: {input_path}")
        
        process_json_file(input_path, args.output_dir, args.dry_run)
    
    else:
        logger.error(f"Input path does not exist: {input_path}")

def process_json_file(file_path, output_dir, dry_run=False):
    """Process a single JSON file containing tasks."""
    logger.info(f"Processing JSON file: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        # Handle both single task and array of tasks
        tasks = content if isinstance(content, list) else [content]
        
        if not tasks:
            logger.warning(f"No tasks found in {file_path}")
            return
        
        logger.info(f"Found {len(tasks)} tasks in {file_path}")
        
        for i, node in enumerate(tasks):
            try:
                validate_task(node)
                if not dry_run:
                    mcp_node_to_md(node, output_dir)
                else:
                    logger.info(f"[DRY RUN] Would create task file for: {node.get('id')}")
            except ValueError as e:
                logger.error(f"Error processing task {i} in {file_path}: {e}")
                continue
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
    except IOError as e:
        logger.error(f"IO error reading {file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")

if __name__ == "__main__":
    main()