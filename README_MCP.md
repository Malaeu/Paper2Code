# Task Management System for Paper2Code

This directory contains a Multi-Agent Coordination Protocol (MCP) implementation for Paper2Code, allowing different agents to work together on tasks with dependencies.

## Overview

The task management system coordinates work between different types of agents:
- `vision`: Handles image processing tasks (Claude CLI)
- `lit_review`: Performs literature review tasks (Claude CLI)
- `chat`: General chat interactions (Claude CLI)
- `code_patch`: Code editing tasks (Codex CLI)
- `code_auto`: Fully automated code generation (Codex CLI)

## Setup

1. Install dependencies:
```bash
pip install pyyaml
npm i -g @openai/codex@latest  # For latest Codex CLI with bug fixes
```

2. Pull Docker images to avoid hanging tasks:
```bash
./scripts/setup_docker.sh
```

## Usage

1. Create a workflow JSON file (see `tasks/enhanced_workflow.json` for an example)

2. Convert workflow to task files:
```bash
python scripts/mcp_to_md.py -i tasks/enhanced_workflow.json -o tasks/
```

3. Run the task management system:
```bash
./scripts/run_enhanced.sh
```

## Fixing Common Codex Hanging Issues

If Codex CLI is hanging for 400-500 seconds, check these common issues:

| Issue | Symptom in stderr | Solution |
|-------|-------------------|----------|
| Model queue | "Waiting for available slot..." | Limit parallel tasks (MAX_PAR=6 in watcher_codex.py) |
| Token delay | Token-by-token output | Update Codex CLI: `npm i -g @openai/codex@latest` |
| Docker image | "Pulling seatbelt image..." | Run `./scripts/setup_docker.sh` to pre-pull images |
| Hanging tests | Endless pytest | Add `--pytest-timeout 120` (already configured) |
| Rate limits | "rate_limit_exceeded" | Auto-fallback to o4-mini (already configured) |
| CLI bugs | Exit code ≠0 | Auto-retry with `retries: 2` in task frontmatter |

## Monitoring Tasks

To check for hanging tasks:
```bash
./scripts/find_hanging_tasks.sh
```

For automatic cleanup of hanging tasks, run the cleaner periodically:
```bash
./scripts/task_cleaner.sh
```

You can set this up as a cron job:
```
*/5 * * * * cd /path/to/Paper2Code && ./scripts/task_cleaner.sh
```

## Recommended Task Configuration

Include these in your task frontmatter for robustness:

```yaml
---
id: example_task
agent: code_patch
timeout: 600  # 10 minutes max
retries: 2    # Retry up to 2 times
parallel_ok: true  # Allow parallel execution with other tasks
---
```

## Extended Image Processing Pipeline

The enhanced workflow includes additional steps for PDF processing:

1. `page2png`: Converts PDF to high-quality images
2. `vision_annot`: Detects figures/tables and extracts metadata
3. `figure_coder`: Generates matplotlib code to reproduce figures
4. `semantic_linker`: Links text passages to figures/tables
5. `json_merger`: Creates enriched JSON for Paper2Code
6. `paper2code`: Runs the main Paper2Code pipeline with enhanced data

This allows for better figure reproduction and data extraction from papers.