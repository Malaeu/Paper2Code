# Paper2Code Task Management System

This directory contains tasks for the Paper2Code Multi-Agent Coordination Protocol (MCP) system. The MCP system coordinates multiple AI agents to process academic papers and implement the described techniques in code.

## Task Types

The system supports five types of agent tasks:

1. `vision`: Analyzes PDF documents and images to extract information
2. `lit_review`: Conducts literature review and research on specified topics
3. `chat`: Engages in dialogue to clarify requirements or provide explanations
4. `code_patch`: Implements specific code changes or fixes
5. `code_auto`: Automatically generates code, tests, or documentation

## Task Workflow

Tasks are processed in the following way:

1. JSON task specifications are converted to Markdown files using `mcp_to_md.py`
2. Task watchers (`watcher_claude.py` and `watcher_codex.py`) monitor the tasks directory
3. When a task's dependencies are satisfied, the appropriate watcher processes it
4. Results are stored in the task's Markdown file and in separate stdout/stderr files

## Task File Format

Each task is represented by a Markdown file with YAML frontmatter:

```markdown
---
id: task_id
agent: agent_type
status: pending|running|done|error
created_at: timestamp
started_at: timestamp  # Added when task starts running
completed_at: timestamp  # Added when task completes
parallel_ok: true|false
timeout: seconds
depends: [task_id_1, task_id_2, ...]
priority: low|normal|high
files: [file1.py, file2.py]  # For code_patch and code_auto agents
---

# PROMPT
Task description/prompt goes here...

# CONTEXT
Optional context information...

# COMMAND
```bash
Optional command to run
```

# RESULT
Task results will be added here...
```

## Using the System

### 1. Creating Tasks

Create JSON task specifications and convert them to Markdown:

```bash
python3 scripts/mcp_to_md.py -i tasks/example_workflow.json -o tasks/
```

### 2. Running Task Watchers

Start the task watchers in separate terminals:

```bash
# Terminal 1
python3 scripts/watcher_claude.py

# Terminal 2
python3 scripts/watcher_codex.py
```

### 3. Monitoring Tasks

Check the status of tasks by examining the Markdown files and log files.

## Example

See the `example_workflow.json` file for a sample workflow that demonstrates the task dependencies and different agent types.