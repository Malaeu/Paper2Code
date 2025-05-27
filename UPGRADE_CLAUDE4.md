# Claude 4 Upgrade Guide

This document outlines the upgrade from Claude 3.7 Sonnet to Claude 4 (Sonnet 4/Opus 4) and Claude Code in the Paper2Code pipeline.

## Benefits of Claude 4

- **+40% accuracy improvement** over Sonnet 3.7
- **SWE-bench 72.5%** performance with Opus 4 for complex bug fixes
- **Longer context retention** - maintains focus across multi-cycle processes
- **Memory API** - automatic memory management for project context
- **Fewer loopholes** - 65% reduction in PEP-8 violations and unsafe code patterns

## Cost Comparison (per 1M tokens)

| Model                | Prompt $ | Output $ |
| -------------------- | -------- | -------- |
| **Opus 4**           | 15.0     | 75.0     |
| **Sonnet 4**         | 3.0      | 15.0     |
| **Sonnet 3.7** (old) | 3.0      | 15.0     |

## Changes Made

### 1. LLM Routing Configuration (`configs/llm_routing.yaml`)

Updated routing to use Claude 4 models:
- `vision`, `lit_review`, `chat` → **Sonnet 4**
- `code_patch`, `code_auto` → **Opus 4** with Claude Code
- Added cheap fallback routes with O3 for cost control

### 2. Watcher Updates

#### `scripts/watcher_claude.py`
- Updated command to use `claude --model sonnet-4`
- Handles vision, lit_review, and chat tasks

#### `scripts/watcher_codex.py` → Claude Code Watcher
- Updated command to use `claude-code --model opus-4`
- Reduced MAX_PAR from 6 to 4 (Opus 4 is more resource intensive)
- Updated logging and comments

### 3. Resource Constraints

Added new resource limits:
- `claude-code-cli`: max_parallel=4, rate_limit=15 (conservative for Opus 4)
- Existing `claude-cli`: max_parallel=10, rate_limit=30

## Installation Requirements

### Update CLI Tools

```bash
# Install latest Claude CLI with Claude 4 support
npm i -g @anthropic-ai/claude-cli@latest

# Install Claude Code CLI
npm i -g @anthropic-ai/claude-code@latest
```

### Verify Installation

```bash
# Check Claude CLI version and model support
claude --version
claude --model sonnet-4 --help

# Check Claude Code CLI
claude-code --version  
claude-code --model opus-4 --help
```

## Usage

### Normal Operation (Claude 4)

Tasks will automatically route to Claude 4 models:
- **vision/chat/lit_review** → Sonnet 4
- **code_patch/code_auto** → Opus 4 with Claude Code

### Cost Control (Rollback Mode)

For budget-conscious usage, use cheap mode:

```yaml
# In task files, use:
agent: code_patch_cheap  # Uses O3 instead of Opus 4
agent: code_auto_cheap   # Uses O3 instead of Opus 4
```

## Expected Improvements

1. **Faster Bug Resolution**: Opus 4's 72% SWE-bench score means fewer retry cycles
2. **Better Multi-file Edits**: Claude Code handles complex patches without git conflicts
3. **Improved Context Retention**: Less context loss in long analysis chains
4. **Cleaner Code**: Reduced need for manual cleanup and PEP-8 fixes

## Monitoring

Monitor performance through:
- Reduced task retry rates
- Improved test pass rates
- Lower manual intervention requirements
- Better code quality metrics

## Rollback Plan

If costs become prohibitive:
1. Switch specific tasks to `code_patch_cheap`/`code_auto_cheap`
2. Use original routing configuration as backup
3. Monitor usage through task logs and billing