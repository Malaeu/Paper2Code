#!/usr/bin/env python3
"""
Watcher for Claude-CLI tasks: scans tasks/*.md for agent: vision|lit_review|chat,
status: pending, checks dependencies, locks the task, sets status to running,
runs claude-cli, captures output, updates status to done/error, and releases lock.
"""
import os
import time
import subprocess
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("claude_watcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watcher_claude")

# Directory containing markdown tasks (one level up from scripts/)
TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tasks'))
LOCK_SUFFIX = '.lock'
SLEEP_INTERVAL = 5.0  # seconds between scans

# Claude agents supported by this watcher
SUPPORTED_AGENTS = ('vision', 'lit_review', 'chat')

def read_frontmatter(path):
    """Read YAML frontmatter and body from a markdown file."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines or lines[0].strip() != '---':
        return {}, ''.join(lines)
    for idx in range(1, len(lines)):
        if lines[idx].strip() == '---':
            end = idx
            break
    else:
        return {}, ''.join(lines)
    fm = yaml.safe_load(''.join(lines[1:end])) or {}
    body = ''.join(lines[end+1:])
    return fm, body

def write_frontmatter(path, fm, body):
    """Write YAML frontmatter and body back to the markdown file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('---\n')
        yaml.safe_dump(fm, f, sort_keys=False, allow_unicode=True)
        f.write('---\n')
        f.write(body)

def check_dependencies(fm):
    """Check if all dependencies are satisfied (status is 'done')."""
    depends = fm.get('depends', [])
    if not depends:
        return True
    
    for dep_id in depends:
        # Find task file with this ID
        for fname in os.listdir(TASK_DIR):
            if not fname.endswith('.md'):
                continue
            
            dep_path = os.path.join(TASK_DIR, fname)
            dep_fm, _ = read_frontmatter(dep_path)
            
            if dep_fm.get('id') == dep_id:
                if dep_fm.get('status') != 'done':
                    logger.info(f"Dependency {dep_id} not yet satisfied (status: {dep_fm.get('status')})")
                    return False
                break
        else:
            logger.warning(f"Dependency {dep_id} not found")
            return False
    
    return True

def process_task(path):
    try:
        fm, body = read_frontmatter(path)
        # only process pending claude agent tasks
        if fm.get('agent') not in SUPPORTED_AGENTS or fm.get('status') != 'pending':
            return False
        
        # Check if dependencies are satisfied
        if not check_dependencies(fm):
            logger.info(f"Skipping task {fm.get('id')} due to unsatisfied dependencies")
            return False
        
        # Check for parallel_ok flag
        if not fm.get('parallel_ok', False):
            # Check if any other tasks are running
            for fname in os.listdir(TASK_DIR):
                if not fname.endswith('.md'):
                    continue
                other_path = os.path.join(TASK_DIR, fname)
                if other_path == path:
                    continue
                other_fm, _ = read_frontmatter(other_path)
                if other_fm.get('status') == 'running':
                    logger.info(f"Skipping task {fm.get('id')} because another task is running and parallel_ok=False")
                    return False
        
        lockfile = path + LOCK_SUFFIX
        try:
            open(lockfile, 'w').close()
        except IOError:
            logger.warning(f"Could not acquire lock for {path}")
            return False
        
        try:
            logger.info(f"Processing task {fm.get('id')} with agent {fm.get('agent')}")
            fm['status'] = 'running'
            fm['started_at'] = datetime.now().isoformat()
            write_frontmatter(path, fm, body)
            
            # Prepare command based on agent type
            cmd = ['claude', '--model', 'sonnet', 'run']
            
            # Add agent-specific flags
            if fm.get('agent') == 'vision':
                cmd.append('--vision')
            elif fm.get('agent') == 'lit_review':
                cmd.append('--lit-review')
            
            # Add timeout if specified
            if 'timeout' in fm:
                cmd.extend(['--timeout', str(fm['timeout'])])
                
            # Add the task file
            cmd.append(path)
            
            # Run the command
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # Save outputs
            with open(path + '.stdout', 'w', encoding='utf-8') as out_f:
                out_f.write(result.stdout)
            with open(path + '.stderr', 'w', encoding='utf-8') as err_f:
                err_f.write(result.stderr)
            
            # Update task status
            fm['status'] = 'done' if result.returncode == 0 else 'error'
            fm['completed_at'] = datetime.now().isoformat()
            fm['exit_code'] = result.returncode
            
            write_frontmatter(path, fm, body)
            logger.info(f"Task {fm.get('id')} completed with status {fm['status']}")
            return True
        finally:
            try:
                os.remove(lockfile)
            except OSError:
                logger.warning(f"Failed to remove lock file {lockfile}")
    except Exception as e:
        logger.exception(f"Error processing task {path}: {e}")
    
    return False

def main():
    logger.info("Starting Claude task watcher")
    os.makedirs(TASK_DIR, exist_ok=True)
    
    while True:
        task_count = 0
        processed_count = 0
        
        for fname in os.listdir(TASK_DIR):
            if not fname.endswith('.md'):
                continue
            
            task_count += 1
            full_path = os.path.join(TASK_DIR, fname)
            
            if process_task(full_path):
                processed_count += 1
        
        if task_count > 0:
            logger.debug(f"Scanned {task_count} tasks, processed {processed_count}")
        
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()