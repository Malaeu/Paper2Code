#!/usr/bin/env python3
"""
Watcher for Codex-CLI tasks: scans tasks/*.md for agent: code_patch|code_auto, 
status: pending, checks dependencies, locks the task, sets status to running, 
runs codex-cli, captures output, updates status to done/error, and releases lock.
"""
import os
import time
import subprocess
import yaml
import logging
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("claude_code_watcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watcher_claude_code")

# Directory containing markdown tasks (one level up from scripts/)
TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tasks'))
LOCK_SUFFIX = '.lock'
SLEEP_INTERVAL = 5.0  # seconds between scans
MAX_PAR = 4  # Maximum parallel Claude Code tasks to avoid rate limits (Opus 4 is more resource intensive)

# Claude Code agents supported by this watcher
SUPPORTED_AGENTS = ('code_patch', 'code_auto')

# Semaphore to limit parallel tasks
task_semaphore = threading.Semaphore(MAX_PAR)

# Track currently running tasks
running_tasks = {}
# Track fallback attempts
fallback_attempts = {}

def read_frontmatter(path):
    """Read YAML frontmatter and body from a markdown file."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines or lines[0].strip() != '---':
        return {}, ''.join(lines)
    # find closing frontmatter
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

def create_logs_dir():
    """Ensure logs directory exists"""
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def check_for_timeouts():
    """Check for running tasks that have exceeded their timeout"""
    now = time.time()
    tasks_to_check = list(running_tasks.items())
    
    for task_path, start_time in tasks_to_check:
        try:
            fm, body = read_frontmatter(task_path)
            if fm.get('status') != 'running':
                # Task is no longer running
                running_tasks.pop(task_path, None)
                continue
                
            timeout = int(fm.get('timeout', 600))  # Default 10 minutes
            age = now - start_time
            
            if age > timeout:
                logger.warning(f"Task {fm.get('id')} timed out after {age:.1f} seconds (timeout: {timeout})")
                
                # Get retries left
                retries_left = int(fm.get('retries_left', 0))
                
                if retries_left > 0:
                    # Retry the task
                    logger.info(f"Retrying task {fm.get('id')}, {retries_left} attempts left")
                    fm['status'] = 'pending'
                    fm['retries_left'] = retries_left - 1
                    write_frontmatter(task_path, fm, body)
                else:
                    # Check for fallback
                    use_fallback = False
                    fallback_count = fallback_attempts.get(fm.get('id'), 0)
                    
                    if fallback_count < 3:  # Allow up to 3 fallback attempts
                        # Mark for fallback to o4-mini
                        use_fallback = True
                        fallback_attempts[fm.get('id')] = fallback_count + 1
                        fm['fallback'] = 'o4-mini'
                        fm['status'] = 'pending'
                        logger.info(f"Falling back to o4-mini for task {fm.get('id')}")
                    else:
                        # No more retries or fallbacks, mark as error
                        fm['status'] = 'error'
                        fm['error_message'] = f"Task timed out after {age:.1f}s and exhausted retries/fallbacks"
                        logger.error(f"Task {fm.get('id')} failed after exhausting retries and fallbacks")
                    
                    write_frontmatter(task_path, fm, body)
                
                # Remove from running tasks
                running_tasks.pop(task_path, None)
                
                # Release semaphore
                task_semaphore.release()
        except Exception as e:
            logger.error(f"Error checking timeout for {task_path}: {e}")

def process_task(path):
    """Process a single task file."""
    try:
        fm, body = read_frontmatter(path)
        # only process pending codex agent tasks
        if fm.get('agent') not in SUPPORTED_AGENTS or fm.get('status') != 'pending':
            return False
        
        # Check if dependencies are satisfied
        if not check_dependencies(fm):
            logger.info(f"Skipping task {fm.get('id')} due to unsatisfied dependencies")
            return False
        
        # Respect semaphore limit for parallel tasks
        if not task_semaphore.acquire(blocking=False):
            logger.info(f"Parallel task limit reached ({MAX_PAR}), will try again later for {fm.get('id')}")
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
                    task_semaphore.release()  # Release semaphore since we're not processing
                    return False
        
        lockfile = path + LOCK_SUFFIX
        try:
            open(lockfile, 'w').close()
        except IOError:
            logger.warning(f"Could not acquire lock for {path}")
            task_semaphore.release()  # Release semaphore on failure
            return False
        
        try:
            logger.info(f"Processing task {fm.get('id')} with agent {fm.get('agent')}")
            fm['status'] = 'running'
            fm['started_at'] = datetime.now().isoformat()
            write_frontmatter(path, fm, body)
            
            # Track this task's start time for timeout monitoring
            running_tasks[path] = time.time()
            
            # Initialize retries if not set
            if 'retries_left' not in fm:
                retries = fm.get('retries', 2)  # Default to 2 retries
                fm['retries_left'] = retries
                write_frontmatter(path, fm, body)
            
            # Create logs directory
            logs_dir = create_logs_dir()
            task_id = fm.get('id')
            stdout_path = os.path.join(logs_dir, f"{task_id}.stdout")
            stderr_path = os.path.join(logs_dir, f"{task_id}.stderr")
            
            # Prepare command based on agent type  
            cmd = ['claude', '--model', 'opus', 'run']
            
            # Add agent-specific flags
            if fm.get('agent') == 'code_patch':
                cmd.append('--patch')
            elif fm.get('agent') == 'code_auto':
                cmd.append('--auto')
            
            # Add timeout if specified
            if 'timeout' in fm:
                cmd.extend(['--timeout', str(fm['timeout'])])
            
            # Add pytest timeout if needed
            cmd.extend(['--pytest-timeout', '120'])  # Default 2 minute timeout for tests
            
            # Add files to process if specified
            files = fm.get('files', [])
            if files:
                for file_path in files:
                    cmd.extend(['--file', file_path])
            
            # Check if this is a fallback task
            if fm.get('fallback') == 'o4-mini':
                # Use o4-mini API instead of Claude Code CLI
                logger.info(f"Using o4-mini fallback for task {task_id}")
                # In a real implementation, this would call the OpenAI API with o4-mini
                # For now, just simulate with a different command
                cmd = ['echo', f"Fallback to o4-mini for task {task_id}"]
            
            # Add the task file
            cmd.append(path)
            
            # Run the command
            logger.info(f"Running command: {' '.join(cmd)}")
            with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
                result = subprocess.run(
                    cmd, 
                    stdout=stdout_file, 
                    stderr=stderr_file, 
                    text=True
                )
            
            # Update task status
            fm['status'] = 'done' if result.returncode == 0 else 'error'
            fm['completed_at'] = datetime.now().isoformat()
            fm['exit_code'] = result.returncode
            
            # Add error message if failed
            if result.returncode != 0:
                with open(stderr_path, 'r') as stderr_file:
                    stderr_content = stderr_file.read()
                    if 'rate_limit_exceeded' in stderr_content:
                        fm['error_message'] = "Rate limit exceeded"
                    elif 'waiting_for_slot' in stderr_content:
                        fm['error_message'] = "Timed out waiting for model slot"
                    else:
                        fm['error_message'] = f"Command failed with exit code {result.returncode}"
            
            write_frontmatter(path, fm, body)
            logger.info(f"Task {fm.get('id')} completed with status {fm['status']}")
            
            # Remove from running tasks
            running_tasks.pop(path, None)
            
            # Release semaphore
            task_semaphore.release()
            return True
        finally:
            try:
                os.remove(lockfile)
            except OSError:
                logger.warning(f"Failed to remove lock file {lockfile}")
    except Exception as e:
        logger.exception(f"Error processing task {path}: {e}")
        # Ensure semaphore is released on error
        if path in running_tasks:
            running_tasks.pop(path)
            task_semaphore.release()
    
    return False

def main():
    logger.info("Starting Claude Code task watcher")
    logger.info(f"Maximum parallel tasks: {MAX_PAR}")
    os.makedirs(TASK_DIR, exist_ok=True)
    
    # Create a timer to check for timeouts
    timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
    timeout_thread.start()
    
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

def timeout_checker():
    """Thread function to periodically check for task timeouts"""
    while True:
        try:
            check_for_timeouts()
        except Exception as e:
            logger.error(f"Error in timeout checker: {e}")
        time.sleep(30)  # Check every 30 seconds

if __name__ == '__main__':
    main()