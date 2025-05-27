#!/bin/bash
# Script to find hanging tasks that have exceeded their timeout

TASKS_DIR="../tasks"

echo "Looking for hanging tasks..."

for f in $TASKS_DIR/*.md; do
  if [ -f "$f" ]; then
    # Extract timeout and status using grep to avoid dependencies
    timeout=$(grep -P '(?<=timeout: )[0-9]+' "$f" | head -1)
    status=$(grep -P '(?<=status: )[a-z]+' "$f" | head -1)
    
    # Default timeout if not found
    if [ -z "$timeout" ]; then
      timeout=600
    fi
    
    # Skip if not running
    if [ "$status" != "running" ]; then
      continue
    fi
    
    # Calculate age in seconds
    file_mod_time=$(stat -c %Y "$f")
    current_time=$(date +%s)
    age=$((current_time - file_mod_time))
    
    # Check if exceeded timeout
    if [ "$age" -gt "$timeout" ]; then
      echo "HANGING: $f (age: ${age}s, timeout: ${timeout}s)"
      
      # Check for lock file
      if [ -f "${f}.lock" ]; then
        echo "  Lock file exists: ${f}.lock"
      fi
      
      # Look for stderr logs
      task_id=$(basename "$f" | sed 's/\(.*\)-.*\.md/\1/')
      stderr_log="../logs/${task_id}.stderr"
      
      if [ -f "$stderr_log" ]; then
        echo "  Log file: $stderr_log"
        echo "  Last 5 lines of stderr:"
        tail -n 5 "$stderr_log" | sed 's/^/    /'
      else
        echo "  No stderr log found"
      fi
      
      echo ""
    fi
  fi
done

echo "Done checking for hanging tasks."