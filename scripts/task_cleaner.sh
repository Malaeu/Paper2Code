#!/bin/bash
# Automatic cleaner script for hanging tasks
# Recommended to run this as a cron job every 5 minutes

TASKS_DIR="../tasks"
LOG_FILE="../logs/task_cleaner.log"

# Create logs directory if it doesn't exist
mkdir -p "../logs"

echo "$(date): Running task cleaner..." >> "$LOG_FILE"

for f in $TASKS_DIR/*.md; do
  if [ -f "$f" ]; then
    # Extract timeout and status using grep
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
      echo "$(date): Task $f has hung for ${age}s (timeout: ${timeout}s)" >> "$LOG_FILE"
      
      # Get task details for logging
      task_id=$(basename "$f" | sed 's/\(.*\)-.*\.md/\1/')
      agent=$(grep -P '(?<=agent: )[a-z_]+' "$f" | head -1)
      
      # Update status to error using sed
      sed -i 's/status: running/status: error/' "$f"
      
      # Add error message
      error_msg="Task timed out after ${age}s (exceeded timeout of ${timeout}s)"
      if grep -q "error_message:" "$f"; then
        # Update existing error message
        sed -i "s/error_message:.*/error_message: \"$error_msg\"/" "$f"
      else
        # Add new error message before second "---"
        sed -i "/---/!b;n;1,2{/---/i\\error_message: \"$error_msg\"\\
}" "$f"
      fi
      
      # Remove lock file if it exists
      if [ -f "${f}.lock" ]; then
        rm -f "${f}.lock"
        echo "$(date): Removed lock file for $task_id" >> "$LOG_FILE"
      fi
      
      echo "$(date): Marked task $task_id ($agent) as error" >> "$LOG_FILE"
    fi
  fi
done

echo "$(date): Task cleaner completed" >> "$LOG_FILE"