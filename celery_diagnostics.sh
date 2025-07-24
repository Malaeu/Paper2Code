#!/bin/bash

# Script for diagnosing Celery issues

echo "Running Celery diagnostics..."
echo "============================="

# Activate virtual environment
source .venv_env/bin/activate

# Set Redis broker URL
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Directory for storing test results
DIAG_DIR="/tmp/celery_diagnostics"
mkdir -p $DIAG_DIR

# Check Redis server status
echo "1. Checking Redis server status..."
if redis-cli ping | grep -q PONG; then
    echo "✅ Redis server is running"
else
    echo "❌ Redis server is not running. Please start Redis with: sudo service redis-server start"
    exit 1
fi

# Check Celery worker status
echo -e "\n2. Checking Celery worker status..."
cd webapp
WORKER_STATUS=$(celery -A app:celery status 2>&1)
if echo "$WORKER_STATUS" | grep -q "1 node online"; then
    echo "✅ Celery worker is running"
else
    echo "❌ Celery worker is not running or has issues"
    echo "Worker status output:"
    echo "$WORKER_STATUS"
fi

# Check for registered tasks
echo -e "\n3. Checking registered tasks..."
REGISTERED_TASKS=$(celery -A app:celery inspect registered 2>&1)
if echo "$REGISTERED_TASKS" | grep -q "task"; then
    echo "✅ Found registered tasks:"
    echo "$REGISTERED_TASKS" | grep -A 20 "task" | head -10
else
    echo "❌ No registered tasks found"
    echo "Registered tasks output:"
    echo "$REGISTERED_TASKS"
fi

# Inspect active tasks
echo -e "\n4. Checking active tasks..."
ACTIVE_TASKS=$(celery -A app:celery inspect active 2>&1)
echo "Active tasks output:"
echo "$ACTIVE_TASKS"

# Test a simple task
echo -e "\n5. Testing a simple Celery task..."
cat > $DIAG_DIR/test_task.py << EOL
from celery import Celery
import time
import os

app = Celery('test_app', 
             broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
             backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))

@app.task
def add(x, y):
    time.sleep(1)  # Simulate some work
    return x + y

# Run the task
if __name__ == '__main__':
    result = add.delay(4, 4)
    print(f"Task ID: {result.id}")
    print("Waiting for result...")
    try:
        result_value = result.get(timeout=5)
        print(f"Result: {result_value}")
        if result_value == 8:
            print("✅ Test task completed successfully")
        else:
            print(f"❌ Test task returned unexpected result: {result_value}")
    except Exception as e:
        print(f"❌ Error getting task result: {e}")
EOL

cd $DIAG_DIR
echo "Running test task..."
python test_task.py

# Check Redis queue content
echo -e "\n6. Checking Redis queue content..."
echo "Redis keys:"
redis-cli keys "*"

echo -e "\n7. Checking Redis results backend..."
RESULT_KEYS=$(redis-cli keys "celery-task-meta-*")
if [ -n "$RESULT_KEYS" ]; then
    echo "✅ Found task results in Redis"
    echo "Sample result:"
    for key in $RESULT_KEYS; do
        redis-cli get "$key" | head -50
        break  # Just show one sample
    done
else
    echo "❌ No task results found in Redis"
fi

echo -e "\nDiagnostics complete!"
echo "If you see issues, check the following:"
echo "1. Make sure Redis server is running"
echo "2. Make sure Celery worker is running with: cd webapp && celery -A app:celery worker --loglevel=info"
echo "3. Check that your app is properly configured with Celery"
echo "4. Ensure task definitions use the @celery.task decorator"
echo "5. Verify that all required Python packages are installed"
echo "6. Make sure environment variables like OPENAI_API_KEY are set"