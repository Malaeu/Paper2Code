#!/bin/bash

# Comprehensive diagnostics for all Paper2Code components

echo "PAPER2CODE COMPONENT DIAGNOSTICS"
echo "================================"
echo "Running diagnostics on $(date)"
echo

# Activate virtual environment
source .venv_env/bin/activate

# Check Python version
echo "1. Python Environment"
echo "--------------------"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo

# Check Redis
echo "2. Redis Server"
echo "--------------"
if redis-cli ping | grep -q PONG; then
    echo "✅ Redis server is running"
    echo "Redis keys: $(redis-cli keys '*' | wc -l)"
    echo "Sample keys:"
    redis-cli keys '*' | head -5
else
    echo "❌ Redis server is NOT running"
fi
echo

# Check required packages
echo "3. Required Packages"
echo "-------------------"
PACKAGES=("celery" "redis" "flask" "openai" "flask_sqlalchemy" "flask_migrate" "flask_login" "werkzeug" "numpy" "pandas" "pyarrow")

for pkg in "${PACKAGES[@]}"; do
    VERSION=$(pip freeze | grep -i "$pkg" | sed 's/.*==//g')
    if [ -n "$VERSION" ]; then
        echo "✅ $pkg: $VERSION"
    else
        echo "❌ $pkg: Not installed"
    fi
done
echo

# Check Celery worker
echo "4. Celery Status"
echo "---------------"
echo "Starting a test Celery worker..."
celery -A test_cli_celery worker --detach --loglevel=INFO -n diagnostic_worker@%h &> /dev/null
sleep 2

if celery -A test_cli_celery status | grep -q "diagnostic_worker"; then
    echo "✅ Celery worker started successfully"
    echo "Worker status:"
    celery -A test_cli_celery status
else
    echo "❌ Celery worker not running correctly"
fi

# Stop the diagnostic worker
celery -A test_cli_celery control shutdown &> /dev/null
echo

# Test a basic Flask app
echo "5. Flask Framework"
echo "----------------"
echo "from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello from Paper2Code!'
if __name__ == '__main__':
    print('Flask app initialized successfully')
    print('Flask version:', app.version)" > /tmp/test_flask.py

FLASK_OUTPUT=$(python /tmp/test_flask.py)
if echo "$FLASK_OUTPUT" | grep -q "initialized successfully"; then
    echo "✅ Flask framework works correctly"
    echo "$FLASK_OUTPUT"
else
    echo "❌ Flask initialization failed"
    echo "$FLASK_OUTPUT"
fi
echo

# Check environment variables
echo "6. Environment Variables"
echo "----------------------"
ENV_VARS=("OPENAI_API_KEY" "CELERY_BROKER_URL" "CELERY_RESULT_BACKEND" "FLASK_APP" "FLASK_ENV")

for var in "${ENV_VARS[@]}"; do
    value="${!var}"
    if [ -n "$value" ]; then
        # Mask sensitive values
        if [ "$var" == "OPENAI_API_KEY" ]; then
            masked="${value:0:5}...${value: -5}"
            echo "✅ $var: $masked"
        else
            echo "✅ $var: $value"
        fi
    else
        echo "❌ $var: Not set"
    fi
done
echo

# Check file access
echo "7. Project Files"
echo "--------------"
FILES=("webapp/app.py" "webapp/minimal_app.py" "codes/utils.py" "codes/adapt_mapping.py")

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists ($(wc -l < "$file") lines)"
    else
        echo "❌ $file not found"
    fi
done
echo

# Check database
echo "8. Database"
echo "----------"
if [ -f "webapp/app.db" ]; then
    echo "✅ SQLite database exists (size: $(du -h webapp/app.db | cut -f1))"
    echo "Tables:"
    echo ".tables" | sqlite3 webapp/app.db 2>/dev/null || echo "  Could not query database"
else
    echo "❌ SQLite database not found"
fi
echo

echo "9. Disk Space"
echo "------------"
echo "Available disk space on project partition:"
df -h . | grep -v Filesystem
echo

echo "DIAGNOSTICS COMPLETE"
echo "===================="
echo "Recommendations:"
echo "1. For OpenAI API issues, check if your API key is correctly formatted and has not expired"
echo "2. For numpy/pandas compatibility issues, consider:"
echo "   - Using a separate virtual environment with numpy<2.0"
echo "   - Using minimal_app.py which works without pandas dependency"
echo "3. For Celery issues, make sure Redis is running and correctly configured"
echo
echo "For detailed logs, run: ./celery_diagnostics.sh"
echo "To monitor Celery: ./monitor_celery.sh"
echo "To run minimal app: ./start_minimal_app_with_celery.sh"