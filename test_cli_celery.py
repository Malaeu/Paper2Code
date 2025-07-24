#!/usr/bin/env python3
"""
Simple CLI test for Celery without requiring the actual application
"""

import os
import time
import json
from celery import Celery

# Create simple Celery app
app = Celery('cli_test')
app.conf.broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.conf.result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

@app.task
def add(x, y):
    """Simple addition task"""
    time.sleep(1)  # Simulate work
    result = x + y
    print(f"Task executed: {x} + {y} = {result}")
    return result

@app.task
def echo(message):
    """Echo a message"""
    print(f"Task received message: {message}")
    return {"message": message, "success": True}

# Direct execution test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [worker|run]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "worker":
        print("Starting worker...")
        app.worker_main(["worker", "--loglevel=info", "-n", "cli_test_worker@%h"])
    
    elif command == "run":
        print("Submitting tasks...")
        result1 = add.delay(4, 5)
        result2 = echo.delay("Hello from CLI test")
        
        print(f"Task IDs: {result1.id}, {result2.id}")
        print("Waiting for results...")
        
        try:
            add_result = result1.get(timeout=5)
            echo_result = result2.get(timeout=5)
            
            print("\nResults:")
            print(f"Addition: {add_result}")
            print(f"Echo: {json.dumps(echo_result, indent=2)}")
            print("\nAll tasks completed successfully!")
        except Exception as e:
            print(f"\nError: {e}")
            print("\nMake sure a worker is running with: python test_cli_celery.py worker")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)