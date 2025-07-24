#!/usr/bin/env python3
"""
Test script for Celery
"""

import os
import sys
import time
from datetime import datetime
from celery import Celery

# Configure Celery app
app = Celery('test_app')
app.conf.broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.conf.result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

@app.task(bind=True)
def debug_task(self):
    """Simple test task that returns time info"""
    print(f'Request: {self.request!r}')
    return {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'message': 'Celery is working!'
    }

@app.task(bind=True)
def add(self, x, y):
    """Simple addition task with a delay"""
    print(f'Adding {x} + {y}')
    # Simulate some work
    time.sleep(2)
    result = x + y
    print(f'Result: {result}')
    return result

if __name__ == '__main__':
    # Run a test task when script is executed directly
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        # Start a worker
        print("Starting Celery worker...")
        argv = ['worker', '--loglevel=info', '-n', 'test_worker@%h']
        app.worker_main(argv)
    else:
        # Run test tasks
        print("Submitting test tasks...")
        
        debug_result = debug_task.delay()
        print(f"Debug task ID: {debug_result.id}")
        
        add_result = add.delay(4, 4)
        print(f"Addition task ID: {add_result.id}")
        
        # Wait for results
        print("\nWaiting for results...")
        try:
            debug_data = debug_result.get(timeout=5)
            print(f"Debug task result: {debug_data}")
            
            add_data = add_result.get(timeout=5)
            print(f"Addition task result: {add_data}")
            
            print("\n✅ All tasks completed successfully!")
        except Exception as e:
            print(f"\n❌ Error getting task results: {e}")
            print("Make sure a Celery worker is running in another terminal with:")
            print("cd webapp && python test_celery.py worker")