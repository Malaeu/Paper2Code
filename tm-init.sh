#!/bin/bash

# Initialize TaskMaster AI for Paper2Code project
echo "Initializing TaskMaster AI for Paper2Code..."

# Check if package.json exists, if not create one
if [ ! -f "package.json" ]; then
    echo "Creating package.json..."
    npm init -y
fi

# Install TaskMaster AI
npm install --save-dev task-master-ai

# Add scripts to package.json
npx json -I -f package.json -e 'this.scripts=Object.assign(this.scripts||{}, {
    "tm:start": "npx task-master-ai start",
    "tm:status": "npx task-master-ai status",
    "tm:analyze": "npx task-master-ai analyze",
    "tm:plan": "npx task-master-ai plan"
})'

# Create basic task configuration if it doesn't exist
if [ ! -f "taskmaster.json" ]; then
    echo "taskmaster.json already exists."
fi

echo "TaskMaster AI initialization complete!"
echo "To start using TaskMaster AI, run: npm run tm:start"