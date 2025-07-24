# Paper2Code Web Application

This web application provides a user-friendly interface for the Paper2Code system, allowing users to convert scientific papers into executable code. The application includes project export functionality, directory configuration management, and enhanced user features.

## Detailed Setup Guide

### System Requirements

1. **Linux System Dependencies**:
   ```bash
   # Install required system packages
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-pip python3-venv
   sudo apt-get install -y redis-server
   sudo apt-get install -y build-essential libssl-dev libffi-dev
   sudo apt-get install -y poppler-utils  # For PDF processing
   ```

2. **Python Version**:
   - Python 3.8 or newer is required
   - Check your version with: `python3 --version`

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Paper2Code.git
   cd Paper2Code
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r webapp/requirements.txt
   ```

4. **Start Redis Server**:
   Redis is required for the Celery task queue, which processes paper uploads and code generation.
   ```bash
   # Check if Redis is running:
   sudo systemctl status redis-server
   
   # If not running, start it:
   sudo systemctl start redis-server
   
   # To make Redis start on boot:
   sudo systemctl enable redis-server
   ```

5. **Environment Variables Setup**:
   Create a `.env` file in the webapp directory:
   ```bash
   cd webapp
   touch .env
   
   # Edit the .env file and add these configurations:
   echo "FLASK_APP=app.py" >> .env
   echo "FLASK_ENV=development" >> .env
   echo "SECRET_KEY=your_secure_random_key_here" >> .env
   echo "OPENAI_API_KEY=your_openai_api_key" >> .env
   echo "UPLOAD_FOLDER=$(pwd)/uploads" >> .env
   ```

   **Note about environment variables**:
   - `FLASK_APP`: Tells Flask which file contains your application (required)
   - `FLASK_ENV=development`: Enables development features like auto-reload and enhanced error messages
   - `SECRET_KEY`: Required for session security (generate a random string)
   - `OPENAI_API_KEY`: Required for LLM processing of papers

6. **Initialize the Database**:
   ```bash
   # Make sure you're in the webapp directory:
   cd webapp  # Skip if already in webapp directory
   
   # Initialize the database migrations:
   python migrations_init.py
   
   # This creates the initial migration structure
   ```

   **What does migrations_init.py do?**
   This script initializes the Flask-Migrate extension for database migrations, creating the initial migration structure without applying any changes yet.

7. **Apply Database Migrations**:
   ```bash
   # Apply the migrations to create the database tables:
   flask db upgrade
   ```

8. **Create Required Directories**:
   ```bash
   mkdir -p uploads/{papers,datasets,outputs,temp}
   mkdir -p logs/projects
   ```

### Running the Application

You'll need three terminal windows for a complete setup (or run Redis as a service):

1. **Terminal 1: Redis Server** (Skip if running as a service):
   ```bash
   redis-server
   ```

2. **Terminal 2: Celery Worker**:
   ```bash
   cd Paper2Code/webapp
   source venv/bin/activate  # If using virtual environment
   
   # Start the Celery worker:
   celery -A app.celery worker --loglevel=info
   ```

   **What does Celery do?**
   Celery handles asynchronous tasks like paper processing, planning, and code generation, which can take some time to complete.

3. **Terminal 3: Flask Web Server**:
   ```bash
   cd Paper2Code/webapp
   source venv/bin/activate  # If using virtual environment
   
   # Start the Flask application:
   flask run --host=0.0.0.0 --port=5000
   ```

4. **Access the Application**:
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

5. **Register an account** and start using the application

### Using Docker (Alternative)

If you prefer using Docker, you can build and run the application with:

```bash
# Make sure you have Docker and Docker Compose installed
docker-compose up -d
```

## Feature Guide

### Project Export Feature

The Project Export functionality allows you to download your entire project as a ZIP archive:

1. Navigate to the project details page by clicking on a project from your dashboard
2. For completed projects, you can export in two ways:
   - Click the "Download as ZIP" button in the generated code section
   - Click the "Export Project" button in the Actions menu

3. In the export dialog:
   - Check "Include log files" to include processing logs in the export
   - Click "Export" to start the export process
   - Once complete, click "Download" to save the ZIP to your computer

4. The exported ZIP contains:
   - Generated code
   - The original paper file
   - Planning and configuration files
   - Project metadata
   - Processing logs (if selected)

## Troubleshooting

### Common Issues and Solutions

1. **Redis Connection Error**:
   ```
   Error: Connection refused to Redis at localhost:6379
   ```
   **Solution**:
   - Verify Redis is running: `sudo systemctl status redis-server`
   - Install if needed: `sudo apt install redis-server`
   - Start the service: `sudo systemctl start redis-server`

2. **Database Migration Errors**:
   ```
   Error: Can't locate revision identified by '...'
   ```
   **Solution**:
   - Reset the migration directory and initialize again:
     ```bash
     rm -rf migrations
     python migrations_init.py
     flask db upgrade
     ```

3. **Celery Worker Not Starting**:
   ```
   Error importing 'app.celery'
   ```
   **Solution**:
   - Make sure you're in the correct directory (webapp)
   - Verify all requirements are installed
   - Check `app/__init__.py` for celery import errors

4. **Permission Issues with Upload Directory**:
   ```
   PermissionError: [Errno 13] Permission denied: '/path/to/uploads'
   ```
   **Solution**:
   - Change ownership or permissions:
     ```bash
     chmod -R 755 webapp/uploads
     ```

5. **OpenAI API Key Issues**:
   ```
   openai.error.AuthenticationError: Incorrect API key provided
   ```
   **Solution**:
   - Verify the API key in your `.env` file
   - Check if the environment variable is being loaded correctly
   - You can manually set it with: `export OPENAI_API_KEY=your_key`

## License

This project is licensed under the same license as the main Paper2Code repository.