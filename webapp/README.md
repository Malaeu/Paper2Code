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
   - Python 3.10, 3.11, or 3.12+ are all supported
   - Check your version with: `python3 --version`
   
   **Python Version Notes:**
   - Modern versions of PyMuPDF (>=1.23.5) fully support Python 3.12+
   - Other dependencies might occasionally have issues with the latest Python versions
   - If you encounter installation problems, updating pip is usually the solution, not downgrading Python

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Paper2Code.git
   cd Paper2Code
   ```

2. **Create a Virtual Environment** (highly recommended):
   ```bash
   # Create a virtual environment named .venv_env
   python3 -m venv .venv_env
   
   # Activate the virtual environment
   source .venv_env/bin/activate  # On Windows: .venv_env\Scripts\activate
   
   # After activation, your prompt should change to show (.venv_env)
   # This indicates you're now working within the virtual environment
   
   # After you're done working, you can deactivate the environment with:
   # deactivate
   ```
   
   **Why use a virtual environment?**
   - Isolates the project dependencies from your system Python
   - Prevents conflicts between different projects
   - Makes it easier to manage package versions
   - Ensures everyone working on the project has the same environment
   - Allows clean uninstallation of all dependencies when no longer needed

3. **Install Python Dependencies**:
   ```bash
   # First update pip and essential tools
   pip install --upgrade pip setuptools wheel
   
   # Install system dependencies for PyMuPDF and other packages
   sudo apt-get install -y \
       build-essential \
       python3-dev \
       libmupdf-dev \
       libfreetype6-dev \
       libharfbuzz-dev \
       libjpeg-dev \
       libpng-dev \
       pkg-config
       
   # Now install Python packages
   pip install -r requirements.txt
   pip install -r webapp/requirements.txt
   ```
   
   **Note about package installation:**
   Before installing dependencies, make sure you have the latest pip:
   
   ```bash
   # Always update pip, setuptools, and wheel before installing dependencies
   pip install --upgrade pip setuptools wheel
   
   # Then install the dependencies
   pip install -r requirements.txt
   pip install -r webapp/requirements.txt
   ```
   
   If you encounter PyMuPDF errors (particularly with older versions):
   
   ```bash
   # Check your requirements.txt file for outdated PyMuPDF versions
   grep -i pymupdf webapp/requirements.txt
   
   # If a specific old version is pinned, consider updating to at least 1.23.5
   # Edit webapp/requirements.txt to change pymupdf==1.23.4 to pymupdf>=1.23.5
   
   # Alternatively, install the latest version directly
   pip install pymupdf>=1.23.5
   ```
   
   The specific error with "Preparing metadata (pyproject.toml)" usually indicates you need to update pip or that the requirements.txt has pinned an outdated version.

4. **Set Up Redis Server**:
   Redis is required for the Celery task queue, which processes paper uploads and code generation.
   
   ```bash
   # Install Redis if not already installed:
   sudo apt update
   sudo apt install -y redis-server
   
   # Check if Redis is running:
   sudo systemctl status redis-server
   
   # If not running, start it:
   sudo systemctl start redis-server
   
   # To make Redis start on boot:
   sudo systemctl enable redis-server
   
   # Verify Redis is working:
   redis-cli ping
   # Should return "PONG"
   ```
   
   **Why Redis and Celery are required:**
   - Paper2Code performs complex, time-consuming operations like PDF parsing and code generation
   - These operations would timeout in a normal web request
   - Redis serves as the message broker between the web app and background workers
   - Celery manages these background tasks, handling queuing and execution
   - This architecture ensures the web interface remains responsive while processing happens in the background

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
   - `FLASK_ENV=development`: Enables development mode with:
     - Auto-reloading when code changes
     - Detailed error pages with traceback
     - Debug toolbar integration
     - Warning: Don't use in production as it enables features that could expose sensitive data
   - `SECRET_KEY`: Required for session security (generate a random string)
   - `OPENAI_API_KEY`: Required for LLM processing of papers
   
   For production use, you would set `FLASK_ENV=production` instead.

6. **Initialize the Database**:
   ```bash
   # Make sure you're in the webapp directory:
   cd webapp  # Skip if already in webapp directory
   
   # Initialize the database migrations:
   python migrations_init.py
   
   # This creates the initial migration structure
   ```

   **What does migrations_init.py do?**
   - This script initializes the Flask-Migrate extension for database migrations
   - Creates the initial migration directory structure needed for Flask-Migrate
   - Sets up the environment for tracking database schema changes
   - Does not make any actual changes to the database yet

7. **Apply Database Migrations**:
   ```bash
   # Apply the migrations to create the database tables:
   flask db upgrade
   
   # This command will:
   # - Create the SQLite database file if it doesn't exist
   # - Create all necessary tables based on the migration files
   # - Set up indexes and constraints
   # - Prepare the database for use by the application
   ```
   
   If you encounter errors about missing revisions, you can reset and recreate the migrations:
   ```bash
   rm -rf migrations
   python migrations_init.py
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
   source .venv_env/bin/activate  # If using virtual environment
   
   # Start the Celery worker:
   celery -A app.celery worker --loglevel=info
   ```

   **What does Celery do?**
   Celery handles asynchronous tasks like paper processing, planning, and code generation, which can take some time to complete.

3. **Terminal 3: Flask Web Server**:
   ```bash
   cd Paper2Code/webapp
   source .venv_env/bin/activate  # If using virtual environment
   
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

## Features

1. **User Management**:
   - User registration and authentication
   - Profile management
   - Account security features

2. **Project Management**:
   - Create projects by uploading scientific papers
   - Monitor processing progress
   - View and download generated code

3. **Paper Processing**:
   - Automated paper analysis
   - Code generation based on paper methodology
   - Repository creation

4. **Project Export** (New Feature):
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

5. **Directory Configuration**:
   - Configure storage locations for uploaded files
   - Manage file paths for generated code
   - Monitor disk usage

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

6. **Python Version Compatibility Issues**:
   ```
   error: subprocess-exited-with-error × Preparing metadata (pyproject.toml) did not run successfully.
   ```
   **Solution**:
   - This error is most likely caused by outdated packages (e.g., PyMuPDF <1.23.5) pinned in requirements.txt
   - **Modern versions of PyMuPDF (>=1.23.5) fully support Python 3.12+**
   - Update your pip and package tools first:
     ```bash
     pip install --upgrade pip setuptools wheel
     ```
   - Check requirements.txt for pinned outdated versions:
     ```bash
     grep -i pymupdf webapp/requirements.txt
     ```
   - If you find an older version specified, update it or remove the version constraint
   - Install dependencies again:
     ```bash
     pip install -r requirements.txt
     pip install -r webapp/requirements.txt
     ```
   - Only in rare cases with other dependencies might you need Python 3.10

## License

This project is licensed under the same license as the main Paper2Code repository.