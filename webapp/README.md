# Paper2Code Web Interface

This web application provides a user-friendly interface for the Paper2Code system, allowing users to adapt scientific methodologies from papers to their own datasets without needing to use the command line.

## Features

- Simple upload interface for papers and datasets
- Intelligent dataset analysis and variable mapping
- Interactive configuration of adaptation parameters
- Human-in-the-loop review of adaptation plans
- Seamless code generation
- Repository download as a ZIP file

## Installation

### Prerequisites

- Python 3.8+
- Redis (for task queue)
- Virtual environment (recommended)

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Paper2Code.git
   cd Paper2Code/webapp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   export OPENAI_API_KEY=your-api-key
   ```

5. Start Redis server:
   ```bash
   redis-server
   ```

6. Start Celery worker:
   ```bash
   celery -A app.celery worker --loglevel=info
   ```

7. Run the Flask application:
   ```bash
   flask run
   ```

8. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload**: Upload your scientific paper (PDF, JSON, or markdown) and dataset (CSV, Parquet, Excel, or JSON)
2. **Configure**: Set adaptation parameters and variable mappings
3. **Review**: Review the generated adaptation plan
4. **Generate**: Generate code based on the adaptation plan
5. **Download**: Download the complete repository as a ZIP file

## Project Structure

```
webapp/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── static/                 # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── img/
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── configure.html
│   ├── plan.html
│   ├── result.html
│   └── waiting.html
└── uploads/                # Directory for uploaded files
    ├── papers/
    ├── datasets/
    └── outputs/
```

## Deployment

For production deployment, it's recommended to use:

- Gunicorn or uWSGI as a WSGI server
- Nginx as a reverse proxy
- Supervisor to manage processes
- A proper Redis setup with persistence
- Environment variables for configuration

### Example Production Command

```bash
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

## License

This project is licensed under the same license as the main Paper2Code repository.