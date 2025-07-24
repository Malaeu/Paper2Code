# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper2Code is a multi-agent LLM system that automatically transforms scientific papers into executable code repositories. It uses a three-stage pipeline (planning, analysis, code generation) with specialized agents for each phase.

## Essential Commands

### Quick Start
```bash
# Setup environment
python3 -m venv .venv_env
source .venv_env/bin/activate
pip install -r requirements.txt
pip install -r webapp/requirements.txt

# Start Redis (required for Celery)
sudo service redis-server start

# Run the full application
./run_full_app.sh
```

### Running the Pipeline

```bash
# OpenAI API (estimated cost $0.50–$0.70 with o3-mini)
export OPENAI_API_KEY="your_key"
cd scripts && bash run.sh

# Open-source models with vLLM
cd scripts && bash run_llm.sh

# Custom adaptation (apply paper methodology to your dataset)
./scripts/run_custom_adapt.sh

# Two-phase adaptation with plan review
./scripts/run_direct_adapt.sh    # Generate plan
./scripts/run_with_plan.sh       # Use plan to generate code
```

### Web Application

```bash
# Terminal 1: Redis (skip if running as service)
redis-server

# Terminal 2: Celery worker
cd webapp
celery -A app.celery worker --loglevel=info

# Terminal 3: Flask app
cd webapp
flask run --host=0.0.0.0 --port=5000

# Access at http://localhost:5000
```

### Testing

```bash
# Run all CLI tests
./test_all_cli.sh

# Test pipeline
./test_pipeline_cli.sh

# Run unit tests
python -m unittest tests.segar.test_segar_pipeline

# Test Celery functionality
python test_cli_celery.py
```

### Database Management

```bash
cd webapp
flask db init                    # Initialize migrations
flask db migrate -m "message"    # Create migration
flask db upgrade                 # Apply migrations
```

## High-Level Architecture

### Core Pipeline Flow
```
PDF/LaTeX → Planning → Analysis → Code Generation → Repository
              ↓          ↓            ↓
          plan.json  analysis/    complete code
                     *.txt        repository
```

### Key Components

1. **Three-Stage Pipeline** (`codes/`):
   - `1_planning.py`: Creates implementation plan from paper
   - `2_analyzing.py`: Analyzes each component in detail
   - `3_coding.py`: Generates actual code implementation

2. **Adaptation System** (`codes/adapt_*.py`):
   - Maps paper variables to user's dataset variables
   - Maintains methodological rigor while adapting to new data
   - Supports CSV, Parquet, Excel, JSON datasets

3. **Web Application** (`webapp/`):
   - Flask + Celery + Redis + PostgreSQL stack
   - Async processing for long-running tasks
   - Project management and export functionality

4. **External Services**:
   - GROBID: PDF → structured document conversion
   - OpenAI/Anthropic APIs: LLM processing
   - S2ORC: Scientific document parsing

### Pipeline Execution Pattern

Each stage follows this pattern:
1. Load previous stage outputs
2. Prepare prompts with context
3. Make LLM API calls (with caching for efficiency)
4. Save outputs for next stage
5. Track costs and usage

### Adaptation Workflow

When adapting a paper's methodology to a new dataset:
1. Analyze dataset structure automatically
2. Generate variable mappings using AI
3. Create adaptation plan preserving methodology
4. Generate code that works with new variables

### Service Architecture

The web app uses a service-oriented architecture:
- **Pipeline Service**: Orchestrates paper processing
- **Directory Service**: Manages file operations
- **Model Service**: Handles LLM configurations
- **Export Service**: Creates downloadable artifacts
- **Email Service**: User notifications

### Async Processing

Long-running tasks use Celery:
- Background paper processing
- Progress tracking via Redis
- Real-time status updates
- Error handling and retries

## Important Notes

- Always ensure Redis is running before starting the web app or Celery
- GROBID must be running for PDF processing (port 8070)
- Use virtual environments to avoid dependency conflicts
- Check `.env` files for required API keys
- No linting tools are currently configured; consider adding flake8/black
- The project supports both OpenAI and open-source models
- Cost tracking is built into the pipeline for API usage monitoring