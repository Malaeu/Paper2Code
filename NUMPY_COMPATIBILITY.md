# NumPy 2.x Compatibility Issues and Solutions

This document explains the compatibility issues between NumPy 2.x and various Paper2Code dependencies, and offers several solutions.

## Problem Description

When trying to run the Paper2Code adaptation pipeline with NumPy 2.x (e.g., NumPy 2.2.5), you may encounter errors like:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.5 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
```

This is because some dependencies (particularly pandas and pyarrow) were compiled for NumPy 1.x and are not compatible with NumPy 2.x. Additionally, there may be issues with the OpenAI client initialization due to changes in parameter handling.

## Affected Components

1. **adapt_planning.py** - Imports from adapt_mapping.py which uses pandas
2. **adapt_mapping.py** - Directly imports pandas and uses it for dataset analysis
3. **utils.py** - OpenAI client initialization may have issues with newer versions of httpx

## Solutions

### Solution 1: Use the Direct Adaptation Script

A lightweight `direct_adapt.py` script is provided as an alternative to the standard adaptation pipeline. This script:

- Does not depend on pandas or pyarrow
- Uses a simple approach to initialize the OpenAI client
- Extracts only key sections from papers to reduce token usage
- Provides a basic adaptation plan

See [DIRECT_ADAPTATION.md](DIRECT_ADAPTATION.md) for usage instructions.

### Solution 2: Create a Compatible Virtual Environment

If you need the full functionality of the adaptation pipeline, create a virtual environment with compatible package versions:

```bash
# Create and activate a new virtual environment
python -m venv .venv_compatible
source .venv_compatible/bin/activate

# Install compatible packages
pip install "numpy<2.0.0" "pandas<2.0.0" "pyarrow<11.0.0"
pip install openai==1.3.0 httpx==0.24.1

# Install other requirements
pip install -r requirements.txt
```

### Solution 3: Fix OpenAI Client Initialization

If you encounter specific issues with the OpenAI client initialization, modify the `create_openai_client()` function in `utils.py`:

```python
def create_openai_client():
    """
    Create and return an OpenAI client using the API key from environment variables.
    
    Returns:
        OpenAI: The OpenAI client
    """
    import os
    from openai import OpenAI
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or environment variables.")
    
    # Create a minimal client to avoid compatibility issues
    client = OpenAI()
    client.api_key = api_key
    
    # Remove any problematic attributes that might be causing issues
    if hasattr(client._client, 'proxies'):
        delattr(client._client, 'proxies')
    
    return client
```

### Solution 4: Use Docker

The project includes Docker configuration files that create an isolated environment with compatible package versions. To use Docker:

```bash
# Build the Docker image
docker build -t paper2code -f docker/Dockerfile.webapp .

# Run the Docker container
docker run -it -p 5000:5000 -v $(pwd):/app paper2code
```

## OpenAI Model Availability

Some models referenced in the code might not be available to all users. If you encounter errors like:

```
The model `o3-mini-2025-04-16` does not exist or you do not have access to it.
```

Try using a more widely available model such as:
- gpt-3.5-turbo
- gpt-4 (if you have access)
- gpt-4o (if you have access)

You can specify the model when running the direct adaptation script:

```bash
python direct_adapt.py --paper examples/Transformer.pdf --output_dir adapt_output --model gpt-3.5-turbo
```

## Token Usage and Context Length

When adapting large papers, you may encounter token usage limits. The direct adaptation script addresses this by:

1. Extracting only the title and abstract from papers
2. Adding a brief summary of the key methodology
3. Limiting the dataset description to 2000 characters

For custom adaptations with larger papers, you may need to:
1. Extract only the most relevant sections
2. Use a model with higher context limits (e.g., gpt-4-32k or gpt-4o)
3. Split the paper into multiple API calls and combine the results

## Long-Term Solution

As NumPy 2.x becomes more widely adopted, package maintainers will likely update their packages for compatibility. Until then, using a virtual environment with NumPy 1.x is the most reliable solution for running the full Paper2Code adaptation pipeline.