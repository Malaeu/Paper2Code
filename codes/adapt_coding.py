#!/usr/bin/env python3
"""
Adaptation Coding Module

This module is responsible for generating the actual code for the adapted methodology,
based on the analysis from either the automatic mapping or the two-phase approach.
"""

import argparse
import json
import os
import re
import sys
import yaml
from pathlib import Path

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import necessary utilities
from codes.utils import create_openai_client, get_response_content

def load_config(config_path):
    """
    Load and parse the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def load_analysis(analysis_dir):
    """
    Load the adaptation analysis from the analysis directory.
    
    Args:
        analysis_dir (str): Path to the analysis directory
        
    Returns:
        dict: Loaded analysis information
    """
    analysis_info = {}
    
    # Load the analysis markdown file
    analysis_path = os.path.join(analysis_dir, "adaptation_analysis.md")
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as file:
            analysis_info["analysis_text"] = file.read()
    
    # Load variable mapping if available
    mapping_path = os.path.join(analysis_dir, "variable_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as file:
            analysis_info["variable_mapping"] = json.load(file)
    
    # Load config if available
    config_path = os.path.join(analysis_dir, "adapt_config.yaml")
    if os.path.exists(config_path):
        analysis_info["config"] = load_config(config_path)
    
    return analysis_info

def generate_repo_structure(analysis_info, output_dir):
    """
    Generate the repository structure based on the analysis.
    
    Args:
        analysis_info (dict): Analysis information
        output_dir (str): Output directory path
        
    Returns:
        dict: Repository structure
    """
    # Extract repo name from config if available
    repo_name = "AdaptedModel"
    if "config" in analysis_info and "output" in analysis_info["config"]:
        repo_name = analysis_info["config"]["output"].get("repo_name", "AdaptedModel")
    
    # Define basic repository structure
    repo_structure = {
        "name": repo_name,
        "directories": [
            "data",
            "models",
            "utils",
            "configs",
            "evaluation",
            "scripts"
        ],
        "files": [
            "main.py",
            "requirements.txt",
            "README.md",
            "configs/default.yaml"
        ]
    }
    
    # Create directory structure
    for directory in repo_structure["directories"]:
        os.makedirs(os.path.join(output_dir, directory), exist_ok=True)
    
    # Create empty files
    for file in repo_structure["files"]:
        file_path = os.path.join(output_dir, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write("")
    
    return repo_structure

def generate_code_files(client, analysis_info, repo_structure, output_dir, gpt_version):
    """
    Generate the actual code files for the repository.
    
    Args:
        client: OpenAI API client
        analysis_info (dict): Analysis information
        repo_structure (dict): Repository structure
        output_dir (str): Output directory path
        gpt_version (str): GPT model version to use
        
    Returns:
        list: Generated file paths
    """
    # Extract variable mapping and analysis text
    variable_mapping = analysis_info.get("variable_mapping", {})
    analysis_text = analysis_info.get("analysis_text", "")
    
    # Create the prompt for generating code
    system_prompt = """You are an expert scientific code generation system.
Your task is to generate high-quality, modular Python code that implements 
a scientific methodology based on the provided adaptation analysis.

Ensure your code follows these principles:
1. Well-structured and modular
2. Properly documented with docstrings
3. Follows Python best practices
4. Handles errors gracefully
5. Includes necessary validation and testing components
6. Maintains methodological rigor from the original paper
"""

    user_prompt = f"""# Code Generation Task

Based on the adaptation analysis below, please generate the complete code 
for implementing the adapted methodology. 

## Adaptation Analysis
{analysis_text}

## Variable Mapping
```json
{json.dumps(variable_mapping, indent=2)}
```

## Repository Structure
The code should be organized according to this structure:
```
{repo_structure["name"]}/
  ├── data/               # Data loading and processing
  ├── models/             # Model implementations
  ├── utils/              # Utility functions
  ├── configs/            # Configuration files
  ├── evaluation/         # Evaluation metrics and tools
  ├── scripts/            # Helper scripts
  ├── main.py             # Main entry point
  ├── requirements.txt    # Dependencies
  └── README.md           # Documentation
```

Please generate the following files:
1. data/dataset_loader.py - For loading and preprocessing the dataset
2. models/model.py - Implementation of the core models
3. evaluation/metrics.py - Evaluation metrics for the models
4. utils/helpers.py - Helper functions
5. configs/default.yaml - Default configuration
6. main.py - Main script that ties everything together
7. requirements.txt - Dependencies
8. README.md - Documentation

Focus on implementing the methodology correctly while adapting to the new variable names.
For each file, start with a detailed comment explaining its purpose and relation to the 
original methodology.

Generate each file one at a time, starting with the most fundamental components.
"""

    # Generate the code
    print(f"Generating code using {gpt_version}...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model=gpt_version,
        messages=messages,
        temperature=0.2,
        max_tokens=4000
    )
    
    response_text = get_response_content(response)
    
    # Extract code blocks for each file
    file_pattern = r'```python\s*#\s*File:\s*([^\n]+)\s*\n(.*?)```'
    markdown_pattern = r'```markdown\s*#\s*File:\s*([^\n]+)\s*\n(.*?)```'
    yaml_pattern = r'```yaml\s*#\s*File:\s*([^\n]+)\s*\n(.*?)```'
    
    # Extract Python files
    python_files = re.findall(file_pattern, response_text, re.DOTALL)
    generated_files = []
    
    for file_path, file_content in python_files:
        file_path = file_path.strip()
        full_path = os.path.join(output_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as file:
            file.write(file_content)
        
        generated_files.append(full_path)
        print(f"Generated: {file_path}")
    
    # Extract Markdown files
    markdown_files = re.findall(markdown_pattern, response_text, re.DOTALL)
    
    for file_path, file_content in markdown_files:
        file_path = file_path.strip()
        full_path = os.path.join(output_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as file:
            file.write(file_content)
        
        generated_files.append(full_path)
        print(f"Generated: {file_path}")
    
    # Extract YAML files
    yaml_files = re.findall(yaml_pattern, response_text, re.DOTALL)
    
    for file_path, file_content in yaml_files:
        file_path = file_path.strip()
        full_path = os.path.join(output_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as file:
            file.write(file_content)
        
        generated_files.append(full_path)
        print(f"Generated: {file_path}")
    
    # If no files were extracted using the patterns, generate basic files
    if not generated_files:
        print("Warning: No file blocks found in the response. Generating basic files...")
        generate_basic_files(output_dir, analysis_info)
    
    return generated_files

def generate_basic_files(output_dir, analysis_info):
    """
    Generate basic files if the API response didn't include proper file blocks.
    
    Args:
        output_dir (str): Output directory path
        analysis_info (dict): Analysis information
    """
    # Create basic main.py
    main_content = """
import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the adapted methodology")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # TODO: Implement the adapted methodology
    
    print("Adapted methodology implementation")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(output_dir, "main.py"), 'w') as file:
        file.write(main_content)
    
    # Create basic README.md
    readme_content = """# Adapted Methodology
    
This repository contains an implementation of an adapted scientific methodology.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config configs/default.yaml
```

## Configuration

Edit the configuration file in `configs/default.yaml` to adjust parameters.
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w') as file:
        file.write(readme_content)
    
    # Create basic requirements.txt
    requirements_content = """
numpy
pandas
scikit-learn
matplotlib
pyyaml
"""
    
    with open(os.path.join(output_dir, "requirements.txt"), 'w') as file:
        file.write(requirements_content)
    
    # Create basic config file
    config_content = """# Default configuration
dataset:
  path: data/dataset.csv
  format: csv

model:
  type: default
  params:
    # Model parameters

training:
  # Training parameters
  
evaluation:
  # Evaluation parameters
"""
    
    with open(os.path.join(output_dir, "configs/default.yaml"), 'w') as file:
        file.write(config_content)

def generate_missing_files(client, analysis_info, output_dir, generated_files, gpt_version):
    """
    Generate any missing core files that weren't created in the first pass.
    
    Args:
        client: OpenAI API client
        analysis_info (dict): Analysis information
        output_dir (str): Output directory path
        generated_files (list): Already generated file paths
        gpt_version (str): GPT model version to use
        
    Returns:
        list: Additional generated file paths
    """
    core_files = [
        "data/dataset_loader.py",
        "models/model.py",
        "evaluation/metrics.py", 
        "utils/helpers.py",
        "main.py"
    ]
    
    additional_files = []
    
    for core_file in core_files:
        full_path = os.path.join(output_dir, core_file)
        if full_path not in generated_files and not os.path.exists(full_path):
            print(f"Generating missing core file: {core_file}...")
            
            # Extract file type and purpose
            file_parts = core_file.split('/')
            file_type = file_parts[0] if len(file_parts) > 1 else "main"
            file_name = file_parts[-1]
            
            # Create specific prompt for this file type
            system_prompt = """You are an expert scientific code generation system.
Your task is to generate a specific Python module for implementing a scientific methodology.
Focus on creating well-structured, properly documented code that follows Python best practices.
"""

            user_prompt = f"""# Code Generation for {core_file}

Based on the adaptation analysis below, please generate the code for the {core_file} file.
This file should handle the {file_type} component of the implementation.

## Adaptation Analysis
{analysis_info.get("analysis_text", "")}

## Variable Mapping
```json
{json.dumps(analysis_info.get("variable_mapping", {}), indent=2)}
```

Please generate complete, well-documented Python code for this file. The code should:
1. Follow Python best practices
2. Include proper error handling
3. Be well-documented with docstrings
4. Implement the required functionality based on the adaptation analysis

Return just the Python code without markdown formatting.
"""

            # Generate the code
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=gpt_version,
                messages=messages,
                temperature=0.2,
                max_tokens=3000
            )
            
            code_content = get_response_content(response)
            
            # Clean up the code (remove markdown code blocks if present)
            code_content = re.sub(r'^```python\s*', '', code_content, flags=re.MULTILINE)
            code_content = re.sub(r'^```\s*$', '', code_content, flags=re.MULTILINE)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write the file
            with open(full_path, 'w') as file:
                file.write(code_content)
            
            additional_files.append(full_path)
            print(f"Generated: {core_file}")
    
    return additional_files

def update_readme(output_dir, analysis_info):
    """
    Update the README.md file with detailed information.
    
    Args:
        output_dir (str): Output directory path
        analysis_info (dict): Analysis information
    """
    readme_path = os.path.join(output_dir, "README.md")
    
    # Extract repo name from config if available
    repo_name = "Adapted Model"
    if "config" in analysis_info and "output" in analysis_info["config"]:
        repo_name = analysis_info["config"]["output"].get("repo_name", "Adapted Model")
    
    # Create README content
    readme_content = f"""# {repo_name}

## Overview
This repository contains an implementation of an adapted scientific methodology.
The code has been automatically generated based on a detailed adaptation analysis
that maps concepts from an original scientific paper to a new dataset and variables.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config configs/default.yaml
```

## Configuration

Edit the configuration file in `configs/default.yaml` to adjust parameters.

## Components

- `data/`: Data loading and preprocessing
- `models/`: Model implementations
- `utils/`: Utility functions
- `configs/`: Configuration files
- `evaluation/`: Evaluation metrics and tools
- `scripts/`: Helper scripts
- `main.py`: Main entry point

## Variable Mapping

The following variable mapping has been applied:

```json
{json.dumps(analysis_info.get("variable_mapping", {}), indent=2)}
```

## Generated with Paper2Code

This repository was automatically generated using the Paper2Code adaptation system,
which adapts scientific methodologies to new datasets and variables while maintaining
methodological rigor.
"""
    
    with open(readme_path, 'w') as file:
        file.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description="Generate code based on adaptation analysis")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--analysis_dir", required=True, help="Directory with adaptation analysis")
    parser.add_argument("--output_dir", required=True, help="Output directory for coding artifacts")
    parser.add_argument("--output_repo_dir", required=True, help="Output directory for the generated repository")
    parser.add_argument("--gpt_version", default="o3-mini-2025-04-16", help="GPT model version to use")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_repo_dir, exist_ok=True)
    
    # Load analysis information
    analysis_info = load_analysis(args.analysis_dir)
    
    # Load config
    config = load_config(args.config)
    analysis_info["config"] = config
    
    # Create OpenAI client
    client = create_openai_client()
    
    # Generate repository structure
    repo_structure = generate_repo_structure(analysis_info, args.output_repo_dir)
    
    # Generate code files
    generated_files = generate_code_files(client, analysis_info, repo_structure, args.output_repo_dir, args.gpt_version)
    
    # Generate any missing core files
    additional_files = generate_missing_files(client, analysis_info, args.output_repo_dir, generated_files, args.gpt_version)
    
    # Update README with detailed information
    update_readme(args.output_repo_dir, analysis_info)
    
    # Copy the config and analysis to the output directory for reference
    import shutil
    shutil.copy(args.config, os.path.join(args.output_dir, "adapt_config.yaml"))
    
    analysis_path = os.path.join(args.analysis_dir, "adaptation_analysis.md")
    if os.path.exists(analysis_path):
        shutil.copy(analysis_path, os.path.join(args.output_dir, "adaptation_analysis.md"))
    
    mapping_path = os.path.join(args.analysis_dir, "variable_mapping.json")
    if os.path.exists(mapping_path):
        shutil.copy(mapping_path, os.path.join(args.output_dir, "variable_mapping.json"))
    
    print(f"Code generation completed. Repository created at {args.output_repo_dir}")
    
    # List the generated files
    print("\nGenerated files:")
    for file in sorted(generated_files + additional_files):
        print(f"- {os.path.relpath(file, args.output_repo_dir)}")

if __name__ == "__main__":
    main()