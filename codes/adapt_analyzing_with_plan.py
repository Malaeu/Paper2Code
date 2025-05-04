#!/usr/bin/env python3
"""
Adaptation Analysis with Pre-Generated Plan

This module implements the second phase of the two-phase adaptation approach.
It takes a pre-generated adaptation plan and performs a detailed analysis
to prepare for code generation.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
import sys

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

def load_adaptation_plan(plan_path):
    """
    Load the pre-generated adaptation plan.
    
    Args:
        plan_path (str): Path to the adaptation plan JSON or markdown file
        
    Returns:
        dict: Parsed adaptation plan
    """
    try:
        if plan_path.endswith('.json'):
            with open(plan_path, 'r') as file:
                plan = json.load(file)
            return plan
        elif plan_path.endswith('.md'):
            with open(plan_path, 'r') as file:
                content = file.read()
                
            # Extract JSON blocks from markdown
            import re
            json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
            if not json_blocks:
                raise ValueError("No JSON blocks found in markdown file")
                
            # Parse the first JSON block (should be the plan)
            plan = json.loads(json_blocks[0])
            return plan
        else:
            raise ValueError("Unsupported file format. Must be .json or .md")
    except Exception as e:
        print(f"Error loading adaptation plan: {e}")
        sys.exit(1)

def load_template(template_name):
    """
    Load a template from the templates directory.
    
    Args:
        template_name (str): Name of the template file
        
    Returns:
        str: Template content
    """
    template_path = os.path.join(os.path.dirname(__file__), 'templates', template_name)
    try:
        with open(template_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading template {template_name}: {e}")
        sys.exit(1)

def generate_analysis(client, config, plan, template, gpt_version):
    """
    Generate a detailed analysis based on the adaptation plan.
    
    Args:
        client: OpenAI API client
        config (dict): Configuration dictionary
        plan (dict): Adaptation plan
        template (str): Analysis template
        gpt_version (str): GPT model version to use
        
    Returns:
        str: Generated analysis
    """
    # Extract key elements from plan and config
    paper_info = config.get('paper', {})
    dataset_info = config.get('dataset', {})
    variable_mapping = config.get('variable_mapping', {}).get('original_to_adapted', {})
    
    # Create the prompt
    system_prompt = """You are an expert scientific methodology adaptation system. 
Your task is to analyze a pre-generated adaptation plan and provide a detailed specification 
for adapting a scientific methodology to a new dataset with different variables.

Focus on maintaining methodological rigor while adapting to new variable names and data structures.
Be specific about implementation details for each component.
"""

    user_prompt = f"""# Adaptation Analysis Task

## Paper Information
- Title: {paper_info.get('title', 'Not specified')}
- Methodology: {paper_info.get('methodology', 'Not specified')}

## Dataset Information
- Format: {dataset_info.get('format', 'Not specified')}
- Path: {dataset_info.get('path', 'Not specified')}

## Variable Mapping
```json
{json.dumps(variable_mapping, indent=2)}
```

## Adaptation Plan
```json
{json.dumps(plan, indent=2)}
```

Based on the variable mapping and adaptation plan above, please create a detailed analysis 
for implementing the adapted methodology. Use the template below to structure your analysis.

{template}

Fill in all the placeholders in the template with detailed, specific implementation guidance.
Focus particularly on:
1. How to map the original variables to the adapted variables correctly
2. How to maintain the core methodological approach
3. Implementation details for each component (data loading, model architecture, training, evaluation)
4. Specific challenges that might arise and their solutions
5. A clear implementation plan for the coding phase

Your analysis should be comprehensive enough to guide the code generation process in the next phase.
"""

    # Make the API call
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
    
    return get_response_content(response)

def save_analysis(analysis, output_dir):
    """
    Save the generated analysis to the output directory.
    
    Args:
        analysis (str): Generated analysis text
        output_dir (str): Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as markdown
    analysis_path = os.path.join(output_dir, "adaptation_analysis.md")
    with open(analysis_path, 'w') as file:
        file.write(analysis)
    
    # Extract and save JSON blocks for easier parsing
    import re
    json_blocks = re.findall(r'```json\n(.*?)\n```', analysis, re.DOTALL)
    
    if json_blocks:
        # Save variable mapping if found
        try:
            variable_mapping = json.loads(json_blocks[0])
            mapping_path = os.path.join(output_dir, "variable_mapping.json")
            with open(mapping_path, 'w') as file:
                json.dump(variable_mapping, file, indent=2)
        except:
            pass
    
    print(f"Analysis saved to {analysis_path}")
    return analysis_path

def main():
    parser = argparse.ArgumentParser(description="Generate adaptation analysis from pre-generated plan")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--plan_path", required=True, help="Path to the pre-generated adaptation plan")
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis artifacts")
    parser.add_argument("--gpt_version", default="o3-mini-2025-04-16", help="GPT model version to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and plan
    config = load_config(args.config)
    plan = load_adaptation_plan(args.plan_path)
    
    # Load the analysis template
    template = load_template("adapt_analyzing_plan_template.md")
    
    # Create OpenAI client
    client = create_openai_client()
    
    # Generate the analysis
    print(f"Generating adaptation analysis using {args.gpt_version}...")
    analysis = generate_analysis(client, config, plan, template, args.gpt_version)
    
    # Save the analysis
    analysis_path = save_analysis(analysis, args.output_dir)
    
    # Copy the config to the output directory for reference
    import shutil
    config_dest = os.path.join(args.output_dir, "adapt_config.yaml")
    shutil.copy(args.config, config_dest)
    
    print(f"Analysis completed and saved to {analysis_path}")
    print(f"Configuration copied to {config_dest}")

if __name__ == "__main__":
    main()