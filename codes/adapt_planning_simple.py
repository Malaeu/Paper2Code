#!/usr/bin/env python3
"""
Simplified Adaptation Planning Module without pandas dependency

This module is a simplified version of adapt_planning.py that doesn't rely on pandas,
making it compatible with NumPy 2.x and avoiding compatibility issues.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import necessary utilities
from codes.utils import create_openai_client, get_response_content

def load_paper_json(json_path):
    """
    Load the paper JSON file.
    
    Args:
        json_path (str): Path to the paper JSON file
        
    Returns:
        dict: Paper JSON content
    """
    try:
        with open(json_path, 'r') as file:
            paper_json = json.load(file)
        return paper_json
    except Exception as e:
        print(f"Error loading paper JSON: {e}")
        sys.exit(1)

def load_dataset_description(description_path):
    """
    Load the dataset description markdown file.
    
    Args:
        description_path (str): Path to the dataset description markdown file
        
    Returns:
        str: Dataset description content
    """
    try:
        with open(description_path, 'r') as file:
            description = file.read()
        return description
    except Exception as e:
        print(f"Error loading dataset description: {e}")
        return ""

def generate_simple_variable_mapping():
    """
    Generate a simple variable mapping template.
    
    Returns:
        dict: A simple variable mapping template
    """
    return {
        "original_to_adapted": {
            "input_variable": "feature1",
            "target_variable": "target",
            "categorical_feature": "feature2"
        },
        "methodology_adjustments": {
            "maintain_key_analysis_feature": True,
            "iterations": 1000
        }
    }

def generate_plan(client, paper_json, dataset_info, gpt_version):
    """
    Generate an adaptation plan using the OpenAI API.
    
    Args:
        client: OpenAI API client
        paper_json (dict): Paper JSON content
        dataset_info (dict): Dataset information
        gpt_version (str): GPT model version to use
        
    Returns:
        dict: Generated adaptation plan
    """
    # Extract paper text
    paper_text = ""
    if "abstract" in paper_json:
        paper_text += "## Abstract\n" + paper_json["abstract"] + "\n\n"
    
    if "body_text" in paper_json:
        for section in paper_json["body_text"]:
            if "section" in section and section["section"]:
                paper_text += f"## {section['section']}\n"
            paper_text += section["text"] + "\n\n"
    
    # Create dataset description text
    dataset_description = dataset_info.get("description", "")
    if not dataset_description and "description_path" in dataset_info and dataset_info["description_path"]:
        dataset_description = load_dataset_description(dataset_info["description_path"])
    
    # Generate a simple variable mapping
    variable_mapping = generate_simple_variable_mapping()
    
    dataset_text = f"""
## Dataset Information
- Path: {dataset_info.get("path", "Not specified")}
- Format: {dataset_info.get("format", "Not specified")}

## Dataset Description
{dataset_description}

## Dataset Variable Mapping
```json
{json.dumps(variable_mapping, indent=2)}
```
"""

    # Create the prompt
    system_prompt = """You are an expert scientific methodology adaptation system. 
Your task is to create a detailed plan for adapting a scientific methodology from a paper 
to a new dataset with different variables. 

Your plan should focus on maintaining the methodological rigor of the original paper
while adapting to the new dataset structure and variable names.
"""

    user_prompt = f"""# Adaptation Planning Task

I need a detailed plan for adapting the methodology from the following scientific paper 
to a new dataset with different variables.

## Paper Content
{paper_text}

{dataset_text}

Based on the paper and dataset information above, please create a detailed adaptation plan.
The plan should include:

1. The core methodological elements that must be preserved
2. How the variables should be mapped (based on the provided mapping)
3. Any adjustments needed to accommodate the new dataset structure
4. Specific recommendations for implementation
5. Potential challenges and solutions

Format your response as a detailed markdown document with JSON sections for structured data.
Begin with a high-level overview of the adaptation approach, then provide specific details
for each component of the methodology.

Include a JSON block with structured information about the adaptation approach, like this:

```json
{{
  "adaptation_strategy": {{
    "core_methodology": "...",
    "variable_mapping_strategy": "...",
    "required_adjustments": [...],
    "implementation_recommendations": [...],
    "potential_challenges": [...],
    "solutions": [...]
  }},
  "methodology_components": [
    {{
      "name": "component_name",
      "original_approach": "...",
      "adaptation_approach": "...",
      "implementation_details": "..."
    }},
    ...
  ]
}}
```

Focus on being specific and detailed in your recommendations.
"""

    # Make the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"Generating adaptation plan using {gpt_version}...")
    # Check model name to determine correct parameters
    if gpt_version.startswith("o"):
        # For Anthropic models (o3, o4-mini), use their specific parameters
        # Anthropic models may not support certain parameters or values
        response = client.chat.completions.create(
            model=gpt_version,
            messages=messages,
            max_completion_tokens=4000  # Use the correct parameter for anthropic models
            # No temperature parameter for Anthropic models
        )
    else:
        # For OpenAI models, use max_tokens
        response = client.chat.completions.create(
            model=gpt_version,
            messages=messages,
            temperature=0.2,
            max_tokens=4000
        )
    
    response_text = get_response_content(response)
    
    # Extract JSON from the response
    import re
    json_blocks = re.findall(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    
    if not json_blocks:
        print("Warning: No JSON blocks found in the response. Using full text as plan.")
        return {"full_plan_text": response_text}
    
    try:
        plan_json = json.loads(json_blocks[0])
        return {
            "plan_json": plan_json,
            "full_plan_text": response_text
        }
    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON from the response. Using full text as plan.")
        return {"full_plan_text": response_text}

def save_plan(plan, output_dir, paper_name):
    """
    Save the generated plan to the output directory.
    
    Args:
        plan (dict): Generated plan
        output_dir (str): Output directory path
        paper_name (str): Name of the paper
        
    Returns:
        str: Path to the saved plan file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full text as markdown
    plan_text_path = os.path.join(output_dir, f"{paper_name}_adaptation_plan.md")
    with open(plan_text_path, 'w') as file:
        file.write(plan["full_plan_text"])
    
    # Save structured JSON if available
    if "plan_json" in plan:
        plan_json_path = os.path.join(output_dir, f"{paper_name}_adaptation_plan.json")
        with open(plan_json_path, 'w') as file:
            json.dump(plan["plan_json"], file, indent=2)
    
    print(f"Adaptation plan saved to {plan_text_path}")
    return plan_text_path

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Determine paper JSON path
    paper_json_path = args.paper_json
    if not paper_json_path and "paper" in config and "json_path" in config["paper"]:
        paper_json_path = config["paper"]["json_path"]
    
    if not paper_json_path:
        print("Error: Paper JSON path not provided")
        sys.exit(1)
    
    # Load paper JSON
    paper_json = load_paper_json(paper_json_path)
    
    # Determine dataset information
    dataset_info = {}
    if "dataset" in config:
        dataset_info = config["dataset"]
    else:
        dataset_info = {
            "path": args.dataset_path,
            "format": args.dataset_format,
            "description_path": args.dataset_description
        }
    
    # Create OpenAI client
    client = create_openai_client()
    
    # Generate the plan
    plan = generate_plan(client, paper_json, dataset_info, args.gpt_version)
    
    # Save the plan
    plan_path = save_plan(plan, args.output_dir, args.paper_name)
    
    print(f"Adaptation planning completed. Plan saved to {plan_path}")
    return plan_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adaptation plan for a paper and dataset")
    parser.add_argument("--config", help="Path to the YAML configuration file")
    parser.add_argument("--paper_json", help="Path to the paper JSON file")
    parser.add_argument("--dataset_path", help="Path to the dataset file")
    parser.add_argument("--dataset_format", default="csv", help="Dataset format (csv, parquet, excel, json)")
    parser.add_argument("--dataset_description", help="Path to the dataset description markdown file")
    parser.add_argument("--output_dir", required=True, help="Output directory for planning artifacts")
    parser.add_argument("--paper_name", default="AdaptedPaper", help="Name of the paper for file naming")
    parser.add_argument("--gpt_version", default="o3-mini-2025-04-16", help="GPT model version to use")
    
    args = parser.parse_args()
    main(args)