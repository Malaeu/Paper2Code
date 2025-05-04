#!/usr/bin/env python3
"""
Direct Adaptation Script

This script implements the first phase of the two-phase adaptation approach.
It takes dataset descriptions and generates an adaptation plan directly,
without parsing the raw dataset files.
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

def load_paper_content(paper_path):
    """
    Load the paper content from a file.
    
    Args:
        paper_path (str): Path to the paper file (PDF, JSON, or markdown)
        
    Returns:
        str: Paper content as text
    """
    try:
        if paper_path.endswith('.json'):
            with open(paper_path, 'r') as file:
                paper_json = json.load(file)
            
            # Extract text from JSON structure
            paper_text = ""
            if "abstract" in paper_json:
                paper_text += "## Abstract\n" + paper_json["abstract"] + "\n\n"
            
            if "body_text" in paper_json:
                for section in paper_json["body_text"]:
                    if "section" in section and section["section"]:
                        paper_text += f"## {section['section']}\n"
                    paper_text += section["text"] + "\n\n"
            
            return paper_text
        
        elif paper_path.endswith('.md'):
            with open(paper_path, 'r') as file:
                paper_text = file.read()
            return paper_text
        
        else:
            print(f"Unsupported paper format: {paper_path}")
            print("Please provide a paper in JSON or markdown format.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error loading paper content: {e}")
        sys.exit(1)

def load_dataset_description(description_path):
    """
    Load the dataset description from a file.
    
    Args:
        description_path (str): Path to the dataset description file
        
    Returns:
        str: Dataset description as text
    """
    try:
        with open(description_path, 'r') as file:
            description = file.read()
        return description
    except Exception as e:
        print(f"Error loading dataset description: {e}")
        return ""

def generate_adaptation_plan(client, paper_content, dataset_description, variable_mapping, gpt_version):
    """
    Generate an adaptation plan using the OpenAI API.
    
    Args:
        client: OpenAI API client
        paper_content (str): Paper content as text
        dataset_description (str): Dataset description as text
        variable_mapping (dict): Variable mapping information
        gpt_version (str): GPT model version to use
        
    Returns:
        dict: Generated adaptation plan
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
{paper_content}

## Dataset Description
{dataset_description}

## Variable Mapping
```json
{json.dumps(variable_mapping, indent=2)}
```

Based on the paper content and dataset description above, please create a detailed adaptation plan.
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
        tuple: Paths to the saved plan files (markdown, JSON)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full text as markdown
    plan_text_path = os.path.join(output_dir, f"{paper_name}_adaptation_plan.md")
    with open(plan_text_path, 'w') as file:
        file.write(plan["full_plan_text"])
    
    plan_json_path = None
    
    # Save structured JSON if available
    if "plan_json" in plan:
        plan_json_path = os.path.join(output_dir, f"{paper_name}_adaptation_plan.json")
        with open(plan_json_path, 'w') as file:
            json.dump(plan["plan_json"], file, indent=2)
    
    print(f"Adaptation plan saved to {plan_text_path}")
    if plan_json_path:
        print(f"Structured plan saved to {plan_json_path}")
    
    return plan_text_path, plan_json_path

def update_config(config_path, plan_paths):
    """
    Update the configuration file with the paths to the generated plans.
    
    Args:
        config_path (str): Path to the configuration file
        plan_paths (tuple): Paths to the saved plan files (markdown, JSON)
    """
    try:
        # Load existing config
        config = load_config(config_path)
        
        # Add advanced section if it doesn't exist
        if "advanced" not in config:
            config["advanced"] = {}
        
        # Update paths
        plan_text_path, plan_json_path = plan_paths
        config["advanced"]["adaptation_plan_path"] = plan_text_path
        
        # Save updated config
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Configuration updated with plan paths at {config_path}")
    
    except Exception as e:
        print(f"Error updating configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate adaptation plan directly from dataset description")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--paper_content", help="Path to the paper content file (JSON or markdown)")
    parser.add_argument("--dataset_description", help="Path to the dataset description file")
    parser.add_argument("--output_dir", help="Output directory for planning artifacts")
    parser.add_argument("--paper_name", default="AdaptedPaper", help="Name of the paper for file naming")
    parser.add_argument("--gpt_version", default="o3-mini-2025-04-16", help="GPT model version to use")
    parser.add_argument("--update_config", action="store_true", help="Update the config file with plan paths")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine paper content path
    paper_content_path = args.paper_content
    if not paper_content_path and "paper" in config and "json_path" in config["paper"]:
        paper_content_path = config["paper"]["json_path"]
    
    if not paper_content_path:
        print("Error: Paper content path not provided")
        print("Please provide a paper content path via --paper_content or in the config file")
        sys.exit(1)
    
    # Load paper content
    paper_content = load_paper_content(paper_content_path)
    
    # Determine dataset description path
    dataset_description_path = args.dataset_description
    if not dataset_description_path and "dataset" in config and "description_path" in config["dataset"]:
        dataset_description_path = config["dataset"]["description_path"]
    
    # Load dataset description if available
    dataset_description = ""
    if dataset_description_path:
        dataset_description = load_dataset_description(dataset_description_path)
    else:
        # If no description file, create a basic description from config
        if "dataset" in config:
            dataset_description = f"""
# Dataset Information

- Path: {config['dataset'].get('path', 'Not specified')}
- Format: {config['dataset'].get('format', 'Not specified')}

## Structure
This dataset contains variables that will be mapped to the original paper's variables.
"""
    
    # Determine variable mapping
    variable_mapping = {}
    if "variable_mapping" in config and "original_to_adapted" in config["variable_mapping"]:
        variable_mapping = config["variable_mapping"]["original_to_adapted"]
    
    # Determine output directory
    output_dir = args.output_dir
    if not output_dir and "output" in config and "output_dir" in config["output"]:
        output_dir = config["output"]["output_dir"]
    
    if not output_dir:
        output_dir = "outputs"
    
    # Determine paper name
    paper_name = args.paper_name
    if "output" in config and "repo_name" in config["output"]:
        paper_name = config["output"]["repo_name"]
    
    # Create output directory
    output_dir = os.path.join(output_dir, paper_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create OpenAI client
    client = create_openai_client()
    
    # Generate the plan
    plan = generate_adaptation_plan(client, paper_content, dataset_description, variable_mapping, args.gpt_version)
    
    # Save the plan
    plan_paths = save_plan(plan, output_dir, paper_name)
    
    # Update config if requested
    if args.update_config:
        update_config(args.config, plan_paths)
    
    print(f"Adaptation planning completed. Plan saved to {plan_paths[0]}")
    
    # Return the paths to the generated plan files
    return plan_paths

if __name__ == "__main__":
    main()