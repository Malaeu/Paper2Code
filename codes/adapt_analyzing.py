#!/usr/bin/env python3
"""
Adaptation analysis script for Paper2Code.
Analyzes each component of the original code and creates an adapted version.
"""

import json
import argparse
import os
import sys
from openai import OpenAI
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--adapted_name', type=str, required=True)
    parser.add_argument('--gpt_version', type=str, required=True)
    parser.add_argument('--pdf_json_path', type=str, required=True)
    parser.add_argument('--mapping_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    paper_name = args.paper_name
    adapted_name = args.adapted_name
    gpt_version = args.gpt_version
    pdf_json_path = args.pdf_json_path
    mapping_file = args.mapping_file
    output_dir = args.output_dir
    
    # Load API key from environment
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Load JSON data
    with open(pdf_json_path, 'r') as f:
        paper_content = json.load(f)
    
    # Load variable mapping
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Load adaptation plan
    adaptation_plan_path = f'{output_dir}/adaptation_plan.md'
    if not os.path.exists(adaptation_plan_path):
        print(f"Error: Adaptation plan not found at {adaptation_plan_path}")
        print("Please run adapt_planning.py first.")
        sys.exit(1)
    
    with open(adaptation_plan_path, 'r') as f:
        adaptation_plan = f.read()
    
    # File components to analyze
    components = [
        "configs/default.yaml",
        "utils/logger.py",
        "data/dataset_loader.py", 
        "data/feature_engineering.py",
        "data/cohort_builder.py",
        "data/imputation.py",
        "models/model_factory.py",
        "models/orsf_wrapper.py",
        "trainer.py",
        "evaluation/metrics.py",
        "evaluation/calibration.py",
        "main.py"
    ]
    
    # Create artifacts directory
    artifact_output_dir = f'{output_dir}/analyzing_artifacts'
    os.makedirs(artifact_output_dir, exist_ok=True)
    
    for component in components:
        print(f"Analyzing adaptation for {component}")
        
        # Create component directory if it contains a path
        if '/' in component:
            component_dir = os.path.join(artifact_output_dir, os.path.dirname(component))
            os.makedirs(component_dir, exist_ok=True)
        
        # Create system prompt
        system_prompt = f"""You are an expert software architect and data scientist.
You will analyze how to adapt a specific component of a codebase to a new application context.
The component is from a paper implementation, and needs to be modified to work with a different dataset.

The original paper analyzes {paper_name}, and we are adapting it to analyze {adapted_name}.
You are specifically working on the {component} component.
"""
        
        # Create user prompt
        user_prompt = f"""
## Component to Adapt
{component}

## Original Paper
{json.dumps(paper_content, indent=2)}

## Adaptation Mapping
{json.dumps(mapping_data, indent=2)}

## Adaptation Plan
{adaptation_plan}

## Task
Provide a detailed analysis of how to adapt the {component} component to work with the new context:

1. Describe the original purpose and functionality of this component
2. List all variable names that need to be replaced based on the mapping
3. Outline any structural changes needed in this component
4. Identify any methodological adjustments specific to this component
5. Provide pseudocode or a detailed logic blueprint for the adapted component

Your analysis should be comprehensive enough that a programmer could implement the adaptation just from your description.
Ensure the adaptation maintains the methodological rigor of the original while working with the new variables and dataset.
"""
        
        # Create messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call API
        completion = client.chat.completions.create(
            model=gpt_version,
            messages=messages
        )
        
        # Process response
        completion_json = json.loads(completion.model_dump_json())
        print_response(completion_json)
        
        # Save component analysis
        component_analysis = completion.choices[0].message.content
        component_file = component.replace('/', '_')
        with open(f'{artifact_output_dir}/{component_file}_adaptation_analysis.md', 'w') as f:
            f.write(component_analysis)
        
        # Save response JSON
        with open(f'{output_dir}/{component_file}_adaptation_analysis_response.json', 'w') as f:
            json.dump(completion_json, f)
        
        # Save trajectories
        trajectories = messages + [{"role": "assistant", "content": component_analysis}]
        with open(f'{output_dir}/{component_file}_adaptation_analysis_trajectories.json', 'w') as f:
            json.dump(trajectories, f)
        
        print(f"Adaptation analysis for {component} saved.")
    
    print(f"All component adaptation analyses saved to {artifact_output_dir}")

if __name__ == "__main__":
    main()