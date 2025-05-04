#!/usr/bin/env python3
"""
Adaptation mapping module for Paper2Code.
Analyzes user datasets and generates/confirms variable mappings for methodology adaptation.
"""

import json
import argparse
import os
import sys
import pandas as pd
import yaml
from openai import OpenAI
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

def detect_dataset_structure(dataset_path):
    """
    Automatically detect the structure of the dataset.
    Returns column names, types, and sample values.
    """
    try:
        # Determine file type
        file_ext = os.path.splitext(dataset_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(dataset_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(dataset_path)
        elif file_ext == '.json':
            df = pd.read_json(dataset_path)
        else:
            return {"error": f"Unsupported file format: {file_ext}"}
        
        # Get column information
        columns = list(df.columns)
        types = df.dtypes.to_dict()
        types = {col: str(dtype) for col, dtype in types.items()}
        
        # Get a few sample values for each column
        samples = {}
        for col in columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                # For large arrays or complex objects, just take string representation of first few
                if hasattr(unique_vals[0], '__len__') and not isinstance(unique_vals[0], str):
                    samples[col] = [str(val)[:100] for val in unique_vals[:3]]
                else:
                    samples[col] = [str(val) for val in unique_vals[:3]]
        
        # Get basic stats
        stats = {
            "row_count": len(df),
            "column_count": len(columns)
        }
        
        return {
            "columns": columns,
            "types": types,
            "samples": samples,
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e)}

def generate_mapping_template(paper_content, dataset_structure, original_name, adapted_name, gpt_version, client, dataset_path=None):
    """
    Generate a mapping template based on the paper content and dataset structure.
    """
    system_prompt = f"""You are an expert data scientist specializing in methodology adaptation.
You will analyze a scientific paper and a user's dataset to create a variable mapping between them.
The original paper studies {original_name}, and the user wants to adapt it to study {adapted_name}.

Your task is to identify key variables in the original paper and suggest corresponding variables from the user's dataset.
Focus on creating a direct, one-to-one mapping between conceptually similar variables.
"""
    
    # Load dataset descriptions if available
    dataset_descriptions = []
    
    # Check if dataset_path is provided
    description_paths = []
    if dataset_path:
        description_paths = [
            os.path.join(os.path.dirname(dataset_path), "dataset_description.md"),
            os.path.join(os.path.dirname(dataset_path), "dataset_description_engl.md"),
            os.path.join(os.path.dirname(dataset_path), "variable_descriptions.md"),
            os.path.join(os.path.dirname(dataset_path), "dataset_def.md")
        ]
    
    for path in description_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    dataset_descriptions.append(f"From {os.path.basename(path)}:\n{f.read()}")
            except Exception as e:
                print(f"Warning: Could not read dataset description from {path}: {e}")
    
    dataset_descriptions_text = "\n\n====================\n\n".join(dataset_descriptions)
    
    user_prompt = f"""
## Original Paper
{json.dumps(paper_content, indent=2)}

## User's Dataset Structure
Columns: {dataset_structure['columns']}
Types: {dataset_structure['types']}
Sample Values: {dataset_structure['samples']}
Stats: {dataset_structure['stats']}

## Dataset Descriptions
{dataset_descriptions_text if dataset_descriptions else "No additional dataset descriptions provided."}

## Task
Create a mapping between variables in the original paper and equivalent variables in the user's dataset:

1. Identify the key variables/concepts in the original paper (e.g., demographic variables, outcomes, predictors)
2. Find corresponding variables in the user's dataset based on name similarity, data type, and sample values
3. Carefully use the dataset descriptions to understand the meaning of each variable in the user's dataset
4. Create a mapping in this format:
   ```json
   {{
       "original_to_adapted": {{
           "original_var1": "user_var1",
           "original_var2": "user_var2",
           "original_category1": "user_category1"
       }},
       "adapted_dataset_path": "path/to/dataset",
       "adapted_dataset_format": "csv",
       "methodology_adjustments": {{
           "maintain_key_analysis_feature": true,
           "iterations": 1000
       }}
   }}
   ```

5. Include only variables that have a clear correspondence
6. For categorical variables, include mappings between category values (e.g., "Black" → "female", "White" → "male")
7. For the original paper on heart failure prediction, consider mappings to liver disease progression metrics in the user's dataset
8. Include methodology adjustments that should be maintained in the adaptation

Only output the JSON mapping without any additional explanation or text.
"""
    
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
    
    # Extract mapping JSON from response
    response_content = completion.choices[0].message.content
    
    # Try to extract JSON from the response
    try:
        # If response is already a JSON string
        mapping = json.loads(response_content)
    except json.JSONDecodeError:
        # If JSON is inside markdown code blocks
        if "```json" in response_content and "```" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
            try:
                mapping = json.loads(json_str)
            except json.JSONDecodeError:
                # If that fails too, return a basic template
                mapping = {
                    "original_to_adapted": {},
                    "adapted_dataset_path": "",
                    "adapted_dataset_format": "",
                    "methodology_adjustments": {}
                }
        else:
            # Return a basic template
            mapping = {
                "original_to_adapted": {},
                "adapted_dataset_path": "",
                "adapted_dataset_format": "",
                "methodology_adjustments": {}
            }
    
    return mapping, completion_json, messages + [{"role": "assistant", "content": response_content}]

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--adapted_name', type=str, required=True)
    parser.add_argument('--gpt_version', type=str, required=True)
    parser.add_argument('--pdf_json_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_mapping_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    paper_name = args.paper_name
    adapted_name = args.adapted_name
    gpt_version = args.gpt_version
    pdf_json_path = args.pdf_json_path
    dataset_path = args.dataset_path
    output_mapping_path = args.output_mapping_path
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load API key from environment
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Load JSON data
    with open(pdf_json_path, 'r') as f:
        paper_content = json.load(f)
    
    # Check if user already provided a mapping
    if os.path.exists(output_mapping_path):
        print(f"Mapping file already exists at {output_mapping_path}")
        print("Using existing mapping file.")
        with open(output_mapping_path, 'r') as f:
            mapping = json.load(f)
        return
    
    # Detect dataset structure
    print(f"Analyzing dataset structure at {dataset_path}...")
    dataset_structure = detect_dataset_structure(dataset_path)
    
    if "error" in dataset_structure:
        print(f"Error analyzing dataset: {dataset_structure['error']}")
        sys.exit(1)
    
    # Generate mapping template
    print(f"Generating variable mapping from {paper_name} to {adapted_name}...")
    mapping, completion_json, trajectories = generate_mapping_template(
        paper_content, dataset_structure, paper_name, adapted_name, gpt_version, client, dataset_path
    )
    
    # Update mapping with dataset path
    mapping["adapted_dataset_path"] = dataset_path
    mapping["adapted_dataset_format"] = os.path.splitext(dataset_path)[1].replace('.', '')
    
    # Save mapping template
    with open(output_mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    # Save response JSON
    with open(f'{output_dir}/mapping_generation_response.json', 'w') as f:
        json.dump(completion_json, f)
    
    # Save trajectories
    with open(f'{output_dir}/mapping_generation_trajectories.json', 'w') as f:
        json.dump(trajectories, f)
    
    print(f"Variable mapping template created at {output_mapping_path}")
    print("Please review and edit this mapping before continuing with adaptation.")

if __name__ == "__main__":
    main()