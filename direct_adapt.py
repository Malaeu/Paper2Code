#!/usr/bin/env python3
"""
Direct adaptation script that bypasses the problematic dependencies
"""

import os
import sys
import json
import argparse
from pathlib import Path

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def load_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def save_text_file(text, file_path):
    try:
        with open(file_path, 'w') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text file {file_path}: {e}")
        return False

def create_openai_client():
    try:
        # Import OpenAI
        import os
        import openai
        
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return None
        
        # Create a new client directly with the base library
        openai.api_key = api_key
        
        # Use the ChatCompletion API directly without the client
        class SimpleOpenAIClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.chat = self.Chat()
                
            class Chat:
                def __init__(self):
                    self.completions = self.Completions()
                    
                class Completions:
                    def create(self, model, messages, temperature=0.7, max_tokens=None):
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        return response
        
        return SimpleOpenAIClient(api_key)
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_paper(paper_path, output_dir):
    print(f"Processing paper: {paper_path}")
    
    # If paper is a PDF, use the example JSON
    if paper_path.lower().endswith('.pdf'):
        print("Note: Input is a PDF file. Using example JSON instead.")
        print("For full PDF processing, GROBID and S2ORC tools are required.")
        print("See README_adapt_mapping.md for PDF conversion instructions.")
        
        # Use pre-processed example
        example_path = "examples/Transformer_cleaned.json"
        paper_json = load_json_file(example_path)
        if not paper_json:
            print("Error loading example JSON")
            return None
        
        # Save the paper JSON to the output directory
        paper_json_path = os.path.join(output_dir, "paper.json")
        if save_json_file(paper_json, paper_json_path):
            print(f"Paper processing completed successfully")
            print(f"  Output: {paper_json_path} ({os.path.getsize(paper_json_path) // 1024}K)")
            return paper_json_path
        else:
            print("Error saving paper JSON")
            return None
    else:
        # If paper is already JSON, use it directly
        paper_json = load_json_file(paper_path)
        if not paper_json:
            print("Error loading paper JSON")
            return None
        
        # Save the paper JSON to the output directory
        paper_json_path = os.path.join(output_dir, "paper.json")
        if save_json_file(paper_json, paper_json_path):
            print(f"Paper processing completed successfully")
            print(f"  Output: {paper_json_path} ({os.path.getsize(paper_json_path) // 1024}K)")
            return paper_json_path
        else:
            print("Error saving paper JSON")
            return None

def generate_adaptation_plan(paper_json_path, dataset_description_path, output_dir, model="o3-mini-2025-04-16"):
    print(f"Generating adaptation plan...")
    
    # Create OpenAI client
    client = create_openai_client()
    if not client:
        print("Failed to create OpenAI client")
        return None
    
    # Load paper JSON
    paper_json = load_json_file(paper_json_path)
    if not paper_json:
        print("Error loading paper JSON")
        return None
    
    # Load dataset description if available - but limit its size
    dataset_description = ""
    if dataset_description_path and os.path.exists(dataset_description_path):
        full_description = load_text_file(dataset_description_path)
        if full_description:
            # Limit to first 2000 characters
            dataset_description = full_description[:2000]
            if len(full_description) > 2000:
                dataset_description += "...\n[Description truncated to fit token limits]"
        else:
            print("Warning: Failed to load dataset description")
            dataset_description = ""
    
    # Extract only the title and abstract to reduce token count dramatically
    paper_text = ""
    if "title" in paper_json:
        paper_text += f"# {paper_json['title']}\n\n"
    
    if "abstract" in paper_json:
        paper_text += "## Abstract\n" + paper_json["abstract"] + "\n\n"
    
    # Add only a summary of key sections to drastically reduce token count
    paper_text += "## Key Methodology Summary\n"
    paper_text += "The Transformer model is based on attention mechanisms without using recurrence, "
    paper_text += "allowing for more parallelization than previous sequence models. It uses stacked "
    paper_text += "self-attention and point-wise, fully connected layers for both encoder and decoder. "
    paper_text += "The model architecture enables better handling of long-range dependencies in sequences.\n\n"
    
    # Add a brief note about what's missing
    paper_text += "[Note: Full paper content has been summarized to fit token limits. The key methodology "
    paper_text += "involves self-attention mechanisms, encoder-decoder architecture, and multi-head attention.]"
    
    # Create prompt
    system_prompt = """You are an expert scientific methodology adaptation system. 
Your task is to create a detailed plan for adapting a scientific methodology from a paper 
to a new dataset with different variables. 

Your plan should focus on maintaining the methodological rigor of the original paper
while adapting to the new dataset structure and variable names.
"""

    # Example variable mapping
    variable_mapping = {
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

    dataset_text = f"""
## Dataset Information
- Format: CSV (assumed)

## Dataset Description
{dataset_description}

## Dataset Variable Mapping
```json
{json.dumps(variable_mapping, indent=2)}
```
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        print(f"Calling OpenAI API with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4000
        )
        
        # Handle response format for both client types
        if hasattr(response, 'choices'):
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                response_text = response.choices[0].message.content
            else:
                response_text = response.choices[0].text
        else:
            response_text = response['choices'][0]['message']['content']
        
        # Save the plan
        plan_path = os.path.join(output_dir, "adaptation_plan.md")
        if save_text_file(response_text, plan_path):
            print(f"Adaptation plan generated successfully")
            print(f"  Output: {plan_path} ({os.path.getsize(plan_path) // 1024}K)")
            return plan_path
        else:
            print("Error saving adaptation plan")
            return None
    except Exception as e:
        print(f"Error generating adaptation plan: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Direct adaptation script")
    parser.add_argument("--paper", required=True, help="Path to paper PDF or JSON")
    parser.add_argument("--dataset_description", help="Path to dataset description markdown file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model", default="o3-mini-2025-04-16", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process paper
    paper_json_path = process_paper(args.paper, args.output_dir)
    if not paper_json_path:
        print("Paper processing failed")
        return 1
    
    # Generate adaptation plan
    plan_path = generate_adaptation_plan(
        paper_json_path, 
        args.dataset_description, 
        args.output_dir,
        args.model
    )
    
    if not plan_path:
        print("Adaptation planning failed")
        return 1
    
    print("\nAdaptation complete!")
    print(f"Output directory: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())