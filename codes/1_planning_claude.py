from anthropic import Anthropic
import json
from tqdm import tqdm
import argparse
import os
import sys
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--claude_model',type=str, default="claude-3-5-sonnet-20241022")
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")

args = parser.parse_args()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

paper_name = args.paper_name
claude_model = args.claude_model
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir

if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

plan_msg = [
    {"role": "user", "content": f"""
You are a helpful assistant tasked with understanding scientific papers and creating implementation plans.

Paper Content: {json.dumps(paper_content) if paper_format == "JSON" else paper_content}

Please analyze this paper and create a detailed implementation plan that includes:

1. **Architecture Overview**: High-level description of the system architecture
2. **Key Components**: List of main components/modules to implement
3. **Dependencies**: Required libraries and frameworks
4. **File Structure**: Proposed directory and file organization
5. **Implementation Steps**: Step-by-step implementation plan
6. **Configuration**: Key parameters and settings needed

Please provide a comprehensive plan in JSON format with the following structure:
{{
    "paper_title": "...",
    "architecture_overview": "...",
    "key_components": [
        {{
            "name": "...",
            "description": "...",
            "dependencies": [...]
        }}
    ],
    "file_structure": {{
        "directories": [...],
        "files": [...]
    }},
    "implementation_steps": [
        {{
            "step": 1,
            "description": "...",
            "components": [...]
        }}
    ],
    "configuration": {{
        "parameters": [...],
        "settings": [...]
    }}
}}
"""}
]

print(f"[INFO] Starting planning stage for {paper_name} using {claude_model}")

try:
    response = client.messages.create(
        model=claude_model,
        max_tokens=4000,
        messages=plan_msg
    )
    
    planning_result = response.content[0].text
    
    # Save the planning result
    os.makedirs(output_dir, exist_ok=True)
    
    planning_trajectories = {
        "planning_result": planning_result,
        "model": claude_model,
        "paper_name": paper_name
    }
    
    with open(f'{output_dir}/planning_trajectories.json', 'w', encoding='utf8') as f:
        json.dump(planning_trajectories, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Planning completed and saved to {output_dir}/planning_trajectories.json")
    print_response(planning_result)
    
except Exception as e:
    print(f"[ERROR] Planning failed: {str(e)}")
    sys.exit(1)