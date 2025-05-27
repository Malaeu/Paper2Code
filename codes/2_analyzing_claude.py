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

# Load planning result
try:
    with open(f'{output_dir}/planning_trajectories.json', 'r', encoding='utf8') as f:
        planning_data = json.load(f)
        planning_result = planning_data.get("planning_result", "")
except FileNotFoundError:
    print("[WARNING] Planning result not found. Proceeding without it.")
    planning_result = ""

analysis_msg = [
    {"role": "user", "content": f"""
You are a helpful assistant tasked with detailed analysis of scientific papers for implementation.

Paper Content: {json.dumps(paper_content) if paper_format == "JSON" else paper_content}

Planning Context: {planning_result}

Please perform a detailed analysis of this paper focusing on:

1. **Algorithm Details**: Step-by-step breakdown of key algorithms
2. **Mathematical Formulations**: Important equations and their implementations
3. **Data Structures**: Required data structures and formats
4. **Model Architecture**: Detailed model/system architecture
5. **Training/Execution Procedures**: How the system should be trained/executed
6. **Evaluation Metrics**: How to measure performance
7. **Implementation Considerations**: Technical details and considerations

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "algorithms": [
        {{
            "name": "...",
            "description": "...",
            "steps": [...],
            "complexity": "..."
        }}
    ],
    "mathematical_formulations": [
        {{
            "equation": "...",
            "description": "...",
            "implementation_notes": "..."
        }}
    ],
    "data_structures": [
        {{
            "name": "...",
            "type": "...",
            "description": "...",
            "format": "..."
        }}
    ],
    "model_architecture": {{
        "overview": "...",
        "components": [...],
        "connections": [...]
    }},
    "procedures": {{
        "training": "...",
        "inference": "...",
        "evaluation": "..."
    }},
    "evaluation_metrics": [...],
    "implementation_considerations": [...]
}}
"""}
]

print(f"[INFO] Starting analysis stage for {paper_name} using {claude_model}")

try:
    response = client.messages.create(
        model=claude_model,
        max_tokens=4000,
        messages=analysis_msg
    )
    
    analysis_result = response.content[0].text
    
    # Save the analysis result
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_trajectories = {
        "analysis_result": analysis_result,
        "model": claude_model,
        "paper_name": paper_name
    }
    
    with open(f'{output_dir}/analyzing_trajectories.json', 'w', encoding='utf8') as f:
        json.dump(analysis_trajectories, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Analysis completed and saved to {output_dir}/analyzing_trajectories.json")
    print_response(analysis_result)
    
except Exception as e:
    print(f"[ERROR] Analysis failed: {str(e)}")
    sys.exit(1)