from anthropic import Anthropic
import json
from tqdm import tqdm
import argparse
import os
import sys
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--claude_model',type=str, default="claude-3-5-opus-20241022")  # Use Opus for coding
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")
parser.add_argument('--output_repo_dir',type=str, default="")

args = parser.parse_args()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

paper_name = args.paper_name
claude_model = args.claude_model
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
output_repo_dir = args.output_repo_dir

if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

# Load planning and analysis results
try:
    with open(f'{output_dir}/planning_trajectories.json', 'r', encoding='utf8') as f:
        planning_data = json.load(f)
        planning_result = planning_data.get("planning_result", "")
except FileNotFoundError:
    print("[WARNING] Planning result not found.")
    planning_result = ""

try:
    with open(f'{output_dir}/analyzing_trajectories.json', 'r', encoding='utf8') as f:
        analysis_data = json.load(f)
        analysis_result = analysis_data.get("analysis_result", "")
except FileNotFoundError:
    print("[WARNING] Analysis result not found.")
    analysis_result = ""

coding_msg = [
    {"role": "user", "content": f"""
You are a helpful assistant tasked with implementing code based on scientific papers.

Paper Content: {json.dumps(paper_content) if paper_format == "JSON" else paper_content}

Planning Context: {planning_result}

Analysis Context: {analysis_result}

Please generate a complete, working implementation based on the paper, planning, and analysis. Focus on:

1. **Main Implementation Files**: Core algorithm implementations
2. **Configuration Files**: Settings and parameters
3. **Utility Functions**: Helper functions and utilities
4. **Test Files**: Basic tests to verify functionality
5. **Documentation**: README and code documentation
6. **Requirements**: Dependencies and setup instructions

Please provide the implementation as a JSON structure with the following format:
{{
    "files": [
        {{
            "path": "relative/path/to/file.py",
            "content": "# Complete file content here\\n...",
            "description": "Description of this file"
        }}
    ],
    "structure": {{
        "directories": [...],
        "main_entry_point": "...",
        "key_files": [...]
    }},
    "setup_instructions": [
        "step 1",
        "step 2",
        "..."
    ]
}}

Make sure the code is:
- Complete and executable
- Well-documented with comments
- Follows Python best practices
- Includes proper error handling
- Has clear variable names and structure
"""}
]

print(f"[INFO] Starting coding stage for {paper_name} using {claude_model}")

try:
    response = client.messages.create(
        model=claude_model,
        max_tokens=8000,  # More tokens for code generation
        messages=coding_msg
    )
    
    coding_result = response.content[0].text
    
    # Save the coding result
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_repo_dir, exist_ok=True)
    
    coding_trajectories = {
        "coding_result": coding_result,
        "model": claude_model,
        "paper_name": paper_name
    }
    
    with open(f'{output_dir}/coding_trajectories.json', 'w', encoding='utf8') as f:
        json.dump(coding_trajectories, f, indent=2, ensure_ascii=False)
    
    # Try to extract and save actual files from the coding result
    try:
        # Look for JSON structure in the response
        import re
        json_match = re.search(r'\{.*\}', coding_result, re.DOTALL)
        if json_match:
            code_structure = json.loads(json_match.group())
            
            # Create files based on the structure
            if "files" in code_structure:
                for file_info in code_structure["files"]:
                    file_path = os.path.join(output_repo_dir, file_info["path"])
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf8') as f:
                        f.write(file_info["content"])
                    
                    print(f"[SUCCESS] Created file: {file_path}")
    except Exception as e:
        print(f"[WARNING] Could not extract files from coding result: {str(e)}")
    
    print(f"[SUCCESS] Coding completed and saved to {output_dir}/coding_trajectories.json")
    print_response(coding_result)
    
except Exception as e:
    print(f"[ERROR] Coding failed: {str(e)}")
    sys.exit(1)