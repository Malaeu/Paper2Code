#!/usr/bin/env python3
"""
JSON Merger for Paper2Code - combines vision annotation results, code snippets,
and semantic passage links into an enriched Paper2Data format.
"""

import os
import sys
import json
import glob
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("json_merger.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("json_merger")

def load_json_file(file_path):
    """Load a JSON file, handling errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def load_code_snippets(snippets_dir):
    """Load all Python code snippets from the snippets directory."""
    snippets = {}
    
    # Find all Python files in the snippets directory
    snippet_files = glob.glob(os.path.join(snippets_dir, "fig_*.py"))
    
    for file_path in snippet_files:
        try:
            # Extract figure number from filename
            file_name = os.path.basename(file_path)
            fig_num = file_name.replace("fig_", "").replace(".py", "")
            
            # Read the code
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            snippets[fig_num] = code
            logger.info(f"Loaded code snippet for figure {fig_num}")
        
        except Exception as e:
            logger.error(f"Error loading code snippet {file_path}: {e}")
    
    return snippets

def merge_data(vision_data, passages_data, snippets_dir, plan_json_path=None):
    """
    Merge vision data, passages, and code snippets into the Paper2Data format.
    Optionally incorporate with an existing plan.json file.
    """
    # Load code snippets
    code_snippets = load_code_snippets(snippets_dir)
    
    # Initialize the enriched data structure
    enriched_data = {
        "pages": [],
        "figures": [],
        "tables": [],
        "passages": passages_data.get("passages", []) if passages_data else []
    }
    
    # Process vision data
    for item in vision_data:
        page_num = item.get("page")
        
        # Add to pages list
        page_entry = {
            "id": page_num,
            "content_type": item.get("object", "none")
        }
        enriched_data["pages"].append(page_entry)
        
        # Add to figures or tables list
        if item.get("object") == "figure":
            fig_id = f"fig_{page_num}"
            figure_entry = {
                "id": fig_id,
                "page": page_num,
                "chart_type": item.get("chart_type", "unknown"),
                "font_family": item.get("font_family", "unknown"),
                "palette": item.get("palette", []),
                "code": code_snippets.get(page_num, "# No code available for this figure")
            }
            
            # Add links to passages if available
            if passages_data:
                figure_entry["passages"] = [
                    p["id"] for p in passages_data.get("passages", [])
                    if p.get("references", []) and fig_id in p.get("references", [])
                ]
            
            enriched_data["figures"].append(figure_entry)
        
        elif item.get("object") == "table":
            table_entry = {
                "id": f"table_{page_num}",
                "page": page_num,
                "font_family": item.get("font_family", "unknown")
            }
            
            # Add links to passages if available
            if passages_data:
                table_entry["passages"] = [
                    p["id"] for p in passages_data.get("passages", [])
                    if p.get("references", []) and f"table_{page_num}" in p.get("references", [])
                ]
            
            enriched_data["tables"].append(table_entry)
    
    # If a plan.json path is provided, merge with it
    if plan_json_path and os.path.exists(plan_json_path):
        plan_data = load_json_file(plan_json_path)
        if plan_data:
            # Merge the plan data with our enriched data
            # The strategy here would depend on the specific structure of plan.json
            # For now, we'll simply add our enriched data as a new section
            plan_data["paper2data"] = enriched_data
            return plan_data
    
    return enriched_data

def main():
    parser = argparse.ArgumentParser(description="Merge vision data, passages, and code snippets into enriched Paper2Data format")
    parser.add_argument("--vision", required=True, help="Path to vision.json (output from vision_annotator)")
    parser.add_argument("--passages", required=True, help="Path to passages.json (output from semantic_linker)")
    parser.add_argument("--snippets", required=True, help="Directory containing figure code snippets")
    parser.add_argument("--plan", help="Optional path to plan.json to incorporate enriched data into")
    parser.add_argument("--output", required=True, help="Path to write the enriched.json output")
    
    args = parser.parse_args()
    
    # Load data
    vision_data = load_json_file(args.vision)
    passages_data = load_json_file(args.passages)
    
    if not vision_data:
        logger.error(f"Failed to load vision data from {args.vision}")
        return 1
    
    if not passages_data:
        logger.warning(f"No passages data found at {args.passages}, continuing without it")
    
    if not os.path.isdir(args.snippets):
        logger.warning(f"Snippets directory {args.snippets} does not exist, continuing without code snippets")
        args.snippets = "."
    
    # Merge data
    enriched_data = merge_data(
        vision_data, 
        passages_data, 
        args.snippets,
        args.plan
    )
    
    # Write output
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2)
        logger.info(f"Successfully wrote enriched data to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error writing enriched data to {args.output}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())