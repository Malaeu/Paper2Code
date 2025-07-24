#!/usr/bin/env python3
"""
Visual Data Extraction module for Paper2Code Enhanced.

This script processes pre-extracted images of figures, uses a multimodal LLM to detect
figures and tables, extracts their captions and context, and generates detailed
JSON descriptions for each visual element.
"""

import os
import json
import argparse
from PIL import Image # For image handling and cropping

# --- CONFIGURATION ---
VISUAL_ANALYSIS_MODEL = "o4-mini-2025-04-16"

# --- HELPER FUNCTIONS ---

def get_page_images(pages_dir):
    """Lists all PNG page images in the specified directory."""
    page_files = []
    for f_name in sorted(os.listdir(pages_dir)):
        if f_name.lower().endswith(".png"):
            page_files.append(os.path.join(pages_dir, f_name))
    return page_files

def build_detailed_description_prompt(caption_text, surrounding_text=""):
    """Builds the master prompt for getting detailed JSON description of a visual element."""
    prompt_template = f"""\
[System] You are a scientific visual analyst.
[User]
Attached is a visual element (figure or table) and its caption.
Caption: "{caption_text}"
Surrounding context (if available): "{surrounding_text}"

Provide a detailed JSON description of this visual element, suitable for
allowing another AI to understand it or attempt to recreate a similar visual.
Fill in the following fields. If information for a field is not clearly available,
use null or an empty string/list as appropriate for the field type.

JSON Structure:
{{
    "type": "<e.g., Kaplan-Meier plot, bar chart, table, flowchart, etc.>",
    "title_in_figure": "<Title text found directly on the figure/table, if any>",
    "x_axis": {{"label": "<X-axis label, if applicable>", "units": "<X-axis units>", "range": "<X-axis range>"}},
    "y_axis": {{"label": "<Y-axis label, if applicable>", "units": "<Y-axis units>", "range": "<Y-axis range>"}},
    "legend": [{{"label": "<Legend item 1 label>", "style": "<e.g., solid blue line>"}}, ...],
    "nodes_and_edges": {{ "nodes": [{{ "id": "node1", "label": "Step 1", "shape": "rectangle" }}], "edges": [{{"from": "node1", "to": "node2", "label": "arrow"}}] }}, 
    "data_representation": "<Description of what data is shown and how>",
    "key_patterns": "<Key visual patterns, trends, or comparisons evident>",
    "statistical_annotations": "<e.g., p-values, error bars, significance stars mentioned>",
    "main_conclusion": "<Main conclusion derivable from this visual, often from caption/context>",
    "style_description": {{"colors": ["<list of prominent colors>"], "fonts": "<font style description>", "overall_look": "<e.g., academic, clean, dense>"}},
    "recreation_prompt_hint": "<A concise hint for an AI trying to recreate this, e.g., 'Flowchart with 6 steps showing model development.'>"
}}
"""
    return prompt_template

def call_llm_for_detailed_description(prompt_text, image_path_or_pil_image, model_name):
    """
    Calls the multimodal LLM for detailed description of a *specific, cropped* visual element.
    Returns parsed JSON or raw string on error.
    """
    image_display_name = image_path_or_pil_image if isinstance(image_path_or_pil_image, str) else "PIL Image object"
    print(f"\n--- SIMULATING DETAILED LLM CALL for {model_name} ---")
    print(f"Image: {image_display_name}")
    
    # For the true Figure 1 (flowchart), the simulated response needs to be updated.
    # This is a placeholder for now, assumes it's still the Kaplan-Meier plot for simulation purposes.
    # We will update this when we have the actual cropped flowchart image.
    simulated_json_str = '''
    {
        "type": "flowchart",
        "title_in_figure": "Figure 1. Analysis overview for identifying best-performing risk prediction model.",
        "x_axis": null,
        "y_axis": null,
        "legend": [{"label": "Cohort", "style": "blue box"}, {"label": "Training/Derivation steps", "style": "orange box"}, {"label": "Validation steps", "style": "green box"}],
        "nodes_and_edges": {
            "nodes": [
                {"id": "step1", "label": "Step 1: Split into 50:50 Training and Validation. Impute missing data separately", "shape": "rectangle"},
                {"id": "jhs_cohort", "label": "4141 Individuals in the JHS derivation cohort", "shape": "rectangle"},
                {"id": "aric_cohort", "label": "7858 Individuals in ARIC the derivation cohort", "shape": "rectangle"},
                {"id": "training_set", "label": "Training Set", "shape": "rectangle"},
                {"id": "validation_set", "label": "Validation Set", "shape": "rectangle"},
                {"id": "step2", "label": "Step 2: Apply candidate learning algorithms...", "shape": "rectangle"},
                {"id": "step3", "label": "Step 3: Validate using validation dataset", "shape": "rectangle"},
                {"id": "step4", "label": "Step 4: Calculate C-index", "shape": "rectangle"},
                {"id": "step5", "label": "Step 5: Select best performing algorithm and retrain...", "shape": "rectangle"},
                {"id": "step6", "label": "Step 6: Validate algorithm in external datasets", "shape": "rectangle"}
            ],
            "edges": [
                {"from": "jhs_cohort", "to": "step1"},
                {"from": "aric_cohort", "to": "training_set"},
                {"from": "step1", "to": "training_set"},
                {"from": "step1", "to": "validation_set"},
                {"from": "training_set", "to": "step2"},
                {"from": "validation_set", "to": "step3"},
                {"from": "step3", "to": "step4"},
                {"from": "step4", "to": "step1", "label": "Repeat 1000 times"},
                {"from": "step4", "to": "step5"},
                {"from": "step2", "to": "step5"},
                {"from": "step5", "to": "step6"}
            ]
        },
        "data_representation": "Flowchart illustrating the steps for developing and validating a risk prediction model.",
        "key_patterns": "Iterative process involving training, validation, and external testing across different cohorts (JHS and ARIC).",
        "statistical_annotations": "C-index mentioned as a performance metric.",
        "main_conclusion": "A multi-step process is used to identify the best-performing risk prediction model.",
        "style_description": {"colors": ["blue", "orange", "green", "black"], "fonts": "Sans-serif", "overall_look": "Diagrammatic, academic"},
        "recreation_prompt_hint": "Create a flowchart depicting a 6-step model development and validation process, involving two cohorts (JHS and ARIC), training/validation sets, and an iterative loop."
    }
    '''
    print("--- DETAILED SIMULATION END ---")
    try:
        return json.loads(simulated_json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding DETAILED simulated JSON: {e}")
        return simulated_json_str

def call_llm_to_detect_elements_on_page(page_image_path, model_name):
    """
    Placeholder for LLM call to detect all visual elements on a full page image.
    Should return a list of dictionaries, each describing a detected element:
    e.g., [{'id': 'pageN_figM', 'type': 'figure'/'table', 'bbox': [x1,y1,x2,y2], 
            'caption': 'text', 'context_text': 'text'}]
    For initial test, we will simulate finding the REAL Figure 1 (flowchart).
    """
    print(f"\n--- SIMULATING ELEMENT DETECTION LLM CALL for {model_name} on {os.path.basename(page_image_path)} ---")
    
    # Simulate finding the flowchart (actual Figure 1) on a specific page
    # We need to know which page it's on. Let's assume it's on 'page_with_flowchart.png'
    # and its bounding box for cropping.
    # The caption is also crucial.
    if os.path.basename(page_image_path) == "page_with_flowchart.png": # Replace with actual page filename
        simulated_elements = [
            {
                "id": "page_X_fig_1_flowchart",
                "type": "figure",
                "bbox": [50, 1500, 2400, 3100], # Example BBox: [left, top, right, bottom] -> ADJUST THIS!
                "caption": "Figure 1. Analysis overview for identifying best-performing risk prediction model. ARIC indicates Atherosclerosis Risk in Communities; GBT, gradient boosted trees; JHS, Jackson Heart Study; and oRSF, oblique random survival forest.",
                "context_text": "The analysis overview is shown in Figure 1."
            }
            # Potentially other elements detected on this page
        ]
        print(f"Simulated: Found flowchart on {os.path.basename(page_image_path)}")
        return simulated_elements
    else:
        print(f"Simulated: No specific elements targeted for detection on {os.path.basename(page_image_path)}")
        return []

# --- MAIN LOGIC ---
def main():
    parser = argparse.ArgumentParser(description="Detect visual elements on PDF page images and extract detailed descriptions.")
    parser.add_argument("--page_images_dir", type=str, required=True,
                        help="Directory containing PNG images of PDF pages.")
    parser.add_argument("--output_json_path", type=str, required=True,
                        help="Path to save the output JSON with visual element descriptions.")
    # parser.add_argument("--target_figure_id", type=str, default="Figure 1", help="Focus on a specific figure ID if needed for testing, e.g., 'Figure 1'")

    args = parser.parse_args()

    if not os.path.isdir(args.page_images_dir):
        print(f"Error: Page images directory not found: {args.page_images_dir}")
        return

    page_image_files = get_page_images(args.page_images_dir)
    if not page_image_files:
        print(f"No PNG images found in {args.page_images_dir}")
        return

    all_detailed_visual_elements = []

    print(f"Found {len(page_image_files)} page images to process.")

    for page_path in page_image_files:
        print(f"\nProcessing page: {os.path.basename(page_path)}")
        
        detected_elements = call_llm_to_detect_elements_on_page(page_path, VISUAL_ANALYSIS_MODEL)

        for element in detected_elements:
            print(f"  Detected element: {element.get('id', 'N/A')}, type: {element.get('type', 'N/A')}")
            
            # Crop the image using PIL and the bounding box
            try:
                page_img_pil = Image.open(page_path)
                bbox = element.get('bbox')
                if not bbox or len(bbox) != 4:
                    print(f"    Skipping element {element.get('id')} due to missing/invalid bbox.")
                    continue
                
                cropped_element_pil = page_img_pil.crop(bbox)
                # cropped_element_pil.save(f"./temp_crop_{element.get('id')}.png") # Optional: save crop for inspection
                print(f"    Successfully cropped element {element.get('id')}.")

                caption = element.get('caption', 'No caption found.')
                context = element.get('context_text', '')
                prompt_text = build_detailed_description_prompt(caption, context)
                
                detailed_description_json = call_llm_for_detailed_description(
                    prompt_text, 
                    cropped_element_pil, # Pass the PIL image object
                    VISUAL_ANALYSIS_MODEL
                )

                if isinstance(detailed_description_json, dict):
                    print(f"    Successfully received and parsed detailed LLM response for {element.get('id')}.")
                else:
                    print(f"    Detailed LLM response for {element.get('id')} was not valid JSON.")
                    if isinstance(detailed_description_json, str):
                        detailed_description_json = {"raw_llm_response": detailed_description_json}
                
                all_detailed_visual_elements.append({
                    "source_page": os.path.basename(page_path),
                    "detected_element_id": element.get('id'),
                    "detected_type": element.get('type'),
                    "detected_caption": caption,
                    "llm_detailed_description": detailed_description_json
                })

            except Exception as e:
                print(f"    Error processing element {element.get('id', 'N/A')} on page {os.path.basename(page_path)}: {e}")

    with open(args.output_json_path, 'w') as f:
        json.dump(all_detailed_visual_elements, f, indent=4)

    if all_detailed_visual_elements:
        print(f"\nProcessing complete. Output saved to {args.output_json_path}")
    else:
        print(f"\nProcessing complete. No detailed descriptions generated. Output file: {args.output_json_path}")

    print("Next step: Review the JSON. Implement real LLM calls for detection and refine bbox/page for Figure 1.")

if __name__ == "__main__":
    main()
