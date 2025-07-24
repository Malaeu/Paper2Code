#!/usr/bin/env python3
"""
Extract figures from PDFs and get LLM descriptions for them.
This enhances the JSON output with image paths and LLM-generated descriptions.
"""

import os
import json
import base64
import argparse
from pdf2image import convert_from_path
from openai import OpenAI
import fitz # PyMuPDF: Ensure this is installed (e.g., pip install PyMuPDF)

def render_pdf_pages(pdf_path, output_dir):
    """Render all PDF pages as PNGs and return list of dicts with page/image info."""
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    print(f"Rendering PDF pages from {pdf_path} to {pages_dir}")
    page_images = convert_from_path(pdf_path, dpi=300)
    page_info = []
    for i, page in enumerate(page_images, start=1):
        img_name = f"page_{i}.png"
        img_path = os.path.join(pages_dir, img_name)
        if os.path.exists(img_path):
            print(f"[SKIP] Page {i} already exists as {img_path}")
        else:
            page.save(img_path, "PNG")
            print(f"[OK] Saved page {i} as {img_path}")
        page_info.append({
            "page": i,
            "image_path": img_path
        })
    return page_info

# Original Figure-Extraktion bleibt erhalten (siehe unten)
def extract_figures(pdf_path, output_dir):
    """Extract figures from a PDF file using PyMuPDF."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Extracting figures from {pdf_path} to {figures_dir}")
    pdf = fitz.open(pdf_path)
    figure_info = []
    for page_num in range(pdf.page_count):
        page = pdf[page_num]
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                fig_num = len(figure_info) + 1
                image_filename = f"figure{fig_num}.{image_ext}"
                image_path = os.path.join(figures_dir, image_filename)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                figure_info.append({
                    "fig_num": fig_num,
                    "page": page_num + 1,
                    "image_path": f"./figures/{image_filename}",
                    "xref": xref,
                    "ext": image_ext
                })
                print(f"Extracted figure {fig_num} from page {page_num + 1} with extension .{image_ext}")
            except Exception as e:
                print(f"Error extracting image at page {page_num + 1}, index {img_idx}: {e}")
    return figure_info

def get_llm_descriptions(figure_info, json_data, paper_title, output_dir, gpt_version="o3-mini"):
    """Get LLM descriptions for figures using the OpenAI API."""
    # Ensure OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        # Optionally, exit or raise an error
        # return json_data # Or handle as appropriate
        raise EnvironmentError("OPENAI_API_KEY not set. Please set it before running the script.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    print(f"Getting LLM descriptions using {gpt_version}")
    
    # Create ref_entries if it doesn't exist
    if "ref_entries" not in json_data:
        json_data["ref_entries"] = {}
        
    ref_entries = json_data["ref_entries"]
    
    # First attempt to match extracted figures with existing ref_entries
    for fig in figure_info:
        matched = False
        for ref_id, ref_entry in ref_entries.items():
            if ref_entry.get("type") == "figure" and ref_entry.get("page") == fig["page"]:
                # Found a potential match
                matched = True
                image_path = fig["image_path"]
                
                # Read the image and encode it as base64
                with open(os.path.join(output_dir, "figures", f"figure{fig['fig_num']}.{fig['ext']}"), "rb") as img_file:
                    image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                # Generate a prompt for the LLM
                prompt = f"""
This is Figure {ref_entry.get('fig_num', 'N/A')} from the scientific paper "{paper_title}".
The figure caption is: "{ref_entry.get('text', 'No caption available')}"

Analyze this figure and provide a detailed technical description of what it shows.
Your analysis should:
1. Explain the visual data, patterns, or processes depicted
2. Connect the figure to the main concepts in the paper
3. Interpret any technical details such as graphs, plots, or diagrams
4. Highlight key findings or insights that this figure demonstrates
5. Avoid just repeating the caption - provide deeper analysis

Your description should be scientific, precise, and include technical details that would be relevant for code implementation.
"""
                
                try:
                    # Get the LLM description using the Vision API
                    response = client.chat.completions.create(
                        model=gpt_version,
                        messages=[
                            {"role": "system", "content": "You are a scientific assistant that provides detailed descriptions of figures from academic papers."},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/{fig['ext']};base64,{image_base64}"}}
                            ]}
                        ]
                    )
                    
                    llm_caption = response.choices[0].message.content
                    
                    # Update the ref_entry with the image path and LLM caption
                    ref_entry["image_path"] = image_path
                    ref_entry["llm_caption"] = llm_caption
                    
                    print(f"Added LLM description for {ref_id} (Figure {ref_entry.get('fig_num', 'N/A')})")
                    break
                    
                except Exception as e:
                    print(f"Error getting LLM description: {e}")
        
        # If no matching ref_entry was found, create a new one
        if not matched:
            print(f"Creating new ref_entry for figure {fig['fig_num']} on page {fig['page']}")
            
            image_path = fig["image_path"]
            
            # Read the image and encode it as base64
            with open(os.path.join(output_dir, "figures", f"figure{fig['fig_num']}.{fig['ext']}"), "rb") as img_file:
                image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Generate a prompt for the LLM to analyze the image
            prompt = f"""
This is a figure from page {fig['page']} of the scientific paper "{paper_title}".
No caption was available in the parsed document.

Analyze this figure and provide a detailed technical description of what it shows.
Your analysis should:
1. Explain the visual data, patterns, or processes depicted in detail
2. Identify the figure type (graph, diagram, chart, illustration, etc.)
3. Interpret any technical details such as axes, scales, or relationships shown
4. Make connections to potential scientific concepts based on what you see
5. Describe how this figure might contribute to the paper's methodology or findings

Your description should be scientific, precise, and include technical details that would be relevant for code implementation.
"""
            
            try:
                # Get the LLM description using the Vision API
                response = client.chat.completions.create(
                    model=gpt_version,
                    messages=[
                        {"role": "system", "content": "You are a scientific assistant that provides detailed descriptions of figures from academic papers."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{fig['ext']};base64,{image_base64}"}}
                        ]}
                    ]
                )
                
                llm_caption = response.choices[0].message.content
                
                # Create a new ref_entry
                ref_id = f"FIGREF_EXTRACTED_{fig['fig_num']}"
                ref_entries[ref_id] = {
                    "type": "figure",
                    "text": "No caption available in the parsed document",
                    "fig_num": str(fig['fig_num']),
                    "page": fig['page'],
                    "image_path": image_path,
                    "llm_caption": llm_caption,
                    "extracted_by_script": True
                }
                
                print(f"Created new ref_entry {ref_id} with LLM description")
                
            except Exception as e:
                print(f"Error getting LLM description: {e}")
    
    return json_data

def enhance_json_with_figures(pdf_path, json_path, output_dir, gpt_version="o3-mini"):
    """Enhance the JSON output with figure information and LLM descriptions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load existing JSON data
    print(f"Loading JSON data from {json_path}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    paper_title = json_data.get("title", "Unknown Paper")

    # Extract figures using PyMuPDF
    # Figures will be saved in a 'figures' subdirectory of output_dir
    figure_info = extract_figures(pdf_path, output_dir)

    # Get LLM descriptions and update json_data
    # This function modifies json_data in place by adding 'llm_caption' and 'image_path' to ref_entries
    # or creating new ref_entries for extracted figures.
    # Figure image paths in figure_info are relative like './figures/figureX.ext'
    # These are relative to the 'output_dir' where figures are saved.
    json_data_enhanced = get_llm_descriptions(figure_info, json_data, paper_title, output_dir, gpt_version)

    # Define the output path for the enhanced JSON file
    output_json_filename = "enhanced_paper.json"
    output_json_path = os.path.join(output_dir, output_json_filename)

    # Save the enhanced JSON data
    print(f"Saving enhanced JSON to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(json_data_enhanced, f, indent=4)

    print(f"Enhanced JSON saved successfully to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract figures from PDFs and get LLM descriptions.")
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file (e.g., paper_cleaned.json).")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted figures and the enhanced JSON.")
    parser.add_argument("--gpt_version", default="o3-mini", help="OpenAI GPT version to use for descriptions.")
    # Add a new argument for the API key, with environment variable as fallback
    parser.add_argument("--openai_api_key", help="OpenAI API Key. Defaults to OPENAI_API_KEY environment variable.")

    args = parser.parse_args()

    # Set OpenAI API key from argument or environment variable
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not provided via argument or environment variable.")
        print("Please set it using --openai_api_key or as an environment variable.")
        return # Exit if no API key

    enhance_json_with_figures(args.pdf_path, args.json_path, args.output_dir, args.gpt_version)

if __name__ == "__main__":
    main()