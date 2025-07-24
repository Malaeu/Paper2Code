import json
import argparse
import logging

def remove_spans(data):
    """
    Remove span information and clean up data (compatible with both GROBID and MinerU formats).
    """
    # If data is a dictionary, recursively check its keys
    if isinstance(data, dict):
        # Remove specific keys if present (GROBID legacy + some MinerU metadata)
        for key in ["cite_spans", "ref_spans", "eq_spans", "authors", "bib_entries", \
                    "year", "venue", "identifiers", "_pdf_hash", "header", "enhanced_by"]:
            data.pop(key, None)
        
        # Clean MinerU specific metadata while preserving essential info
        if "mineru_metadata" in data:
            # Keep essential metadata but remove file paths for cleaner output
            mineru_meta = data["mineru_metadata"]
            if "files" in mineru_meta:
                mineru_meta.pop("files", None)
        
        # Recursively apply to child dictionaries or lists
        for key, value in data.items():
            data[key] = remove_spans(value)
    # If data is a list, apply the function to each item
    elif isinstance(data, list):
        return [remove_spans(item) for item in data]
    return data

def validate_paper2code_format(data):
    """
    Validate that the JSON has the expected Paper2Code format.
    """
    required_fields = ["paper_id", "metadata", "abstract", "body_text", "ref_entries"]
    
    for field in required_fields:
        if field not in data:
            logging.warning(f"Missing required field: {field}")
            return False
    
    # Check metadata structure
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or "title" not in metadata:
        logging.warning("Invalid metadata structure")
        return False
    
    # Check if we have content
    if not data.get("body_text") and not data.get("abstract"):
        logging.warning("No content found in body_text or abstract")
        return False
    
    return True

def enhance_mineru_compatibility(data):
    """
    Enhance data structure for better compatibility with downstream processing.
    """
    # Ensure ref_entries have consistent structure
    ref_entries = data.get("ref_entries", {})
    for ref_id, ref_entry in ref_entries.items():
        # Add missing fields for compatibility
        if "mineru_source" in ref_entry and ref_entry["mineru_source"]:
            # This is from MinerU, ensure it has all expected fields
            if ref_entry.get("type") == "figure" and "fig_num" not in ref_entry:
                # Extract figure number from ref_id
                try:
                    fig_num = ref_id.split("_")[-1]
                    ref_entry["fig_num"] = fig_num
                except:
                    ref_entry["fig_num"] = "unknown"
    
    # Ensure text entries have required fields
    for text_item in data.get("body_text", []):
        if "cite_spans" not in text_item:
            text_item["cite_spans"] = []
        if "ref_spans" not in text_item:
            text_item["ref_spans"] = []
        if "section" not in text_item:
            text_item["section"] = "Unknown"
    
    for text_item in data.get("abstract", []):
        if "cite_spans" not in text_item:
            text_item["cite_spans"] = []
        if "ref_spans" not in text_item:
            text_item["ref_spans"] = []
    
    return data

def main(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path 
    
    logging.info(f"Processing: {input_json_path}")

    # Load the JSON data
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return 1

    # Validate format
    if not validate_paper2code_format(data):
        logging.error("Invalid Paper2Code format detected")
        return 1
    
    # Detect source type
    is_mineru = "mineru_metadata" in data
    logging.info(f"Source detected: {'MinerU' if is_mineru else 'GROBID/Legacy'}")
    
    # Apply compatibility enhancements for MinerU
    if is_mineru:
        data = enhance_mineru_compatibility(data)
        logging.info("Applied MinerU compatibility enhancements")
    
    # Clean the data
    cleaned_data = remove_spans(data)
    
    # Add processing metadata
    if "metadata" in cleaned_data:
        cleaned_data["metadata"]["processed_with"] = "Paper2Code 0_pdf_process.py"
        if is_mineru:
            cleaned_data["metadata"]["source"] = "MinerU"
    
    # Log statistics
    stats = {
        "title": cleaned_data.get("metadata", {}).get("title", "Unknown"),
        "body_text_entries": len(cleaned_data.get("body_text", [])),
        "abstract_entries": len(cleaned_data.get("abstract", [])),
        "ref_entries": len(cleaned_data.get("ref_entries", {})),
        "figures": len([r for r in cleaned_data.get("ref_entries", {}).values() if r.get("type") == "figure"]),
        "tables": len([r for r in cleaned_data.get("ref_entries", {}).values() if r.get("type") == "table"]),
        "equations": len([r for r in cleaned_data.get("ref_entries", {}).values() if r.get("type") == "equation"])
    }
    
    logging.info(f"Processing complete: {stats}")

    # Save the cleaned data
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        logging.info(f"[SAVED] {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and clean Paper2Code JSON (supports both GROBID and MinerU formats)"
    )
    parser.add_argument("--input_json_path", type=str, required=True,
                       help="Path to input JSON file")
    parser.add_argument("--output_json_path", type=str, required=True,
                       help="Path to output cleaned JSON file")
    
    args = parser.parse_args()
    exit(main(args))