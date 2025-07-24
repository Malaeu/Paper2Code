#!/usr/bin/env python3
"""
MinerU Processor for Paper2Code
Integrates MinerU PDF processing capabilities to replace GROBID
"""

import os
import json
import argparse
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MinerUProcessor:
    """
    Wrapper for MinerU processing within Paper2Code pipeline.
    Handles PDF processing and converts output to Paper2Code compatible format.
    """
    
    def __init__(self, mineru_path: str = "/media/chirurgie/hdd01/Soft/GitHub/MinerU"):
        """
        Initialize MinerU processor.
        
        Args:
            mineru_path: Path to MinerU installation directory
        """
        self.mineru_path = Path(mineru_path)
        self.venv_path = self.mineru_path / ".venv"
        self.python_path = self.venv_path / "bin" / "python"
        
        # Validate paths
        if not self.mineru_path.exists():
            raise FileNotFoundError(f"MinerU not found at {mineru_path}")
        if not self.venv_path.exists():
            raise FileNotFoundError(f"MinerU venv not found at {self.venv_path}")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_pdf(self, pdf_path: str, output_dir: str, method: str = "ocr") -> Dict[str, str]:
        """
        Process PDF using MinerU and return paths to generated files.
        
        Args:
            pdf_path: Path to input PDF file
            output_dir: Directory for output files
            method: Processing method ('ocr' or 'auto')
            
        Returns:
            Dictionary with paths to generated files
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Validate inputs
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare MinerU command
        cmd = [
            str(self.python_path), "-m", "mineru.cli.common",
            "-p", str(pdf_path),
            "-o", str(output_dir),
            "--method", method
        ]
        
        self.logger.info(f"Running MinerU command: {' '.join(cmd)}")
        
        # Execute MinerU
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.mineru_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minutes timeout
            )
            
            self.logger.info("MinerU processing completed successfully")
            self.logger.debug(f"MinerU stdout: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"MinerU processing failed: {e}")
            self.logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"MinerU processing failed: {e.stderr}")
        
        except subprocess.TimeoutExpired:
            self.logger.error("MinerU processing timed out")
            raise RuntimeError("MinerU processing timed out after 10 minutes")
        
        # Find generated files
        return self._find_output_files(pdf_path, output_dir)
    
    def _find_output_files(self, pdf_path: Path, output_dir: Path) -> Dict[str, str]:
        """
        Find and return paths to MinerU output files.
        
        Args:
            pdf_path: Original PDF path
            output_dir: MinerU output directory
            
        Returns:
            Dictionary mapping file types to their paths
        """
        # Get base filename without extension
        base_name = pdf_path.stem
        
        # Expected output structure from MinerU
        mineru_output_dir = output_dir / base_name / "ocr"
        
        if not mineru_output_dir.exists():
            # Try alternative structure
            mineru_output_dir = output_dir / "ocr"
            
        if not mineru_output_dir.exists():
            raise FileNotFoundError(f"MinerU output directory not found in {output_dir}")
        
        # Map expected files
        file_mapping = {
            'markdown': f"{base_name}.md",
            'content_list': f"{base_name}_content_list.json", 
            'model': f"{base_name}_model.json",
            'middle': f"{base_name}_middle.json",
            'layout_pdf': f"{base_name}_layout.pdf",
            'spans_pdf': f"{base_name}_spans.pdf",
            'images_dir': "images"
        }
        
        # Find actual files
        output_files = {}
        for file_type, filename in file_mapping.items():
            file_path = mineru_output_dir / filename
            if file_path.exists():
                output_files[file_type] = str(file_path)
            else:
                self.logger.warning(f"Expected file not found: {file_path}")
        
        # Ensure we have at least the essential files
        if 'markdown' not in output_files:
            raise FileNotFoundError("MinerU markdown output not found")
        
        self.logger.info(f"Found MinerU output files: {list(output_files.keys())}")
        return output_files
    
    def convert_to_paper2code_format(self, mineru_files: Dict[str, str], output_path: str) -> str:
        """
        Convert MinerU output to Paper2Code compatible JSON format.
        
        Args:
            mineru_files: Dictionary of MinerU output file paths
            output_path: Path for the Paper2Code JSON output
            
        Returns:
            Path to the generated Paper2Code JSON file
        """
        self.logger.info("Converting MinerU output to Paper2Code format")
        
        # Read MinerU content list (primary source)
        content_list = []
        if 'content_list' in mineru_files:
            with open(mineru_files['content_list'], 'r', encoding='utf-8') as f:
                content_list = json.load(f)
        
        # Read additional metadata from model.json if available
        model_data = {}
        if 'model' in mineru_files:
            with open(mineru_files['model'], 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        
        # Convert to Paper2Code format
        paper2code_json = self._build_paper2code_json(content_list, model_data, mineru_files)
        
        # Save the converted JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(paper2code_json, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Paper2Code JSON saved to: {output_path}")
        return output_path
    
    def _build_paper2code_json(self, content_list: List[Dict], model_data: Dict, mineru_files: Dict[str, str]) -> Dict:
        """
        Build Paper2Code compatible JSON from MinerU data.
        
        Args:
            content_list: MinerU content list data
            model_data: MinerU model inference data  
            mineru_files: Paths to MinerU output files
            
        Returns:
            Paper2Code compatible JSON structure
        """
        # Initialize Paper2Code structure
        paper_json = {
            "paper_id": "mineru_processed",
            "metadata": {
                "title": "",
                "authors": [],
                "abstract": ""
            },
            "abstract": [],
            "body_text": [],
            "ref_entries": {},
            "mineru_metadata": {
                "version": "2.1.5",
                "backend": "pipeline", 
                "files": mineru_files
            }
        }
        
        # Process content list
        current_section = "Introduction"
        figure_counter = 1
        table_counter = 1
        equation_counter = 1
        
        for item in content_list:
            item_type = item.get("type", "")
            page_idx = item.get("page_idx", 0)
            
            if item_type == "text":
                text_level = item.get("text_level", 0)
                text_content = item.get("text", "")
                
                # Handle titles and headers
                if text_level == 1:
                    if not paper_json["metadata"]["title"]:
                        paper_json["metadata"]["title"] = text_content.strip()
                    else:
                        current_section = text_content.strip()
                elif text_level == 2:
                    current_section = text_content.strip()
                
                # Handle abstract
                if "abstract" in text_content.lower() and text_level > 0:
                    current_section = "Abstract"
                elif current_section.lower() == "abstract" and text_level == 0:
                    paper_json["abstract"].append({
                        "text": text_content,
                        "cite_spans": [],
                        "ref_spans": [],
                        "section": "Abstract"
                    })
                    paper_json["metadata"]["abstract"] = text_content
                else:
                    # Regular body text
                    paper_json["body_text"].append({
                        "text": text_content,
                        "cite_spans": [],
                        "ref_spans": [],
                        "section": current_section
                    })
            
            elif item_type == "image":
                # Process figures
                ref_id = f"FIGREF_{figure_counter}"
                
                paper_json["ref_entries"][ref_id] = {
                    "type": "figure",
                    "text": " ".join(item.get("img_caption", [])),
                    "image_path": item.get("img_path", ""),
                    "page": page_idx,
                    "fig_num": str(figure_counter),
                    "mineru_source": True
                }
                
                figure_counter += 1
            
            elif item_type == "table":
                # Process tables
                ref_id = f"TABREF_{table_counter}"
                
                paper_json["ref_entries"][ref_id] = {
                    "type": "table", 
                    "text": " ".join(item.get("table_caption", [])),
                    "html": item.get("table_body", ""),
                    "image_path": item.get("img_path", ""),
                    "page": page_idx,
                    "table_num": str(table_counter),
                    "footnote": " ".join(item.get("table_footnote", [])),
                    "mineru_source": True
                }
                
                table_counter += 1
            
            elif item_type == "equation":
                # Process equations
                ref_id = f"EQREF_{equation_counter}"
                
                paper_json["ref_entries"][ref_id] = {
                    "type": "equation",
                    "text": item.get("text", ""),
                    "latex": item.get("text", ""),
                    "image_path": item.get("img_path", ""),
                    "page": page_idx,
                    "eq_num": str(equation_counter),
                    "text_format": item.get("text_format", "latex"),
                    "mineru_source": True
                }
                
                equation_counter += 1
        
        return paper_json


def main():
    """Command line interface for MinerU processor."""
    parser = argparse.ArgumentParser(description="Process PDF using MinerU for Paper2Code")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed files")
    parser.add_argument("--mineru_path", type=str, 
                       default="/media/chirurgie/hdd01/Soft/GitHub/MinerU",
                       help="Path to MinerU installation")
    parser.add_argument("--method", type=str, default="ocr", 
                       choices=["ocr", "auto"], help="Processing method")
    parser.add_argument("--json_output", type=str, 
                       help="Path for Paper2Code JSON output (optional)")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = MinerUProcessor(args.mineru_path)
        
        # Process PDF
        mineru_files = processor.process_pdf(args.pdf_path, args.output_dir, args.method)
        
        # Convert to Paper2Code format if requested
        if args.json_output:
            processor.convert_to_paper2code_format(mineru_files, args.json_output)
        
        print("✅ MinerU processing completed successfully!")
        print(f"📁 Output files: {mineru_files}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())