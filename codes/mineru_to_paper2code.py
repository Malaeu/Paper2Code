#!/usr/bin/env python3
"""
MinerU to Paper2Code JSON Converter
Advanced converter that transforms MinerU rich output into Paper2Code compatible format
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ContentBlock:
    """Represents a content block from MinerU."""
    type: str
    text: str
    page: int
    level: int = 0
    metadata: Dict[str, Any] = None
    

class MinerUToPaper2CodeConverter:
    """
    Advanced converter for MinerU output to Paper2Code JSON format.
    Handles complex document structures, sections, and references.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Section patterns for automatic detection
        self.section_patterns = [
            r'^abstract$',
            r'^introduction$',
            r'^related work$',
            r'^methodology?$',
            r'^method$', 
            r'^approach$',
            r'^experiments?$',
            r'^results?$',
            r'^discussion$',
            r'^conclusion$',
            r'^future work$',
            r'^references?$',
            r'^appendix$'
        ]
        
        # Reference patterns
        self.fig_ref_pattern = re.compile(r'fig(?:ure)?\s*(\d+)', re.IGNORECASE)
        self.table_ref_pattern = re.compile(r'table\s*(\d+)', re.IGNORECASE)
        self.eq_ref_pattern = re.compile(r'eq(?:uation)?\s*(\d+)', re.IGNORECASE)
    
    def convert(self, content_list_path: str, model_path: Optional[str] = None, 
                middle_path: Optional[str] = None, images_dir: Optional[str] = None) -> Dict:
        """
        Convert MinerU output to Paper2Code format.
        
        Args:
            content_list_path: Path to MinerU content_list.json
            model_path: Path to MinerU model.json (optional)
            middle_path: Path to MinerU middle.json (optional)
            images_dir: Path to extracted images directory (optional)
            
        Returns:
            Paper2Code compatible JSON dictionary
        """
        self.logger.info("Starting MinerU to Paper2Code conversion")
        
        # Load MinerU data
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        model_data = {}
        if model_path and Path(model_path).exists():
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        
        middle_data = {}
        if middle_path and Path(middle_path).exists():
            with open(middle_path, 'r', encoding='utf-8') as f:
                middle_data = json.load(f)
        
        # Process content
        paper_json = self._build_paper2code_structure(content_list, model_data, middle_data, images_dir)
        
        self.logger.info("Conversion completed successfully")
        return paper_json
    
    def _build_paper2code_structure(self, content_list: List[Dict], model_data: Dict, 
                                  middle_data: Dict, images_dir: Optional[str]) -> Dict:
        """
        Build the main Paper2Code JSON structure.
        
        Args:
            content_list: MinerU content list
            model_data: MinerU model data
            middle_data: MinerU middle processing data
            images_dir: Path to images directory
            
        Returns:
            Complete Paper2Code JSON structure
        """
        # Initialize structure
        paper_json = {
            "paper_id": f"mineru_{hash(str(content_list)) % 100000}",
            "metadata": {
                "title": "",
                "authors": [],
                "abstract": "",
                "venue": "",
                "year": None
            },
            "abstract": [],
            "body_text": [],
            "ref_entries": {},
            "mineru_metadata": {
                "backend": middle_data.get("_backend", "pipeline"),
                "version": middle_data.get("_version_name", "unknown"),
                "total_pages": len(middle_data.get("pdf_info", [])),
                "images_dir": images_dir
            }
        }
        
        # Process content blocks
        blocks = self._parse_content_blocks(content_list)
        
        # Extract metadata
        paper_json["metadata"] = self._extract_metadata(blocks)
        
        # Process sections
        sections = self._organize_sections(blocks)
        
        # Build abstract and body text
        paper_json["abstract"] = self._build_abstract(sections.get("Abstract", []))
        paper_json["body_text"] = self._build_body_text(sections)
        
        # Process references (figures, tables, equations)
        paper_json["ref_entries"] = self._process_references(blocks, images_dir)
        
        # Add reference spans to text
        paper_json = self._add_reference_spans(paper_json)
        
        return paper_json
    
    def _parse_content_blocks(self, content_list: List[Dict]) -> List[ContentBlock]:
        """
        Parse MinerU content list into structured blocks.
        
        Args:
            content_list: Raw MinerU content list
            
        Returns:
            List of structured ContentBlock objects
        """
        blocks = []
        
        for item in content_list:
            block_type = item.get("type", "")
            page = item.get("page_idx", 0)
            
            if block_type == "text":
                text = item.get("text", "").strip()
                level = item.get("text_level", 0)
                
                if text:  # Skip empty text blocks
                    blocks.append(ContentBlock(
                        type="text",
                        text=text,
                        page=page,
                        level=level
                    ))
            
            elif block_type in ["image", "table", "equation"]:
                blocks.append(ContentBlock(
                    type=block_type,
                    text="",
                    page=page,
                    metadata=item
                ))
        
        return blocks
    
    def _extract_metadata(self, blocks: List[ContentBlock]) -> Dict:
        """
        Extract paper metadata from content blocks.
        
        Args:
            blocks: List of content blocks
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "title": "",
            "authors": [],
            "abstract": "",
            "venue": "",
            "year": None
        }
        
        # Find title (usually first level 1 heading)
        for block in blocks:
            if block.type == "text" and block.level == 1:
                metadata["title"] = block.text
                break
        
        # Find abstract
        abstract_blocks = []
        in_abstract = False
        
        for block in blocks:
            if block.type == "text":
                text_lower = block.text.lower()
                
                if "abstract" in text_lower and block.level > 0:
                    in_abstract = True
                    continue
                elif in_abstract and block.level > 0:
                    # New section started
                    break
                elif in_abstract and block.level == 0:
                    abstract_blocks.append(block.text)
        
        if abstract_blocks:
            metadata["abstract"] = " ".join(abstract_blocks)
        
        return metadata
    
    def _organize_sections(self, blocks: List[ContentBlock]) -> Dict[str, List[ContentBlock]]:
        """
        Organize blocks into sections.
        
        Args:
            blocks: List of content blocks
            
        Returns:
            Dictionary mapping section names to blocks
        """
        sections = {}
        current_section = "Introduction"
        
        for block in blocks:
            if block.type == "text" and block.level > 0:
                # This is a heading
                section_name = self._normalize_section_name(block.text)
                current_section = section_name
                
                if section_name not in sections:
                    sections[section_name] = []
            else:
                # Regular content
                if current_section not in sections:
                    sections[current_section] = []
                
                sections[current_section].append(block)
        
        return sections
    
    def _normalize_section_name(self, section_text: str) -> str:
        """
        Normalize section names to standard format.
        
        Args:
            section_text: Raw section text
            
        Returns:
            Normalized section name
        """
        text_lower = section_text.lower().strip()
        
        # Remove numbering (e.g., "1. Introduction" -> "Introduction")
        text_clean = re.sub(r'^\d+\.?\s*', '', text_lower)
        
        # Check against known patterns
        for pattern in self.section_patterns:
            if re.match(pattern, text_clean):
                return text_clean.title()
        
        # Return original if no match
        return section_text.strip()
    
    def _build_abstract(self, abstract_blocks: List[ContentBlock]) -> List[Dict]:
        """
        Build abstract section.
        
        Args:
            abstract_blocks: List of blocks in abstract section
            
        Returns:
            Abstract section in Paper2Code format
        """
        abstract = []
        
        for block in abstract_blocks:
            if block.type == "text":
                abstract.append({
                    "text": block.text,
                    "cite_spans": [],
                    "ref_spans": [],
                    "section": "Abstract"
                })
        
        return abstract
    
    def _build_body_text(self, sections: Dict[str, List[ContentBlock]]) -> List[Dict]:
        """
        Build body text from organized sections.
        
        Args:
            sections: Dictionary of organized sections
            
        Returns:
            Body text list in Paper2Code format
        """
        body_text = []
        
        # Skip abstract section for body text
        for section_name, blocks in sections.items():
            if section_name.lower() == "abstract":
                continue
            
            for block in blocks:
                if block.type == "text":
                    body_text.append({
                        "text": block.text,
                        "cite_spans": [],
                        "ref_spans": [],
                        "section": section_name
                    })
        
        return body_text
    
    def _process_references(self, blocks: List[ContentBlock], images_dir: Optional[str]) -> Dict:
        """
        Process reference entries (figures, tables, equations).
        
        Args:
            blocks: List of content blocks
            images_dir: Path to images directory
            
        Returns:
            Reference entries dictionary
        """
        ref_entries = {}
        counters = {"figure": 1, "table": 1, "equation": 1}
        
        for block in blocks:
            if block.type == "image":
                ref_id = f"FIGREF_{counters['figure']}"
                
                ref_entries[ref_id] = {
                    "type": "figure",
                    "text": " ".join(block.metadata.get("img_caption", [])),
                    "image_path": self._resolve_image_path(block.metadata.get("img_path"), images_dir),
                    "page": block.page,
                    "fig_num": str(counters["figure"]),
                    "footnote": " ".join(block.metadata.get("img_footnote", [])),
                    "mineru_source": True
                }
                
                counters["figure"] += 1
            
            elif block.type == "table":
                ref_id = f"TABREF_{counters['table']}"
                
                ref_entries[ref_id] = {
                    "type": "table",
                    "text": " ".join(block.metadata.get("table_caption", [])),
                    "html": block.metadata.get("table_body", ""),
                    "image_path": self._resolve_image_path(block.metadata.get("img_path"), images_dir),
                    "page": block.page,
                    "table_num": str(counters["table"]),
                    "footnote": " ".join(block.metadata.get("table_footnote", [])),
                    "mineru_source": True
                }
                
                counters["table"] += 1
            
            elif block.type == "equation":
                ref_id = f"EQREF_{counters['equation']}"
                
                ref_entries[ref_id] = {
                    "type": "equation",
                    "text": block.metadata.get("text", ""),
                    "latex": block.metadata.get("text", ""),
                    "image_path": self._resolve_image_path(block.metadata.get("img_path"), images_dir),
                    "page": block.page,
                    "eq_num": str(counters["equation"]),
                    "text_format": block.metadata.get("text_format", "latex"),
                    "mineru_source": True
                }
                
                counters["equation"] += 1
        
        return ref_entries
    
    def _resolve_image_path(self, img_path: Optional[str], images_dir: Optional[str]) -> str:
        """
        Resolve image path relative to output directory.
        
        Args:
            img_path: Original image path from MinerU
            images_dir: Base images directory
            
        Returns:
            Resolved image path
        """
        if not img_path:
            return ""
        
        # If absolute path, make it relative to images_dir
        if images_dir and Path(img_path).is_absolute():
            try:
                return str(Path(img_path).relative_to(Path(images_dir).parent))
            except ValueError:
                pass
        
        return img_path
    
    def _add_reference_spans(self, paper_json: Dict) -> Dict:
        """
        Add reference spans to text based on detected patterns.
        
        Args:
            paper_json: Paper2Code JSON structure
            
        Returns:
            Updated JSON with reference spans
        """
        # Create reference mapping
        ref_mapping = {}
        for ref_id, ref_data in paper_json["ref_entries"].items():
            ref_type = ref_data["type"]
            if ref_type == "figure":
                ref_mapping[f"fig{ref_data['fig_num']}"] = ref_id
                ref_mapping[f"figure{ref_data['fig_num']}"] = ref_id
            elif ref_type == "table":
                ref_mapping[f"table{ref_data['table_num']}"] = ref_id
            elif ref_type == "equation":
                ref_mapping[f"eq{ref_data['eq_num']}"] = ref_id
                ref_mapping[f"equation{ref_data['eq_num']}"] = ref_id
        
        # Process text sections
        for section in ["abstract", "body_text"]:
            for item in paper_json[section]:
                item["ref_spans"] = self._find_reference_spans(item["text"], ref_mapping)
        
        return paper_json
    
    def _find_reference_spans(self, text: str, ref_mapping: Dict[str, str]) -> List[Dict]:
        """
        Find reference spans in text.
        
        Args:
            text: Text to search
            ref_mapping: Mapping of reference patterns to IDs
            
        Returns:
            List of reference span objects
        """
        ref_spans = []
        
        # Find figure references
        for match in self.fig_ref_pattern.finditer(text):
            fig_key = f"fig{match.group(1)}"
            if fig_key in ref_mapping:
                ref_spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "ref_id": ref_mapping[fig_key]
                })
        
        # Find table references
        for match in self.table_ref_pattern.finditer(text):
            table_key = f"table{match.group(1)}"
            if table_key in ref_mapping:
                ref_spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "ref_id": ref_mapping[table_key]
                })
        
        # Find equation references
        for match in self.eq_ref_pattern.finditer(text):
            eq_key = f"eq{match.group(1)}"
            if eq_key in ref_mapping:
                ref_spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "ref_id": ref_mapping[eq_key]
                })
        
        return ref_spans


def main():
    """Command line interface for the converter."""
    parser = argparse.ArgumentParser(description="Convert MinerU output to Paper2Code JSON format")
    parser.add_argument("--content_list", type=str, required=True, 
                       help="Path to MinerU content_list.json")
    parser.add_argument("--model", type=str, help="Path to MinerU model.json (optional)")
    parser.add_argument("--middle", type=str, help="Path to MinerU middle.json (optional)")
    parser.add_argument("--images_dir", type=str, help="Path to images directory (optional)")
    parser.add_argument("--output", type=str, required=True, help="Output path for Paper2Code JSON")
    
    args = parser.parse_args()
    
    try:
        converter = MinerUToPaper2CodeConverter()
        result = converter.convert(
            content_list_path=args.content_list,
            model_path=args.model,
            middle_path=args.middle,
            images_dir=args.images_dir
        )
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversion completed: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())