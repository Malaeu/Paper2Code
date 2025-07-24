#!/usr/bin/env python3
"""
MinerU Image Enhancer with Gemini Vision
Enhances extracted images from MinerU with detailed descriptions using Gemini 2.5 Flash
"""

import os
import json
import argparse
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")


class MinerUImageEnhancer:
    """
    Enhances MinerU extracted images with detailed descriptions using Gemini Vision.
    Provides context-aware analysis for better LLM understanding.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the image enhancer.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai library not available")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Configure Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # Configuration
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1000,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.logger.info(f"Initialized Gemini Vision enhancer with model: {model_name}")
    
    def enhance_paper2code_json(self, json_path: str, images_base_dir: Optional[str] = None, 
                               output_path: Optional[str] = None) -> str:
        """
        Enhance a Paper2Code JSON file with Gemini Vision descriptions.
        
        Args:
            json_path: Path to Paper2Code JSON file
            images_base_dir: Base directory for image paths (if relative)
            output_path: Output path for enhanced JSON (defaults to input + '_enhanced')
            
        Returns:
            Path to enhanced JSON file
        """
        self.logger.info(f"Enhancing Paper2Code JSON: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_json = json.load(f)
        
        # Get paper context
        paper_title = paper_json.get("metadata", {}).get("title", "Unknown Paper")
        paper_abstract = paper_json.get("metadata", {}).get("abstract", "")
        
        # Enhance images in ref_entries
        enhanced_count = 0
        for ref_id, ref_entry in paper_json.get("ref_entries", {}).items():
            if ref_entry.get("type") == "figure" and ref_entry.get("image_path"):
                try:
                    # Get surrounding context
                    context = self._get_image_context(paper_json, ref_id)
                    
                    # Enhance image
                    description = self._analyze_image(
                        image_path=ref_entry["image_path"],
                        caption=ref_entry.get("text", ""),
                        context=context,
                        paper_title=paper_title,
                        paper_abstract=paper_abstract,
                        images_base_dir=images_base_dir
                    )
                    
                    if description:
                        ref_entry["llm_description"] = description
                        ref_entry["enhanced_by"] = f"gemini-{self.model_name}"
                        enhanced_count += 1
                        
                        self.logger.info(f"Enhanced {ref_id}: {ref_entry['image_path']}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Failed to enhance {ref_id}: {e}")
                    continue
        
        # Save enhanced JSON
        if not output_path:
            output_path = json_path.replace('.json', '_enhanced.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(paper_json, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Enhanced {enhanced_count} images. Saved to: {output_path}")
        return output_path
    
    def _get_image_context(self, paper_json: Dict, ref_id: str) -> str:
        """
        Get textual context around an image reference.
        
        Args:
            paper_json: Paper2Code JSON data
            ref_id: Reference ID to find context for
            
        Returns:
            Contextual text around the reference
        """
        context_parts = []
        
        # Search in body text for mentions
        for text_item in paper_json.get("body_text", []):
            text = text_item.get("text", "")
            ref_spans = text_item.get("ref_spans", [])
            
            # Check if this text mentions our reference
            for span in ref_spans:
                if span.get("ref_id") == ref_id:
                    # Add surrounding text
                    context_parts.append(f"Section '{text_item.get('section', 'Unknown')}': {text[:500]}")
                    break
        
        return " ... ".join(context_parts[:3])  # Limit context length
    
    def _analyze_image(self, image_path: str, caption: str, context: str, 
                      paper_title: str, paper_abstract: str, 
                      images_base_dir: Optional[str] = None) -> Optional[str]:
        """
        Analyze an image using Gemini Vision.
        
        Args:
            image_path: Path to the image file
            caption: Original caption from MinerU
            context: Surrounding textual context
            paper_title: Paper title for context
            paper_abstract: Paper abstract for context
            images_base_dir: Base directory for resolving relative paths
            
        Returns:
            Detailed image description or None if analysis failed
        """
        try:
            # Resolve image path
            full_image_path = self._resolve_image_path(image_path, images_base_dir)
            
            if not Path(full_image_path).exists():
                self.logger.warning(f"Image not found: {full_image_path}")
                return None
            
            # Load and encode image
            with open(full_image_path, 'rb') as img_file:
                image_data = img_file.read()
            
            # Check file size (Gemini has limits)
            if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
                self.logger.warning(f"Image too large: {full_image_path}")
                return None
            
            # Create prompt
            prompt = self._create_analysis_prompt(caption, context, paper_title, paper_abstract)
            
            # Analyze with Gemini
            response = self.model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_data}],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.text:
                return response.text.strip()
            else:
                self.logger.warning("Empty response from Gemini")
                return None
        
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return None
    
    def _resolve_image_path(self, image_path: str, images_base_dir: Optional[str]) -> str:
        """
        Resolve image path to absolute path.
        
        Args:
            image_path: Original image path (relative or absolute)
            images_base_dir: Base directory for relative paths
            
        Returns:
            Absolute path to image
        """
        path = Path(image_path)
        
        if path.is_absolute():
            return str(path)
        
        if images_base_dir:
            base_path = Path(images_base_dir)
            if not path.parts[0] == "images":
                # If path doesn't start with "images", it might be relative to base
                return str(base_path / image_path)
            else:
                # Path starts with "images", resolve relative to parent of base
                return str(base_path.parent / image_path)
        
        return str(path)
    
    def _create_analysis_prompt(self, caption: str, context: str, paper_title: str, 
                               paper_abstract: str) -> str:
        """
        Create analysis prompt for Gemini Vision.
        
        Args:
            caption: Original image caption
            context: Surrounding textual context
            paper_title: Paper title
            paper_abstract: Paper abstract
            
        Returns:
            Formatted prompt for Gemini
        """
        prompt = f"""Analyze this figure from the scientific paper "{paper_title}".

PAPER CONTEXT:
Abstract: {paper_abstract[:500]}...

FIGURE INFORMATION:
Caption: {caption}
Context: {context}

ANALYSIS TASK:
Provide a detailed technical description of this figure. Focus on:

1. VISUAL CONTENT: What does the image show? (graphs, diagrams, charts, etc.)
2. DATA INTERPRETATION: What quantitative data or patterns are presented?
3. METHODOLOGY INSIGHT: How does this relate to the paper's methodology?
4. TECHNICAL DETAILS: Key measurements, scales, labels, annotations
5. IMPLEMENTATION RELEVANCE: What code implementation insights can be derived?

REQUIREMENTS:
- Be precise and technical
- Include specific numbers/values if visible
- Explain relationships between elements
- Focus on information useful for reproducing the methodology
- Avoid just repeating the caption
- Keep response under 800 words

IMPORTANT: If this appears to be a table, focus on the data structure and values. If it's a graph, describe axes, trends, and data points. If it's a diagram, explain the process or architecture shown."""

        return prompt
    
    def enhance_mineru_content_list(self, content_list_path: str, images_base_dir: str,
                                   paper_title: str = "", output_path: Optional[str] = None) -> str:
        """
        Enhance MinerU content_list.json with image descriptions.
        
        Args:
            content_list_path: Path to MinerU content_list.json
            images_base_dir: Directory containing extracted images
            paper_title: Paper title for context
            output_path: Output path for enhanced content list
            
        Returns:
            Path to enhanced content list
        """
        self.logger.info(f"Enhancing MinerU content list: {content_list_path}")
        
        # Load content list
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        # Enhance images
        enhanced_count = 0
        for item in content_list:
            if item.get("type") == "image" and item.get("img_path"):
                try:
                    description = self._analyze_image(
                        image_path=item["img_path"],
                        caption=" ".join(item.get("img_caption", [])),
                        context="",  # No context available in content list format
                        paper_title=paper_title,
                        paper_abstract="",
                        images_base_dir=images_base_dir
                    )
                    
                    if description:
                        item["llm_description"] = description
                        item["enhanced_by"] = f"gemini-{self.model_name}"
                        enhanced_count += 1
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Failed to enhance image {item.get('img_path')}: {e}")
                    continue
        
        # Save enhanced content list
        if not output_path:
            output_path = content_list_path.replace('.json', '_enhanced.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Enhanced {enhanced_count} images. Saved to: {output_path}")
        return output_path


def main():
    """Command line interface for the image enhancer."""
    parser = argparse.ArgumentParser(description="Enhance MinerU images with Gemini Vision")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input JSON file (Paper2Code or MinerU content_list)")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory containing extracted images")
    parser.add_argument("--output", type=str, help="Output path for enhanced JSON")
    parser.add_argument("--api_key", type=str, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-exp",
                       help="Gemini model name")
    parser.add_argument("--paper_title", type=str, default="",
                       help="Paper title for context")
    parser.add_argument("--format", type=str, choices=["paper2code", "content_list"],
                       default="paper2code", help="Input format")
    
    args = parser.parse_args()
    
    try:
        enhancer = MinerUImageEnhancer(api_key=args.api_key, model_name=args.model)
        
        if args.format == "paper2code":
            output_path = enhancer.enhance_paper2code_json(
                json_path=args.input,
                images_base_dir=args.images_dir,
                output_path=args.output
            )
        else:
            output_path = enhancer.enhance_mineru_content_list(
                content_list_path=args.input,
                images_base_dir=args.images_dir,
                paper_title=args.paper_title,
                output_path=args.output
            )
        
        print(f"✅ Image enhancement completed: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())