#!/usr/bin/env python3
"""
Table Processor for MinerU Output
Processes HTML tables from MinerU and converts them to structured formats for analysis
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Try to import HTML processing libraries
try:
    from bs4 import BeautifulSoup
    import pandas as pd
    HTML_PROCESSING_AVAILABLE = True
except ImportError:
    HTML_PROCESSING_AVAILABLE = False
    print("Warning: beautifulsoup4 or pandas not installed. Install with: pip install beautifulsoup4 pandas")


@dataclass
class TableData:
    """Represents structured table data."""
    headers: List[str]
    rows: List[List[str]]
    caption: str
    footnote: str
    page: int
    table_num: str


class TableProcessor:
    """
    Processes HTML tables from MinerU output into structured formats.
    Converts tables to CSV, JSON, and pandas-friendly formats.
    """
    
    def __init__(self):
        """Initialize the table processor."""
        if not HTML_PROCESSING_AVAILABLE:
            raise ImportError("Required libraries not available. Install with: pip install beautifulsoup4 pandas")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def process_paper2code_json(self, json_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process tables from Paper2Code JSON and extract structured data.
        
        Args:
            json_path: Path to Paper2Code JSON file
            output_dir: Directory to save extracted table files
            
        Returns:
            Dictionary with processed table information
        """
        self.logger.info(f"Processing tables from: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_json = json.load(f)
        
        # Extract tables
        tables = self._extract_tables_from_json(paper_json)
        
        # Process each table
        processed_tables = {}
        for table_id, table_data in tables.items():
            try:
                structured_table = self._process_html_table(table_data)
                processed_tables[table_id] = structured_table
                
                # Save individual table files if output directory specified
                if output_dir:
                    self._save_table_files(table_id, structured_table, output_dir)
                
            except Exception as e:
                self.logger.error(f"Failed to process table {table_id}: {e}")
                continue
        
        # Create summary
        summary = {
            "total_tables": len(tables),
            "processed_tables": len(processed_tables),
            "tables": processed_tables
        }
        
        # Save summary if output directory specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            summary_path = Path(output_dir) / "tables_summary.json"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Table summary saved to: {summary_path}")
        
        self.logger.info(f"Processed {len(processed_tables)}/{len(tables)} tables successfully")
        return summary
    
    def _extract_tables_from_json(self, paper_json: Dict) -> Dict[str, Dict]:
        """
        Extract table entries from Paper2Code JSON.
        
        Args:
            paper_json: Paper2Code JSON data
            
        Returns:
            Dictionary of table entries
        """
        tables = {}
        
        for ref_id, ref_entry in paper_json.get("ref_entries", {}).items():
            if ref_entry.get("type") == "table":
                tables[ref_id] = ref_entry
        
        return tables
    
    def _process_html_table(self, table_entry: Dict) -> TableData:
        """
        Process HTML table into structured format.
        
        Args:
            table_entry: Table entry from Paper2Code JSON
            
        Returns:
            Structured TableData object
        """
        html_content = table_entry.get("html", "")
        caption = table_entry.get("text", "")
        footnote = table_entry.get("footnote", "")
        page = table_entry.get("page", 0)
        table_num = table_entry.get("table_num", "unknown")
        
        if not html_content:
            self.logger.warning(f"No HTML content found for table {table_num}")
            return TableData([], [], caption, footnote, page, table_num)
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            self.logger.warning(f"No table element found in HTML for table {table_num}")
            return TableData([], [], caption, footnote, page, table_num)
        
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            headers = [self._clean_cell_text(cell.get_text()) for cell in header_cells]
        
        # Extract rows
        rows = []
        all_rows = table.find_all('tr')
        
        # Skip header row if we found headers
        start_idx = 1 if headers else 0
        
        for row in all_rows[start_idx:]:
            cells = row.find_all(['td', 'th'])
            row_data = [self._clean_cell_text(cell.get_text()) for cell in cells]
            if row_data:  # Skip empty rows
                rows.append(row_data)
        
        # If no explicit headers found, try to infer from first row
        if not headers and rows:
            # Check if first row looks like headers (contains text, not just numbers)
            first_row = rows[0]
            if self._looks_like_header_row(first_row):
                headers = first_row
                rows = rows[1:]
        
        return TableData(headers, rows, caption, footnote, page, table_num)
    
    def _clean_cell_text(self, text: str) -> str:
        """
        Clean text from table cells.
        
        Args:
            text: Raw cell text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)  # Control characters
        
        return cleaned
    
    def _looks_like_header_row(self, row: List[str]) -> bool:
        """
        Determine if a row looks like headers.
        
        Args:
            row: List of cell values
            
        Returns:
            True if row looks like headers
        """
        if not row:
            return False
        
        # Count numeric vs text cells
        numeric_count = 0
        text_count = 0
        
        for cell in row:
            cell_clean = cell.strip()
            if not cell_clean:
                continue
            
            # Try to parse as number
            try:
                float(cell_clean.replace(',', ''))
                numeric_count += 1
            except ValueError:
                text_count += 1
        
        # Headers are more likely to be text than numbers
        return text_count > numeric_count
    
    def _save_table_files(self, table_id: str, table_data: TableData, output_dir: str):
        """
        Save table data to multiple formats.
        
        Args:
            table_id: Table identifier
            table_data: Structured table data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        safe_name = re.sub(r'[^\w\-_]', '_', table_id)
        
        # Save as CSV
        csv_path = output_path / f"{safe_name}.csv"
        self._save_as_csv(table_data, csv_path)
        
        # Save as JSON
        json_path = output_path / f"{safe_name}.json"
        self._save_as_json(table_data, json_path)
        
        # Save metadata
        metadata_path = output_path / f"{safe_name}_metadata.json"
        metadata = {
            "table_id": table_id,
            "caption": table_data.caption,
            "footnote": table_data.footnote,
            "page": table_data.page,
            "table_num": table_data.table_num,
            "dimensions": {
                "rows": len(table_data.rows),
                "columns": len(table_data.headers) if table_data.headers else (len(table_data.rows[0]) if table_data.rows else 0)
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved table {table_id} to: {csv_path}, {json_path}, {metadata_path}")
    
    def _save_as_csv(self, table_data: TableData, csv_path: Path):
        """Save table as CSV file."""
        try:
            # Create DataFrame
            if table_data.headers and table_data.rows:
                # Ensure all rows have the same number of columns as headers
                max_cols = len(table_data.headers)
                normalized_rows = []
                
                for row in table_data.rows:
                    normalized_row = row[:max_cols]  # Truncate if too long
                    while len(normalized_row) < max_cols:  # Pad if too short
                        normalized_row.append("")
                    normalized_rows.append(normalized_row)
                
                df = pd.DataFrame(normalized_rows, columns=table_data.headers)
            elif table_data.rows:
                # No headers, use generic column names
                df = pd.DataFrame(table_data.rows)
            else:
                # Empty table
                df = pd.DataFrame()
            
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {e}")
    
    def _save_as_json(self, table_data: TableData, json_path: Path):
        """Save table as JSON file."""
        try:
            table_json = {
                "caption": table_data.caption,
                "footnote": table_data.footnote,
                "page": table_data.page,
                "table_num": table_data.table_num,
                "headers": table_data.headers,
                "rows": table_data.rows,
                "data": []
            }
            
            # Create structured data records
            if table_data.headers and table_data.rows:
                for row in table_data.rows:
                    record = {}
                    for i, header in enumerate(table_data.headers):
                        value = row[i] if i < len(row) else ""
                        record[header] = value
                    table_json["data"].append(record)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(table_json, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
    
    def analyze_table_structure(self, json_path: str) -> Dict[str, Any]:
        """
        Analyze table structures in a Paper2Code JSON file.
        
        Args:
            json_path: Path to Paper2Code JSON file
            
        Returns:
            Analysis results
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_json = json.load(f)
        
        tables = self._extract_tables_from_json(paper_json)
        
        analysis = {
            "total_tables": len(tables),
            "tables_with_html": 0,
            "tables_with_images": 0,
            "average_dimensions": {"rows": 0, "cols": 0},
            "table_details": []
        }
        
        total_rows = 0
        total_cols = 0
        
        for table_id, table_entry in tables.items():
            detail = {
                "id": table_id,
                "has_html": bool(table_entry.get("html")),
                "has_image": bool(table_entry.get("image_path")),
                "caption_length": len(table_entry.get("text", "")),
                "footnote_length": len(table_entry.get("footnote", "")),
                "page": table_entry.get("page", 0)
            }
            
            if detail["has_html"]:
                analysis["tables_with_html"] += 1
                try:
                    table_data = self._process_html_table(table_entry)
                    detail["rows"] = len(table_data.rows)
                    detail["cols"] = len(table_data.headers) if table_data.headers else 0
                    total_rows += detail["rows"]
                    total_cols += detail["cols"]
                except:
                    detail["rows"] = 0
                    detail["cols"] = 0
            
            if detail["has_image"]:
                analysis["tables_with_images"] += 1
            
            analysis["table_details"].append(detail)
        
        # Calculate averages
        if analysis["tables_with_html"] > 0:
            analysis["average_dimensions"]["rows"] = total_rows / analysis["tables_with_html"]
            analysis["average_dimensions"]["cols"] = total_cols / analysis["tables_with_html"]
        
        return analysis


def main():
    """Command line interface for the table processor."""
    parser = argparse.ArgumentParser(description="Process tables from MinerU Paper2Code output")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input Paper2Code JSON file")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for extracted tables")
    parser.add_argument("--analyze", action="store_true",
                       help="Only analyze table structures without extracting")
    
    args = parser.parse_args()
    
    try:
        processor = TableProcessor()
        
        if args.analyze:
            analysis = processor.analyze_table_structure(args.input)
            print(json.dumps(analysis, indent=2))
        else:
            result = processor.process_paper2code_json(args.input, args.output_dir)
            print(f"✅ Processed {result['processed_tables']}/{result['total_tables']} tables")
            
            if args.output_dir:
                print(f"📁 Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())