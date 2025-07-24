#!/usr/bin/env python3
"""
Formula Processor for MinerU Output
Processes LaTeX formulas from MinerU and provides analysis and conversion capabilities
"""

import json
import argparse
import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Try to import LaTeX processing libraries
try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy not installed. Install with: pip install sympy")


@dataclass
class FormulaData:
    """Represents structured formula data."""
    latex: str
    text_format: str
    page: int
    eq_num: str
    image_path: str
    parsed_expression: Optional[str] = None
    variables: List[str] = None
    complexity_score: float = 0.0


class FormulaProcessor:
    """
    Processes LaTeX formulas from MinerU output.
    Analyzes mathematical expressions and extracts variable information.
    """
    
    def __init__(self):
        """Initialize the formula processor."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Common mathematical patterns
        self.variable_patterns = [
            r'\\[a-zA-Z]+\{([^}]+)\}',  # \alpha{x}, \beta{y}
            r'([a-zA-Z])(?:_\{[^}]+\})?(?:\^\{[^}]+\})?',  # x, x_i, x^2, x_i^2
            r'\\([a-zA-Z]+)',  # \alpha, \beta, \gamma
        ]
        
        # Mathematical operators and functions
        self.operators = [
            'sum', 'prod', 'int', 'lim', 'max', 'min', 'arg',
            'sin', 'cos', 'tan', 'exp', 'log', 'ln',
            'frac', 'sqrt', 'partial'
        ]
        
        # Complexity indicators
        self.complexity_indicators = {
            'integrals': [r'\\int', r'\\iint', r'\\iiint'],
            'summations': [r'\\sum', r'\\prod'],
            'fractions': [r'\\frac'],
            'derivatives': [r'\\partial', r'\\frac\{d', r'\\frac\{\\partial'],
            'matrices': [r'\\begin\{matrix\}', r'\\begin\{pmatrix\}', r'\\begin\{bmatrix\}'],
            'limits': [r'\\lim', r'\\to', r'\\infty'],
            'subscripts_superscripts': [r'_\{', r'\^\{'],
            'greek_letters': [r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\epsilon', 
                            r'\\theta', r'\\lambda', r'\\mu', r'\\sigma', r'\\phi']
        }
    
    def process_paper2code_json(self, json_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process formulas from Paper2Code JSON and extract analysis.
        
        Args:
            json_path: Path to Paper2Code JSON file
            output_dir: Directory to save formula analysis files
            
        Returns:
            Dictionary with processed formula information
        """
        self.logger.info(f"Processing formulas from: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_json = json.load(f)
        
        # Extract formulas
        formulas = self._extract_formulas_from_json(paper_json)
        
        # Process each formula
        processed_formulas = {}
        for formula_id, formula_data in formulas.items():
            try:
                structured_formula = self._process_latex_formula(formula_data)
                processed_formulas[formula_id] = structured_formula
                
            except Exception as e:
                self.logger.error(f"Failed to process formula {formula_id}: {e}")
                continue
        
        # Create analysis summary
        analysis = self._analyze_formulas(processed_formulas)
        
        # Create comprehensive summary
        summary = {
            "total_formulas": len(formulas),
            "processed_formulas": len(processed_formulas),
            "analysis": analysis,
            "formulas": {k: self._serialize_formula_data(v) for k, v in processed_formulas.items()}
        }
        
        # Save files if output directory specified
        if output_dir:
            self._save_formula_files(summary, output_dir)
        
        self.logger.info(f"Processed {len(processed_formulas)}/{len(formulas)} formulas successfully")
        return summary
    
    def _extract_formulas_from_json(self, paper_json: Dict) -> Dict[str, Dict]:
        """
        Extract formula entries from Paper2Code JSON.
        
        Args:
            paper_json: Paper2Code JSON data
            
        Returns:
            Dictionary of formula entries
        """
        formulas = {}
        
        for ref_id, ref_entry in paper_json.get("ref_entries", {}).items():
            if ref_entry.get("type") == "equation":
                formulas[ref_id] = ref_entry
        
        return formulas
    
    def _process_latex_formula(self, formula_entry: Dict) -> FormulaData:
        """
        Process LaTeX formula into structured format.
        
        Args:
            formula_entry: Formula entry from Paper2Code JSON
            
        Returns:
            Structured FormulaData object
        """
        latex = formula_entry.get("latex", "")
        text_format = formula_entry.get("text_format", "latex")
        page = formula_entry.get("page", 0)
        eq_num = formula_entry.get("eq_num", "unknown")
        image_path = formula_entry.get("image_path", "")
        
        # Clean LaTeX
        cleaned_latex = self._clean_latex(latex)
        
        # Extract variables
        variables = self._extract_variables(cleaned_latex)
        
        # Calculate complexity
        complexity = self._calculate_complexity(cleaned_latex)
        
        # Try to parse with SymPy if available
        parsed_expression = None
        if SYMPY_AVAILABLE and cleaned_latex:
            parsed_expression = self._parse_with_sympy(cleaned_latex)
        
        return FormulaData(
            latex=cleaned_latex,
            text_format=text_format,
            page=page,
            eq_num=eq_num,
            image_path=image_path,
            parsed_expression=parsed_expression,
            variables=variables,
            complexity_score=complexity
        )
    
    def _clean_latex(self, latex: str) -> str:
        """
        Clean and normalize LaTeX formula.
        
        Args:
            latex: Raw LaTeX string
            
        Returns:
            Cleaned LaTeX string
        """
        if not latex:
            return ""
        
        # Remove outer dollar signs
        latex = re.sub(r'^\$+|^\\\[|\\\]$|\$+$', '', latex.strip())
        
        # Normalize whitespace
        latex = re.sub(r'\s+', ' ', latex).strip()
        
        # Fix common LaTeX issues
        latex = latex.replace('\\\\', '\\')  # Double backslashes
        latex = re.sub(r'\\text\{([^}]+)\}', r'\1', latex)  # Remove \text{}
        
        return latex
    
    def _extract_variables(self, latex: str) -> List[str]:
        """
        Extract mathematical variables from LaTeX.
        
        Args:
            latex: LaTeX formula string
            
        Returns:
            List of unique variables
        """
        variables = set()
        
        for pattern in self.variable_patterns:
            matches = re.findall(pattern, latex)
            for match in matches:
                if isinstance(match, tuple):
                    for item in match:
                        if item and not item.isdigit():
                            variables.add(item)
                else:
                    if match and not match.isdigit() and match not in self.operators:
                        variables.add(match)
        
        # Filter out common operators and numbers
        filtered_variables = []
        for var in variables:
            if (len(var) <= 3 and  # Short variable names
                var not in self.operators and
                not var.isdigit() and
                not re.match(r'^\d+$', var)):
                filtered_variables.append(var)
        
        return sorted(list(set(filtered_variables)))
    
    def _calculate_complexity(self, latex: str) -> float:
        """
        Calculate complexity score for a LaTeX formula.
        
        Args:
            latex: LaTeX formula string
            
        Returns:
            Complexity score (0.0 to 10.0)
        """
        if not latex:
            return 0.0
        
        score = 0.0
        
        # Count complexity indicators
        for category, patterns in self.complexity_indicators.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, latex))
            
            # Weight different categories
            weights = {
                'integrals': 3.0,
                'summations': 2.5,
                'fractions': 1.5,
                'derivatives': 2.0,
                'matrices': 3.0,
                'limits': 2.0,
                'subscripts_superscripts': 0.5,
                'greek_letters': 0.3
            }
            
            score += count * weights.get(category, 1.0)
        
        # Add base complexity for length
        score += len(latex) / 100.0
        
        # Normalize to 0-10 scale
        return min(score, 10.0)
    
    def _parse_with_sympy(self, latex: str) -> Optional[str]:
        """
        Parse LaTeX with SymPy if possible.
        
        Args:
            latex: LaTeX formula string
            
        Returns:
            Parsed expression as string or None if parsing failed
        """
        try:
            # Try to parse with SymPy
            expr = parse_latex(latex)
            return str(expr)
        except Exception as e:
            self.logger.debug(f"SymPy parsing failed for '{latex}': {e}")
            return None
    
    def _analyze_formulas(self, formulas: Dict[str, FormulaData]) -> Dict[str, Any]:
        """
        Analyze collection of formulas.
        
        Args:
            formulas: Dictionary of processed formulas
            
        Returns:
            Analysis results
        """
        if not formulas:
            return {"error": "No formulas to analyze"}
        
        # Basic statistics
        complexities = [f.complexity_score for f in formulas.values()]
        all_variables = []
        for f in formulas.values():
            if f.variables:
                all_variables.extend(f.variables)
        
        # Count variable usage
        variable_counts = {}
        for var in all_variables:
            variable_counts[var] = variable_counts.get(var, 0) + 1
        
        # Most common variables
        common_variables = sorted(variable_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        
        # Complexity distribution
        complexity_distribution = {
            "simple": len([c for c in complexities if c < 2.0]),
            "moderate": len([c for c in complexities if 2.0 <= c < 5.0]),
            "complex": len([c for c in complexities if c >= 5.0])
        }
        
        analysis = {
            "statistics": {
                "average_complexity": sum(complexities) / len(complexities),
                "max_complexity": max(complexities),
                "min_complexity": min(complexities),
                "total_unique_variables": len(set(all_variables)),
                "total_variable_instances": len(all_variables)
            },
            "complexity_distribution": complexity_distribution,
            "common_variables": common_variables,
            "formula_types": {
                "with_variables": len([f for f in formulas.values() if f.variables]),
                "parseable": len([f for f in formulas.values() if f.parsed_expression]),
                "with_images": len([f for f in formulas.values() if f.image_path])
            }
        }
        
        return analysis
    
    def _serialize_formula_data(self, formula: FormulaData) -> Dict:
        """Convert FormulaData to serializable dictionary."""
        return {
            "latex": formula.latex,
            "text_format": formula.text_format,
            "page": formula.page,
            "eq_num": formula.eq_num,
            "image_path": formula.image_path,
            "parsed_expression": formula.parsed_expression,
            "variables": formula.variables or [],
            "complexity_score": formula.complexity_score
        }
    
    def _save_formula_files(self, summary: Dict, output_dir: str):
        """
        Save formula analysis to files.
        
        Args:
            summary: Formula analysis summary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete summary
        summary_path = output_path / "formulas_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save LaTeX compilation file
        latex_file = output_path / "all_formulas.tex"
        self._create_latex_compilation(summary["formulas"], latex_file)
        
        # Save variable analysis
        variables_file = output_path / "variables_analysis.json"
        with open(variables_file, 'w', encoding='utf-8') as f:
            json.dump(summary["analysis"], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Formula analysis saved to: {output_dir}")
    
    def _create_latex_compilation(self, formulas: Dict, output_path: Path):
        """
        Create a LaTeX file with all formulas for compilation testing.
        
        Args:
            formulas: Dictionary of formula data
            output_path: Output file path
        """
        latex_content = [
            "\\documentclass{article}",
            "\\usepackage{amsmath}",
            "\\usepackage{amsfonts}",
            "\\usepackage{amssymb}",
            "\\begin{document}",
            "\\title{Extracted Formulas}",
            "\\maketitle",
            ""
        ]
        
        for formula_id, formula_data in formulas.items():
            latex_content.extend([
                f"\\section*{{Formula {formula_data['eq_num']} (Page {formula_data['page']})}}",
                f"\\label{{{formula_id}}}",
                "\\begin{equation}",
                formula_data['latex'],
                "\\end{equation}",
                f"Variables: {', '.join(formula_data['variables'])}",
                f"Complexity: {formula_data['complexity_score']:.2f}",
                ""
            ])
        
        latex_content.append("\\end{document}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))


def main():
    """Command line interface for the formula processor."""
    parser = argparse.ArgumentParser(description="Process formulas from MinerU Paper2Code output")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input Paper2Code JSON file")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for formula analysis")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze without saving detailed files")
    
    args = parser.parse_args()
    
    try:
        processor = FormulaProcessor()
        
        result = processor.process_paper2code_json(
            args.input, 
            args.output_dir if not args.analyze_only else None
        )
        
        # Print summary
        print(f"✅ Processed {result['processed_formulas']}/{result['total_formulas']} formulas")
        
        if result['analysis'] and 'statistics' in result['analysis']:
            stats = result['analysis']['statistics']
            print(f"📊 Average complexity: {stats['average_complexity']:.2f}")
            print(f"🔤 Unique variables: {stats['total_unique_variables']}")
            
            if result['analysis']['common_variables']:
                print("🔝 Most common variables:")
                for var, count in result['analysis']['common_variables'][:5]:
                    print(f"   {var}: {count} times")
        
        if args.output_dir and not args.analyze_only:
            print(f"📁 Analysis saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())