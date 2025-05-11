#!/usr/bin/env python3
"""
auto_mapper.py

This module provides:
1. AutoMapper class for automatically mapping variables from a user's dataset to concepts
   in a scientific paper using semantic similarity and heuristics.
2. Critic class for evaluating the quality of variable mappings using LLM-based analysis.
"""
import os
import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path to import data_processing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.model_selector import ModelSelector, ModelPurpose

logger = logging.getLogger(__name__)

class AutoMapper:
    def __init__(self, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", llm_client: Optional[Any] = None):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_client = llm_client # For future LLM-based enhancements
        # Basic unit patterns (extendable)
        self.unit_patterns = {
            'years': r'\b(years|yrs|age)\b',
            'pg/mL': r'\b(pg/ml|picograms/milliliter)\b',
            'mg/dL': r'\b(mg/dl|milligrams/deciliter)\b',
            'percentage': r'%|\b(percent|percentage)\b',
            'binary': r'\b(status|flag|binary|is_|has_)\b'
        }

    def _extract_text_from_paper_concept(self, concept: Dict[str, Any]) -> str:
        """Helper to create a descriptive string for a paper concept for embedding."""
        parts = [
            concept.get("raw_text_source", ""),
            concept.get("description", ""),
            f"Type: {concept.get('type', 'unknown')}",
            f"Units: {concept.get('units', 'unknown')}"
        ]
        return ". ".join(filter(None, parts))

    def _extract_text_from_user_column(self, column_profile: Dict[str, Any]) -> str:
        """Helper to create a descriptive string for a user column for embedding."""
        parts = [
            column_profile.get("column_name", ""),
            f"Data Type: {column_profile.get('dtype', 'unknown')}",
            f"Guessed Units: {column_profile.get('units_guessed', 'unknown')}",
            # Adding a few sample values might help context if they are not too noisy
            # f"Sample values: {', '.join(map(str, column_profile.get('sample_values', [])[:2]))}"
        ]
        return ". ".join(filter(None, parts))

    def load_paper_concepts(self, paper_json_path_or_data: Any) -> List[Dict[str, Any]]:
        """
        Loads and structures variable concepts from the paper's JSON representation.
        This is a placeholder and needs to be adapted to the actual structure of 'enriched_paper.json'.
        For PoC, it might expect a simple list of concept dictionaries.
        Example concept structure:
        {
          "variable_id": "age", // Unique ID for the concept
          "raw_text_source": "Age (years)", // As it appears in paper table/text
          "description": "Baseline age of study participants",
          "units": "years",
          "type": "continuous" // "continuous", "categorical", "binary"
        }
        """
        if isinstance(paper_json_path_or_data, str):
            with open(paper_json_path_or_data, 'r') as f:
                paper_data = json.load(f)
            # Assuming paper_data has a key like 'variable_descriptions'
            # This part is highly dependent on the actual structure of enriched_paper.json
            # For now, let's assume it's a list under a specific key or a predefined list for PoC
            concepts = paper_data.get("variable_descriptions", [])
            if not concepts and "ref_entries" in paper_data: # Try to infer from table-like figures
                for ref_id, ref_entry in paper_data["ref_entries"].items():
                    if ref_entry.get("type") == "table" or (ref_entry.get("type") == "figure" and "table" in ref_entry.get("llm_caption","").lower()):
                        # This would require more sophisticated parsing of table text/captions
                        # For PoC, this is a simplification
                        if "llm_caption" in ref_entry:
                             # A very naive extraction, real implementation needs robust table parsing
                            potential_vars = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]+)\s*\((.*?)\)\b', ref_entry["llm_caption"])
                            for var_name, var_unit in potential_vars:
                                concepts.append({
                                    "variable_id": var_name.lower().replace(" ","_"),
                                    "raw_text_source": f"{var_name} ({var_unit})",
                                    "description": f"Variable {var_name} from table {ref_id}",
                                    "units": var_unit,
                                    "type": "unknown" # Type inference would be needed
                                })
            if not concepts:
                 print(f"Warning: No 'variable_descriptions' found in {paper_json_path_or_data}. Using dummy concepts for PoC.")
                 # Fallback for PoC if actual parsing logic is not ready
                 return [
                    {"variable_id": "age", "raw_text_source": "Age (years)", "description": "Age of participant", "units": "years", "type": "continuous"},
                    {"variable_id": "nt_probnp", "raw_text_source": "NT-proBNP (pg/mL)", "description": "NT-proBNP level", "units": "pg/mL", "type": "continuous"},
                    {"variable_id": "diabetes", "raw_text_source": "Diabetes status", "description": "Presence of diabetes", "units": None, "type": "binary"}
                 ]

        elif isinstance(paper_json_path_or_data, list): # Allow passing a list of dicts directly
            concepts = paper_json_path_or_data
        else:
            raise ValueError("paper_json_path_or_data must be a file path or a list of concept dicts.")
        
        # Ensure all concepts have a unique variable_id
        for i, concept in enumerate(concepts):
            if "variable_id" not in concept:
                concept["variable_id"] = concept.get("raw_text_source", f"concept_{i}").lower().replace(" ","_").replace("(","").replace(")","").replace("/","_")
        return concepts


    def _guess_units_from_name(self, column_name: str) -> Optional[str]:
        col_lower = column_name.lower()
        for unit_key, pattern in self.unit_patterns.items():
            if re.search(pattern, col_lower):
                return unit_key
        # Try to extract from parentheses or brackets
        match = re.search(r'[\(\[]([^)]+)[\)\]]$', column_name)
        if match:
            return match.group(1).strip()
        return None

    def _is_binary_heuristic(self, series: pd.Series) -> bool:
        if series.nunique() == 2:
            return True
        # Check for common binary representations (e.g., Yes/No, True/False, 0/1 strings)
        if series.dtype == 'object':
            unique_values = {str(v).lower() for v in series.dropna().unique()}
            if unique_values.issubset({'yes', 'no'}) or \
               unique_values.issubset({'true', 'false'}) or \
               unique_values.issubset({'0', '1'}):
                return True
        return False

    def profile_user_dataset(self, dataset_path: str, dataset_format: str = 'csv') -> List[Dict[str, Any]]:
        """Profiles user dataset columns."""
        if dataset_format == 'csv':
            df = pd.read_csv(dataset_path)
        elif dataset_format == 'parquet':
            df = pd.read_parquet(dataset_path)
        # Add more formats as needed (Excel, JSON)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        column_profiles = []
        for col_name in df.columns:
            series = df[col_name]
            profile = {
                "column_name": col_name,
                "dtype": str(series.dtype),
                "units_guessed": self._guess_units_from_name(col_name),
                "sample_values": [str(v) for v in series.dropna().unique()[:3]],
                "stats": {
                    "missing_percentage": series.isnull().mean() * 100,
                    "unique_count": series.nunique(),
                    "is_binary_heuristic": self._is_binary_heuristic(series)
                }
            }
            if pd.api.types.is_numeric_dtype(series.dtype):
                profile["stats"]["min"] = series.min()
                profile["stats"]["max"] = series.max()
                profile["stats"]["mean"] = series.mean()
                profile["stats"]["median"] = series.median()
            column_profiles.append(profile)
        return column_profiles

    def _apply_heuristics(self, similarity_score: float, paper_concept: Dict, user_column: Dict) -> Tuple[float, str]:
        """Applies heuristics to adjust similarity score."""
        bonus = 0.0
        details = []

        # Unit matching bonus
        paper_units = paper_concept.get("units")
        user_units = user_column.get("units_guessed")
        if paper_units and user_units:
            # Normalize common unit representations for comparison
            norm_paper_units = paper_units.lower().replace(" ", "")\
                            .replace("years","year").replace("yrs","year")\
                            .replace("pg/ml","pgml").replace("picograms/milliliter","pgml")\
                            .replace("mg/dl","mgdl").replace("milligrams/deciliter","mgdl")

            norm_user_units = user_units.lower().replace(" ", "")\
                            .replace("years","year").replace("yrs","year")\
                            .replace("pg/ml","pgml").replace("picograms/milliliter","pgml")\
                            .replace("mg/dl","mgdl").replace("milligrams/deciliter","mgdl")

            if norm_paper_units == norm_user_units:
                bonus += 0.05
                details.append("Unit Match Bonus: +0.05")
            elif any(u in norm_paper_units for u in norm_user_units.split('/')) or \
                 any(u in norm_user_units for u in norm_paper_units.split('/')): # Partial match for compound units
                bonus += 0.02
                details.append("Partial Unit Match Bonus: +0.02")


        # Type matching bonus/penalty (simple version)
        paper_type = paper_concept.get("type") # e.g. "continuous", "binary", "categorical"
        user_is_binary = user_column.get("stats", {}).get("is_binary_heuristic", False)
        user_dtype = user_column.get("dtype")

        if paper_type == "binary" and user_is_binary:
            bonus += 0.05
            details.append("Binary Type Match Bonus: +0.05")
        elif paper_type == "binary" and not user_is_binary:
            bonus -= 0.03 # Penalty if paper says binary but user data isn't
            details.append("Binary Type Mismatch Penalty: -0.03")
        
        if paper_type == "continuous" and "int" in user_dtype or "float" in user_dtype:
            bonus += 0.03
            details.append("Continuous Type Match Bonus: +0.03")

        adjusted_score = min(1.0, max(0.0, similarity_score + bonus)) # Cap score between 0 and 1
        return adjusted_score, ", ".join(details) if details else "No heuristics applied"

    def map_variables(self,
                      paper_concepts: List[Dict[str, Any]],
                      user_column_profiles: List[Dict[str, Any]],
                      confidence_threshold: float = 0.85,
                      top_n_matches: int = 3) -> List[Dict[str, Any]]:
        """
        Maps paper concepts to user dataset columns using embeddings and heuristics.
        """
        paper_concept_texts = [self._extract_text_from_paper_concept(c) for c in paper_concepts]
        user_column_texts = [self._extract_text_from_user_column(ucp) for ucp in user_column_profiles]

        if not paper_concept_texts or not user_column_texts:
            print("Warning: Empty paper concepts or user column profiles. Cannot perform mapping.")
            return []

        paper_vecs = self.embedding_model.encode(paper_concept_texts, convert_to_tensor=True, normalize_embeddings=True)
        col_vecs = self.embedding_model.encode(user_column_texts, convert_to_tensor=True, normalize_embeddings=True)

        # Compute cosine-similarities
        cos_sim_matrix = util.pytorch_cos_sim(paper_vecs, col_vecs).cpu().numpy()

        all_mappings = []
        for i, paper_concept in enumerate(paper_concepts):
            # Get similarity scores for this paper_concept with all user_columns
            sim_scores_for_concept = cos_sim_matrix[i]
            
            # Apply heuristics and store adjusted scores
            candidate_matches = []
            for j, user_column_profile in enumerate(user_column_profiles):
                original_sim_score = float(sim_scores_for_concept[j])
                adjusted_score, heuristic_details = self._apply_heuristics(original_sim_score, paper_concept, user_column_profile)
                candidate_matches.append({
                    "column_profile": user_column_profile,
                    "original_score": original_sim_score,
                    "adjusted_score": adjusted_score,
                    "heuristic_details": heuristic_details
                })
            
            # Sort candidates by adjusted_score descending
            sorted_candidates = sorted(candidate_matches, key=lambda x: x["adjusted_score"], reverse=True)
            
            top_matches = sorted_candidates[:top_n_matches]
            
            suggested_match_info = None
            needs_review = True
            if top_matches and top_matches[0]["adjusted_score"] >= confidence_threshold:
                suggested_match_info = top_matches[0]
                needs_review = False

            all_mappings.append({
                "paper_variable": paper_concept,
                "top_user_columns": [{
                    "column_name": match["column_profile"]["column_name"],
                    "score": round(match["adjusted_score"], 4),
                    "details": f"Cosine: {round(match['original_score'],4)}. {match['heuristic_details']}"
                } for match in top_matches],
                "suggested_mapping": suggested_match_info["column_profile"]["column_name"] if suggested_match_info else None,
                "confidence": suggested_match_info["adjusted_score"] if suggested_match_info else (top_matches[0]["adjusted_score"] if top_matches else 0.0),
                "needs_review": needs_review
            })
            
        return all_mappings

    def format_mapping_for_review_cli(self, proposed_mappings: List[Dict[str, Any]]) -> str:
        """Formats the proposed mappings for CLI review."""
        output = "Proposed Variable Mappings (Review Needed):\n"
        output += "============================================\n"
        for i, mapping_info in enumerate(proposed_mappings):
            pv = mapping_info['paper_variable']
            output += f"\n{i+1}. Paper Concept: '{pv.get('raw_text_source', pv.get('variable_id'))}' (Type: {pv.get('type','N/A')}, Units: {pv.get('units','N/A')})\n"
            output += f"   Description: {pv.get('description', 'N/A')}\n"
            
            if mapping_info['suggested_mapping']:
                output += f"   --> Suggested Match: '{mapping_info['suggested_mapping']}' (Confidence: {mapping_info['confidence']:.2f}) {'[AUTO-ACCEPTED]' if not mapping_info['needs_review'] else '[NEEDS REVIEW]'}\n"
            else:
                output += f"   --> No confident match found. Needs review.\n"

            output += f"   Top {len(mapping_info['top_user_columns'])} candidates:\n"
            for candidate in mapping_info['top_user_columns']:
                output += f"       - '{candidate['column_name']}' (Score: {candidate['score']:.2f}, Details: {candidate['details']})\n"
        output += "============================================\n"
        return output

class Critic:
    """
    A class to evaluate proposed variable mapping hypotheses using LLM.
    Provides a critical assessment of the quality and validity of mappings
    between dataset columns and scientific paper variables.
    """
    def __init__(self, model_selector: Optional[ModelSelector] = None, test_mode: bool = False):
        """
        Initialize the Critic component.

        Args:
            model_selector: Optional ModelSelector instance. If None, a new one will be created.
            test_mode: If True, uses a mock LLM client for testing without API keys
        """
        self.model_selector = model_selector or ModelSelector()
        self.test_mode = test_mode
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the LLM client for the Critic component."""
        # If in test mode, set up a mock client
        if self.test_mode:
            self.provider = "TEST"
            self.model_name = "test-model"
            self.llm_client = None
            logger.info("Initialized Critic in TEST mode (no API calls will be made)")
            return

        # Select the most appropriate model for the Critic purpose
        model_name = self.model_selector.select_model(purpose=ModelPurpose.CRITIC)
        client_config = self.model_selector.get_client_config(model_name)

        # Get API key from environment variable
        api_key = os.getenv(client_config['api_key_env_var'])
        if not api_key:
            raise ValueError(f"API key not found in environment variable {client_config['api_key_env_var']}")

        # Initialize client based on provider
        provider = client_config['provider']
        if provider == self.model_selector._get_provider_enum("OPENAI"):
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=api_key)
            self.model_name = client_config['model_api_name']
            self.provider = "OPENAI"
        elif provider == self.model_selector._get_provider_enum("ANTHROPIC"):
            from anthropic import Anthropic
            self.llm_client = Anthropic(api_key=api_key)
            self.model_name = client_config['model_api_name']
            self.provider = "ANTHROPIC"
        elif provider == self.model_selector._get_provider_enum("GOOGLE"):
            from google import genai
            self.llm_client = genai.Client(api_key=api_key)
            self.model_name = client_config['model_api_name']
            self.provider = "GOOGLE"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"Initialized Critic with {self.provider} model: {self.model_name}")

    def _format_column_profile(self, column_profile: Dict[str, Any]) -> str:
        """Format column profile information for the prompt."""
        col_info = [f"Column Name: {column_profile.get('column_name', 'unknown')}"]
        col_info.append(f"Data Type: {column_profile.get('dtype', 'unknown')}")

        stats = column_profile.get('stats', {})
        if stats:
            col_info.append("Statistics:")
            for stat_name, stat_value in stats.items():
                col_info.append(f"- {stat_name}: {stat_value}")

        if 'units_guessed' in column_profile and column_profile['units_guessed']:
            col_info.append(f"Guessed Units: {column_profile['units_guessed']}")

        if 'sample_values' in column_profile and column_profile['sample_values']:
            col_info.append(f"Sample Values: {', '.join(map(str, column_profile['sample_values']))}")

        return "\n".join(col_info)

    def _format_paper_variable(self, paper_variable: Dict[str, Any]) -> str:
        """Format paper variable information for the prompt."""
        var_info = [f"Variable ID: {paper_variable.get('variable_id', 'unknown')}"]
        var_info.append(f"Raw Text: {paper_variable.get('raw_text_source', 'unknown')}")

        if 'description' in paper_variable and paper_variable['description']:
            var_info.append(f"Description: {paper_variable['description']}")

        if 'units' in paper_variable and paper_variable['units']:
            var_info.append(f"Units: {paper_variable['units']}")

        if 'type' in paper_variable and paper_variable['type']:
            var_info.append(f"Type: {paper_variable['type']}")

        return "\n".join(var_info)

    def create_evaluation_prompt(self,
                                column_profile: Dict[str, Any],
                                paper_variable: Dict[str, Any],
                                proposer_hypothesis: str) -> str:
        """
        Create a prompt for the LLM to evaluate a mapping hypothesis.

        Args:
            column_profile: Profile information about the dataset column
            paper_variable: Information about the variable from the scientific paper
            proposer_hypothesis: The hypothesis about how these variables map to each other

        Returns:
            str: A formatted prompt for the LLM
        """
        formatted_column = self._format_column_profile(column_profile)
        formatted_variable = self._format_paper_variable(paper_variable)

        prompt = f"""You are a Critic component in a dataset variable mapping system. Your task is to evaluate a hypothesis about mapping a dataset column to a variable from a scientific paper.

COLUMN INFORMATION:
{formatted_column}

PAPER VARIABLE:
{formatted_variable}

PROPOSED MAPPING HYPOTHESIS:
{proposer_hypothesis}

EVALUATION CRITERIA:
1. Semantic Relevance (0-10): How semantically related are the column and paper variable?
2. Type Compatibility (0-10): Are the data types compatible?
3. Unit Compatibility (0-10): Are the units compatible or convertible?
4. Range Plausibility (0-10): Do the column's value ranges make sense for this variable?
5. Confidence (0-10): Overall confidence in this mapping

INSTRUCTIONS:
- Carefully analyze the column profile and compare it with the paper variable
- Consider the semantic meaning, data types, units, and statistical properties
- Evaluate the hypothesis from the Proposer based on the criteria above
- Suggest improvements or alternatives if the mapping is suboptimal
- Format your response as a structured JSON

RESPONSE FORMAT:
{{
  "scores": {{
    "semantic_relevance": <0-10>,
    "type_compatibility": <0-10>,
    "unit_compatibility": <0-10>,
    "range_plausibility": <0-10>,
    "overall_confidence": <0-10>
  }},
  "overall_score": <0-10>,
  "reasoning": "<detailed explanation>",
  "suggestion": "<improved mapping or 'none' if hypothesis is optimal>"
}}
"""
        return prompt

    def evaluate_mapping(self,
                        column_profile: Dict[str, Any],
                        paper_variable: Dict[str, Any],
                        proposer_hypothesis: str) -> Dict[str, Any]:
        """
        Evaluate a proposed mapping between a dataset column and a paper variable.

        Args:
            column_profile: Profile information about the dataset column
            paper_variable: Information about the variable from the scientific paper
            proposer_hypothesis: The hypothesis about how these variables map to each other

        Returns:
            Dict: The evaluation results with scores and reasoning
        """
        prompt = self.create_evaluation_prompt(column_profile, paper_variable, proposer_hypothesis)

        # If in test mode, return a mock response based on the input data
        if self.test_mode:
            logger.info("Using TEST mode to evaluate mapping (no API call)")

            # Extract key information for the mock scoring logic
            col_name = column_profile.get('column_name', '').lower()
            var_id = paper_variable.get('variable_id', '').lower()
            var_desc = paper_variable.get('description', '').lower()

            # Simple heuristic to generate a realistic score
            semantic_relevance = 7  # Default good score

            # Adjust scores based on keyword matching
            if col_name in var_id or var_id in col_name:
                semantic_relevance = 9  # Excellent match
            elif any(word in col_name for word in var_desc.split()):
                semantic_relevance = 8  # Good match

            # Type compatibility
            col_dtype = column_profile.get('dtype', '')
            var_type = paper_variable.get('type', '').lower()

            type_compatibility = 7  # Default good score
            if 'continuous' in var_type and ('float' in col_dtype or 'int' in col_dtype):
                type_compatibility = 9
            elif 'binary' in var_type and column_profile.get('stats', {}).get('is_binary_heuristic', False):
                type_compatibility = 10
            elif 'categorical' in var_type and 'object' in col_dtype:
                type_compatibility = 8

            # Unit compatibility
            col_units = column_profile.get('units_guessed', '').lower()
            var_units = paper_variable.get('units', '').lower()

            unit_compatibility = 6  # Default score
            if col_units and var_units and col_units == var_units:
                unit_compatibility = 10
            elif col_units and var_units and (col_units in var_units or var_units in col_units):
                unit_compatibility = 8

            # Create mock response
            overall_score = int((semantic_relevance + type_compatibility + unit_compatibility) / 3)

            return {
                "scores": {
                    "semantic_relevance": semantic_relevance,
                    "type_compatibility": type_compatibility,
                    "unit_compatibility": unit_compatibility,
                    "range_plausibility": 7,  # Default reasonable score
                    "overall_confidence": overall_score
                },
                "overall_score": overall_score,
                "reasoning": f"Test mode evaluation: Column '{col_name}' evaluated against variable '{var_id}'. " +
                            f"Semantic match is {'strong' if semantic_relevance >= 8 else 'moderate'}, " +
                            f"type compatibility is {'excellent' if type_compatibility >= 8 else 'acceptable'}, " +
                            f"unit compatibility is {'matching' if unit_compatibility >= 8 else 'convertible'}.",
                "suggestion": "None" if overall_score >= 8 else f"Consider checking if '{col_name}' truly corresponds to '{var_id}'"
            }

        # Normal mode with actual LLM API calls
        try:
            if self.provider == "OPENAI":
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content

            elif self.provider == "ANTHROPIC":
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=0.2,
                    system="You are a helpful assistant that responds in JSON format.",
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text

            elif self.provider == "GOOGLE":
                genai = self.llm_client
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                result_text = response.text

            # Parse the JSON response
            result = json.loads(result_text)

            return result

        except Exception as e:
            logger.error(f"Error evaluating mapping: {str(e)}")
            # Return a default error response
            return {
                "scores": {
                    "semantic_relevance": 0,
                    "type_compatibility": 0,
                    "unit_compatibility": 0,
                    "range_plausibility": 0,
                    "overall_confidence": 0
                },
                "overall_score": 0,
                "reasoning": f"Error evaluating mapping: {str(e)}",
                "suggestion": "Unable to evaluate due to error"
            }

    def format_evaluation_result(self, result: Dict[str, Any]) -> str:
        """Format the evaluation result for display."""
        output = "===== Mapping Evaluation =====\n"
        output += f"Overall Score: {result.get('overall_score', 'N/A')}/10\n\n"

        scores = result.get('scores', {})
        output += "Detailed Scores:\n"
        for criterion, score in scores.items():
            output += f"- {criterion.replace('_', ' ').title()}: {score}/10\n"

        output += f"\nReasoning: {result.get('reasoning', 'N/A')}\n"

        suggestion = result.get('suggestion', 'N/A')
        if suggestion.lower() != 'none':
            output += f"\nSuggestion: {suggestion}\n"
        else:
            output += "\nNo suggestions for improvement - this mapping appears optimal.\n"

        output += "=============================\n"
        return output


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Example Usage (PoC)
    mapper = AutoMapper()

    # 1. Define paper concepts (manually for PoC or load from a dummy JSON)
    # In a real scenario, this would come from a parsed paper JSON
    dummy_paper_concepts = [
        {"variable_id": "age_yrs", "raw_text_source": "Age (years)", "description": "Participant age at baseline", "units": "years", "type": "continuous"},
        {"variable_id": "sbp_mmhg", "raw_text_source": "Systolic Blood Pressure (mmHg)", "description": "Systolic blood pressure", "units": "mmHg", "type": "continuous"},
        {"variable_id": "gender_cat", "raw_text_source": "Gender", "description": "Participant gender", "units": None, "type": "categorical"},
        {"variable_id": "diabetes_bin", "raw_text_source": "Diabetes Mellitus", "description": "Diabetes diagnosis status", "units": None, "type": "binary"}
    ]

    # 2. Create a dummy user CSV file for testing
    dummy_data = {
        'patient_id': [1, 2, 3, 4, 5],
        'age_in_years': [65, 58, 72, 61, 70],
        'systolic_bp': [140, 130, 150, 120, 160],
        'sex': ['Male', 'Female', 'Female', 'Male', 'Male'],
        'has_diabetes': [1, 0, 1, 0, 1] # Binary 0 or 1
    }
    dummy_csv_path = "dummy_user_data.csv"
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)

    # 3. Profile user dataset
    print(f"Profiling dataset: {dummy_csv_path}")
    user_cols = mapper.profile_user_dataset(dummy_csv_path)
    # print("\nUser Column Profiles:")
    # for col_profile in user_cols:
    #     print(col_profile)

    # 4. Perform mapping
    print("\nPerforming variable mapping...")
    proposed_mappings = mapper.map_variables(dummy_paper_concepts, user_cols, confidence_threshold=0.5) # Lower threshold for PoC

    # 5. Display for review (CLI version)
    review_text = mapper.format_mapping_for_review_cli(proposed_mappings)
    print(review_text)

    # 6. Test the Critic component (using test mode if no API keys)
    print("\nTesting Critic component...")
    # Create critic in test mode to avoid requiring API keys
    critic = Critic(test_mode=True)

    # Evaluate a mapping (use the first mapping as an example)
    if proposed_mappings:
        # Test with all proposed mappings
        for i, mapping in enumerate(proposed_mappings[:2]):  # Limit to first 2 for brevity
            paper_var = mapping["paper_variable"]
            col_name = mapping["suggested_mapping"] or mapping["top_user_columns"][0]["column_name"]

            # Find the column profile
            col_profile = next((col for col in user_cols if col["column_name"] == col_name), None)

            if col_profile:
                print(f"\n----- Testing mapping #{i+1}: '{col_name}' -> '{paper_var['raw_text_source']}' -----")

                # Create a simple hypothesis for testing
                hypothesis = f"The column '{col_name}' in the dataset maps to the paper variable '{paper_var['raw_text_source']}' because they both represent the same concept and have compatible data types and units."

                # Evaluate the mapping
                evaluation = critic.evaluate_mapping(col_profile, paper_var, hypothesis)

                # Display the result
                print(critic.format_evaluation_result(evaluation))
    else:
        print("No mappings available to test")

    # Cleanup dummy file
    os.remove(dummy_csv_path)

    print("\nAutoMapper PoC finished.")
    # Next steps would be to integrate this into adapt_mapping.py and run_custom_adapt.sh
    # and provide a way for the user to confirm/edit these mappings.
    # The confirmed mappings would then be saved in the format expected by adapt_planning.py etc.
    # e.g., {"original_to_adapted": {"Age (years)": "age_in_years", ...}, ...}