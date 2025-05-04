# Automatic Dataset Analysis and Variable Mapping

This module provides functionality for automatically analyzing datasets and generating variable mappings for scientific methodology adaptation.

## Overview

The automatic mapping system analyzes input datasets and proposes intelligent variable mappings that connect the original paper's variables to equivalent variables in the dataset. This enables the adaptation of scientific methodologies to new domains while maintaining methodological rigor.

## Features

- **Multi-format support**: Analyze CSV, Parquet, Excel, and JSON datasets
- **Automatic detection**: Identify column names, data types, and sample values
- **Intelligent mapping**: Propose variable mappings based on semantic similarity
- **Human-in-the-loop**: Generate mappings for review and customization
- **Flexible configuration**: Support for various adaptation strategies

## Usage

### Basic Usage

```python
from codes.adapt_mapping import analyze_dataset, generate_mapping

# Analyze the dataset
dataset_analysis = analyze_dataset('/path/to/dataset.csv', 'csv')

# Generate variable mapping
variable_mapping = generate_mapping({
    'race': 'gender',
    'Black': 'female',
    'White': 'male'
}, dataset_analysis)

# Save the mapping
import json
with open('variable_mapping.json', 'w') as f:
    json.dump(variable_mapping, f, indent=2)
```

### Command Line Usage

```bash
python -m codes.adapt_mapping --dataset_path /path/to/dataset.csv --dataset_format csv --output_path mapping.json
```

## Integration with Two-Phase Approach

The automatic mapping system integrates with the two-phase adaptation approach:

1. **Phase 1 - Plan Generation**:
   - Uses dataset description to create an adaptation plan
   - Can incorporate automatically generated mappings

2. **Phase 2 - Code Generation**:
   - Uses the adaptation plan to generate code
   - Applies the variable mappings consistently

## API Reference

### `analyze_dataset(dataset_path, dataset_format='csv')`

Analyzes a dataset and returns a structured representation.

**Parameters:**
- `dataset_path`: Path to the dataset file
- `dataset_format`: Format of the dataset ('csv', 'parquet', 'excel', 'json')

**Returns:**
- Dictionary containing dataset structure information

### `generate_mapping(initial_mapping, dataset_analysis)`

Generates a complete variable mapping based on initial mapping and dataset analysis.

**Parameters:**
- `initial_mapping`: Dictionary with initial variable mappings
- `dataset_analysis`: Dataset analysis from `analyze_dataset()`

**Returns:**
- Dictionary containing complete variable mapping

## Example Output

A generated variable mapping might look like:

```json
{
  "original_to_adapted": {
    "race": "gender",
    "Black": "female",
    "White": "male",
    "natriuretic_peptide": "blood_marker_a",
    "troponin": "blood_marker_b",
    "JHS": "dataset_a",
    "ARIC": "dataset_b"
  },
  "adapted_dataset_path": "/path/to/mydata.csv",
  "adapted_dataset_format": "csv",
  "methodology_adjustments": {
    "maintain_landmark_analysis": true,
    "maintain_monte_carlo_cv": true,
    "iterations": 1000
  }
}
```

## Extending the Module

To add support for additional dataset formats:

1. Update the `_read_dataset()` function in `adapt_mapping.py`
2. Add format-specific parsing logic
3. Update the dataset format validation in the main functions