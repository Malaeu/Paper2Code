# Automatic Variable Mapping for Methodology Adaptation

The `adapt_mapping.py` module is part of Paper2Code's methodology adaptation feature. It analyzes a user's dataset and automatically generates variable mappings between the original paper's methodology and the user's dataset.

## Features

- Automatic detection of dataset structure
- Support for multiple file formats (CSV, Parquet, Excel, JSON)
- AI-powered variable matching based on names, types, and sample values
- Creation of comprehensive mapping files for methodology adaptation

## Usage

### Command Line

```bash
python adapt_mapping.py \
    --paper_name "Segar" \
    --adapted_name "GenderBasedModel" \
    --gpt_version "o3-2025-04-16" \
    --pdf_json_path "/path/to/paper.json" \
    --dataset_path "/path/to/mydata.csv" \
    --output_mapping_path "/path/to/mapping.json" \
    --output_dir "/path/to/output/dir"
```

### Parameters

- `--paper_name`: Name of the original paper
- `--adapted_name`: Name of your adaptation
- `--gpt_version`: GPT model version to use for mapping generation
- `--pdf_json_path`: Path to the processed JSON of the original paper
- `--dataset_path`: Path to your dataset file
- `--output_mapping_path`: Where to save the generated mapping
- `--output_dir`: Directory for additional outputs

## Output

The module generates a JSON mapping file with the following structure:

```json
{
    "original_to_adapted": {
        "original_var1": "your_var1",
        "original_var2": "your_var2",
        "original_category1": "your_category1"
    },
    "adapted_dataset_path": "/path/to/your/dataset.csv",
    "adapted_dataset_format": "csv",
    "methodology_adjustments": {
        "maintain_feature1": true,
        "iterations": 1000
    }
}
```

## How It Works

1. **Dataset Analysis**: The module first analyzes your dataset to identify:
   - Column names and types
   - Representative sample values
   - Basic dataset statistics

2. **AI Mapping Generation**: Using an LLM, the module:
   - Identifies key variables in the original paper
   - Matches them with variables in your dataset
   - Creates mappings for both variables and categorical values
   - Suggests methodology adjustments

3. **User Review**: After generation, you should:
   - Review the mapping file
   - Make any necessary adjustments
   - Ensure all critical variables are properly mapped

## Integration with Adaptation Pipeline

This module is automatically called by `run_custom_adapt.sh` when no mapping file exists. After generating the mapping, the script will prompt you to review it before proceeding with the full adaptation.

## Tips for Better Mappings

- Use descriptive column names in your dataset
- Include both demographic and outcome variables in your dataset
- Ensure your dataset has similar structure to what's described in the paper
- Review the generated mapping carefully before proceeding