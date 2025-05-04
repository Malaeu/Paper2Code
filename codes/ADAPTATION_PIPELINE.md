# Paper2Code Methodology Adaptation Pipeline

This document provides an overview of the methodology adaptation pipeline in Paper2Code, which allows users to adapt research methodologies from scientific papers to their own datasets and contexts.

## Complete Pipeline Overview

The adaptation pipeline consists of the following steps:

1. **Dataset Preparation**
   - User provides their own dataset or uses the sample dataset
   - Multiple formats supported: CSV, Parquet, Excel, JSON

2. **Automatic Variable Mapping**
   - Intelligent analysis of dataset structure
   - AI-powered matching of paper variables to user variables
   - Generation of complete mapping template
   - User review and customization

3. **Adaptation Planning**
   - High-level plan for adapting the paper methodology
   - Focus on maintaining methodological rigor
   - Creation of detailed adaptation roadmap

4. **Component Analysis**
   - Detailed analysis of each code component
   - Identification of necessary changes per component
   - Pseudocode generation for adapted implementations

5. **Code Generation**
   - Generation of complete, adapted code repository
   - Structured according to the original methodology
   - Tailored to work with user's dataset and variables

## Pipeline Components

### 1. Dataset Analysis and Mapping (`adapt_mapping.py`)

This module analyzes the user's dataset and generates a variable mapping between the original paper and the user's data:

```
Input: Original paper JSON + User dataset
Output: Variable mapping JSON file
```

Key features:
- Automatic detection of dataset structure
- AI-powered variable matching
- User-reviewable mapping template

### 2. Adaptation Planning (`adapt_planning.py`)

This module creates a comprehensive adaptation plan:

```
Input: Original paper JSON + Variable mapping
Output: Adaptation plan (markdown)
```

Key features:
- Methodological adaptation guidelines
- Data processing approach
- Model development strategy
- Evaluation framework

### 3. Component Analysis (`adapt_analyzing.py`) 

This module analyzes each component of the original methodology:

```
Input: Original paper JSON + Variable mapping + Adaptation plan
Output: Component-specific adaptation analyses
```

Key features:
- Component-by-component analysis
- Variable substitution planning
- Structural adjustment identification
- Pseudocode generation

### 4. Code Generation (`adapt_coding.py`)

This module generates the adapted code for each component:

```
Input: Original paper JSON + Variable mapping + Adaptation plan + Component analyses
Output: Complete adapted code repository
```

Key features:
- Working code generation
- Proper variable substitution
- Methodologically rigorous implementation
- Ready-to-run repository

## Running the Pipeline

The entire pipeline can be executed with a single command:

```bash
./scripts/run_custom_adapt.sh
```

This script:
1. Creates a sample dataset if none exists
2. Analyzes the dataset and generates a mapping template
3. Prompts the user to review the mapping
4. Runs the complete adaptation pipeline
5. Generates a ready-to-use code repository

## Technical Implementation

- **LLM Integration**: Uses o3-2025-04-16 for all adaptation steps
- **Pandas Integration**: For dataset analysis and structure detection
- **Human-in-the-Loop**: User reviews and customizes the variable mapping
- **Modular Design**: Each step can be run independently
- **File Format Support**: Works with multiple data formats

## Usage Examples

### From Race-Based to Gender-Based Models

```json
{
    "original_to_adapted": {
        "race": "gender",
        "Black": "female",
        "White": "male"
    }
}
```

### From Binary to Multi-Class Classification

```json
{
    "original_to_adapted": {
        "outcome_binary": "outcome_multiclass",
        "positive": "class_a",
        "negative": ["class_b", "class_c"]
    }
}
```

### From One Medical Context to Another

```json
{
    "original_to_adapted": {
        "heart_failure": "diabetes",
        "ejection_fraction": "HbA1c",
        "hypertension": "hypertension"
    }
}
```

## Best Practices

1. **Dataset Preparation**:
   - Use descriptive column names
   - Include all relevant variables mentioned in the paper
   - Clean data before adaptation

2. **Mapping Review**:
   - Carefully review the generated mapping
   - Ensure all critical variables are properly mapped
   - Add any missing mappings manually

3. **Code Customization**:
   - The generated code is a starting point
   - You may need to make minor adjustments
   - Test the adapted code with your data