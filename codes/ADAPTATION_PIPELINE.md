# Paper2Code Adaptation Pipeline

This document provides a comprehensive explanation of the Paper2Code adaptation pipeline, which enables users to adapt scientific methodologies from papers to their own datasets.

## Overview

The adaptation pipeline consists of two main approaches:

1. **Standard Approach**: Analyzes dataset files directly and generates variable mappings
2. **Two-Phase Approach**: Separates adaptation into plan generation and code generation phases

Both approaches maintain methodological rigor while adapting to new datasets and variables.

## Standard Approach

The standard approach works in a single pass:

1. **Dataset Analysis**: Examines the dataset to detect its structure
2. **Variable Mapping**: Proposes mappings between paper variables and dataset variables
3. **Adaptation Planning**: Creates a plan for adapting the methodology
4. **Code Generation**: Generates a complete repository with adapted code

### Workflow

```
Dataset File → Dataset Analysis → Variable Mapping → Adaptation Planning → Code Generation → Adapted Repository
```

### Usage

```bash
./scripts/run_custom_adapt.sh
```

## Two-Phase Approach

The two-phase approach separates the process into:

1. **Phase 1: Plan Generation**
   - Dataset Description → Adaptation Plan (JSON/Markdown)
   
2. **Phase 2: Code Generation**
   - Adaptation Plan → Detailed Analysis → Code Generation

This approach allows for human review and editing of the adaptation plan before code generation, providing more control over the adaptation process.

### Workflow

```
Phase 1: Dataset Description → LLM → Adaptation Plan (JSON/Markdown)
          ↓
          Human Review/Edit
          ↓
Phase 2: Adaptation Plan → Detailed Analysis → Code Generation → Adapted Repository
```

### Configuration

The two-phase approach uses a YAML configuration file:

```yaml
paper:
  title: "Paper Title"
  methodology: "Methodology Description"

dataset:
  path: "path/to/dataset.csv"
  format: "csv"
  description_path: "path/to/description.md"  # Optional

variable_mapping:
  original_to_adapted:
    "race": "gender"
    "Black": "female"
    "White": "male"

methodology:
  maintain_landmark_analysis: true
  maintain_monte_carlo_cv: true
  iterations: 1000

output:
  repo_name: "AdaptedModel"
  output_dir: "outputs"
```

### Usage

**Phase 1: Generate Adaptation Plan**

```bash
./scripts/run_direct_adapt.sh --config custom_adapt/adapt_config.yaml
```

This generates an adaptation plan (JSON and Markdown) that can be reviewed and edited.

**Phase 2: Generate Code Using Plan**

```bash
./scripts/run_with_plan.sh --config custom_adapt/adapt_config.yaml
```

This uses the previously generated plan to create a detailed analysis and generate code.

## Components

### 1. Dataset Analysis (`adapt_mapping.py`)

- Detects dataset format (CSV, Parquet, Excel, JSON)
- Analyzes column names, data types, and values
- Provides a structured representation of the dataset

### 2. Variable Mapping (`adapt_mapping.py`)

- Maps original paper variables to dataset variables
- Uses AI to propose intelligent mappings
- Generates a JSON mapping file

### 3. Direct Adaptation (`direct_adapt.py`)

- Sends dataset descriptions directly to the API
- Generates adaptation plans without raw file parsing
- Supports standalone operation for plan generation

### 4. Planning with Analysis (`adapt_planning.py`)

- Creates detailed adaptation plans
- Identifies core methodological elements to preserve
- Specifies required adjustments for the new dataset

### 5. Analysis with Plan (`adapt_analyzing_with_plan.py`)

- Takes a pre-generated plan as input
- Performs detailed analysis for implementation
- Creates structured analysis documents

### 6. Code Generation (`adapt_coding.py`)

- Generates complete repository structure
- Creates well-documented, modular code
- Implements the adapted methodology with new variables

## Directory Structure

The generated repository follows a standardized structure:

```
AdaptedModel_repo/
  ├── data/               # Data loading and preprocessing
  │   └── dataset_loader.py
  ├── models/             # Model implementations
  │   └── model.py
  ├── utils/              # Utility functions
  │   └── helpers.py
  ├── configs/            # Configuration files
  │   └── default.yaml
  ├── evaluation/         # Evaluation metrics and tools
  │   └── metrics.py
  ├── scripts/            # Helper scripts
  ├── main.py             # Main entry point
  ├── requirements.txt    # Dependencies
  └── README.md           # Documentation
```

## Methodological Considerations

Both approaches maintain key methodological elements:

- **Monte Carlo Cross-Validation**: Preserves rigorous validation
- **Landmark Analysis**: Maintains time-dependent feature handling
- **Model Selection**: Preserves the same model types
- **Evaluation Metrics**: Maintains statistical evaluation approaches

## Future Enhancements

Planned enhancements to the adaptation pipeline:

1. **Web Interface**: For plan editing and review
2. **Plan Library**: Pre-generated plans for common scenarios
3. **Interactive Mode**: Step-by-step adaptation guidance
4. **Multi-paper Integration**: Combine methodologies from multiple papers
5. **Automatic Testing**: Generate tests for adapted code