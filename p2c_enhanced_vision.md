# Paper2Code Enhanced: Intelligent Research Analysis and Code Generation Platform

## 1. Introduction and Main Idea

The primary challenge in computational research is the difficulty in reproducing, adapting, and extending analyses presented in scientific publications. Existing tools may extract code or text, but often lack the deep semantic understanding required to apply methodologies to new datasets or research questions effectively.

**Paper2Code Enhanced** aims to create an intelligent platform that notifies users of this. It will:
1.  Perform a deep, multi-modal analysis of scientific papers.
2.  Enable users to precisely configure how the paper's methodology applies to their own data and research focus.
3.  Automatically generate adaptable, high-quality analysis code.
4.  Build a library of reusable analytical components derived from processed papers.
5.  Ultimately, provide a visual "roadmap" for research analysis, streamlining the path from publication to insight.

The core vision is to empower researchers to rapidly leverage existing scientific knowledge and apply it to their own work, accelerating discovery.

## 2. Key Features and Capabilities

*   **Deep Paper Analysis:**
    *   GROBID for initial structural parsing of PDFs (text, metadata, references, tables).
    *   Advanced LLM-powered analysis for:
        *   **Figures:** Detailed descriptions of type (e.g., histogram, scatter plot, alluvial diagram), style, fonts, key elements, and inferred conclusions. The goal is to create descriptions rich enough for an LLM to later generate a similar figure based on new data.
        *   **Tables:** Similar detailed analysis of structure, content, and statistical findings.
        *   **Text:** Semantic understanding, extraction of key methodologies, hypotheses, results, and limitations.
    *   Output: A comprehensive "enriched_paper.json" (or similar artifact) containing all structured and semantic information.
*   **User-Driven Analysis Configuration:**
    *   A detailed configuration file (e.g., `project_analysis_config.yaml`) allowing users to:
        *   Define their dataset and variables.
        *   Map concepts and variables from the paper to their own data.
        *   Specify their unique research questions and analysis focus.
        *   Provide context for adaptation (e.g., "in Segar paper, 'race' should be mapped to 'sex' in my dataset").
*   **Adaptive Code Generation:**
    *   LLMs leveraging both the "enriched_paper.json" and the user's `project_analysis_config.yaml` to generate executable analysis code (e.g., Python, R).
    *   Initial focus on reproducing adapted versions of analyses from the paper.
    *   Capability for more customized code generation based on user goals.
*   **Reusable Analytical Library:**
    *   Successful analytical components (e.g., a function to create a specific type of plot with certain styling, a specific statistical modeling approach) will be cataloged.
    *   The system can identify and suggest/reuse these components when processing new papers with similar analytical elements.
*   **Visual Research Roadmap:**
    *   A longer-term goal to visualize the analytical steps from a paper and how they connect, allowing users to interactively build or modify analysis pipelines.
*   **Automated Results Generation:**
    *   Execution of generated code to produce tables, figures, and textual summaries in a style consistent with the source paper, but using the user's data.

## 3. Architecture and Workflow (Conceptual)

1.  **Input:**
    *   User uploads a scientific paper (PDF).
    *   User provides their dataset (e.g., CSV).
    *   User defines their research context and mappings (`project_analysis_config.yaml`).
2.  **Paper Processing Pipeline:**
    *   **GROBID:** PDF to structured text/JSON.
    *   **LLM Enhancement:** The GROBID output is fed to an LLM for deep analysis of figures, tables, and text, resulting in "enriched_paper.json".
3.  **Configuration & Mapping:**
    *   User interacts with the system (potentially via a UI or by editing `project_analysis_config.yaml`) to link paper concepts to their data.
4.  **Code Generation:**
    *   The core LLM engine takes "enriched_paper.json" and `project_analysis_config.yaml` as input.
    *   Generates analysis scripts.
5.  **Execution & Output:**
    *   Generated scripts are run on the user's data.
    *   Results (plots, tables, summaries) are presented to the user.
6.  **Library Update (Future):**
    *   Validated analytical components from the generated code can be added to the reusable library.

## 4. Current State and Initial File Structure (Draft)

The project is an evolution of the original Paper2Code. Key existing components include GROBID integration and Python scripts for processing.

```
/media/chirurgie/hdd01/Soft/GitHub/Paper2Code/
├── p2c_enhanced_vision.md     # This document
├── README.md
├── codes/
│   ├── 0_pdf_process.py         # GROBID processing
│   ├── enhance_paper_llm.py     # NEW: LLM-based deep analysis (figures, tables, text)
│   ├── adapt_mapping.py         # MODIFIED/REPLACED: Handles project_analysis_config.yaml
│   ├── generate_analysis_code.py # NEW: Generates analysis code
│   └── utils.py
├── examples/
│   └── segar/                   # Use case for development
│       ├── paper.pdf
│       ├── fv_export.csv
│       ├── project_analysis_config.yaml
│       ├── processed_paper_files/ # Output of GROBID, "enriched_paper.json"
│       └── outputs/             # Generated code & results for the Segar example
├── scripts/
│   └── run_custom_adapt.sh      # MODIFIED: Orchestrates the new pipeline
├── s2orc-doc2json/
├── paper2code_env/
├── library/                     # FUTURE: Reusable analytical components
└── tests/
```

## 5. Development Plan and Next Steps (Phased Approach)

1.  **Phase 1: Vision & Core Enhancement (Current Focus)**
    *   Formalize project vision (this document).
    *   Develop and implement `enhance_paper_llm.py` to produce a rich "enriched_paper.json", focusing initially on detailed figure descriptions from the Segar paper.
    *   Test direct code generation (e.g., figure reproduction) using *only* "enriched_paper.json" (without `project_analysis_config.yaml`) to validate the richness of descriptions.
2.  **Phase 2: User Configuration and Adapted Analysis**
    *   Refine and integrate `project_analysis_config.yaml` fully.
    *   Develop `generate_analysis_code.py` to use both "enriched_paper.json" and `project_analysis_config.yaml` for generating adapted analyses (e.g., Segar example with sex instead of race).
3.  **Phase 3: Broadening Scope & Library Development**
    *   Expand capabilities to handle diverse types of figures, tables, and analyses.
    *   Begin design and implementation of the reusable analytical library.
4.  **Phase 4: UI and User Experience**
    *   Develop a user interface for easier interaction.
5.  **Phase 5: Visual Roadmap & Advanced Features**
    *   Implement the visual analysis roadmap concept.

## 6. Example Use Case: Adapting Segar et al. for a New Dataset

1.  **User Input:**
    *   `Segar_paper.pdf`.
    *   `my_liver_patient_data.csv` (contains variables like `sex`, `age`, `muscle_volume`, `vat_volume`, `outcome_status`, `time_to_event`).
    *   `project_analysis_config.yaml`:
        *   Defines outcome as mortality on waiting list.
        *   Maps `sex` in user data to the concept of `Race` in Segar for stratification.
        *   Maps user's body composition volume metrics to relevant concepts.
        *   Specifies that figures analogous to Segar's should be generated for M vs. W.
2.  **System Processing:**
    *   `Segar_paper.pdf` is processed by GROBID and `enhance_paper_llm.py` -> `segar_enriched.json`.
    *   `generate_analysis_code.py` uses `segar_enriched.json` and the user's `project_analysis_config.yaml`.
3.  **Output:**
    *   Python/R scripts to perform survival analysis, generate Kaplan-Meier curves (similar to Segar's but for M/W), create tables comparing baseline characteristics, etc., all based on `my_liver_patient_data.csv`.
    *   Generated figures and tables.

