## Paper2Code Benchmark

### Overview
The **Paper2Code Benchmark** is designed to evaluate the ability to reproduce methods and experiments described in scientific papers.

We collected **90 papers** from **ICML 2024**, **NeurIPS 2024**, and **ICLR 2024**, selecting only those with publicly available GitHub repositories.  
To ensure manageable complexity, we filtered for repositories with fewer than **70,000 tokens**.  
Using a model-based evaluation, we selected the **top 30 papers** from each conference based on repository quality.

A full list of the benchmark papers is provided in `dataset_info.json`.
For more details, refer to Section 4.1 "Paper2Code Benchmark" of the [paper](https://arxiv.org/abs/2504.17192).

### How to Use
- Unzip the `paper2code_data.zip` file:
```bash
unzip paper2code_data.zip
```

### Data Structure
Each conference folder is organized as follows:
- `[PAPER].json` — Parsed version of the paper
- `[PAPER]_cleaned.json` — Preprocessed version for PaperCoder

```bash
├── iclr2024 
├── icml2024
└── nips2024
    ├── adaptive-randomized-smoothing.json
    ├── adaptive-randomized-smoothing_cleaned.json
    ├── ... 
    ├── YOLA.json
    └── YOLA_cleaned.json
```
