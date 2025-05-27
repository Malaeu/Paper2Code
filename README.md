# 📄 Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning

![PaperCoder Overview](./assets/papercoder_overview.png)

📄 [Read the paper on arXiv](https://arxiv.org/abs/2504.17192)

**PaperCoder** is a multi-agent LLM system that transforms paper into a code repository.
It follows a three-stage pipeline: planning, analysis, and code generation, each handled by specialized agents.  
Our method outperforms strong baselines on both Paper2Code and PaperBench and produces faithful, high-quality implementations.

---

## 🗺️ Table of Contents

- [⚡ Quick Start](#-quick-start)
- [📚 Detailed Setup Instructions](#-detailed-setup-instructions)
- [📦 Paper2Code Benchmark Datasets](#-paper2code-benchmark-datasets)
- [📊 Model-based Evaluation of Repositories](#-model-based-evaluation-of-repositories-generated-by-papercoder)
- [🔀 LLM Router](#-llm-router)

---

## ⚡ Quick Start
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  

### Using OpenAI API
- 💵 Estimated cost for using o3-mini: $0.50–$0.70

```bash
pip install openai

export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
```

### Using Open Source Models with vLLM
- If you encounter any issues installing vLLM, please refer to the [official vLLM repository](https://github.com/vllm-project/vllm).
- The default model is `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

```bash
pip install vllm

cd scripts
bash run_llm.sh
```

### Output Folder Structure (Only Important Files)
```bash
outputs
├── Transformer
│   ├── analyzing_artifacts
│   ├── coding_artifacts
│   └── planning_artifacts
└── Transformer_repo  # Final output repository
```
---

## 📚 Detailed Setup Instructions

### 🛠️ Environment Setup

- 💡 To use the `o3-mini` version, make sure you have the latest `openai` package installed.
- 📦 Install only what you need:
  - For OpenAI API: `openai`
  - For open-source models: `vllm`
      - If you encounter any issues installing vLLM, please refer to the [official vLLM repository](https://github.com/vllm-project/vllm).


```bash
pip install openai 
pip install vllm 
```

- Or, if you prefer, you can install all dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 📄 (Option) Convert PDF to JSON
The following process describes how to convert a paper PDF into JSON format.
If you have access to the LaTeX source and plan to use it with PaperCoder, you may skip this step and proceed to [🚀 Running PaperCoder](#-running-papercoder).

Note: In our experiments, we converted all paper PDFs to JSON format. The original workflow relied on the
[`s2orc-doc2json`](https://github.com/allenai/s2orc-doc2json) repository. As of 2025 more capable open-source
libraries exist. We provide two approaches below.

#### Legacy approach (compatible with older instructions)

1. Clone `s2orc-doc2json` and run its processing service:

```bash
git clone https://github.com/allenai/s2orc-doc2json.git
cd ./s2orc-doc2json/grobid-0.7.3
./gradlew run
```

2. Convert the PDF into JSON format using the bundled script:

```bash
mkdir -p ./s2orc-doc2json/output_dir/paper_coder
python ./s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
    -i ${PDF_PATH} \
    -t ./s2orc-doc2json/temp_dir/ \
    -o ./s2orc-doc2json/output_dir/paper_coder
```

#### Hybrid approach (recommended for 2025)

1. Install modern PDF processing libraries.

```bash
pip install PyMuPDF pdfplumber layoutparser
```

2. Ensure the latest `grobid` server (v0.8 or later) is running.

3. Use the script [`codes/pdf_to_json_hybrid.py`](./codes/pdf_to_json_hybrid.py) to combine
page-level text extraction with metadata from `grobid` and produce a single JSON file:

```bash
python codes/pdf_to_json_hybrid.py \
    --pdf_path ${PDF_PATH} \
    --output_json ./paper_coder_output/paper.json \
    --grobid_url http://localhost:8070
```

This hybrid pipeline leverages modern layout analysis tools for accurate page content
while still using `grobid` for reliable metadata extraction.

#### Simple approach (no `grobid`)

1. Install lightweight dependencies.

```bash
pip install PyMuPDF pdf2image pytesseract camelot-py
```

2. Run the script [`codes/pdf_to_json_simple.py`](./codes/pdf_to_json_simple.py):

```bash
python codes/pdf_to_json_simple.py \
    --pdf_path ${PDF_PATH} \
    --output_json ./paper_coder_output/paper.json
```

This method relies solely on PyMuPDF and OCR, optionally using `camelot` to
extract tables.

### 🚀 Running PaperCoder
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  
  If you want to run PaperCoder on your own paper, please modify the environment variables accordingly.

#### Using OpenAI API
- 💵 Estimated cost for using o3-mini: $0.50–$0.70


```bash
# Using the PDF-based JSON format of the paper
export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
```

```bash
# Using the LaTeX source of the paper
export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run_latex.sh
```


#### Using Open Source Models with vLLM
- The default model is `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

```bash
# Using the PDF-based JSON format of the paper
cd scripts
bash run_llm.sh
```

```bash
# Using the LaTeX source of the paper
cd scripts
bash run_latex_llm.sh
```

---

## 📦 Paper2Code Benchmark Datasets
- Huggingface dataset: [paper2code](https://huggingface.co/datasets/iaminju/paper2code)
  
- You can find the description of the Paper2Code benchmark dataset in [data/paper2code](https://github.com/going-doer/Paper2Code/tree/main/data/paper2code). 
- For more details, refer to Section 4.1 "Paper2Code Benchmark" in the [paper](https://arxiv.org/abs/2504.17192).


---

## 📊 Model-based Evaluation of Repositories Generated by PaperCoder

- We evaluate repository quality using a model-based approach, supporting both reference-based and reference-free settings.  
  The model critiques key implementation components, assigns severity levels, and generates a 1–5 correctness score averaged over 8 samples using **o3-mini-high**.

- For more details, please refer to Section 4.3.1 (*Paper2Code Benchmark*) of the paper.
- **Note:** The following examples evaluate the sample repository (**Transformer_repo**).  
  Please modify the relevant paths and arguments if you wish to evaluate a different repository.

### 🛠️ Environment Setup
```bash
pip install tiktoken
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```


### 📝 Reference-free Evaluation
- `target_repo_dir` is the generated repository.

```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --eval_result_dir ../results \
    --eval_type ref_free \
    --generated_n 8 \
    --papercoder
```

### 📝 Reference-based Evaluation
- `target_repo_dir` is the generated repository.
- `gold_repo_dir` should point to the official repository (e.g., author-released code).

```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --gold_repo_dir ../examples/Transformer_gold_repo \
    --eval_result_dir ../results \
    --eval_type ref_based \
    --generated_n 8 \
    --papercoder
```


### 📄 Example Output
```bash
========================================
🌟 Evaluation Summary 🌟
📄 Paper name: Transformer
🧪 Evaluation type: ref_based
📁 Target repo directory: ../outputs/Transformer_repo
📊 Evaluation result:
        📈 Score: 4.5000
        ✅ Valid: 8/8
========================================
🌟 Usage Summary 🌟
[Evaluation] Transformer - ref_based
🛠️ Model: o3-mini
📥 Input tokens: 44318 (Cost: $0.04874980)
📦 Cached input tokens: 0 (Cost: $0.00000000)
📤 Output tokens: 26310 (Cost: $0.11576400)
💵 Current total cost: $0.16451380
🪙 Accumulated total cost so far: $0.16451380
============================================
```

## 🔀 LLM Router
The router configuration lives in [`llm_router/config.yaml`](./llm_router/config.yaml).

| Task Pattern | Primary Model | Fallback |
|--------------|--------------|---------|
| `chat\|faq\|rag` | `gemini_flash_25` | `claude_sonnet_35` |
| `code\|unit_tests` | `claude_sonnet_37` | `o4mini` |
| `long_doc>300k` | `gpt41` | `claude_sonnet_35` |
| `tool_reasoning` | `o4mini` | `gemini_flash_25` |

Override the config by setting `LLM_CFG`:

```bash
export LLM_CFG=/path/to/custom.yaml
```
If `PyYAML` is unavailable, the router falls back to a minimal built-in parser.

## 💵 Official AI Model API Pricing (May 2025)

The following prices were collected from official documentation in May 2025. All values are shown per **million tokens**.

### OpenAI Models
- **o4-mini-2025-04-16**: Input `$1.10`, Output `$4.40` – fast, cost‑efficient reasoning with multimodal support.
- **gpt-4.1-2025-04-14**: Input `$2.00`, Output `$8.00` – improved coding and instruction following with a 1M token context window.
- **o3-2025-04-16**: Input `$10.00` (cached input `$2.50`), Output `$40.00` – OpenAI's most powerful reasoning model with a 200K token context window.

### Google Gemini Models
- **Gemini 2.5 Flash (preview)**:
  - Input: Text/Image/Video `$0.15`, Audio `$1.00`
  - Output: Non-thinking mode `$0.60`, Thinking mode `$3.50`
  - First Flash model with thinking capabilities (preview).
- **Gemini 2.5 Pro (preview)**:
  - Input ≤ 200k tokens `$1.25`, > 200k tokens `$2.50`
  - Output ≤ 200k tokens `$10.00`, > 200k tokens `$15.00`
  - Most advanced Gemini reasoning model with a 1M token context window.

Prices may change as these models move from preview to general availability. Consult the respective provider pages for the latest information.
