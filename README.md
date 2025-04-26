# 📄 Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning

![PaperCoder Overview](./assets/papercoder_overview.png)

📄 [Read the paper on arXiv](https://arxiv.org/abs/2504.17192)

**PaperCoder** is a multi-agent LLM system that transforms paper into code repository.
It follows a three-stage pipeline: planning, analysis, and code generation, each handled by specialized agents.  
Our method outperforms strong baselines both on Paper2Code and PaperBench and produces faithful, high-quality implementations.

---

## ⚡ QuickStart
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  
- 💵 Estimated cost for using o3-mini: $0.50–$0.70

```bash
pip install openai

export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
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

- Note: If you wish to use the `o3-mini` version, please make sure to install the latest version of the OpenAI package.

```bash
pip install openai
```

### 📄 Convert PDF to JSON

1. Clone the `s2orc-doc2json` repository to convert your PDF file into a structured JSON format.  
   (For detailed configuration, please refer to the [official repository](https://github.com/allenai/s2orc-doc2json).)

```bash
git clone https://github.com/allenai/s2orc-doc2json.git
```

2. Running the PDF processing service.

```bash
cd ./s2orc-doc2json/grobid-0.7.3
./gradlew run
```

3. Convert your PDF into JSON format.

```bash
mkdir -p ./s2orc-doc2json/output_dir/paper_coder
python ./s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
    -i ${PDF_PATH} \
    -t ./s2orc-doc2json/temp_dir/ \ 
    -o ./s2orc-doc2json/output_dir/paper_coder
```

### 🚀 Runing PaperCoder
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  
  If you want to run PaperCoder on your own paper, please modify the environment variables accordingly.

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
```

