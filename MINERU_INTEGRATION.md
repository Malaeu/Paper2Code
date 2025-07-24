# MinerU Integration for Paper2Code

This document describes the MinerU integration that replaces GROBID for PDF processing in Paper2Code.

## 🎯 Overview

MinerU is a modern PDF processing tool with advanced OCR and layout analysis capabilities. This integration provides:

- **Better OCR Quality**: Superior text extraction with vision-language models
- **Rich Media Support**: Automatic extraction of images, tables, and formulas  
- **Advanced Layout Analysis**: Better understanding of document structure
- **Gemini Vision Enhancement**: Optional image analysis with Google's Gemini models
- **No External Dependencies**: No need to run GROBID servers

## 🏗️ Architecture

```
PDF Input → MinerU Processor → Paper2Code JSON → Enhanced Processing
    ↓              ↓                   ↓              ↓
Raw PDF    ┌─────────────┐     ┌─────────────┐  ┌─────────────┐
          │ MinerU OCR  │     │ Format      │  │ Gemini      │
          │ Layout      │ →   │ Converter   │→ │ Vision      │
          │ Extraction  │     │ (JSON)      │  │ Enhancer    │
          └─────────────┘     └─────────────┘  └─────────────┘
                 ↓                   ↓              ↓
          ┌─────────────┐     ┌─────────────┐  ┌─────────────┐
          │ Images/     │     │ Paper2Code  │  │ Enhanced    │
          │ Tables/     │     │ Compatible  │  │ Descriptions│
          │ Formulas    │     │ JSON        │  │ + Analysis  │
          └─────────────┘     └─────────────┘  └─────────────┘
```

## 📦 Components

### Core Modules

1. **`codes/mineru_processor.py`** - Main MinerU integration wrapper
2. **`codes/mineru_to_paper2code.py`** - Format converter for Paper2Code compatibility
3. **`codes/mineru_image_enhancer.py`** - Gemini Vision image analysis
4. **`codes/table_processor.py`** - HTML table processing and extraction
5. **`codes/formula_processor.py`** - LaTeX formula analysis
6. **`codes/mineru_config.py`** - Configuration management

### Configuration

- **`config/mineru_config.yaml`** - Main configuration file

### Updated Scripts

- **`scripts/run_custom_adapt.sh`** - Uses MinerU instead of GROBID
- **`scripts/run_custom_enhanced.sh`** - Enhanced processing with MinerU
- **`codes/0_pdf_process.py`** - Updated for MinerU compatibility
- **`webapp/app/services/pipeline/pipeline_service.py`** - Web interface integration

## 🚀 Quick Start

### Prerequisites

1. **MinerU Installation**: MinerU should be installed at `/media/chirurgie/hdd01/Soft/GitHub/MinerU`
2. **Python Dependencies**: Install required packages:
   ```bash
   pip install google-generativeai beautifulsoup4 pandas sympy pyyaml
   ```
3. **API Keys** (optional):
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"  # For image enhancement
   ```

### Basic Usage

#### 1. Process PDF with MinerU
```bash
python codes/mineru_processor.py \
    --pdf_path "path/to/paper.pdf" \
    --output_dir "output/" \
    --json_output "paper.json"
```

#### 2. Run Full Adaptation Pipeline
```bash
# Set up your parameters in the script
bash scripts/run_custom_adapt.sh
```

#### 3. Enhance Images with Gemini Vision
```bash
python codes/mineru_image_enhancer.py \
    --input "paper.json" \
    --images_dir "output/mineru_output" \
    --output "enhanced_paper.json"
```

## 🔧 Configuration

### MinerU Settings

Edit `config/mineru_config.yaml`:

```yaml
mineru:
  installation_path: "/path/to/MinerU"
  venv_path: "/path/to/MinerU/.venv"
  method: "ocr"  # or "auto"
  timeout: 600

gemini:
  model: "gemini-2.0-flash-exp"
  vision_enhancement: true
  rate_limit_delay: 0.5
```

### Environment Variables

```bash
# MinerU paths
export MINERU_PATH="/path/to/MinerU"
export MINERU_VENV="/path/to/MinerU/.venv"

# Gemini API
export GEMINI_API_KEY="your_api_key"

# Processing method
export MINERU_METHOD="ocr"
```

## 📊 Output Formats

### MinerU Raw Output
```
output_dir/
├── paper.md                    # Main markdown content
├── paper_content_list.json     # Structured content list
├── paper_model.json           # ML inference results  
├── paper_middle.json          # Processing intermediates
├── paper_layout.pdf           # Layout visualization
├── paper_spans.pdf            # Text spans visualization
└── images/                    # Extracted images
    ├── hash1.jpg
    └── hash2.jpg
```

### Paper2Code JSON Output
```json
{
  "paper_id": "mineru_12345",
  "metadata": {
    "title": "Paper Title",
    "abstract": "Abstract text...",
    "source": "MinerU"
  },
  "abstract": [...],
  "body_text": [...],
  "ref_entries": {
    "FIGREF_1": {
      "type": "figure",
      "text": "Figure caption",
      "image_path": "./images/hash.jpg",
      "llm_description": "Detailed Gemini analysis...",
      "page": 2,
      "mineru_source": true
    },
    "TABREF_1": {
      "type": "table",
      "text": "Table caption",
      "html": "<table>...</table>",
      "image_path": "./images/table.jpg",
      "page": 3,
      "mineru_source": true
    }
  },
  "mineru_metadata": {
    "backend": "pipeline",
    "version": "2.1.5",
    "total_pages": 12
  }
}
```

## 🛠️ Advanced Usage

### Table Processing
```bash
python codes/table_processor.py \
    --input "paper.json" \
    --output_dir "tables/" \
    --analyze
```

### Formula Analysis
```bash
python codes/formula_processor.py \
    --input "paper.json" \
    --output_dir "formulas/"
```

### Configuration Management
```bash
# Validate configuration
python codes/mineru_config.py --validate

# Show current settings
python codes/mineru_config.py --show

# Apply environment overrides
python codes/mineru_config.py --apply-env
```

## 🔍 Troubleshooting

### Common Issues

1. **MinerU Not Found**
   ```
   ERROR: MinerU not found at /path/to/MinerU
   ```
   **Solution**: Check `mineru.installation_path` in config

2. **Virtual Environment Issues**
   ```
   ERROR: MinerU venv not found
   ```
   **Solution**: Verify `mineru.venv_path` points to correct venv

3. **Processing Timeout**
   ```
   ERROR: MinerU processing timed out
   ```
   **Solution**: Increase `mineru.timeout` in config or use smaller PDFs

4. **Gemini API Errors**
   ```
   WARNING: Gemini Vision enhancement failed
   ```
   **Solution**: Check `GEMINI_API_KEY` and rate limits

### Debugging

Enable debug mode in config:
```yaml
development:
  debug_mode: true
  verbose: true
  save_intermediates: true

logging:
  level: DEBUG
```

### Performance Tuning

For large PDFs:
```yaml
performance:
  use_gpu: true
  memory_limit: 8192  # Increase memory limit
  
mineru:
  timeout: 1200  # Increase timeout
```

## 🔄 Migration from GROBID

### Automatic Migration

The integration is designed to be backward compatible:

1. **Scripts Updated**: `run_custom_adapt.sh` automatically uses MinerU
2. **Format Compatible**: Output is compatible with existing Paper2Code pipeline
3. **Fallback Support**: Can fallback to GROBID if configured

### Manual Migration

1. **Update Scripts**: Replace GROBID calls with MinerU processor
2. **Update Dependencies**: Install MinerU and optional dependencies
3. **Configure Paths**: Set MinerU installation path in config
4. **Test Pipeline**: Run with sample PDFs to verify functionality

## 📈 Performance Comparison

| Feature | GROBID | MinerU |
|---------|---------|---------|
| OCR Quality | Good | Excellent |
| Layout Analysis | Basic | Advanced |
| Image Extraction | Manual | Automatic |
| Table Processing | Limited | Rich HTML |
| Formula Support | Basic | LaTeX + Analysis |
| Setup Complexity | High (Server) | Low (Direct) |
| Processing Speed | Fast | Moderate |
| GPU Support | No | Yes |

## 🤝 Contributing

To contribute to MinerU integration:

1. **Test Coverage**: Add tests for new features
2. **Documentation**: Update this guide for new functionality  
3. **Configuration**: Add new config options as needed
4. **Error Handling**: Improve error messages and recovery

## 📚 References

- [MinerU Documentation](https://github.com/opendatalab/MinerU)
- [Paper2Code Original](https://github.com/going-doer/Paper2Code)
- [Gemini API Documentation](https://ai.google.dev/docs)

---

**Note**: This integration significantly improves PDF processing quality but may require more computational resources than GROBID. GPU acceleration is recommended for optimal performance.