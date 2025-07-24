# PDF Processing Guide for Paper2Code

This document explains how to properly convert PDF research papers to the JSON format required by the Paper2Code pipeline.

## PDF Processing Workflow

The Paper2Code system processes scientific papers using this workflow:

1. **Convert PDF to JSON** using S2ORC and GROBID tools
2. **Clean the JSON** using the 0_pdf_process.py script
3. **(Optional) Extract figures** for enhanced understanding

## Step 1: PDF to JSON Conversion

Paper2Code uses the [S2ORC-doc2json](https://github.com/allenai/s2orc-doc2json) tools with [GROBID](https://github.com/kermitt2/grobid) for initial PDF parsing.

### Setup GROBID (if not using Docker)

```bash
# Clone S2ORC repository
git clone https://github.com/allenai/s2orc-doc2json.git
cd s2orc-doc2json

# Set up GROBID
cd grobid-0.7.3
./gradlew run
```

### Convert PDF to JSON

```bash
# From the main project directory 
python s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
    -i path/to/your/paper.pdf \
    -t temp_dir/ \
    -o output_dir/
```

This will generate a JSON file in the specified output directory.

## Step 2: Clean JSON with 0_pdf_process.py

```bash
python codes/0_pdf_process.py \
    --input_json_path path/to/generated.json \
    --output_json_path path/to/cleaned.json
```

## Step 3: (Optional) Extract and Analyze Figures

For enhanced functionality, you can also extract and analyze figures from the PDF:

```bash
python codes/extract_figures.py \
    --pdf_path path/to/paper.pdf \
    --json_path path/to/cleaned.json \
    --output_dir path/to/output_dir \
    --gpt_version o4-mini-2025-04-16
```

## Docker Alternative

If you're using the Docker setup, GROBID is included as a service and the PDF processing is handled automatically.

## Troubleshooting

- **JSON Decode Errors**: Make sure you're using the correct processing order (PDF → S2ORC → 0_pdf_process.py)
- **GROBID Connection Issues**: Ensure GROBID service is running on port 8070
- **Missing Sections**: Some PDFs may not parse correctly due to unusual formatting

## Notes

- The adaptation pipeline scripts will attempt to use example files if direct PDF processing isn't available
- For complete control over PDF processing, set up GROBID and S2ORC properly