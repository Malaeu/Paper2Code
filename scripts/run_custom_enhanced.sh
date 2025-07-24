#!/bin/bash

# API key is already in environment

GPT_VERSION="o3-2025-04-16"
IMAGE_GPT_VERSION="o4-mini-2025-04-16"

PAPER_NAME="Segar"
PDF_PATH="/Users/Lordof44/Projects/segar-et-al-development-and-validation-of-machine-learning-based-race-specific-models-to-predict-10-year-risk-of-heart.pdf"
CUSTOM_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper"
PDF_JSON_PATH="${CUSTOM_DIR}/paper.json"
PDF_JSON_CLEANED_PATH="${CUSTOM_DIR}/paper_cleaned.json"
ENHANCED_JSON_PATH="${CUSTOM_DIR}/enhanced_paper.json"
OUTPUT_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/Segar_enhanced"
OUTPUT_REPO_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/Segar_repo_enhanced"

mkdir -p $CUSTOM_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

# Copy the paper to the custom directory
echo "------- Copying Paper -------"
cp "$PDF_PATH" "${CUSTOM_DIR}/paper.pdf"
PDF_PATH="${CUSTOM_DIR}/paper.pdf"

# Process PDF with MinerU (replaces GROBID)
echo "------- Processing PDF with MinerU -------"
echo "Using MinerU for advanced OCR and layout analysis (no GROBID needed)..."

python codes/mineru_processor.py \
    --pdf_path "$PDF_PATH" \
    --output_dir "${CUSTOM_DIR}/mineru_output" \
    --json_output ${PDF_JSON_PATH}

# Check if PDF processing was successful
if [ ! -f "${PDF_JSON_PATH}" ]; then
    echo "ERROR: MinerU PDF processing failed. Check the logs above."
    exit 1
fi

echo "------- Preprocess -------"
python codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH}

# Enhance images with Gemini Vision if API key is available
if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "------- Enhancing Images with Gemini Vision -------"
    
    # Install google-generativeai if not already installed
    pip install google-generativeai
    
    # Enhance images with detailed descriptions using MinerU extracted images
    python codes/mineru_image_enhancer.py \
        --input ${PDF_JSON_CLEANED_PATH} \
        --images_dir "${CUSTOM_DIR}/mineru_output" \
        --output ${ENHANCED_JSON_PATH} \
        --format paper2code
    
    # Use the enhanced JSON for the rest of the pipeline
    if [ -f "$ENHANCED_JSON_PATH" ]; then
        echo "Using enhanced JSON with Gemini Vision descriptions"
        PDF_JSON_CLEANED_PATH=${ENHANCED_JSON_PATH}
    else
        echo "WARNING: Gemini Vision enhancement failed, using regular cleaned JSON"
    fi
else
    echo "GEMINI_API_KEY not set, skipping image enhancement"
fi

echo "------- PaperCoder -------"
python codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR}