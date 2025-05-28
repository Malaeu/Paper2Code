#!/bin/bash
# Modern PDF to JSON conversion script
# Uses vision models instead of GROBID

set -e

# Default values
PDF_PATH="${1:-../examples/Transformer.pdf}"
OUTPUT_DIR="${2:-../outputs}"
MODEL="${3:-gemini-2.5-flash}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Modern PDF to JSON Converter${NC}"
echo -e "${YELLOW}Using model: ${MODEL}${NC}"

# Check if PDF exists
if [ ! -f "$PDF_PATH" ]; then
    echo -e "${RED}Error: PDF file not found: $PDF_PATH${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract filename without extension
FILENAME=$(basename "$PDF_PATH" .pdf)
OUTPUT_PATH="$OUTPUT_DIR/${FILENAME}_modern.json"

# Install dependencies if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
pip show pdf2image >/dev/null 2>&1 || pip install pdf2image
pip show pytesseract >/dev/null 2>&1 || pip install pytesseract
pip show aiohttp >/dev/null 2>&1 || pip install aiohttp
pip show tqdm >/dev/null 2>&1 || pip install tqdm

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo -e "${RED}Tesseract not found. Installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install tesseract
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y tesseract-ocr
    else
        echo -e "${RED}Please install Tesseract manually${NC}"
        exit 1
    fi
fi

# Run conversion
echo -e "${GREEN}Converting PDF to JSON...${NC}"

# Check if API key is set for vision models
if [ -n "$GEMINI_API_KEY" ]; then
    echo -e "${GREEN}Using Gemini Vision API${NC}"
    python ../codes/pdf_to_json_modern.py \
        --input "$PDF_PATH" \
        --output "$OUTPUT_PATH" \
        --model "$MODEL"
else
    echo -e "${YELLOW}No API key found. Using OCR fallback...${NC}"
    python ../codes/pdf_to_json_modern.py \
        --input "$PDF_PATH" \
        --output "$OUTPUT_PATH" \
        --ocr-only
fi

# Check if output was created
if [ -f "$OUTPUT_PATH" ]; then
    echo -e "${GREEN}✅ Success! Output saved to: $OUTPUT_PATH${NC}"
    
    # Show file size and preview
    SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    echo -e "${YELLOW}File size: $SIZE${NC}"
    echo -e "${YELLOW}Preview:${NC}"
    head -n 20 "$OUTPUT_PATH"
else
    echo -e "${RED}❌ Error: Output file was not created${NC}"
    exit 1
fi

# Compare with old GROBID method if available
OLD_OUTPUT="../s2orc-doc2json/output_dir/paper_coder/${FILENAME}.json"
if [ -f "$OLD_OUTPUT" ]; then
    echo -e "${YELLOW}Comparing with GROBID output...${NC}"
    OLD_SIZE=$(du -h "$OLD_OUTPUT" | cut -f1)
    echo -e "GROBID size: $OLD_SIZE vs Modern: $SIZE"
fi

echo -e "${GREEN}🎉 Conversion complete!${NC}"