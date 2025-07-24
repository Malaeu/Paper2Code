#!/bin/bash

# Script to process PDFs for the Paper2Code pipeline
# This script handles the conversion of PDF documents to the required JSON format

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}        PAPER2CODE PDF PROCESSOR         ${NC}"
echo -e "${BLUE}==========================================${NC}"

# Function to print usage information
usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  -i, --input PATH    Input PDF file path (required)"
    echo -e "  -o, --output PATH   Output JSON file path (default: same as input with .json extension)"
    echo -e "  -h, --help          Show this help message"
    echo -e "  -d, --docker        Use Docker for GROBID (if available)"
    echo -e "  -g, --grobid PATH   Path to GROBID installation (default: ./s2orc-doc2json/grobid-0.7.3)"
    echo -e "  -s, --s2orc PATH    Path to S2ORC installation (default: ./s2orc-doc2json)"
    exit 1
}

# Default values
INPUT_PDF=""
OUTPUT_JSON=""
USE_DOCKER=false
GROBID_PATH="./s2orc-doc2json/grobid-0.7.3"
S2ORC_PATH="./s2orc-doc2json"
TEMP_DIR="temp_pdf_processing"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_PDF="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_JSON="$2"
            shift
            shift
            ;;
        -d|--docker)
            USE_DOCKER=true
            shift
            ;;
        -g|--grobid)
            GROBID_PATH="$2"
            shift
            shift
            ;;
        -s|--s2orc)
            S2ORC_PATH="$2"
            shift
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Check if input is provided
if [ -z "$INPUT_PDF" ]; then
    echo -e "${RED}Error: Input PDF file is required${NC}"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_PDF" ]; then
    echo -e "${RED}Error: Input file '$INPUT_PDF' does not exist${NC}"
    exit 1
fi

# Set default output path if not provided
if [ -z "$OUTPUT_JSON" ]; then
    # Remove .pdf extension and add .json
    OUTPUT_JSON="${INPUT_PDF%.pdf}.json"
    echo -e "${YELLOW}No output path specified. Using: $OUTPUT_JSON${NC}"
fi

# Create temp directory
mkdir -p "$TEMP_DIR"
echo -e "${GREEN}Created temporary directory: $TEMP_DIR${NC}"

# Process the PDF
echo -e "\n${YELLOW}[1/3] PROCESSING PDF WITH GROBID${NC}"

if [ "$USE_DOCKER" = true ]; then
    echo -e "Using Docker for GROBID processing..."
    
    # Check if GROBID container is running
    if ! docker ps | grep -q grobid; then
        echo -e "${YELLOW}GROBID container not running. Starting it...${NC}"
        docker run --rm --init --ulimit core=0 -d -p 8070:8070 --name grobid lfoppiano/grobid:0.7.3
        sleep 10  # Wait for GROBID to start
    fi
    
    # Check if s2orc-doc2json is installed, if not clone it
    if [ ! -d "$S2ORC_PATH" ]; then
        echo -e "${YELLOW}S2ORC not found, cloning repository...${NC}"
        git clone https://github.com/allenai/s2orc-doc2json.git
    fi
    
    # Create a temporary directory for intermediate results
    INTERMEDIATE_DIR="$TEMP_DIR/intermediate"
    mkdir -p "$INTERMEDIATE_DIR"
    
    # Process PDF using S2ORC with Docker GROBID
    echo -e "Converting PDF to JSON format..."
    python $S2ORC_PATH/doc2json/grobid2json/process_pdf.py \
        -i "$INPUT_PDF" \
        -t "$TEMP_DIR" \
        -o "$INTERMEDIATE_DIR" \
        -g http://localhost:8070
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: PDF processing failed${NC}"
        exit 1
    fi
    
    # Find the generated JSON file
    INTERMEDIATE_JSON=$(find "$INTERMEDIATE_DIR" -name "*.json" | head -n 1)
    
    if [ -z "$INTERMEDIATE_JSON" ]; then
        echo -e "${RED}Error: No JSON file was generated${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ PDF processed successfully with GROBID${NC}"
    echo -e "  Intermediate output: $INTERMEDIATE_JSON"
    
else
    # Check if GROBID and S2ORC paths exist
    if [ ! -d "$GROBID_PATH" ] || [ ! -d "$S2ORC_PATH" ]; then
        echo -e "${RED}Error: GROBID or S2ORC paths not found.${NC}"
        echo -e "${YELLOW}Please install GROBID and S2ORC or use Docker mode.${NC}"
        echo -e "You can clone S2ORC with: git clone https://github.com/allenai/s2orc-doc2json.git"
        exit 1
    fi
    
    # Start GROBID
    echo -e "Starting GROBID service..."
    (cd "$GROBID_PATH" && ./gradlew run > "$TEMP_DIR/grobid.log" 2>&1) &
    GROBID_PID=$!
    
    # Wait for GROBID to start
    echo -e "Waiting for GROBID to start (this might take a minute)..."
    sleep 30
    
    # Create a temporary directory for intermediate results
    INTERMEDIATE_DIR="$TEMP_DIR/intermediate"
    mkdir -p "$INTERMEDIATE_DIR"
    
    # Process PDF using S2ORC
    echo -e "Converting PDF to JSON format..."
    python "$S2ORC_PATH/doc2json/grobid2json/process_pdf.py" \
        -i "$INPUT_PDF" \
        -t "$TEMP_DIR" \
        -o "$INTERMEDIATE_DIR"
    
    # Stop GROBID
    kill $GROBID_PID
    
    # Find the generated JSON file
    INTERMEDIATE_JSON=$(find "$INTERMEDIATE_DIR" -name "*.json" | head -n 1)
    
    if [ -z "$INTERMEDIATE_JSON" ]; then
        echo -e "${RED}Error: No JSON file was generated${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ PDF processed successfully with GROBID${NC}"
    echo -e "  Intermediate output: $INTERMEDIATE_JSON"
fi

# Clean the JSON using 0_pdf_process.py
echo -e "\n${YELLOW}[2/3] CLEANING JSON${NC}"
echo -e "Processing JSON with 0_pdf_process.py..."

python codes/0_pdf_process.py \
    --input_json_path "$INTERMEDIATE_JSON" \
    --output_json_path "$OUTPUT_JSON"

if [ $? -eq 0 ] && [ -f "$OUTPUT_JSON" ]; then
    echo -e "${GREEN}✓ JSON cleaning completed successfully${NC}"
    echo -e "  Output: $OUTPUT_JSON ($(du -h "$OUTPUT_JSON" | cut -f1))"
else
    echo -e "${RED}✗ JSON cleaning failed${NC}"
    exit 1
fi

# Optionally extract figures if extract_figures.py exists
if [ -f "codes/extract_figures.py" ]; then
    echo -e "\n${YELLOW}[3/3] EXTRACTING FIGURES (OPTIONAL)${NC}"
    echo -e "Would you like to extract and analyze figures from the PDF? (y/n)"
    read -r extract_figures
    
    if [[ "$extract_figures" =~ ^[Yy]$ ]]; then
        FIGURES_DIR="${OUTPUT_JSON%.json}_figures"
        mkdir -p "$FIGURES_DIR"
        
        echo -e "Extracting figures from PDF..."
        python codes/extract_figures.py \
            --pdf_path "$INPUT_PDF" \
            --json_path "$OUTPUT_JSON" \
            --output_dir "$FIGURES_DIR" \
            --gpt_version o4-mini-2025-04-16
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Figure extraction completed successfully${NC}"
            echo -e "  Figures saved to: $FIGURES_DIR"
        else
            echo -e "${RED}✗ Figure extraction failed${NC}"
            echo -e "${YELLOW}Continuing with the main JSON output...${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping figure extraction${NC}"
    fi
else
    echo -e "\n${YELLOW}Figure extraction module not found, skipping this step${NC}"
fi

# Clean up temporary files
echo -e "\n${YELLOW}Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"

echo -e "\n${BLUE}==========================================${NC}"
echo -e "${BLUE}          PROCESSING COMPLETE            ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "The processed JSON is available at:"
echo -e "  ${GREEN}$OUTPUT_JSON${NC}"
echo -e "\nYou can now use this JSON file with the Paper2Code adaptation pipeline:"
echo -e "  ${BLUE}./test_adaptation_cli.sh${NC}"
echo -e "  ${YELLOW}export PAPER2CODE_PAPER=$OUTPUT_JSON${NC}"

# Make the script executable
chmod +x "$0"