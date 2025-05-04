#!/bin/bash

# API key is already in environment

GPT_VERSION="o3-2025-04-16"
IMAGE_GPT_VERSION="o4-mini-2025-04-16"

# Original paper parameters
PAPER_NAME="Segar"
PAPER_PDF_PATH="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper/paper.pdf"
PAPER_JSON_PATH="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper/paper.json"
PAPER_JSON_CLEANED_PATH="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper/paper_cleaned.json"
PAPER_ENHANCED_JSON_PATH="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper/enhanced_paper.json"

# User's adaptation parameters
ADAPTED_NAME="GenderBasedModel"
ADAPTED_PATH="/Users/Lordof44/Documents/MyData"
ADAPTED_DATA_FILE="${ADAPTED_PATH}/mydata.csv"
ADAPTED_MAPPING_FILE="${ADAPTED_PATH}/variable_mapping.json"
OUTPUT_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/${ADAPTED_NAME}"
OUTPUT_REPO_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/${ADAPTED_NAME}_repo"

# Create required directories
mkdir -p "${ADAPTED_PATH}"
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo "Creating adaptation from $PAPER_NAME to $ADAPTED_NAME"

# First, check if paper JSON already exists, otherwise process paper
if [ ! -f "$PAPER_JSON_CLEANED_PATH" ]; then
    echo "------- Processing Original Paper -------"
    
    # Check if GROBID is running
    echo "IMPORTANT: Make sure GROBID is running in another terminal with the command:"
    echo "cd \$HOME/grobid-0.7.3 && ./gradlew run"
    echo "Press Enter when GROBID is running..."
    read -p ""
    
    # Process PDF to JSON
    cd /Users/Lordof44/Documents/GitHub/Paper2Code
    source paper2code_env/bin/activate
    python s2orc-doc2json/doc2json/grobid2json/process_pdf.py -i "$PAPER_PDF_PATH" -t custom_paper/temp_dir/ -o custom_paper/
    
    # Preprocess
    python codes/0_pdf_process.py \
        --input_json_path ${PAPER_JSON_PATH} \
        --output_json_path ${PAPER_JSON_CLEANED_PATH}
    
    # Extract figures if not already done
    if [ ! -f "$PAPER_ENHANCED_JSON_PATH" ]; then
        echo "------- Extracting Figures and Getting LLM Descriptions -------"
        # Install PyMuPDF if not already installed
        pip install PyMuPDF
        
        # Run the figure extraction script
        python codes/extract_figures.py \
            --pdf_path "$PAPER_PDF_PATH" \
            --json_path ${PAPER_JSON_CLEANED_PATH} \
            --output_dir custom_paper \
            --gpt_version ${IMAGE_GPT_VERSION}
    fi
    
    # Use the enhanced JSON if available
    if [ -f "$PAPER_ENHANCED_JSON_PATH" ]; then
        echo "Using enhanced JSON with figure descriptions"
        PAPER_JSON_CLEANED_PATH=${PAPER_ENHANCED_JSON_PATH}
    fi
else
    echo "------- Using existing processed paper -------"
    # Use the enhanced JSON if available
    if [ -f "$PAPER_ENHANCED_JSON_PATH" ]; then
        echo "Using enhanced JSON with figure descriptions"
        PAPER_JSON_CLEANED_PATH=${PAPER_ENHANCED_JSON_PATH}
    fi
fi

# Check if user dataset exists
if [ ! -f "$ADAPTED_DATA_FILE" ]; then
    echo "------- Creating Sample Dataset -------"
    # Create a sample dataset for the user to modify or replace
    cat > "$ADAPTED_DATA_FILE" << EOL
id,gender,age,bmi,blood_marker_a,blood_marker_b,smoker,diabetes,hypertension,outcome
1,female,65,28.5,450,0.05,1,0,1,1
2,female,58,32.1,523,0.08,0,1,1,1
3,female,72,25.6,380,0.02,0,1,1,0
4,male,61,29.3,200,0.01,1,0,0,0
5,male,70,27.8,180,0.03,0,0,1,1
6,male,55,31.5,210,0.04,1,1,0,0
EOL
    echo "Sample dataset created at $ADAPTED_DATA_FILE"
    echo "You can replace this with your actual dataset before continuing."
    echo "Make sure to update the column names to match your data."
fi

# Check if variable mapping exists or create it automatically
if [ ! -f "$ADAPTED_MAPPING_FILE" ]; then
    echo "------- Analyzing Dataset and Creating Variable Mapping -------"
    
    # Check for pandas
    pip install pandas
    
    # Use the adapt_mapping.py to automatically generate mapping
    python codes/adapt_mapping.py \
        --paper_name $PAPER_NAME \
        --adapted_name $ADAPTED_NAME \
        --gpt_version ${GPT_VERSION} \
        --pdf_json_path ${PAPER_JSON_CLEANED_PATH} \
        --dataset_path ${ADAPTED_DATA_FILE} \
        --output_mapping_path ${ADAPTED_MAPPING_FILE} \
        --output_dir ${OUTPUT_DIR}
    
    echo "Variable mapping template created at $ADAPTED_MAPPING_FILE"
    echo "Please review and edit this file to match your data structure before continuing."
    echo "Once reviewed, run this script again to continue with adaptation."
    exit 0
fi

# Ask user if they have reviewed the mapping
read -p "Have you reviewed and edited the variable mapping file? (y/n): " reviewed
if [ "$reviewed" != "y" ]; then
    echo "Please review the mapping file at $ADAPTED_MAPPING_FILE before continuing."
    echo "Run this script again after reviewing the mapping."
    exit 0
fi

echo "------- Adaptation Planning -------"
python codes/adapt_planning.py \
    --paper_name $PAPER_NAME \
    --adapted_name $ADAPTED_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PAPER_JSON_CLEANED_PATH} \
    --mapping_file ${ADAPTED_MAPPING_FILE} \
    --output_dir ${OUTPUT_DIR}

echo "------- Adaptation Analysis -------"
python codes/adapt_analyzing.py \
    --paper_name $PAPER_NAME \
    --adapted_name $ADAPTED_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PAPER_JSON_CLEANED_PATH} \
    --mapping_file ${ADAPTED_MAPPING_FILE} \
    --output_dir ${OUTPUT_DIR}

echo "------- Adaptation Code Generation -------"
python codes/adapt_coding.py \
    --paper_name $PAPER_NAME \
    --adapted_name $ADAPTED_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PAPER_JSON_CLEANED_PATH} \
    --mapping_file ${ADAPTED_MAPPING_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR}

echo "------- Adaptation Complete -------"
echo "Adapted code has been generated at: ${OUTPUT_REPO_DIR}"
echo "You can now run the adapted analysis on your dataset."