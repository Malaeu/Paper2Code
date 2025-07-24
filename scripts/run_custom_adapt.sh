#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# API key is already in environment

GPT_VERSION="o3-2025-04-16"
IMAGE_GPT_VERSION="o4-mini-2025-04-16"

# --- Custom Configuration for 'segar' Example ---
# Root directory of the Paper2Code project
PROJECT_ROOT="/media/chirurgie/hdd01/Soft/GitHub/Paper2Code"

# Directory for the 'segar' example data and inputs
EXAMPLE_DIR="${PROJECT_ROOT}/examples/segar"

# Directory to store processed files from the original paper for this example
PROCESSED_PAPER_DIR="${EXAMPLE_DIR}/processed_paper_files"

# Base directory for all outputs of this script run
OUTPUT_BASE_DIR="${PROJECT_ROOT}/outputs"

# Original paper parameters (specific to the 'segar' example context)
PAPER_NAME="Segar" # Name of the original paper being adapted
PAPER_PDF_PATH="${EXAMPLE_DIR}/paper.pdf" # Path to the PDF of the original paper within the example dir
# Assuming process_pdf.py outputs paper.json to the -o directory
PAPER_JSON_PATH="${PROCESSED_PAPER_DIR}/paper.json" # Intermediate JSON from PDF processing
PAPER_JSON_CLEANED_PATH="${PROCESSED_PAPER_DIR}/paper_cleaned.json" # Cleaned JSON
# Assuming extract_figures.py outputs enhanced_paper.json to its --output_dir
PAPER_ENHANCED_JSON_PATH="${PROCESSED_PAPER_DIR}/enhanced_paper.json" # Enhanced JSON with figure descriptions

# User's adaptation parameters (specific to the 'segar' example)
ADAPTED_NAME="segar" # Name for this specific adaptation instance
ADAPTED_DATA_FILE="${EXAMPLE_DIR}/data.csv" # Path to the user's dataset CSV within the example dir
ADAPTED_MAPPING_FILE="${EXAMPLE_DIR}/mapping.json" # Path to the variable mapping file within the example dir

# Output directories for this adaptation run
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${PAPER_NAME}_${ADAPTED_NAME}_Adaptation"
OUTPUT_REPO_DIR="${OUTPUT_BASE_DIR}/${PAPER_NAME}_${ADAPTED_NAME}_Adaptation_repo"

# Create required directories
mkdir -p "${EXAMPLE_DIR}" # Ensures example directory exists
mkdir -p "${PROCESSED_PAPER_DIR}" # Ensures directory for processed paper files exists
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_REPO_DIR}"

echo "Creating adaptation from $PAPER_NAME to $ADAPTED_NAME"

# First, check if paper JSON already exists, otherwise process paper
if [ ! -f "$PAPER_JSON_CLEANED_PATH" ]; then
    echo "------- Processing Original Paper -------"
# Set JAVA_HOME to the path of your JDK installation
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Update PATH to include the bin directory of JAVA_HOME
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java installation by checking the version
java -version

# Verify Java compiler installation by checking the version
javac -version

# Display the release information of the JDK
cat $JAVA_HOME/release
    
    # Check if GROBID is running
    echo "IMPORTANT: Make sure GROBID is running in another terminal with the command:"
    echo "cd $HOME/grobid-0.7.3 && ./gradlew run"
    if [ "$AUTOMATED_TEST_RUN" != "true" ]; then
        read -p "Press Enter when GROBID is running..."
    else
        echo "AUTOMATED_TEST_RUN is true, skipping 'Press Enter' prompt."
    fi
    
    # Process PDF to JSON
    cd "${PROJECT_ROOT}" # Change to the project root directory
    
    # Activate the virtual environment for s2orc-doc2json
    # This assumes the script is in PROJECT_ROOT/scripts/ and the venv is in PROJECT_ROOT/s2orc-doc2json/
    
    TARGET_PROJECT_ROOT_FOR_S2ORC_VENV=""
    if [ -n "$REAL_PROJECT_ROOT_FOR_VENV" ]; then
        echo "Using REAL_PROJECT_ROOT_FOR_VENV: $REAL_PROJECT_ROOT_FOR_VENV to find s2orc venv"
        TARGET_PROJECT_ROOT_FOR_S2ORC_VENV="$REAL_PROJECT_ROOT_FOR_VENV"
    else
        echo "WARNING: REAL_PROJECT_ROOT_FOR_VENV not set. Falling back to deriving path from script location."
        SCRIPT_DIR_FOR_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
        # BASE_PROJECT_ROOT_FOR_VENV is one level up from the SCRIPT_DIR_FOR_VENV (scripts directory)
        TARGET_PROJECT_ROOT_FOR_S2ORC_VENV="$(dirname "$SCRIPT_DIR_FOR_VENV")"
    fi

    VENV_PATH_S2ORC="${TARGET_PROJECT_ROOT_FOR_S2ORC_VENV}/s2orc-doc2json/venv_doc2json/bin/activate"

    echo "Attempting to activate venv from: $VENV_PATH_S2ORC"
    if [ -f "$VENV_PATH_S2ORC" ]; then
        echo "Activating venv: $VENV_PATH_S2ORC"
        source "$VENV_PATH_S2ORC"
    else
        echo "ERROR: Virtual environment activation script not found at $VENV_PATH_S2ORC"
        echo "Please ensure s2orc-doc2json is correctly placed relative to the project root (${TARGET_PROJECT_ROOT_FOR_S2ORC_VENV}) and its venv (venv_doc2json) is created."
        exit 1
    fi
    
    # process_pdf.py creates its output file (e.g., paper.pdf.json) in the directory specified by -o
    # and names it based on the input file. We then expect PAPER_JSON_PATH to match this.
    # Forcing output name via PAPER_JSON_PATH might be better if script allows, or rename after.
    # For now, assume process_pdf.py creates 'paper.json' if input is 'paper.pdf'
    python s2orc-doc2json/doc2json/grobid2json/process_pdf.py -i "$PAPER_PDF_PATH" -t "${PROCESSED_PAPER_DIR}/temp_grobid/" -o "${PROCESSED_PAPER_DIR}/"
    echo "s2orc-doc2json/doc2json/grobid2json/process_pdf.py exited with code $?"
    # If process_pdf.py creates paper.pdf.json, ensure it's moved to paper.json for consistency
    GROBID_OUTPUT_JSON="${PROCESSED_PAPER_DIR}/$(basename "${PAPER_PDF_PATH}").json"
    if [ -f "$GROBID_OUTPUT_JSON" ]; then
        echo "GROBID output found at $GROBID_OUTPUT_JSON. Moving to ${PAPER_JSON_PATH}"
        mv "$GROBID_OUTPUT_JSON" "${PAPER_JSON_PATH}"
    else
        echo "WARNING: Expected GROBID output $GROBID_OUTPUT_JSON not found."
    fi
    ls -l "${PAPER_JSON_PATH}" || echo "File ${PAPER_JSON_PATH} not found after process_pdf.py and potential mv."

    # Explicit check for PAPER_JSON_PATH after GROBID processing
    if [ ! -s "${PAPER_JSON_PATH}" ]; then # -s checks if file exists and is not empty
        echo "ERROR: ${PAPER_JSON_PATH} does not exist or is empty after GROBID processing and renaming. Aborting."
        exit 1
    fi

    # Preprocess
    python codes/0_pdf_process.py \
        --input_json_path ${PAPER_JSON_PATH} \
        --output_json_path ${PAPER_JSON_CLEANED_PATH}
    echo "codes/0_pdf_process.py exited with code $?"
    ls -l "${PAPER_JSON_CLEANED_PATH}" || echo "File ${PAPER_JSON_CLEANED_PATH} not found after 0_pdf_process.py"
    
    # Extract figures if not already done
    if [ ! -f "$PAPER_ENHANCED_JSON_PATH" ]; then
        echo "------- Extracting Figures and Getting LLM Descriptions -------"
        # Consider moving pip install to a setup script or checking if already installed
        # pip install PyMuPDF
        
        # Run the figure extraction script
        # extract_figures.py is expected to create/update the json file (specified by --json_path)
        # and save it as enhanced_paper.json in --output_dir.
        python codes/extract_figures.py \
            --pdf_path "$PAPER_PDF_PATH" \
            --json_path ${PAPER_JSON_CLEANED_PATH} \
            --output_dir "${PROCESSED_PAPER_DIR}" \
            --gpt_version ${IMAGE_GPT_VERSION}
        echo "codes/extract_figures.py exited with code $?"
        ls -l "${PAPER_ENHANCED_JSON_PATH}" || echo "File ${PAPER_ENHANCED_JSON_PATH} not found after extract_figures.py"
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
    # pip install pandas # Dependencies should be handled at environment setup
    
    python "${PROJECT_ROOT}/codes/adapt_mapping.py" \
        --paper_name "$PAPER_NAME" \
        --adapted_name "$ADAPTED_NAME" \
        --gpt_version ${GPT_VERSION} \
        --pdf_json_path ${PAPER_JSON_CLEANED_PATH} \
        --dataset_path ${ADAPTED_DATA_FILE} \
        --output_mapping_path ${ADAPTED_MAPPING_FILE} \
        --output_dir ${OUTPUT_DIR}
    echo "codes/adapt_mapping.py exited with code $?"
    
    echo "DEBUG: Verifying existence and content of mapping file: ${ADAPTED_MAPPING_FILE}"
    echo "DEBUG: Running 'ls -l ${ADAPTED_MAPPING_FILE}'"
    ls -l "${ADAPTED_MAPPING_FILE}"
    if [ $? -ne 0 ]; then
        echo "DEBUG_LS_FAILED: 'ls -l' command failed for ${ADAPTED_MAPPING_FILE}. This indicates the file likely does not exist at this point."
        # This implies adapt_mapping.py might not have created it, or it's gone.
    else
        echo "DEBUG: Running 'wc -c ${ADAPTED_MAPPING_FILE}' (byte count)"
        wc -c "${ADAPTED_MAPPING_FILE}"
        echo "DEBUG: Running 'head -n 3 ${ADAPTED_MAPPING_FILE}' (content sample)"
        head -n 3 "${ADAPTED_MAPPING_FILE}"
        echo "DEBUG: --- End of content sample ---"
    fi

    # Explicitly check if adapt_mapping.py actually created the file and it's not empty
    if [ ! -s "${ADAPTED_MAPPING_FILE}" ]; then
        echo "CRITICAL_ERROR_FINAL_CHECK: adapt_mapping.py seems to have completed (exit code $?), but the mapping file (${ADAPTED_MAPPING_FILE}) is NOT FOUND or IS EMPTY right before exiting block. Aborting script!"
        exit 1
    fi
    
    echo "Variable mapping template CREATED and VERIFIED (by shell) at $ADAPTED_MAPPING_FILE"
    echo "Please review and edit this file to match your data structure before continuing."
    echo "Once reviewed, run this script again to continue with adaptation."
    echo "Exiting with 0 because mapping file generation stage is complete for this test."
    exit 0
fi

# If the script reaches here, it means $ADAPTED_MAPPING_FILE was assumed to exist from the start
# or the above block was not entered for some reason.
echo "Using existing mapping file at $ADAPTED_MAPPING_FILE" # This line should NOT print in the current test scenario

# Skip the review prompt for testing
# read -p "Have you reviewed and edited the variable mapping file? (y/n): " reviewed
# if [ "$reviewed" != "y" ]; then
#     echo "Please review the mapping file at $ADAPTED_MAPPING_FILE before continuing."
#     echo "Run this script again after reviewing the mapping."
#     exit 0
# fi
echo "Using existing mapping file at $ADAPTED_MAPPING_FILE"

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