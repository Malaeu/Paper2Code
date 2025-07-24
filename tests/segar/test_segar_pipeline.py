import unittest
import subprocess
import os
import tempfile
import shutil
import re
import stat

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'run_custom_adapt.sh')
DUMMY_PAPER_PATH = os.path.join(PROJECT_ROOT, 'tests', 'segar', 'test_data', 'dummy_paper.pdf')
DUMMY_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, 'tests', 'segar', 'test_data', 'dummy_data.csv')

class TestSegarPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a unique temporary directory for this test run
        cls.test_dir = tempfile.mkdtemp(prefix='paper2code_test_segar_')
        
        cls.test_example_dir = os.path.join(cls.test_dir, 'test_example_segar')
        cls.test_output_base_dir = os.path.join(cls.test_dir, 'test_outputs_segar')
        cls.test_processed_paper_dir = os.path.join(cls.test_example_dir, 'processed_paper_files')

        os.makedirs(cls.test_example_dir, exist_ok=True)
        os.makedirs(cls.test_output_base_dir, exist_ok=True)
        os.makedirs(cls.test_processed_paper_dir, exist_ok=True)

        # Define paths for files that might be created by the script
        cls.test_paper_cleaned_json_path = os.path.join(cls.test_processed_paper_dir, 'paper_cleaned.json')
        cls.test_enhanced_paper_json_path = os.path.join(cls.test_processed_paper_dir, 'enhanced_paper.json')
        cls.test_mapping_json_path = os.path.join(cls.test_example_dir, 'mapping.json')

        # Clean up potentially existing files from previous runs to ensure regeneration
        files_to_remove = [
            cls.test_paper_cleaned_json_path,
            cls.test_enhanced_paper_json_path,
            cls.test_mapping_json_path,
            os.path.join(cls.test_processed_paper_dir, 'paper.json') # Grobid output
        ]
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)

        # --- Prepare test paper --- 
        cls.test_paper_path = os.path.join(cls.test_example_dir, 'paper.pdf') # Keep the target name as paper.pdf for the script
        original_segar_pdf_path = os.path.join(PROJECT_ROOT, 'examples', 'segar', 'segar_paper.pdf') # Correct source name
        
        if not os.path.exists(original_segar_pdf_path):
            raise FileNotFoundError(f"Original segar PDF not found at {original_segar_pdf_path}. Cannot run test.")
        shutil.copy(original_segar_pdf_path, cls.test_paper_path)

        # --- Prepare data.csv by copying the actual fv_export.csv --- 
        cls.test_data_path = os.path.join(cls.test_example_dir, 'data.csv')
        original_segar_csv_path = os.path.join(PROJECT_ROOT, 'examples', 'segar', 'fv_export.csv')

        if not os.path.exists(original_segar_csv_path):
            raise FileNotFoundError(f"Original segar CSV not found at {original_segar_csv_path}. Cannot run test.")
        
        shutil.copy(original_segar_csv_path, cls.test_data_path)

        # --- Prepare script --- 
        original_script_path = os.path.join(PROJECT_ROOT, 'scripts', 'run_custom_adapt.sh')
        # Read the original script content
        with open(original_script_path, 'r') as f_original_script:
            script_content = f_original_script.read()

        # Modify paths in the script content - ensure quoting for paths with spaces
        script_content = re.sub(r'^PROJECT_ROOT=.*', f'PROJECT_ROOT="{PROJECT_ROOT}"', script_content, flags=re.MULTILINE)
        script_content = re.sub(r'^EXAMPLE_DIR=.*', f'EXAMPLE_DIR="{cls.test_example_dir}"', script_content, flags=re.MULTILINE)
        script_content = re.sub(r'^PROCESSED_PAPER_DIR=.*', f'PROCESSED_PAPER_DIR="{cls.test_processed_paper_dir}"', script_content, flags=re.MULTILINE)
        script_content = re.sub(r'^OUTPUT_BASE_DIR=.*', f'OUTPUT_BASE_DIR="{cls.test_output_base_dir}"', script_content, flags=re.MULTILINE)
        
        # --- Venv Activation Handling for Test Script ---
        # The run_custom_adapt.sh now has a block like:
        # SCRIPT_DIR_FOR_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
        # BASE_PROJECT_ROOT_FOR_VENV="$(dirname "$SCRIPT_DIR_FOR_VENV")"
        # VENV_PATH_S2ORC="${BASE_PROJECT_ROOT_FOR_VENV}/s2orc-doc2json/venv_doc2json/bin/activate"
        # ...
        # source "$VENV_PATH_S2ORC"
        # We need to ensure that VENV_PATH_S2ORC resolves correctly in the test copy.
        # The most robust way is to replace the 'source "$VENV_PATH_S2ORC"' line
        # or the VENV_PATH_S2ORC definition line with an absolute path for the test.

        # Define the absolute path to the s2orc venv activate script based on PROJECT_ROOT
        s2orc_venv_activate_path = os.path.join(PROJECT_ROOT, 's2orc-doc2json', 'venv_doc2json', 'bin', 'activate')

        # Revert to Option 1: Replace the VENV_PATH_S2ORC definition line
        script_content = re.sub(r'^(VENV_PATH_S2ORC=).*', 
                                f'\1"{s2orc_venv_activate_path}" # Test-modified path',
                                script_content, flags=re.MULTILINE)
        
        # --- End of Venv Activation Handling ---

        # Comment out the 'read -p ""' line to prevent script from halting
        # This regex handles potential leading/trailing whitespace around the command
        script_content = re.sub(r'^\s*read -p ""\s*$', '# read -p "" # Commented out for automated test', script_content, flags=re.MULTILINE)

        cls.test_run_script_path = os.path.join(cls.test_dir, 'run_custom_adapt.sh')
        with open(cls.test_run_script_path, 'w') as test_script_file:
            test_script_file.write(script_content)
        os.chmod(cls.test_run_script_path, 0o755)

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            # Pass for now to inspect output, uncomment to clean up
            # shutil.rmtree(cls.test_dir)
            print(f"Test artifacts are in {cls.test_dir}")
            pass

    def test_pipeline_generates_mapping_file(self):
        """Test that the first part of the pipeline runs and generates mapping.json."""
        # This test assumes GROBID is not running or accessible, 
        # and OPENAI_API_KEY might not be set or valid for the dummy calls.
        # It primarily checks if the script starts, sets up, and attempts to create mapping.json.
        # The script is expected to exit with 0 after trying to create mapping.json if it's not found.

        print(f"Running script: {self.test_run_script_path} from CWD: {os.getcwd()}")
        print(f"Test directory: {self.test_dir}")
        print(f"Project root for script: {PROJECT_ROOT}")
        print(f"Test example dir for script: {self.test_example_dir}")
        print(f"Dummy paper for script: {os.path.join(self.test_example_dir, 'paper.pdf')}")
        print(f"Dummy data for script: {os.path.join(self.test_example_dir, 'data.csv')}")

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = env.get('OPENAI_API_KEY', 'dummy_key_for_testing_only') # Use existing or dummy
        env["REAL_PROJECT_ROOT_FOR_VENV"] = PROJECT_ROOT # Pass the real project root (using global constant)
        env["AUTOMATED_TEST_RUN"] = "true" # Indicate that this is an automated run

        process = subprocess.run(['bash', self.test_run_script_path, 'paper.pdf', 'data.csv'], 
                                 capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, timeout=600) # Increased timeout

        # ALWAYS print stdout and stderr for debugging this elusive issue
        if process:
            print("\n--- run_custom_adapt.sh STDOUT ---")
            print(process.stdout)
            print("--- END run_custom_adapt.sh STDOUT ---\n")
            print("--- run_custom_adapt.sh STDERR ---")
            print(process.stderr)
            print("--- END run_custom_adapt.sh STDERR ---\n")
        else:
            print("Process was None, did not run.")

        self.assertIsNotNone(process, "Subprocess did not run or was not assigned.")
        self.assertTrue(os.path.exists(self.test_mapping_json_path), f"mapping.json not found at {self.test_mapping_json_path}")
        self.assertEqual(process.returncode, 0, f"Script should exit successfully. Stderr:\n{process.stderr if process else 'N/A'}")

    def test_pipeline_full_run_generates_output_repo(self):
        """Test that the full pipeline runs and generates the output repository."""
        # TODO: Implement this test

if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is available if tests involving actual LLM calls are run.
    # For now, the first test tries to avoid them by focusing on mapping.json generation.
    # It's good practice to run tests from the project root directory.
    print(f"Running tests from: {os.getcwd()}")
    print(f"Expected project root: {PROJECT_ROOT}")
    unittest.main()
