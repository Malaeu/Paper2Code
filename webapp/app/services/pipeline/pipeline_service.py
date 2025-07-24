import os
import json
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple

from flask import current_app
from app.models.projects import Project, ProjectStatus
from app.extensions import db
from app.services.pipeline.celery_tasks import process_paper, process_paper_step


class PipelineService:
    """
    Service for handling Paper2Code pipeline operations.
    
    This service integrates with the existing Python code in the codes directory
    to process papers and generate code repositories.
    """
    
    @staticmethod
    def start_pipeline(project_id: int) -> bool:
        """
        Start the paper processing pipeline for a project.
        
        Args:
            project_id: ID of the project to process
            
        Returns:
            bool: True if the pipeline was started successfully, False otherwise
        """
        # Get the project
        project = Project.query.get(project_id)
        if not project:
            return False
            
        # Check if the project is already being processed
        if project.status not in [ProjectStatus.CREATED, ProjectStatus.FAILED]:
            return False
            
        # Update project status to PROCESSING
        project.update_status(ProjectStatus.PROCESSING)
        project.current_task = "Preparing pipeline"
        project.progress = 5
        db.session.commit()
        
        # Start the pipeline using Celery
        process_paper.delay(project_id)
        
        return True
    
    @staticmethod
    def setup_project_paths(project: Project) -> Dict[str, str]:
        """
        Set up the directories and paths for pipeline processing.
        
        Args:
            project: The project to set up paths for
            
        Returns:
            Dict containing paths for pipeline processing
        """
        base_path = current_app.config['UPLOAD_FOLDER']
        project_name = f"project_{project.id}"
        
        # Define paths
        paths = {
            'paper_path': os.path.join(base_path, 'papers', project.paper_path),
            'pdf_json_path': os.path.join(base_path, 'temp', f"{project_name}.json"),
            'pdf_json_cleaned_path': os.path.join(base_path, 'temp', f"{project_name}_cleaned.json"),
            'output_dir': os.path.join(base_path, 'outputs', project_name),
            'output_repo_dir': os.path.join(base_path, 'outputs', f"{project_name}_repo"),
            'log_path': os.path.join(current_app.root_path, 'logs', 'projects', f"{project_name}.log")
        }
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(paths['pdf_json_path']), exist_ok=True)
        os.makedirs(paths['output_dir'], exist_ok=True)
        os.makedirs(paths['output_repo_dir'], exist_ok=True)
        os.makedirs(os.path.dirname(paths['log_path']), exist_ok=True)
        
        # Update project with output and log paths
        project.output_path = paths['output_repo_dir']
        project.log_path = paths['log_path']
        db.session.commit()
        
        return paths
    
    @staticmethod
    def run_pdf_process(project: Project, paths: Dict[str, str]) -> bool:
        """
        Run the PDF preprocessing step.
        
        Args:
            project: The project being processed
            paths: Dictionary of paths for processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set up logging
            logging.basicConfig(
                filename=paths['log_path'],
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s'
            )
            
            # Process PDF using MinerU (replaces GROBID)
            logging.info("Processing PDF with MinerU...")
            
            mineru_processor_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', 'mineru_processor.py')
            mineru_output_dir = os.path.join(os.path.dirname(paths['pdf_json_path']), 'mineru_output')
            
            # Run MinerU processor
            mineru_command = [
                'python', mineru_processor_path,
                '--pdf_path', paths['paper_path'],
                '--output_dir', mineru_output_dir,
                '--json_output', paths['pdf_json_path']
            ]
            
            logging.info(f"Running MinerU command: {' '.join(mineru_command)}")
            
            mineru_result = subprocess.run(
                mineru_command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if mineru_result.returncode != 0:
                logging.error(f"MinerU processing failed: {mineru_result.stderr}")
                raise RuntimeError(f"MinerU processing failed: {mineru_result.stderr}")
            
            logging.info(f"MinerU processing completed: {mineru_result.stdout}")
            
            # Enhance with Gemini Vision if API key is available
            enhanced_json_path = paths['pdf_json_path'].replace('.json', '_enhanced.json')
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            
            if gemini_api_key:
                logging.info("Enhancing images with Gemini Vision...")
                
                image_enhancer_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', 'mineru_image_enhancer.py')
                
                enhance_command = [
                    'python', image_enhancer_path,
                    '--input', paths['pdf_json_path'],
                    '--images_dir', mineru_output_dir,
                    '--output', enhanced_json_path,
                    '--format', 'paper2code'
                ]
                
                enhance_result = subprocess.run(
                    enhance_command,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if enhance_result.returncode == 0 and os.path.exists(enhanced_json_path):
                    logging.info("Gemini Vision enhancement completed successfully")
                    # Use enhanced JSON for further processing
                    paths['pdf_json_path'] = enhanced_json_path
                else:
                    logging.warning(f"Gemini Vision enhancement failed: {enhance_result.stderr}")
            else:
                logging.info("GEMINI_API_KEY not set, skipping image enhancement")
            
            # Now run the 0_pdf_process.py script
            script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', '0_pdf_process.py')
            
            command = [
                'python', script_path,
                '--input_json_path', paths['pdf_json_path'],
                '--output_json_path', paths['pdf_json_cleaned_path']
            ]
            
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info(f"PDF preprocessing completed: {result.stdout}")
            
            # Update project status
            project.progress = 15
            project.current_task = "PDF preprocessing completed"
            db.session.commit()
            
            return True
            
        except Exception as e:
            logging.error(f"Error in PDF preprocessing: {str(e)}")
            project.update_status(ProjectStatus.FAILED, f"PDF preprocessing failed: {str(e)}")
            return False
    
    @staticmethod
    def run_planning(project: Project, paths: Dict[str, str], gpt_version: str = "o3-mini") -> bool:
        """
        Run the planning step of the pipeline.
        
        Args:
            project: The project being processed
            paths: Dictionary of paths for processing
            gpt_version: The GPT model version to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update project status
            project.current_task = "Planning project structure"
            project.status = ProjectStatus.PLANNING
            project.progress = 25
            db.session.commit()
            
            # Set up logging
            logging.basicConfig(
                filename=paths['log_path'],
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s',
                force=True
            )
            
            # Run the planning script
            script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', '1_planning.py')
            
            command = [
                'python', script_path,
                '--paper_name', f"project_{project.id}",
                '--gpt_version', gpt_version,
                '--pdf_json_path', paths['pdf_json_cleaned_path'],
                '--output_dir', paths['output_dir']
            ]
            
            logging.info(f"Running planning command: {' '.join(command)}")
            
            # Set OPENAI_API_KEY in the environment
            env = os.environ.copy()
            if 'OPENAI_API_KEY' not in env:
                # Try to get from current_app.config or use a placeholder
                env['OPENAI_API_KEY'] = current_app.config.get('OPENAI_API_KEY', 'dummy_key')
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            logging.info(f"Planning completed: {result.stdout}")
            
            # Run extract_config script
            extract_config_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', '1.1_extract_config.py')
            
            command = [
                'python', extract_config_path,
                '--paper_name', f"project_{project.id}",
                '--output_dir', paths['output_dir']
            ]
            
            logging.info(f"Running extract config command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            logging.info(f"Config extraction completed: {result.stdout}")
            
            # Copy config.yaml to the output repo directory
            src_config = os.path.join(paths['output_dir'], 'planning_config.yaml')
            dst_config = os.path.join(paths['output_repo_dir'], 'config.yaml')
            
            if os.path.exists(src_config):
                shutil.copy(src_config, dst_config)
                logging.info(f"Copied config from {src_config} to {dst_config}")
            
            # Update project status
            project.progress = 40
            project.current_task = "Planning completed"
            db.session.commit()
            
            return True
            
        except Exception as e:
            logging.error(f"Error in planning: {str(e)}")
            project.update_status(ProjectStatus.FAILED, f"Planning failed: {str(e)}")
            return False
    
    @staticmethod
    def run_analyzing(project: Project, paths: Dict[str, str], gpt_version: str = "o3-mini") -> bool:
        """
        Run the analyzing step of the pipeline.
        
        Args:
            project: The project being processed
            paths: Dictionary of paths for processing
            gpt_version: The GPT model version to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update project status
            project.current_task = "Analyzing paper contents"
            project.status = ProjectStatus.ANALYZING
            project.progress = 55
            db.session.commit()
            
            # Set up logging
            logging.basicConfig(
                filename=paths['log_path'],
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s',
                force=True
            )
            
            # Run the analyzing script
            script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', '2_analyzing.py')
            
            command = [
                'python', script_path,
                '--paper_name', f"project_{project.id}",
                '--gpt_version', gpt_version,
                '--pdf_json_path', paths['pdf_json_cleaned_path'],
                '--output_dir', paths['output_dir']
            ]
            
            logging.info(f"Running analyzing command: {' '.join(command)}")
            
            # Set OPENAI_API_KEY in the environment
            env = os.environ.copy()
            if 'OPENAI_API_KEY' not in env:
                # Try to get from current_app.config or use a placeholder
                env['OPENAI_API_KEY'] = current_app.config.get('OPENAI_API_KEY', 'dummy_key')
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            logging.info(f"Analyzing completed: {result.stdout}")
            
            # Update project status
            project.progress = 70
            project.current_task = "Analysis completed"
            db.session.commit()
            
            return True
            
        except Exception as e:
            logging.error(f"Error in analyzing: {str(e)}")
            project.update_status(ProjectStatus.FAILED, f"Analyzing failed: {str(e)}")
            return False
    
    @staticmethod
    def run_coding(project: Project, paths: Dict[str, str], gpt_version: str = "o3-mini") -> bool:
        """
        Run the coding step of the pipeline.
        
        Args:
            project: The project being processed
            paths: Dictionary of paths for processing
            gpt_version: The GPT model version to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update project status
            project.current_task = "Generating code implementation"
            project.status = ProjectStatus.CODING
            project.progress = 80
            db.session.commit()
            
            # Set up logging
            logging.basicConfig(
                filename=paths['log_path'],
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s',
                force=True
            )
            
            # Run the coding script
            script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'codes', '3_coding.py')
            
            command = [
                'python', script_path,
                '--paper_name', f"project_{project.id}",
                '--gpt_version', gpt_version,
                '--pdf_json_path', paths['pdf_json_cleaned_path'],
                '--output_dir', paths['output_dir'],
                '--output_repo_dir', paths['output_repo_dir']
            ]
            
            logging.info(f"Running coding command: {' '.join(command)}")
            
            # Set OPENAI_API_KEY in the environment
            env = os.environ.copy()
            if 'OPENAI_API_KEY' not in env:
                # Try to get from current_app.config or use a placeholder
                env['OPENAI_API_KEY'] = current_app.config.get('OPENAI_API_KEY', 'dummy_key')
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            logging.info(f"Coding completed: {result.stdout}")
            
            # Update project status
            project.progress = 100
            project.update_status(ProjectStatus.COMPLETED)
            project.current_task = "Pipeline completed successfully"
            db.session.commit()
            
            return True
            
        except Exception as e:
            logging.error(f"Error in coding: {str(e)}")
            project.update_status(ProjectStatus.FAILED, f"Coding failed: {str(e)}")
            return False
    
    @staticmethod
    def extract_paper_metadata(project: Project) -> bool:
        """
        Extract metadata from the processed paper JSON.
        
        Args:
            project: The project to extract metadata for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find the paper JSON file
            base_path = current_app.config['UPLOAD_FOLDER']
            project_name = f"project_{project.id}"
            pdf_json_path = os.path.join(base_path, 'temp', f"{project_name}.json")
            
            if not os.path.exists(pdf_json_path):
                return False
                
            # Read the JSON file
            with open(pdf_json_path, 'r') as f:
                data = json.load(f)
                
            # Extract metadata
            if 'metadata' in data:
                metadata = data['metadata']
                
                if 'title' in metadata:
                    project.paper_title = metadata['title']
                    
                if 'authors' in metadata:
                    author_names = [author.get('name', '') for author in metadata.get('authors', [])]
                    project.paper_authors = ', '.join(author_names)
                    
            # Extract abstract
            if 'abstract' in data:
                abstract_text = ' '.join([section.get('text', '') for section in data.get('abstract', [])])
                project.paper_abstract = abstract_text
                
            db.session.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error extracting paper metadata: {str(e)}")
            return False