"""Service for managing the Paper2Code processing pipeline."""

import uuid
import subprocess
import threading
import os
from typing import Dict, Any, Optional
from flask import current_app

from app.extensions import db
from app.models.projects import Project, ProjectStatus
from app.services.usage_service import UsageService


class PipelineService:
    """Service for managing the Paper2Code processing pipeline."""
    
    @staticmethod
    def start_pipeline(project_id: int) -> bool:
        """
        Start the Paper2Code processing pipeline for a project.
        
        Args:
            project_id: The ID of the project to process
            
        Returns:
            True if the pipeline was started successfully, False otherwise
        """
        try:
            # Get the project
            project = Project.query.get(project_id)
            
            if not project:
                current_app.logger.error(f"Project with ID {project_id} not found")
                return False
            
            # Update project status
            project.update_status(ProjectStatus.PROCESSING)
            
            # Run the pipeline in a separate thread
            thread = threading.Thread(
                target=PipelineService._run_pipeline,
                args=(project_id,)
            )
            thread.daemon = True
            thread.start()
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Error starting pipeline: {str(e)}")
            return False
    
    @staticmethod
    def _run_pipeline(project_id: int) -> None:
        """
        Run the Paper2Code processing pipeline for a project.
        
        Args:
            project_id: The ID of the project to process
        """
        try:
            project = Project.query.get(project_id)
            
            if not project:
                current_app.logger.error(f"Project with ID {project_id} not found")
                return
            
            # Create a unique request ID for tracking
            request_id = str(uuid.uuid4())
            
            # Get project data
            project_data = project.to_dict()
            paper_path = project.get_paper_path()
            
            # Setup environment variables
            env = os.environ.copy()
            env['PAPER_PATH'] = paper_path
            env['PROJECT_ID'] = str(project_id)
            env['PROJECT_TYPE'] = project.project_type.value
            
            # Set project language if specified
            if project.programming_language:
                env['PROJECT_LANGUAGE'] = project.programming_language
            
            # Get selected model or default
            selected_model = project.get_selected_model()
            if selected_model:
                env['MODEL_ID'] = selected_model.get('model_id', '')
                env['MODEL_PROVIDER'] = selected_model.get('provider', '')
            
            # Setup output directories
            output_dir = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project_id),
                'output'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Update project status
            project.update_status(ProjectStatus.PLANNING)
            
            # Phase 1: Planning
            planning_output = PipelineService._run_planning_stage(project, env, request_id)
            if not planning_output.get('success'):
                project.update_status(ProjectStatus.FAILED)
                project.add_error(f"Planning stage failed: {planning_output.get('error')}")
                return
            
            # Track usage for planning stage
            PipelineService._track_stage_usage(
                user_id=project.user_id,
                model_id=env.get('MODEL_ID', 'default'),
                input_tokens=planning_output.get('input_tokens', 0),
                output_tokens=planning_output.get('output_tokens', 0),
                endpoint='pipeline/planning',
                task_type='paper_processing',
                request_id=request_id
            )
            
            # Update project status
            project.update_status(ProjectStatus.ANALYZING)
            
            # Phase 2: Analyzing
            analyzing_output = PipelineService._run_analyzing_stage(project, env, request_id)
            if not analyzing_output.get('success'):
                project.update_status(ProjectStatus.FAILED)
                project.add_error(f"Analyzing stage failed: {analyzing_output.get('error')}")
                return
            
            # Track usage for analyzing stage
            PipelineService._track_stage_usage(
                user_id=project.user_id,
                model_id=env.get('MODEL_ID', 'default'),
                input_tokens=analyzing_output.get('input_tokens', 0),
                output_tokens=analyzing_output.get('output_tokens', 0),
                endpoint='pipeline/analyzing',
                task_type='paper_processing',
                request_id=request_id
            )
            
            # Update project status
            project.update_status(ProjectStatus.CODING)
            
            # Phase 3: Coding
            coding_output = PipelineService._run_coding_stage(project, env, request_id)
            if not coding_output.get('success'):
                project.update_status(ProjectStatus.FAILED)
                project.add_error(f"Coding stage failed: {coding_output.get('error')}")
                return
            
            # Track usage for coding stage
            PipelineService._track_stage_usage(
                user_id=project.user_id,
                model_id=env.get('MODEL_ID', 'default'),
                input_tokens=coding_output.get('input_tokens', 0),
                output_tokens=coding_output.get('output_tokens', 0),
                endpoint='pipeline/coding',
                task_type='paper_processing',
                request_id=request_id
            )
            
            # Set project as completed
            project.update_status(ProjectStatus.COMPLETED)
            
            # Generate download artifacts
            PipelineService._generate_artifacts(project)
            
        except Exception as e:
            current_app.logger.error(f"Error in pipeline: {str(e)}")
            try:
                project = Project.query.get(project_id)
                if project:
                    project.update_status(ProjectStatus.FAILED)
                    project.add_error(f"Pipeline error: {str(e)}")
            except:
                pass
    
    @staticmethod
    def _run_planning_stage(project: Project, env: Dict[str, str], request_id: str) -> Dict[str, Any]:
        """Run the planning stage of the pipeline."""
        try:
            # Set output paths
            plan_output_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'plan.json'
            )
            
            # Path to script
            script_path = os.path.join(
                current_app.config['PROJECT_ROOT'],
                'scripts',
                'run_llm.sh'
            )
            
            # Command to run
            command = [
                'bash',
                script_path,
                '1',  # Stage 1: Planning
                plan_output_path
            ]
            
            # Run command
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
            
            # Check if command was successful
            if proc.returncode != 0:
                current_app.logger.error(f"Planning stage failed: {stderr.decode()}")
                return {
                    'success': False,
                    'error': stderr.decode()
                }
            
            # Parse output to get token usage
            output = stdout.decode()
            input_tokens = PipelineService._extract_token_count(output, 'input')
            output_tokens = PipelineService._extract_token_count(output, 'output')
            
            return {
                'success': True,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
        except Exception as e:
            current_app.logger.error(f"Error in planning stage: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _run_analyzing_stage(project: Project, env: Dict[str, str], request_id: str) -> Dict[str, Any]:
        """Run the analyzing stage of the pipeline."""
        try:
            # Set output paths
            analysis_output_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'analysis.json'
            )
            
            # Path to script
            script_path = os.path.join(
                current_app.config['PROJECT_ROOT'],
                'scripts',
                'run_llm.sh'
            )
            
            # Command to run
            command = [
                'bash',
                script_path,
                '2',  # Stage 2: Analyzing
                analysis_output_path
            ]
            
            # Run command
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
            
            # Check if command was successful
            if proc.returncode != 0:
                current_app.logger.error(f"Analyzing stage failed: {stderr.decode()}")
                return {
                    'success': False,
                    'error': stderr.decode()
                }
            
            # Parse output to get token usage
            output = stdout.decode()
            input_tokens = PipelineService._extract_token_count(output, 'input')
            output_tokens = PipelineService._extract_token_count(output, 'output')
            
            return {
                'success': True,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
        except Exception as e:
            current_app.logger.error(f"Error in analyzing stage: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _run_coding_stage(project: Project, env: Dict[str, str], request_id: str) -> Dict[str, Any]:
        """Run the coding stage of the pipeline."""
        try:
            # Set output paths
            code_output_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'code'
            )
            
            # Create code directory if it doesn't exist
            os.makedirs(code_output_path, exist_ok=True)
            
            # Path to script
            script_path = os.path.join(
                current_app.config['PROJECT_ROOT'],
                'scripts',
                'run_llm.sh'
            )
            
            # Command to run
            command = [
                'bash',
                script_path,
                '3',  # Stage 3: Coding
                code_output_path
            ]
            
            # Run command
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
            
            # Check if command was successful
            if proc.returncode != 0:
                current_app.logger.error(f"Coding stage failed: {stderr.decode()}")
                return {
                    'success': False,
                    'error': stderr.decode()
                }
            
            # Parse output to get token usage
            output = stdout.decode()
            input_tokens = PipelineService._extract_token_count(output, 'input')
            output_tokens = PipelineService._extract_token_count(output, 'output')
            
            return {
                'success': True,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
        except Exception as e:
            current_app.logger.error(f"Error in coding stage: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _generate_artifacts(project: Project) -> None:
        """Generate downloadable artifacts for a completed project."""
        try:
            # Create a zip file of the code
            code_dir = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'code'
            )
            
            zip_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'code.zip'
            )
            
            # Only create zip if code directory exists and has files
            if os.path.exists(code_dir) and os.listdir(code_dir):
                command = ['zip', '-r', zip_path, '.']
                subprocess.run(
                    command, 
                    cwd=code_dir, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Add artifact to project if zip was created
                if os.path.exists(zip_path):
                    project.add_artifact('code.zip', zip_path, 'Code Implementation')
            
            # Save the plan and analysis as artifacts
            plan_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'plan.json'
            )
            
            analysis_path = os.path.join(
                current_app.config['PROJECT_DATA_DIR'],
                str(project.id),
                'output',
                'analysis.json'
            )
            
            if os.path.exists(plan_path):
                project.add_artifact('plan.json', plan_path, 'Implementation Plan')
                
            if os.path.exists(analysis_path):
                project.add_artifact('analysis.json', analysis_path, 'Paper Analysis')
                
            db.session.commit()
            
        except Exception as e:
            current_app.logger.error(f"Error generating artifacts: {str(e)}")
    
    @staticmethod
    def _extract_token_count(output: str, token_type: str) -> int:
        """Extract token count from command output."""
        try:
            # Look for lines like "Input tokens: 1234" or "Output tokens: 5678"
            lines = output.split('\n')
            for line in lines:
                if f"{token_type} tokens:" in line.lower():
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            return int(parts[1].strip())
                        except ValueError:
                            pass
            return 0
        except Exception:
            return 0
    
    @staticmethod
    def _track_stage_usage(
        user_id: int, 
        model_id: str, 
        input_tokens: int, 
        output_tokens: int,
        endpoint: str,
        task_type: str,
        request_id: str
    ) -> None:
        """Track API usage for a pipeline stage."""
        try:
            UsageService.track_api_usage(
                user_id=user_id,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                endpoint=endpoint,
                task_type=task_type,
                request_id=request_id
            )
        except Exception as e:
            current_app.logger.error(f"Error tracking API usage: {str(e)}")