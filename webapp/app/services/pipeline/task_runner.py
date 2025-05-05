import os
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple

from flask import current_app
from app.models.projects import Project, ProjectStatus
from app.extensions import db


class TaskRunner:
    """
    Helper class to run external Python scripts and track their execution.
    
    This class provides methods to run the various steps of the Paper2Code
    pipeline as separate processes and capture their outputs.
    """
    
    @staticmethod
    def run_script(
        script_path: str, 
        args: List[str], 
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Run a Python script with the given arguments.
        
        Args:
            script_path: Path to the script to run
            args: List of command-line arguments to pass to the script
            env: Optional environment variables dictionary
            cwd: Optional working directory
            
        Returns:
            Tuple containing (success_flag, stdout, stderr)
        """
        try:
            # Prepare the command
            command = ['python', script_path] + args
            
            # Prepare environment variables
            if env is None:
                env = os.environ.copy()
                
            # Ensure OPENAI_API_KEY is available
            if 'OPENAI_API_KEY' not in env:
                api_key = current_app.config.get('OPENAI_API_KEY', '')
                if api_key:
                    env['OPENAI_API_KEY'] = api_key
            
            # Run the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd
            )
            
            # Check for errors
            if result.returncode != 0:
                return False, result.stdout, result.stderr
                
            return True, result.stdout, result.stderr
            
        except Exception as e:
            logging.error(f"Error running script {script_path}: {str(e)}")
            return False, "", str(e)
    
    @staticmethod
    def update_project_status(project_id: int, status: ProjectStatus, 
                              progress: int, current_task: str, 
                              error_message: Optional[str] = None) -> bool:
        """
        Update the status of a project in the database.
        
        Args:
            project_id: ID of the project to update
            status: New status to set
            progress: Progress percentage (0-100)
            current_task: Description of the current task
            error_message: Optional error message in case of failure
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            project = Project.query.get(project_id)
            if not project:
                return False
                
            project.status = status
            project.progress = progress
            project.current_task = current_task
            
            if error_message:
                project.error_message = error_message
                
            db.session.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error updating project status: {str(e)}")
            return False