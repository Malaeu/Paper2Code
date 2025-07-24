"""Service for handling project exports."""

import os
import zipfile
import tempfile
import shutil
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Set

from flask import current_app
from werkzeug.utils import secure_filename

from app.models.projects import Project, ProjectStatus
from app.extensions import db


class ProjectExportService:
    """Service for exporting projects as zip archives."""
    
    @staticmethod
    def export_project(project_id: int, include_logs: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Export a project as a zip archive.
        
        Args:
            project_id: ID of the project to export
            include_logs: Whether to include log files in export
            
        Returns:
            Tuple of (success, message, path to zip file or None if failed)
        """
        try:
            # Get the project
            project = Project.query.get(project_id)
            if not project:
                return False, f"Project with ID {project_id} not found", None
                
            # Check if project has output
            if not project.output_path or not os.path.exists(project.output_path):
                return False, "Project has no output files to export", None
                
            # Create a temporary directory for staging files
            temp_dir = tempfile.mkdtemp(prefix=f"export_project_{project_id}_")
            try:
                # Get project paths
                paths = project.get_project_paths()
                
                # Create output directory
                export_dir = os.path.join(temp_dir, f"paper2code_{project.name.replace(' ', '_')}")
                os.makedirs(export_dir, exist_ok=True)
                
                # Copy output directory
                if project.output_path and os.path.exists(project.output_path):
                    output_dir = os.path.join(export_dir, "generated_code")
                    shutil.copytree(project.output_path, output_dir)
                
                # Include paper file
                if paths.get('paper_path') and os.path.exists(paths.get('paper_path')):
                    paper_dir = os.path.join(export_dir, "paper")
                    os.makedirs(paper_dir, exist_ok=True)
                    shutil.copy2(paths.get('paper_path'), paper_dir)
                    
                # Include parsed data (cleaned JSON)
                if paths.get('pdf_json_cleaned_path') and os.path.exists(paths.get('pdf_json_cleaned_path')):
                    data_dir = os.path.join(export_dir, "processed_data")
                    os.makedirs(data_dir, exist_ok=True)
                    shutil.copy2(paths.get('pdf_json_cleaned_path'), data_dir)
                
                # Include planning output
                planning_dir = os.path.join(export_dir, "planning")
                if paths.get('output_dir') and os.path.exists(paths.get('output_dir')):
                    planning_output = os.path.join(paths.get('output_dir'), 'planning_output.json')
                    if os.path.exists(planning_output):
                        os.makedirs(planning_dir, exist_ok=True)
                        shutil.copy2(planning_output, planning_dir)
                        
                    # Include planning config
                    planning_config = os.path.join(paths.get('output_dir'), 'planning_config.yaml')
                    if os.path.exists(planning_config):
                        if not os.path.exists(planning_dir):
                            os.makedirs(planning_dir, exist_ok=True)
                        shutil.copy2(planning_config, planning_dir)
                
                # Include logs if requested
                if include_logs and project.log_path and os.path.exists(project.log_path):
                    logs_dir = os.path.join(export_dir, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    shutil.copy2(project.log_path, logs_dir)
                
                # Create a metadata file
                metadata = {
                    'project_name': project.name,
                    'project_description': project.description,
                    'project_type': project.project_type.value,
                    'paper_title': project.paper_title,
                    'paper_authors': project.paper_authors,
                    'paper_abstract': project.paper_abstract,
                    'programming_language': project.programming_language,
                    'export_date': datetime.utcnow().isoformat(),
                    'status': project.status.value,
                    'completion_date': project.completed_at.isoformat() if project.completed_at else None,
                }
                
                with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create the zip file
                export_folder = os.path.join(current_app.config.get('UPLOAD_FOLDER', current_app.root_path), 'exports')
                os.makedirs(export_folder, exist_ok=True)
                
                timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                safe_name = secure_filename(project.name.replace(' ', '_'))
                zip_filename = f"project_{project.id}_{safe_name}_{timestamp}.zip"
                zip_path = os.path.join(export_folder, zip_filename)
                
                # Create the zip with all files
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(export_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)
                
                # Update the project with the export path
                project.export_path = zip_path
                project.last_exported_at = datetime.utcnow()
                db.session.commit()
                
                return True, f"Project '{project.name}' exported successfully", zip_path
                
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logging.error(f"Error exporting project: {str(e)}")
            return False, f"Error exporting project: {str(e)}", None
    
    @staticmethod
    def get_export_filepath(project_id: int) -> Optional[str]:
        """
        Get the export file path for a project, if it exists.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Path to export file or None if not found
        """
        project = Project.query.get(project_id)
        if not project or not project.export_path:
            return None
            
        if os.path.exists(project.export_path):
            return project.export_path
            
        return None
        
    @staticmethod
    def check_export_size(project: Project) -> int:
        """
        Estimate the size of the project export in bytes.
        
        Args:
            project: The project to check
            
        Returns:
            Estimated size in bytes
        """
        total_size = 0
        
        # Check output directory size
        if project.output_path and os.path.exists(project.output_path):
            for dirpath, _, filenames in os.walk(project.output_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        
        # Add paper size
        paths = project.get_project_paths()
        if paths.get('paper_path') and os.path.exists(paths.get('paper_path')):
            total_size += os.path.getsize(paths.get('paper_path'))
            
        # Add processed data size
        if paths.get('pdf_json_cleaned_path') and os.path.exists(paths.get('pdf_json_cleaned_path')):
            total_size += os.path.getsize(paths.get('pdf_json_cleaned_path'))
            
        # Add planning output size
        if paths.get('output_dir') and os.path.exists(paths.get('output_dir')):
            planning_output = os.path.join(paths.get('output_dir'), 'planning_output.json')
            if os.path.exists(planning_output):
                total_size += os.path.getsize(planning_output)
                
            planning_config = os.path.join(paths.get('output_dir'), 'planning_config.yaml')
            if os.path.exists(planning_config):
                total_size += os.path.getsize(planning_config)
                
        # Add log size
        if project.log_path and os.path.exists(project.log_path):
            total_size += os.path.getsize(project.log_path)
            
        # Add a small buffer for metadata
        total_size += 5 * 1024  # 5KB for metadata
        
        return total_size
        
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        Format a size in bytes to a human-readable string.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable size string
        """
        if size_bytes == 0:
            return "0 B"
            
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
            
        return f"{size_bytes:.2f} {size_names[i]}"