from typing import Dict, Any, Optional
import os
import logging
import time

from app.extensions import celery, db
from app.models.projects import Project, ProjectStatus


@celery.task(bind=True, name='process_paper')
def process_paper(self, project_id: int) -> Dict[str, Any]:
    """
    Celery task to process a paper and generate code repository.
    
    Args:
        project_id: ID of the project to process
        
    Returns:
        Dictionary with task results
    """
    # Create Flask app context
    from flask import current_app
    with current_app.app_context():
        # Import here to avoid circular imports
        from app.services.pipeline.pipeline_service import PipelineService
        
        try:
            # Get the project
            project = Project.query.get(project_id)
            if not project:
                return {'success': False, 'error': 'Project not found'}
            
            # Setup project paths
            paths = PipelineService.setup_project_paths(project)
            
            # Run the pipeline steps
            
            # Step 1: PDF Processing
            self.update_state(state='PROCESSING', meta={'progress': 10, 'step': 'PDF processing'})
            if not PipelineService.run_pdf_process(project, paths):
                return {'success': False, 'error': 'PDF processing failed'}
            
            # Extract paper metadata
            PipelineService.extract_paper_metadata(project)
            
            # Step 2: Planning
            self.update_state(state='PLANNING', meta={'progress': 30, 'step': 'Planning'})
            if not PipelineService.run_planning(project, paths):
                return {'success': False, 'error': 'Planning failed'}
            
            # Step 3: Analyzing
            self.update_state(state='ANALYZING', meta={'progress': 60, 'step': 'Analyzing'})
            if not PipelineService.run_analyzing(project, paths):
                return {'success': False, 'error': 'Analyzing failed'}
            
            # Step 4: Coding
            self.update_state(state='CODING', meta={'progress': 80, 'step': 'Coding'})
            if not PipelineService.run_coding(project, paths):
                return {'success': False, 'error': 'Coding failed'}
            
            # Update project status to completed
            project.update_status(ProjectStatus.COMPLETED)
            project.current_task = 'Processing completed'
            project.progress = 100
            db.session.commit()
            
            return {
                'success': True,
                'project_id': project_id,
                'output_path': project.output_path
            }
            
        except Exception as e:
            logging.error(f"Error processing paper: {str(e)}")
            
            # Update project status to failed
            try:
                project = Project.query.get(project_id)
                if project:
                    project.update_status(ProjectStatus.FAILED, str(e))
            except:
                pass
                
            return {'success': False, 'error': str(e)}


@celery.task(bind=True, name='process_paper_step')
def process_paper_step(self, project_id: int, step: str) -> Dict[str, Any]:
    """
    Celery task to process a specific step of the paper-to-code pipeline.
    
    Args:
        project_id: ID of the project to process
        step: The step to process ('pdf_process', 'planning', 'analyzing', 'coding')
        
    Returns:
        Dictionary with task results
    """
    # Create Flask app context
    from flask import current_app
    with current_app.app_context():
        # Import here to avoid circular imports
        from app.services.pipeline.pipeline_service import PipelineService
        
        try:
            # Get the project
            project = Project.query.get(project_id)
            if not project:
                return {'success': False, 'error': 'Project not found'}
            
            # Setup project paths
            paths = PipelineService.setup_project_paths(project)
            
            # Run the requested step
            if step == 'pdf_process':
                self.update_state(state='PROCESSING', meta={'progress': 10, 'step': 'PDF processing'})
                if not PipelineService.run_pdf_process(project, paths):
                    return {'success': False, 'error': 'PDF processing failed'}
                
                # Extract paper metadata
                PipelineService.extract_paper_metadata(project)
                
            elif step == 'planning':
                self.update_state(state='PLANNING', meta={'progress': 30, 'step': 'Planning'})
                if not PipelineService.run_planning(project, paths):
                    return {'success': False, 'error': 'Planning failed'}
                
            elif step == 'analyzing':
                self.update_state(state='ANALYZING', meta={'progress': 60, 'step': 'Analyzing'})
                if not PipelineService.run_analyzing(project, paths):
                    return {'success': False, 'error': 'Analyzing failed'}
                
            elif step == 'coding':
                self.update_state(state='CODING', meta={'progress': 80, 'step': 'Coding'})
                if not PipelineService.run_coding(project, paths):
                    return {'success': False, 'error': 'Coding failed'}
                
                # Update project status to completed if this is the last step
                project.update_status(ProjectStatus.COMPLETED)
                project.current_task = 'Processing completed'
                project.progress = 100
                db.session.commit()
                
            else:
                return {'success': False, 'error': f'Unknown step: {step}'}
            
            return {
                'success': True,
                'project_id': project_id,
                'step': step
            }
            
        except Exception as e:
            logging.error(f"Error processing step {step}: {str(e)}")
            
            # Update project status to failed
            try:
                project = Project.query.get(project_id)
                if project:
                    project.update_status(ProjectStatus.FAILED, str(e))
            except:
                pass
                
            return {'success': False, 'error': str(e)}