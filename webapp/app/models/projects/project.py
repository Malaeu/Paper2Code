import datetime
import enum
import os
import uuid
from flask import current_app
from werkzeug.utils import secure_filename

from app.extensions import db


class ProjectStatus(enum.Enum):
    """Enum for project statuses."""
    CREATED = 'created'
    PROCESSING = 'processing'
    PLANNING = 'planning'
    ANALYZING = 'analyzing'
    CODING = 'coding'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    
    @property
    def color(self):
        """Return Bootstrap color class for the status."""
        colors = {
            'created': 'secondary',
            'processing': 'info',
            'planning': 'info',
            'analyzing': 'primary',
            'coding': 'warning',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'dark'
        }
        return colors.get(self.value, 'secondary')


class ProjectType(enum.Enum):
    """Enum for project types."""
    CODE_GENERATION = 'code_generation'
    METHODOLOGY_ADAPTATION = 'methodology_adaptation'
    DATASET_ANALYSIS = 'dataset_analysis'
    
    @property
    def display_name(self):
        """Return human-readable display name."""
        names = {
            'code_generation': 'Code Generation',
            'methodology_adaptation': 'Methodology Adaptation',
            'dataset_analysis': 'Dataset Analysis'
        }
        return names.get(self.value, self.value)


class Project(db.Model):
    """Project model for Paper2Code projects."""
    
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # User relationship
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('projects', lazy=True))
    
    # Paper information
    paper_path = db.Column(db.String(255), nullable=True)
    paper_title = db.Column(db.String(255), nullable=True)
    paper_authors = db.Column(db.String(500), nullable=True)
    paper_abstract = db.Column(db.Text, nullable=True)
    
    # Project metadata
    project_type = db.Column(db.Enum(ProjectType), default=ProjectType.CODE_GENERATION, nullable=False)
    status = db.Column(db.Enum(ProjectStatus), default=ProjectStatus.CREATED, nullable=False)
    programming_language = db.Column(db.String(50), default='python', nullable=True)
    
    # Dataset path for methodology adaptation
    dataset_path = db.Column(db.String(255), nullable=True)
    
    # Output information
    output_path = db.Column(db.String(255), nullable=True)
    repository_url = db.Column(db.String(255), nullable=True)
    export_path = db.Column(db.String(255), nullable=True)
    last_exported_at = db.Column(db.DateTime, nullable=True)
    
    # Log and error tracking
    log_path = db.Column(db.String(255), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Task tracking
    current_task = db.Column(db.String(100), nullable=True)
    progress = db.Column(db.Integer, default=0, nullable=False)  # 0-100%
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, 
                           onupdate=datetime.datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    def __init__(self, name, user_id, description=None, project_type=ProjectType.CODE_GENERATION):
        self.name = name
        self.user_id = user_id
        self.description = description
        self.project_type = project_type
        self.status = ProjectStatus.CREATED
        
    def get_paper_filename(self):
        """Return the filename for the paper."""
        if not self.paper_path:
            return None
        return os.path.basename(self.paper_path)
    
    def get_paper_size(self):
        """
        Return the size of the paper file in human-readable format.
        Uses configured directories from DirectoryService.
        """
        if not self.paper_path:
            return None
            
        try:
            # Import here to avoid circular imports
            from app.services.directory_service import DirectoryService
            
            # Get configured upload directory or fall back to default
            upload_dir = DirectoryService.get_directory_path_by_name(
                'uploads', 
                default_path=current_app.config.get('UPLOAD_FOLDER', os.path.join(current_app.root_path, 'uploads'))
            )
            
            path = os.path.join(upload_dir, 'papers', self.paper_path)
            size_bytes = os.path.getsize(path)
            
            # Convert to human-readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024 or unit == 'GB':
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024
        except FileNotFoundError:
            return None
    
    def get_status_color(self):
        """Return the Bootstrap color class for the current status."""
        return self.status.color
    
    def save_paper(self, file):
        """
        Save the uploaded paper file and update project metadata.
        Uses configured directories from DirectoryService.
        """
        if file and file.filename:
            # Import here to avoid circular imports
            from app.services.directory_service import DirectoryService
            
            # Create secure filename
            filename = secure_filename(file.filename)
            # Add unique identifier to prevent filename collisions
            unique_filename = f"{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self.uuid}_{filename}"
            
            # Get configured upload directory or fall back to default
            upload_dir = DirectoryService.get_directory_path_by_name(
                'uploads', 
                default_path=current_app.config.get('UPLOAD_FOLDER', os.path.join(current_app.root_path, 'uploads'))
            )
            
            # Make sure the upload directory exists
            upload_path = os.path.join(upload_dir, 'papers')
            os.makedirs(upload_path, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(upload_path, unique_filename)
            file.save(file_path)
            
            # Update the project
            self.paper_path = unique_filename
            self.paper_title = filename  # Will be updated after processing
            
            return True
        return False
    
    def update_status(self, status, error_message=None):
        """Update project status and related fields."""
        self.status = status
        
        if error_message:
            self.error_message = error_message
            
        if status == ProjectStatus.COMPLETED:
            self.completed_at = datetime.datetime.utcnow()
            self.progress = 100
            
        db.session.commit()
    
    def get_project_paths(self):
        """
        Get all paths related to this project using directory configurations.
        
        Returns:
            Dict of paths for this project
        """
        # Import here to avoid circular imports
        from app.services.directory_service import DirectoryService
        
        project_name = f"project_{self.id}"
        
        # Get configured directories or fall back to defaults
        upload_dir = DirectoryService.get_directory_path_by_name(
            'uploads', 
            default_path=current_app.config.get('UPLOAD_FOLDER', os.path.join(current_app.root_path, 'uploads'))
        )
        
        output_dir = DirectoryService.get_directory_path_by_name(
            'outputs',
            default_path=os.path.join(current_app.config.get('PROJECT_ROOT', os.path.dirname(current_app.root_path)), 'outputs')
        )
        
        logs_dir = DirectoryService.get_directory_path_by_name(
            'logs',
            default_path=os.path.join(current_app.root_path, 'logs')
        )
        
        # Build path dictionary
        paths = {
            'paper_path': os.path.join(upload_dir, 'papers', self.paper_path) if self.paper_path else None,
            'pdf_json_path': os.path.join(upload_dir, 'temp', f"{project_name}.json"),
            'pdf_json_cleaned_path': os.path.join(upload_dir, 'temp', f"{project_name}_cleaned.json"),
            'output_dir': os.path.join(output_dir, project_name),
            'output_repo_dir': self.output_path or os.path.join(output_dir, f"{project_name}_repo"),
            'log_path': self.log_path or os.path.join(logs_dir, 'projects', f"{project_name}.log")
        }
        
        return paths
    
    def check_paths_match_config(self) -> bool:
        """
        Check if the current project paths match the configured directories.
        
        Returns:
            bool: True if paths match configuration, False otherwise
        """
        # Check if we even have the required paths
        if not self.output_path or not self.log_path:
            return False
            
        # Get the expected paths based on current configuration
        paths = self.get_project_paths()
        
        # Check if the stored paths match the expected ones
        # We only check the base directories, not the full paths
        
        # For output path
        expected_output_dir = os.path.dirname(paths['output_repo_dir'])
        current_output_dir = os.path.dirname(self.output_path)
        output_matches = current_output_dir == expected_output_dir
        
        # For log path
        expected_log_dir = os.path.dirname(paths['log_path'])
        current_log_dir = os.path.dirname(self.log_path)
        log_matches = current_log_dir == expected_log_dir
        
        return output_matches and log_matches
    
    def update_paths_to_config(self) -> bool:
        """
        Update project paths to match the current directory configuration.
        This is useful when directory configurations have changed.
        
        Returns:
            bool: True if paths were updated, False if no update was needed
        """
        # Check if paths already match
        if self.check_paths_match_config():
            return False
            
        # Get the new paths
        paths = self.get_project_paths()
        
        # Save the old paths for potential file migration
        old_output_path = self.output_path
        old_log_path = self.log_path
        
        # Update the project with new paths
        self.output_path = paths['output_repo_dir']
        self.log_path = paths['log_path']
        
        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # TODO: Consider migrating files from old paths to new paths
        # This would be implemented here if requested
        
        db.session.commit()
        return True
    
    def to_dict(self):
        """Convert project to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'paper_title': self.paper_title,
            'paper_authors': self.paper_authors,
            'paper_abstract': self.paper_abstract,
            'project_type': self.project_type.value,
            'status': self.status.value,
            'status_color': self.get_status_color(),
            'programming_language': self.programming_language,
            'progress': self.progress,
            'current_task': self.current_task,
            'repository_url': self.repository_url,
            'export_path': self.export_path,
            'last_exported_at': self.last_exported_at.isoformat() if self.last_exported_at else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def __repr__(self):
        return f'<Project {self.name} ({self.status.value})>'