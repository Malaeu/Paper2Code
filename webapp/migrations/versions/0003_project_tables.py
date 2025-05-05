"""Project tables

Revision ID: 0003
Revises: 0002
Create Date: 2025-05-05

"""

# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    # Create projects table
    op.create_table('projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('project_type', sa.Enum('PAPER_TO_CODE', 'CUSTOM_ADAPTATION', name='projecttype'), nullable=False),
        sa.Column('status', sa.Enum('CREATED', 'PROCESSING', 'PLANNING', 'ANALYZING', 'CODING', 'COMPLETED', 'FAILED', 'CANCELLED', name='projectstatus'), nullable=False),
        sa.Column('paper_path', sa.String(length=255), nullable=True),
        sa.Column('dataset_path', sa.String(length=255), nullable=True),
        sa.Column('output_path', sa.String(length=255), nullable=True),
        sa.Column('programming_language', sa.String(length=50), nullable=True),
        sa.Column('model_config_id', sa.Integer(), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['model_config_id'], ['model_configs.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create project_logs table
    op.create_table('project_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('level', sa.String(length=10), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create project_artifacts table
    op.create_table('project_artifacts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('artifact_type', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('project_artifacts')
    op.drop_table('project_logs')
    op.drop_table('projects')