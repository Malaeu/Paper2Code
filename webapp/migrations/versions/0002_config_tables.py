"""Config tables

Revision ID: 0002
Revises: 0001
Create Date: 2025-05-05

"""

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    # Create model_cost_info table
    op.create_table('model_cost_info',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('input_cost_per_1k_tokens', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('output_cost_per_1k_tokens', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_tokens_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_cost', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    
    # Create model_configs table
    op.create_table('model_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('provider', sa.Enum('OPENAI', 'ANTHROPIC', 'LOCAL', 'DEEPSEEK', 'HUGGINGFACE', 'OTHER', name='modelprovider'), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('context_length', sa.Integer(), nullable=False, server_default='8192'),
        sa.Column('supports_vision', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('supports_function_calling', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('requires_api_key', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('command_args', sa.String(length=255), nullable=True),
        sa.Column('gpt_version', sa.String(length=50), nullable=True),
        sa.Column('cost_info_id', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['cost_info_id'], ['model_cost_info.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    
    # Create directory_configs table
    op.create_table('directory_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('path', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create project_settings table
    op.create_table('project_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )


def downgrade():
    op.drop_table('project_settings')
    op.drop_table('directory_configs')
    op.drop_table('model_configs')
    op.drop_table('model_cost_info')