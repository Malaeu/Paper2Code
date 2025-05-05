"""Initial user tables

Revision ID: 0001
Revises: 
Create Date: 2025-05-05 13:35:50

"""

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite


def upgrade():
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=120), nullable=False),
        sa.Column('password_hash', sa.String(length=256), nullable=False),
        sa.Column('role', sa.Enum('USER', 'ADMIN', name='userrole'), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'ACTIVE', 'INACTIVE', 'SUSPENDED', name='userstatus'), nullable=False),
        sa.Column('email_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('verification_token', sa.String(length=100), nullable=True),
        sa.Column('verification_token_expiry', sa.DateTime(), nullable=True),
        sa.Column('two_factor_enabled', sa.Boolean(), nullable=False, default=False),
        sa.Column('two_factor_secret', sa.String(length=32), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('last_login_ip', sa.String(length=45), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, default=0),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('total_api_calls', sa.Integer(), nullable=False, default=0),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('service', sa.String(length=50), nullable=False),
        sa.Column('key_name', sa.String(length=64), nullable=False),
        sa.Column('key_prefix', sa.String(length=10), nullable=False),
        sa.Column('key_hash', sa.String(length=256), nullable=False),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('calls_count', sa.Integer(), nullable=False, default=0),
        sa.Column('tokens_used', sa.Integer(), nullable=False, default=0),
        sa.Column('estimated_cost', sa.Float(), nullable=False, default=0.0),
        sa.Column('has_quota', sa.Boolean(), nullable=False, default=False),
        sa.Column('daily_quota', sa.Integer(), nullable=True),
        sa.Column('monthly_quota', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create api_key_usage table
    op.create_table('api_key_usage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('endpoint', sa.String(length=100), nullable=False),
        sa.Column('tokens_used', sa.Integer(), nullable=False, default=0),
        sa.Column('cost', sa.Float(), nullable=False, default=0.0),
        sa.Column('task_type', sa.String(length=50), nullable=True),
        sa.Column('user_request_id', sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('api_key_usage')
    op.drop_table('api_keys')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
