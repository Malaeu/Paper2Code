"""Create user activity logs table

Revision ID: 0005_user_activity_logs
Revises: 0004_add_pending_email
Create Date: 2025-05-05

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0005_user_activity_logs'
down_revision = '0004_add_pending_email'
branch_labels = None
depends_on = None


def upgrade():
    """Create user_activity_logs table."""
    # Create enum type for activity types
    activity_type_enum = sa.Enum(
        'login', 'logout', 'profile_update', 'password_change', 'email_change',
        'email_verify', 'api_key_create', 'api_key_delete', 'api_key_toggle',
        'password_reset_request', 'password_reset', 'api_call', 'email_sent',
        'project_create', 'project_update', 'project_delete', 'config_update',
        name='activitytype'
    )
    activity_type_enum.create(op.get_bind())
    
    # Create user_activity_logs table
    op.create_table(
        'user_activity_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('activity_type', sa.Enum('login', 'logout', 'profile_update',
                                         'password_change', 'email_change', 'email_verify',
                                         'api_key_create', 'api_key_delete', 'api_key_toggle',
                                         'password_reset_request', 'password_reset',
                                         'api_call', 'email_sent', 'project_create',
                                         'project_update', 'project_delete', 'config_update',
                                         name='activitytype'), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=255), nullable=True),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('meta_data', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index for faster querying
    op.create_index(op.f('ix_user_activity_logs_user_id'), 'user_activity_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_activity_logs_created_at'), 'user_activity_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_user_activity_logs_activity_type'), 'user_activity_logs', ['activity_type'], unique=False)


def downgrade():
    """Drop user_activity_logs table."""
    op.drop_index(op.f('ix_user_activity_logs_activity_type'), table_name='user_activity_logs')
    op.drop_index(op.f('ix_user_activity_logs_created_at'), table_name='user_activity_logs')
    op.drop_index(op.f('ix_user_activity_logs_user_id'), table_name='user_activity_logs')
    op.drop_table('user_activity_logs')
    
    # Drop enum type
    sa.Enum(name='activitytype').drop(op.get_bind())