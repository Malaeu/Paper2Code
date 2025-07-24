"""Add pending_email field to users table

Revision ID: 0004_add_pending_email
Revises: 0003_project_tables
Create Date: 2025-05-05

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0004_add_pending_email'
down_revision = '0003_project_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Add pending_email column to users table."""
    op.add_column('users', sa.Column('pending_email', sa.String(120), nullable=True))


def downgrade():
    """Remove pending_email column from users table."""
    op.drop_column('users', 'pending_email')