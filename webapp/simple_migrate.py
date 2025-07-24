"""
Simple script to create migrations directory structure.
"""
import os
import sys

def create_migrations_structure():
    """Create the basic migrations directory structure."""
    migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
    versions_dir = os.path.join(migrations_dir, 'versions')
    
    # Create directories
    os.makedirs(migrations_dir, exist_ok=True)
    os.makedirs(versions_dir, exist_ok=True)
    
    # Create alembic.ini file
    alembic_ini = os.path.join(migrations_dir, 'alembic.ini')
    with open(alembic_ini, 'w') as f:
        f.write("""# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = migrations

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
prepend_sys_path = .

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; this defaults
# to alembic/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path
# version_locations = %(here)s/bar %(here)s/bat alembic/versions

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = sqlite:///app.db


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
""")
    
    # Create env.py file
    env_py = os.path.join(migrations_dir, 'env.py')
    with open(env_py, 'w') as f:
        f.write("""import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from flask import current_app
config.set_main_option(
    'sqlalchemy.url',
    current_app.config.get('SQLALCHEMY_DATABASE_URI').replace('%', '%%')
)
target_metadata = current_app.extensions['migrate'].db.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    \"\"\"Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    \"\"\"
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    \"\"\"Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    \"\"\"

    # this callback is used to prevent an auto-migration from being generated
    # when there are no changes to the schema
    # reference: http://alembic.zzzcomputing.com/en/latest/cookbook.html
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            process_revision_directives=process_revision_directives,
            **current_app.extensions['migrate'].configure_args
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
""")
    
    # Create script.py.mako template
    script_mako = os.path.join(migrations_dir, 'script.py.mako')
    with open(script_mako, 'w') as f:
        f.write("""\"\"\"${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

\"\"\"""")

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}
""")
    
    print(f"Migration directory structure created at {migrations_dir}")
    
    # Create a first migration for the User model
    first_migration = os.path.join(versions_dir, '0001_initial_user_tables.py')
    with open(first_migration, 'w') as f:
        f.write("""\"\"\"Initial user tables

Revision ID: 0001
Revises: 
Create Date: 2025-05-05

\"\"\"

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
""")

    # Create documentation file
    migrations_readme = os.path.join(migrations_dir, 'README.md')
    with open(migrations_readme, 'w') as f:
        f.write("""# Database Migrations

This directory contains database migrations for the Paper2Code application.

## Migration Commands

To manage migrations, use the following commands:

### Initialize migrations (if needed)
```bash
flask db init
```

### Create a new migration
```bash
flask db migrate -m "Description of changes"
```

### Apply migrations
```bash
flask db upgrade
```

### Revert migrations
```bash
flask db downgrade
```

## Migration Strategy

1. **Before creating a migration**:
   - Always backup your database
   - Make sure all model changes are properly defined

2. **Testing migrations**:
   - Test migrations on a development database first
   - Verify both upgrade and downgrade paths

3. **Production migrations**:
   - Schedule migrations during low-traffic periods
   - Always create a full backup before migration
   - Have a rollback plan ready

## Backup Procedure

Before running migrations in production:

1. Create a full database backup:
   ```bash
   # For SQLite (copy the file)
   cp instance/app.db instance/app.db.backup-YYYY-MM-DD
   
   # For PostgreSQL
   pg_dump -U username -d database_name -f backup-YYYY-MM-DD.sql
   ```

2. Verify the backup is valid:
   ```bash
   # For SQLite
   sqlite3 instance/app.db.backup-YYYY-MM-DD .tables
   
   # For PostgreSQL
   psql -U username -d test_restore -f backup-YYYY-MM-DD.sql
   ```

## Rollback Procedure

If a migration fails or causes issues:

1. Stop the application
2. Run `flask db downgrade` to revert to the previous version
3. If downgrade fails, restore from backup:
   ```bash
   # For SQLite
   cp instance/app.db.backup-YYYY-MM-DD instance/app.db
   
   # For PostgreSQL
   psql -U username -d database_name -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   psql -U username -d database_name -f backup-YYYY-MM-DD.sql
   ```
4. Restart the application

## Migration Guidelines

1. Keep migrations small and focused on specific changes
2. Document complex migrations with comments
3. Handle data migrations carefully to avoid data loss
4. Update application code to be compatible with both pre- and post-migration schema during deployment
5. Test migrations thoroughly before applying to production

""")
    
    return True


if __name__ == '__main__':
    create_migrations_structure()