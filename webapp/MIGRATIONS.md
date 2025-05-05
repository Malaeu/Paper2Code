# Database Migrations Guide

This document provides a comprehensive guide for managing database migrations in the Paper2Code application.

## Overview

Database migrations are essential for tracking and applying database schema changes in a controlled manner. The Paper2Code application uses Flask-Migrate (built on Alembic) to manage database migrations.

## Setup

The migrations system is already set up with the following components:

1. **Migrations Directory**: Contains all migration scripts
2. **Migration Scripts**: Python files in `migrations/versions/` defining schema changes
3. **Management Scripts**: Utility scripts for working with migrations

## Migration Commands

### Basic Commands

Use these commands to manage migrations:

```bash
# Apply migrations to the database
python db_migrate.py upgrade

# Generate a new migration based on model changes
python db_migrate.py migrate

# Revert the last migration
python db_migrate.py downgrade

# Initialize migrations system (if needed)
python db_migrate.py init
```

### Database Management

For backup and restore operations:

```bash
# Create a database backup
python db_management.py backup

# List available backups
python db_management.py list

# Restore from a specific backup
python db_management.py restore backups/app_db_backup_20250505_123456.db

# Test a migration in a temporary database
python db_management.py test_migration
```

## Migration Workflow

Follow this workflow when making database schema changes:

1. **Update Models**: Modify the SQLAlchemy models in the application
2. **Create Backup**: `python db_management.py backup`
3. **Generate Migration**: `python db_migrate.py migrate`
4. **Review Migration**: Carefully check the generated migration script in `migrations/versions/`
5. **Test Migration**: `python db_management.py test_migration`
6. **Apply Migration**: `python db_migrate.py upgrade`

## Migration Development Guidelines

### General Guidelines

1. **Small, Focused Changes**: Keep migrations small and focused on specific changes
2. **Test Thoroughly**: Always test migrations before applying to production
3. **Backup First**: Create a backup before applying any migration
4. **Version Control**: Commit migration scripts to version control
5. **Deployment Planning**: Coordinate database migrations with application deployments

### Common Migration Patterns

#### Adding a Column

When adding a new column to an existing table:

```python
def upgrade():
    op.add_column('table_name', sa.Column('new_column', sa.String(50), nullable=True))
    
def downgrade():
    op.drop_column('table_name', 'new_column')
```

#### Adding a Table

When adding a new table:

```python
def upgrade():
    op.create_table(
        'new_table',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
def downgrade():
    op.drop_table('new_table')
```

#### Renaming a Column

When renaming a column:

```python
def upgrade():
    # SQLite doesn't support direct column renames, so use batch mode
    with op.batch_alter_table('table_name') as batch_op:
        batch_op.alter_column('old_name', new_column_name='new_name')
    
def downgrade():
    with op.batch_alter_table('table_name') as batch_op:
        batch_op.alter_column('new_name', new_column_name='old_name')
```

#### Data Migrations

When you need to modify data as part of a migration:

```python
def upgrade():
    # Add new column
    op.add_column('users', sa.Column('full_name', sa.String(100), nullable=True))
    
    # Update data
    connection = op.get_bind()
    users = connection.execute('SELECT id, first_name, last_name FROM users').fetchall()
    
    for user_id, first_name, last_name in users:
        full_name = f"{first_name} {last_name}".strip()
        connection.execute(
            'UPDATE users SET full_name = :full_name WHERE id = :id',
            {'full_name': full_name, 'id': user_id}
        )
    
    # Make column non-nullable after data migration
    op.alter_column('users', 'full_name', nullable=False)
    
def downgrade():
    op.drop_column('users', 'full_name')
```

## Troubleshooting

### Common Issues

1. **Migration Head Mismatch**: 
   - Error: `Multiple head revisions are present`
   - Solution: Run `flask db merge heads` to create a merge migration

2. **Migration Not Applied**: 
   - Error: `Target database is not up to date`
   - Solution: Run `python db_migrate.py upgrade` to apply pending migrations

3. **SQLite Limitations**:
   - SQLite has limited ALTER TABLE support
   - Use `batch_alter_table` for complex operations
   - Example: 
     ```python
     with op.batch_alter_table('table_name') as batch_op:
         batch_op.alter_column(...)
     ```

4. **Foreign Key Constraints**:
   - Define ForeignKey constraints with `onupdate` and `ondelete` values
   - Example: 
     ```python
     sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
     ```

## Recovery Procedures

### Recovering from Failed Migrations

If a migration fails, follow these steps:

1. **Check Error Messages**: Review error output for specific issues
2. **Downgrade**: Try to downgrade to the previous version: `python db_migrate.py downgrade`
3. **Restore Backup**: If downgrade fails, restore from backup: `python db_management.py restore <backup_file>`
4. **Fix Migration**: Correct issues in the migration script
5. **Retry**: Try upgrading again with the fixed migration

### Database Corruption Recovery

In case of database corruption:

1. **Stop Application**: Take the application offline
2. **Assess Damage**: Try to determine extent of corruption
3. **Restore Backup**: Restore latest working backup: `python db_management.py restore <backup_file>`
4. **Apply Migrations**: If needed, apply migrations to get to the desired schema version
5. **Data Recovery**: Recover any data created after the backup (if possible)
6. **Restart Application**: Bring the application back online

## Best Practices

1. **Regular Backups**: Implement regular, automated database backups
2. **Migration Testing**: Always test migrations in a staging environment
3. **Keep Migrations Reversible**: Ensure downgrade functions work correctly
4. **Version Control**: Commit migrations with related application code changes
5. **Documentation**: Document complex migrations with comments
6. **Separation of Concerns**: Split schema changes and data changes into separate migrations
7. **Error Handling**: Include error handling in data migrations

## Conclusion

A proper database migration system ensures reliable and reproducible database schema changes. By following the guidelines in this document, you can safely manage the evolution of the Paper2Code database schema.