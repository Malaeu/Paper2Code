# Database Migrations

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

