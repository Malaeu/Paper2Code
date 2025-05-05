#!/usr/bin/env python
"""
Database management utility for Paper2Code.

This script provides utilities for managing the application database,
including creating backups, testing migrations, and restoring from backups.

Usage:
  python db_management.py backup               - Create a database backup
  python db_management.py test_migration       - Test migration in a temporary database
  python db_management.py restore BACKUP_FILE  - Restore database from backup
"""

import os
import sys
import shutil
import datetime
import argparse
import tempfile
import subprocess
from pathlib import Path

DB_FILE = 'app.db'
BACKUP_DIR = 'backups'

def ensure_backup_dir():
    """Ensure the backup directory exists."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

def create_backup():
    """Create a database backup."""
    ensure_backup_dir()
    
    # Check if database exists
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found.")
        return False
    
    # Create backup filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(BACKUP_DIR, f"app_db_backup_{timestamp}.db")
    
    # Copy database file
    try:
        shutil.copy2(DB_FILE, backup_file)
        print(f"Backup created: {backup_file}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def test_migration():
    """Test migration in a temporary database."""
    # Check if database exists
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found. Cannot test migration.")
        return False
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db = os.path.join(temp_dir, DB_FILE)
            
            # Copy current database to temp location
            shutil.copy2(DB_FILE, temp_db)
            
            # Apply migration in temp environment
            print(f"Testing migration in temporary database: {temp_db}")
            
            # Set environment variable to point to temp database
            env = os.environ.copy()
            env['TEST_DB_PATH'] = temp_db
            
            # Run migration in temp environment
            result = subprocess.run(
                ["python", "db_migrate.py", "upgrade"],
                env=env,
                capture_output=True,
                text=True
            )
            
            # Check result
            if result.returncode == 0:
                print("Migration test successful!")
                print(result.stdout)
                return True
            else:
                print("Migration test failed!")
                print(result.stderr)
                return False
                
    except Exception as e:
        print(f"Error testing migration: {e}")
        return False

def restore_backup(backup_file):
    """Restore database from backup."""
    # Validate backup file
    if not os.path.exists(backup_file):
        print(f"Error: Backup file '{backup_file}' not found.")
        return False
    
    # Backup current database before restore (safety measure)
    current_backup = create_backup()
    if not current_backup:
        response = input("Failed to create safety backup. Continue with restore? (y/N): ")
        if response.lower() != 'y':
            print("Restore aborted.")
            return False
    
    # Restore from backup
    try:
        # If database exists, remove it first
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        
        # Copy backup to database location
        shutil.copy2(backup_file, DB_FILE)
        print(f"Database restored from backup: {backup_file}")
        return True
    except Exception as e:
        print(f"Error restoring backup: {e}")
        return False

def list_backups():
    """List available backups."""
    ensure_backup_dir()
    
    backups = [f for f in os.listdir(BACKUP_DIR) if f.startswith('app_db_backup_')]
    
    if not backups:
        print("No backups found.")
        return
    
    print("\nAvailable backups:")
    for i, backup in enumerate(sorted(backups)):
        backup_path = os.path.join(BACKUP_DIR, backup)
        size = os.path.getsize(backup_path) / 1024  # Convert to KB
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(backup_path))
        print(f"{i+1}. {backup} ({size:.2f} KB, {modified})")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Paper2Code Database Management Utility')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    
    # Test migration command
    test_parser = subparsers.add_parser('test_migration', help='Test migration in a temporary database')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database from backup')
    restore_parser.add_argument('backup_file', nargs='?', help='Backup file to restore from')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    args = parser.parse_args()
    
    if args.command == 'backup':
        create_backup()
    elif args.command == 'test_migration':
        test_migration()
    elif args.command == 'restore':
        if args.backup_file:
            restore_backup(args.backup_file)
        else:
            list_backups()
            backup_num = input("\nEnter backup number to restore (or 'q' to quit): ")
            if backup_num.lower() == 'q':
                return
            
            try:
                backup_num = int(backup_num) - 1
                backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith('app_db_backup_')])
                if 0 <= backup_num < len(backups):
                    backup_file = os.path.join(BACKUP_DIR, backups[backup_num])
                    restore_backup(backup_file)
                else:
                    print("Invalid backup number.")
            except ValueError:
                print("Invalid input.")
    elif args.command == 'list':
        list_backups()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()