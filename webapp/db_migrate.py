#!/usr/bin/env python
"""
Database migration command-line utility for Paper2Code.

Usage:
  python db_migrate.py init    - Initialize migrations
  python db_migrate.py migrate - Generate migration from models
  python db_migrate.py upgrade - Apply migrations to database
  python db_migrate.py downgrade - Revert last migration
"""

import os
import sys
import argparse
import importlib
import subprocess
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

def create_test_app():
    """Create a minimal Flask app for testing migrations."""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize SQLAlchemy without binding to app yet
    db = SQLAlchemy()
    
    # Import models
    from app.models.auth.user import User, ApiKey, ApiKeyUsage
    from app.models.config.model_config import ModelConfig, ModelCostInfo, DirectoryConfig, ProjectSettings
    from app.models.projects.project import Project
    
    # Bind SQLAlchemy to app
    db.init_app(app)
    
    # Initialize migrations
    migrate = Migrate(app, db)
    
    return app, db, migrate

def run_migration_command(command):
    """Run a Flask-Migrate command."""
    if command == 'init':
        print("Initializing migrations...")
        app, db, migrate = create_test_app()
        with app.app_context():
            from flask_migrate import init
            init('migrations')
        print("Migrations initialized.")
    
    elif command == 'migrate':
        print("Creating migration...")
        app, db, migrate = create_test_app()
        with app.app_context():
            from flask_migrate import migrate as create_migration
            create_migration(message='Auto-generated migration')
        print("Migration created.")
    
    elif command == 'upgrade':
        print("Applying migrations...")
        app, db, migrate = create_test_app()
        with app.app_context():
            from flask_migrate import upgrade
            upgrade()
        print("Migrations applied.")
    
    elif command == 'downgrade':
        print("Reverting last migration...")
        app, db, migrate = create_test_app()
        with app.app_context():
            from flask_migrate import downgrade
            downgrade()
        print("Migration reverted.")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: init, migrate, upgrade, downgrade")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Paper2Code Database Migration Utility')
    parser.add_argument('command', choices=['init', 'migrate', 'upgrade', 'downgrade'],
                        help='Migration command to run')
    
    args = parser.parse_args()
    run_migration_command(args.command)

if __name__ == '__main__':
    main()