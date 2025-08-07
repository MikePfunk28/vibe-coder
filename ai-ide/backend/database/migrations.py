"""
Database migration system for schema updates and version management.
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from .connection import db_manager

logger = logging.getLogger(__name__)

@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    description: str
    up_sql: str
    down_sql: Optional[str] = None
    applied_at: Optional[datetime] = None

class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, migrations_dir: str = None):
        self.migrations_dir = migrations_dir or os.path.join(
            os.path.dirname(__file__), 'migrations'
        )
        self.migrations: List[Migration] = []
        self._load_migrations()
    
    def _load_migrations(self) -> None:
        """Load migration files from the migrations directory."""
        if not os.path.exists(self.migrations_dir):
            os.makedirs(self.migrations_dir)
            logger.info(f"Created migrations directory: {self.migrations_dir}")
            return
        
        migration_files = sorted([
            f for f in os.listdir(self.migrations_dir) 
            if f.endswith('.sql')
        ])
        
        for filename in migration_files:
            try:
                migration = self._parse_migration_file(filename)
                if migration:
                    self.migrations.append(migration)
            except Exception as e:
                logger.error(f"Failed to parse migration {filename}: {e}")
    
    def _parse_migration_file(self, filename: str) -> Optional[Migration]:
        """Parse a migration file and extract version, description, and SQL."""
        filepath = os.path.join(self.migrations_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract version from filename (e.g., "001_initial_schema.sql" -> "001")
        version = filename.split('_')[0]
        
        # Extract description from filename or first comment
        description_parts = filename.replace('.sql', '').split('_')[1:]
        description = ' '.join(description_parts).replace('_', ' ').title()
        
        # Look for -- UP and -- DOWN markers to split SQL
        up_sql = content
        down_sql = None
        
        if '-- DOWN' in content:
            parts = content.split('-- DOWN')
            up_sql = parts[0].replace('-- UP', '').strip()
            down_sql = parts[1].strip()
        elif '-- UP' in content:
            up_sql = content.split('-- UP')[1].strip()
        
        return Migration(
            version=version,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql
        )
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            rows = await db_manager.fetch(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            return [row['version'] for row in rows]
        except Exception as e:
            logger.warning(f"Could not fetch applied migrations: {e}")
            return []
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        try:
            async with db_manager.transaction() as conn:
                # Execute the migration SQL
                await conn.execute(migration.up_sql)
                
                # Record the migration as applied
                await conn.execute(
                    """
                    INSERT INTO schema_migrations (version, description, applied_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (version) DO NOTHING
                    """,
                    migration.version,
                    migration.description,
                    datetime.now()
                )
                
                logger.info(f"Applied migration {migration.version}: {migration.description}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        if not migration.down_sql:
            logger.error(f"Migration {migration.version} has no rollback SQL")
            return False
        
        try:
            async with db_manager.transaction() as conn:
                # Execute the rollback SQL
                await conn.execute(migration.down_sql)
                
                # Remove the migration record
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1",
                    migration.version
                )
                
                logger.info(f"Rolled back migration {migration.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> bool:
        """Apply all pending migrations up to target version."""
        applied_versions = await self.get_applied_migrations()
        
        pending_migrations = [
            m for m in self.migrations 
            if m.version not in applied_versions
        ]
        
        if target_version:
            pending_migrations = [
                m for m in pending_migrations 
                if m.version <= target_version
            ]
        
        if not pending_migrations:
            logger.info("No pending migrations to apply")
            return True
        
        logger.info(f"Applying {len(pending_migrations)} migrations...")
        
        for migration in pending_migrations:
            success = await self.apply_migration(migration)
            if not success:
                logger.error(f"Migration failed at version {migration.version}")
                return False
        
        logger.info("All migrations applied successfully")
        return True
    
    async def migrate_down(self, target_version: str) -> bool:
        """Rollback migrations down to target version."""
        applied_versions = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        migrations_to_rollback = [
            m for m in reversed(self.migrations)
            if m.version in applied_versions and m.version > target_version
        ]
        
        if not migrations_to_rollback:
            logger.info("No migrations to rollback")
            return True
        
        logger.info(f"Rolling back {len(migrations_to_rollback)} migrations...")
        
        for migration in migrations_to_rollback:
            success = await self.rollback_migration(migration)
            if not success:
                logger.error(f"Rollback failed at version {migration.version}")
                return False
        
        logger.info("All rollbacks completed successfully")
        return True
    
    async def get_migration_status(self) -> Dict[str, any]:
        """Get current migration status."""
        applied_versions = await self.get_applied_migrations()
        
        all_versions = [m.version for m in self.migrations]
        pending_versions = [v for v in all_versions if v not in applied_versions]
        
        current_version = max(applied_versions) if applied_versions else None
        latest_version = max(all_versions) if all_versions else None
        
        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "applied_migrations": len(applied_versions),
            "pending_migrations": len(pending_versions),
            "pending_versions": pending_versions,
            "up_to_date": current_version == latest_version
        }
    
    def create_migration(self, description: str, up_sql: str, down_sql: str = None) -> str:
        """Create a new migration file."""
        # Generate version number based on existing migrations
        existing_versions = [int(m.version) for m in self.migrations if m.version.isdigit()]
        next_version = str(max(existing_versions) + 1).zfill(3) if existing_versions else "001"
        
        # Create filename
        filename = f"{next_version}_{description.lower().replace(' ', '_')}.sql"
        filepath = os.path.join(self.migrations_dir, filename)
        
        # Create migration content
        content = f"-- Migration: {description}\n-- Version: {next_version}\n\n-- UP\n{up_sql}\n"
        if down_sql:
            content += f"\n-- DOWN\n{down_sql}\n"
        
        # Write migration file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created migration file: {filename}")
        
        # Reload migrations
        self._load_migrations()
        
        return filepath

# Global migration manager instance
migration_manager = MigrationManager()

# Convenience functions
async def migrate_up(target_version: Optional[str] = None) -> bool:
    """Apply pending migrations."""
    return await migration_manager.migrate_up(target_version)

async def migrate_down(target_version: str) -> bool:
    """Rollback migrations."""
    return await migration_manager.migrate_down(target_version)

async def get_migration_status() -> Dict[str, any]:
    """Get migration status."""
    return await migration_manager.get_migration_status()

def create_migration(description: str, up_sql: str, down_sql: str = None) -> str:
    """Create a new migration."""
    return migration_manager.create_migration(description, up_sql, down_sql)