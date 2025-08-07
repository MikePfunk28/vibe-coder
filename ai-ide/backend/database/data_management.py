"""
Data cleanup, archival policies, and export/import functionality.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, IO
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import gzip
import os
import tempfile
from pathlib import Path

from .connection import db_manager
from .caching import cache_manager
from .repositories.agent_repository import agent_interaction_repo, reasoning_trace_repo
from .repositories.search_repository import code_embedding_repo, web_search_cache_repo

logger = logging.getLogger(__name__)

@dataclass
class CleanupPolicy:
    """Defines cleanup policy for a data type."""
    table_name: str
    retention_days: int
    archive_before_delete: bool = True
    batch_size: int = 1000
    date_column: str = "created_at"

@dataclass
class ArchiveInfo:
    """Information about an archived data set."""
    archive_id: str
    table_name: str
    date_range: tuple[datetime, datetime]
    record_count: int
    file_path: str
    created_at: datetime
    compressed: bool = True

class DataCleanupManager:
    """Manages data cleanup and archival policies."""
    
    def __init__(self, archive_directory: str = None):
        self.archive_directory = archive_directory or os.path.join(
            os.path.dirname(__file__), '..', 'archives'
        )
        Path(self.archive_directory).mkdir(parents=True, exist_ok=True)
        
        # Default cleanup policies
        self.cleanup_policies = {
            'agent_interactions': CleanupPolicy(
                table_name='agent_interactions',
                retention_days=90,
                archive_before_delete=True,
                date_column='timestamp'
            ),
            'reasoning_traces': CleanupPolicy(
                table_name='reasoning_traces',
                retention_days=60,
                archive_before_delete=True,
                date_column='timestamp'
            ),
            'web_search_cache': CleanupPolicy(
                table_name='web_search_cache',
                retention_days=7,
                archive_before_delete=False,
                date_column='cached_at'
            ),
            'code_embeddings': CleanupPolicy(
                table_name='code_embeddings',
                retention_days=180,
                archive_before_delete=True,
                date_column='updated_at'
            ),
            'user_interactions': CleanupPolicy(
                table_name='user_interactions',
                retention_days=365,
                archive_before_delete=True,
                date_column='timestamp'
            ),
            'performance_metrics': CleanupPolicy(
                table_name='performance_metrics',
                retention_days=30,
                archive_before_delete=True,
                date_column='measurement_timestamp'
            )
        }
    
    def set_cleanup_policy(self, table_name: str, policy: CleanupPolicy) -> None:
        """Set or update cleanup policy for a table."""
        self.cleanup_policies[table_name] = policy
        logger.info(f"Updated cleanup policy for {table_name}: {policy.retention_days} days retention")
    
    async def run_cleanup(self, table_name: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Run cleanup for specified table or all tables."""
        results = {}
        
        tables_to_clean = [table_name] if table_name else list(self.cleanup_policies.keys())
        
        for table in tables_to_clean:
            if table not in self.cleanup_policies:
                logger.warning(f"No cleanup policy defined for table: {table}")
                continue
            
            policy = self.cleanup_policies[table]
            
            try:
                result = await self._cleanup_table(policy, dry_run)
                results[table] = result
                
            except Exception as e:
                logger.error(f"Failed to cleanup table {table}: {e}")
                results[table] = {'error': str(e)}
        
        return results
    
    async def _cleanup_table(self, policy: CleanupPolicy, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up a specific table according to its policy."""
        cutoff_date = datetime.now() - timedelta(days=policy.retention_days)
        
        # Count records to be cleaned
        count_query = f"""
            SELECT COUNT(*) FROM {policy.table_name} 
            WHERE {policy.date_column} < $1
        """
        total_records = await db_manager.fetchval(count_query, cutoff_date)
        
        if total_records == 0:
            return {
                'table_name': policy.table_name,
                'records_found': 0,
                'records_archived': 0,
                'records_deleted': 0,
                'dry_run': dry_run
            }
        
        archived_count = 0
        deleted_count = 0
        
        # Archive before delete if policy requires it
        if policy.archive_before_delete and not dry_run:
            archive_info = await self._archive_table_data(policy, cutoff_date)
            if archive_info:
                archived_count = archive_info.record_count
        
        # Delete old records
        if not dry_run:
            deleted_count = await self._delete_old_records(policy, cutoff_date)
            
            # Clear related caches
            await self._clear_related_caches(policy.table_name)
        
        return {
            'table_name': policy.table_name,
            'records_found': total_records,
            'records_archived': archived_count,
            'records_deleted': deleted_count if not dry_run else 0,
            'cutoff_date': cutoff_date.isoformat(),
            'dry_run': dry_run
        }
    
    async def _archive_table_data(self, policy: CleanupPolicy, cutoff_date: datetime) -> Optional[ArchiveInfo]:
        """Archive table data before deletion."""
        try:
            # Generate archive filename
            date_str = cutoff_date.strftime('%Y%m%d')
            archive_filename = f"{policy.table_name}_{date_str}.json.gz"
            archive_path = os.path.join(self.archive_directory, archive_filename)
            
            # Query data to archive
            query = f"""
                SELECT * FROM {policy.table_name} 
                WHERE {policy.date_column} < $1
                ORDER BY {policy.date_column}
            """
            
            records = await db_manager.fetch(query, cutoff_date)
            
            if not records:
                return None
            
            # Convert records to JSON-serializable format
            json_records = []
            for record in records:
                record_dict = dict(record)
                # Convert datetime objects to ISO strings
                for key, value in record_dict.items():
                    if isinstance(value, datetime):
                        record_dict[key] = value.isoformat()
                json_records.append(record_dict)
            
            # Write to compressed archive
            with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                json.dump({
                    'table_name': policy.table_name,
                    'archive_date': datetime.now().isoformat(),
                    'cutoff_date': cutoff_date.isoformat(),
                    'record_count': len(json_records),
                    'records': json_records
                }, f, indent=2)
            
            # Create archive info
            archive_info = ArchiveInfo(
                archive_id=f"{policy.table_name}_{date_str}",
                table_name=policy.table_name,
                date_range=(records[0][policy.date_column], records[-1][policy.date_column]),
                record_count=len(json_records),
                file_path=archive_path,
                created_at=datetime.now(),
                compressed=True
            )
            
            # Store archive metadata
            await self._store_archive_metadata(archive_info)
            
            logger.info(f"Archived {len(json_records)} records from {policy.table_name} to {archive_path}")
            return archive_info
            
        except Exception as e:
            logger.error(f"Failed to archive data for {policy.table_name}: {e}")
            return None
    
    async def _delete_old_records(self, policy: CleanupPolicy, cutoff_date: datetime) -> int:
        """Delete old records from table."""
        try:
            # Delete in batches to avoid long-running transactions
            total_deleted = 0
            
            while True:
                delete_query = f"""
                    DELETE FROM {policy.table_name} 
                    WHERE id IN (
                        SELECT id FROM {policy.table_name} 
                        WHERE {policy.date_column} < $1 
                        LIMIT $2
                    )
                """
                
                result = await db_manager.execute(delete_query, cutoff_date, policy.batch_size)
                
                # Extract number of deleted rows
                deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
                total_deleted += deleted_count
                
                if deleted_count < policy.batch_size:
                    break
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Deleted {total_deleted} old records from {policy.table_name}")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to delete old records from {policy.table_name}: {e}")
            return 0
    
    async def _clear_related_caches(self, table_name: str) -> None:
        """Clear caches related to the cleaned table."""
        try:
            if table_name == "agent_interactions":
                await cache_manager.clear_cache_type('agent')
            elif table_name == "reasoning_traces":
                await cache_manager.clear_cache_type('reasoning')
            elif table_name == "web_search_cache":
                await cache_manager.clear_cache_type('web_search')
            elif table_name == "code_embeddings":
                await cache_manager.clear_cache_type('embedding')
                await cache_manager.clear_cache_type('search')
            
        except Exception as e:
            logger.error(f"Failed to clear caches for {table_name}: {e}")
    
    async def _store_archive_metadata(self, archive_info: ArchiveInfo) -> None:
        """Store archive metadata in database."""
        try:
            # Create archives table if it doesn't exist
            await db_manager.execute("""
                CREATE TABLE IF NOT EXISTS data_archives (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    archive_id VARCHAR(100) UNIQUE NOT NULL,
                    table_name VARCHAR(50) NOT NULL,
                    date_range_start TIMESTAMP,
                    date_range_end TIMESTAMP,
                    record_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    compressed BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Insert archive metadata
            await db_manager.execute("""
                INSERT INTO data_archives 
                (archive_id, table_name, date_range_start, date_range_end, record_count, file_path, compressed)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                archive_info.archive_id,
                archive_info.table_name,
                archive_info.date_range[0],
                archive_info.date_range[1],
                archive_info.record_count,
                archive_info.file_path,
                archive_info.compressed
            )
            
        except Exception as e:
            logger.error(f"Failed to store archive metadata: {e}")
    
    async def get_archive_list(self, table_name: Optional[str] = None) -> List[ArchiveInfo]:
        """Get list of available archives."""
        try:
            query = "SELECT * FROM data_archives"
            params = []
            
            if table_name:
                query += " WHERE table_name = $1"
                params.append(table_name)
            
            query += " ORDER BY created_at DESC"
            
            rows = await db_manager.fetch(query, *params)
            
            archives = []
            for row in rows:
                archives.append(ArchiveInfo(
                    archive_id=row['archive_id'],
                    table_name=row['table_name'],
                    date_range=(row['date_range_start'], row['date_range_end']),
                    record_count=row['record_count'],
                    file_path=row['file_path'],
                    created_at=row['created_at'],
                    compressed=row['compressed']
                ))
            
            return archives
            
        except Exception as e:
            logger.error(f"Failed to get archive list: {e}")
            return []

class DataExportImportManager:
    """Manages data export and import operations."""
    
    def __init__(self, export_directory: str = None):
        self.export_directory = export_directory or os.path.join(
            os.path.dirname(__file__), '..', 'exports'
        )
        Path(self.export_directory).mkdir(parents=True, exist_ok=True)
    
    async def export_table_data(self, table_name: str, filters: Optional[Dict[str, Any]] = None,
                               format: str = 'json', compress: bool = True) -> str:
        """Export table data to file."""
        try:
            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []
            
            if filters:
                conditions = []
                for field, value in filters.items():
                    conditions.append(f"{field} = ${len(params) + 1}")
                    params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            # Fetch data
            records = await db_manager.fetch(query, *params)
            
            if not records:
                raise ValueError(f"No data found for table {table_name}")
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{table_name}_export_{timestamp}.{format}"
            if compress:
                filename += '.gz'
            
            filepath = os.path.join(self.export_directory, filename)
            
            # Convert records to serializable format
            export_data = []
            for record in records:
                record_dict = dict(record)
                for key, value in record_dict.items():
                    if isinstance(value, datetime):
                        record_dict[key] = value.isoformat()
                    elif hasattr(value, 'tolist'):  # numpy arrays
                        record_dict[key] = value.tolist()
                export_data.append(record_dict)
            
            # Write to file
            if format.lower() == 'json':
                export_content = json.dumps({
                    'table_name': table_name,
                    'export_date': datetime.now().isoformat(),
                    'record_count': len(export_data),
                    'filters': filters,
                    'records': export_data
                }, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            if compress:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    f.write(export_content)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(export_content)
            
            logger.info(f"Exported {len(export_data)} records from {table_name} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export table data: {e}")
            raise
    
    async def import_table_data(self, filepath: str, table_name: Optional[str] = None,
                               merge_strategy: str = 'skip') -> Dict[str, Any]:
        """Import table data from file."""
        try:
            # Read file
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Validate data structure
            if 'records' not in data:
                raise ValueError("Invalid export file format: missing 'records' field")
            
            target_table = table_name or data.get('table_name')
            if not target_table:
                raise ValueError("Table name not specified and not found in export file")
            
            records = data['records']
            if not records:
                return {'imported': 0, 'skipped': 0, 'errors': 0}
            
            # Import records
            imported_count = 0
            skipped_count = 0
            error_count = 0
            
            for record in records:
                try:
                    # Convert datetime strings back to datetime objects
                    for key, value in record.items():
                        if isinstance(value, str) and 'T' in value and value.endswith('Z') or '+' in value[-6:]:
                            try:
                                record[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except ValueError:
                                pass  # Keep as string if not a valid datetime
                    
                    # Handle merge strategy
                    if merge_strategy == 'skip':
                        # Check if record exists (assuming 'id' field)
                        if 'id' in record:
                            existing = await db_manager.fetchval(
                                f"SELECT 1 FROM {target_table} WHERE id = $1",
                                record['id']
                            )
                            if existing:
                                skipped_count += 1
                                continue
                    
                    # Insert record
                    fields = list(record.keys())
                    values = list(record.values())
                    placeholders = [f"${i+1}" for i in range(len(values))]
                    
                    insert_query = f"""
                        INSERT INTO {target_table} ({', '.join(fields)})
                        VALUES ({', '.join(placeholders)})
                    """
                    
                    if merge_strategy == 'update':
                        # Use ON CONFLICT to update existing records
                        update_fields = [f"{field} = EXCLUDED.{field}" for field in fields if field != 'id']
                        insert_query += f" ON CONFLICT (id) DO UPDATE SET {', '.join(update_fields)}"
                    elif merge_strategy == 'skip':
                        insert_query += " ON CONFLICT (id) DO NOTHING"
                    
                    await db_manager.execute(insert_query, *values)
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to import record: {e}")
                    error_count += 1
            
            result = {
                'table_name': target_table,
                'total_records': len(records),
                'imported': imported_count,
                'skipped': skipped_count,
                'errors': error_count,
                'merge_strategy': merge_strategy
            }
            
            logger.info(f"Import completed for {target_table}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to import table data: {e}")
            raise

# Global manager instances
cleanup_manager = DataCleanupManager()
export_import_manager = DataExportImportManager()

# Convenience functions
async def run_data_cleanup(table_name: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    """Run data cleanup."""
    return await cleanup_manager.run_cleanup(table_name, dry_run)

async def export_data(table_name: str, filters: Optional[Dict[str, Any]] = None,
                     format: str = 'json', compress: bool = True) -> str:
    """Export table data."""
    return await export_import_manager.export_table_data(table_name, filters, format, compress)

async def import_data(filepath: str, table_name: Optional[str] = None,
                     merge_strategy: str = 'skip') -> Dict[str, Any]:
    """Import table data."""
    return await export_import_manager.import_table_data(filepath, table_name, merge_strategy)