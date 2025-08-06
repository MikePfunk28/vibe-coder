"""
Version Management and Rollback System for Darwin-Gödel Model

This module provides comprehensive version management capabilities including:
- Automated backup system before applying improvements
- Rollback mechanisms for failed improvements
- Improvement history tracking and analysis
- Version comparison and diff generation
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from darwin_godel_model import CodeModification, ImprovementType, PerformanceMetrics


class VersionStatus(str):
    """Status of a version."""
    ACTIVE = "active"
    BACKUP = "backup"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class FileVersion:
    """Represents a version of a single file."""
    id: str
    file_path: str
    content: str
    content_hash: str
    timestamp: datetime
    modification_id: Optional[str] = None
    parent_version_id: Optional[str] = None
    status: VersionStatus = VersionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemVersion:
    """Represents a complete system version snapshot."""
    id: str
    timestamp: datetime
    description: str
    file_versions: List[FileVersion]
    performance_metrics: Optional[PerformanceMetrics] = None
    modification_ids: List[str] = field(default_factory=list)
    status: VersionStatus = VersionStatus.ACTIVE
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionDiff:
    """Represents differences between two versions."""
    from_version_id: str
    to_version_id: str
    file_changes: List[Dict[str, Any]]
    added_files: List[str]
    removed_files: List[str]
    modified_files: List[str]
    performance_delta: Optional[Dict[str, float]] = None


@dataclass
class BackupInfo:
    """Information about a backup operation."""
    id: str
    timestamp: datetime
    backup_path: str
    original_files: List[str]
    backup_size: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionManager:
    """Manages versions, backups, and rollbacks for the Darwin-Gödel system."""
    
    def __init__(self, storage_path: str = "dgm_versions", backup_path: str = "dgm_backups"):
        self.storage_path = Path(storage_path)
        self.backup_path = Path(backup_path)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.storage_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_path / "versions.db"
        self._init_database()
        
        # Current system state
        self.current_version_id: Optional[str] = None
        self.version_cache: Dict[str, SystemVersion] = {}
        
        self.logger.info(f"VersionManager initialized with storage at {self.storage_path}")
    
    def close(self):
        """Close any open resources."""
        # Clear cache to help with cleanup
        self.version_cache.clear()
    
    def _init_database(self):
        """Initialize the SQLite database for version storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_versions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    performance_metrics TEXT,
                    modification_ids TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_versions (
                    id TEXT PRIMARY KEY,
                    system_version_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    modification_id TEXT,
                    parent_version_id TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (system_version_id) REFERENCES system_versions (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    original_files TEXT NOT NULL,
                    backup_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_versions_path 
                ON file_versions (file_path)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_versions_timestamp 
                ON system_versions (timestamp)
            """)
    
    def create_backup(self, files_to_backup: List[str], description: str = "") -> BackupInfo:
        """Create a backup of specified files before applying modifications."""
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now()
        backup_dir = self.backup_path / f"backup_{backup_id}"
        backup_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating backup {backup_id} for {len(files_to_backup)} files")
        
        backed_up_files = []
        total_size = 0
        
        try:
            for file_path in files_to_backup:
                source_path = Path(file_path)
                if source_path.exists():
                    # Create directory structure in backup
                    backup_file_path = backup_dir / source_path.name
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(source_path, backup_file_path)
                    backed_up_files.append(str(source_path))
                    total_size += backup_file_path.stat().st_size
                else:
                    self.logger.warning(f"File not found for backup: {file_path}")
            
            # Calculate checksum
            checksum = self._calculate_backup_checksum(backup_dir)
            
            # Create backup info
            backup_info = BackupInfo(
                id=backup_id,
                timestamp=timestamp,
                backup_path=str(backup_dir),
                original_files=backed_up_files,
                backup_size=total_size,
                checksum=checksum,
                metadata={"description": description}
            )
            
            # Store in database
            self._store_backup_info(backup_info)
            
            self.logger.info(f"Backup {backup_id} created successfully ({total_size} bytes)")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Failed to create backup {backup_id}: {e}")
            # Clean up partial backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise
    
    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for backup verification."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(backup_dir.rglob("*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                hasher.update(file_path.read_bytes())
        
        return hasher.hexdigest()
    
    def _store_backup_info(self, backup_info: BackupInfo):
        """Store backup information in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO backups 
                (id, timestamp, backup_path, original_files, backup_size, checksum, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                backup_info.id,
                backup_info.timestamp.isoformat(),
                backup_info.backup_path,
                json.dumps(backup_info.original_files),
                backup_info.backup_size,
                backup_info.checksum,
                json.dumps(backup_info.metadata)
            ))
    
    def create_version_snapshot(self, 
                              files: Dict[str, str], 
                              description: str,
                              modification_ids: List[str] = None,
                              performance_metrics: PerformanceMetrics = None,
                              tags: Set[str] = None) -> SystemVersion:
        """Create a complete system version snapshot."""
        version_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        self.logger.info(f"Creating version snapshot {version_id}: {description}")
        
        # Create file versions
        file_versions = []
        for file_path, content in files.items():
            file_version = FileVersion(
                id=str(uuid.uuid4()),
                file_path=file_path,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                timestamp=timestamp,
                status=VersionStatus.ACTIVE
            )
            file_versions.append(file_version)
        
        # Create system version
        system_version = SystemVersion(
            id=version_id,
            timestamp=timestamp,
            description=description,
            file_versions=file_versions,
            performance_metrics=performance_metrics,
            modification_ids=modification_ids or [],
            status=VersionStatus.ACTIVE,
            tags=tags or set()
        )
        
        # Store in database
        self._store_system_version(system_version)
        
        # Update current version
        self.current_version_id = version_id
        self.version_cache[version_id] = system_version
        
        self.logger.info(f"Version snapshot {version_id} created with {len(file_versions)} files")
        return system_version
    
    def _store_system_version(self, version: SystemVersion):
        """Store system version in database."""
        with sqlite3.connect(self.db_path) as conn:
            # Store system version
            performance_metrics_json = None
            if version.performance_metrics:
                # Convert datetime to ISO string for JSON serialization
                metrics_dict = asdict(version.performance_metrics)
                if 'timestamp' in metrics_dict:
                    metrics_dict['timestamp'] = metrics_dict['timestamp'].isoformat()
                performance_metrics_json = json.dumps(metrics_dict)
            
            conn.execute("""
                INSERT INTO system_versions 
                (id, timestamp, description, status, performance_metrics, modification_ids, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.id,
                version.timestamp.isoformat(),
                version.description,
                version.status,
                performance_metrics_json,
                json.dumps(version.modification_ids),
                json.dumps(list(version.tags)),
                json.dumps(version.metadata)
            ))
            
            # Store file versions
            for file_version in version.file_versions:
                conn.execute("""
                    INSERT INTO file_versions 
                    (id, system_version_id, file_path, content, content_hash, timestamp, 
                     modification_id, parent_version_id, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_version.id,
                    version.id,
                    file_version.file_path,
                    file_version.content,
                    file_version.content_hash,
                    file_version.timestamp.isoformat(),
                    file_version.modification_id,
                    file_version.parent_version_id,
                    file_version.status,
                    json.dumps(file_version.metadata)
                ))
    
    def apply_modification_with_backup(self, 
                                     modification: CodeModification,
                                     current_files: Dict[str, str]) -> Tuple[bool, Optional[BackupInfo]]:
        """Apply a modification with automatic backup creation."""
        self.logger.info(f"Applying modification {modification.id} with backup")
        
        try:
            # Create backup first
            files_to_backup = [modification.file_path]
            backup_info = self.create_backup(
                files_to_backup, 
                f"Backup before applying modification {modification.id}"
            )
            
            # Create version snapshot before modification
            pre_modification_version = self.create_version_snapshot(
                current_files,
                f"Pre-modification snapshot for {modification.id}",
                tags={"pre_modification", "backup"}
            )
            
            # Apply the modification (in a real system, this would modify actual files)
            # For now, we'll simulate by updating the content
            modified_files = current_files.copy()
            if modification.file_path in modified_files:
                modified_files[modification.file_path] = modification.modified_code
            
            # Create post-modification version
            post_modification_version = self.create_version_snapshot(
                modified_files,
                f"Post-modification snapshot for {modification.id}",
                modification_ids=[modification.id],
                tags={"post_modification", "applied"}
            )
            
            self.logger.info(f"Modification {modification.id} applied successfully")
            return True, backup_info
            
        except Exception as e:
            self.logger.error(f"Failed to apply modification {modification.id}: {e}")
            return False, None
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback system to a specific version."""
        self.logger.info(f"Rolling back to version {version_id}")
        
        try:
            # Load target version
            target_version = self.get_version(version_id)
            if not target_version:
                self.logger.error(f"Version {version_id} not found")
                return False
            
            # Create backup of current state
            current_files = self._get_current_file_contents()
            backup_info = self.create_backup(
                list(current_files.keys()),
                f"Backup before rollback to {version_id}"
            )
            
            # Create rollback version snapshot
            rollback_version = self.create_version_snapshot(
                {fv.file_path: fv.content for fv in target_version.file_versions},
                f"Rollback to version {version_id}",
                tags={"rollback", "restored"}
            )
            
            # Update version status
            self._update_version_status(version_id, VersionStatus.ACTIVE)
            if self.current_version_id:
                self._update_version_status(self.current_version_id, VersionStatus.ROLLED_BACK)
            
            self.current_version_id = rollback_version.id
            
            self.logger.info(f"Successfully rolled back to version {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def _get_current_file_contents(self) -> Dict[str, str]:
        """Get current file contents (simulated for now)."""
        # In a real implementation, this would read actual files
        return {
            "darwin_godel_model.py": "# Current content of darwin_godel_model.py",
            "version_manager.py": "# Current content of version_manager.py"
        }
    
    def _update_version_status(self, version_id: str, status: VersionStatus):
        """Update the status of a version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE system_versions SET status = ? WHERE id = ?
            """, (status, version_id))
    
    def get_version(self, version_id: str) -> Optional[SystemVersion]:
        """Get a specific version by ID."""
        if version_id in self.version_cache:
            return self.version_cache[version_id]
        
        with sqlite3.connect(self.db_path) as conn:
            # Get system version
            cursor = conn.execute("""
                SELECT id, timestamp, description, status, performance_metrics, 
                       modification_ids, tags, metadata
                FROM system_versions WHERE id = ?
            """, (version_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse system version data
            performance_metrics = None
            if row[4]:
                metrics_data = json.loads(row[4])
                # Convert ISO string back to datetime if present
                if 'timestamp' in metrics_data and isinstance(metrics_data['timestamp'], str):
                    metrics_data['timestamp'] = datetime.fromisoformat(metrics_data['timestamp'])
                performance_metrics = PerformanceMetrics(**metrics_data)
            
            # Get file versions
            file_cursor = conn.execute("""
                SELECT id, file_path, content, content_hash, timestamp, 
                       modification_id, parent_version_id, status, metadata
                FROM file_versions WHERE system_version_id = ?
            """, (version_id,))
            
            file_versions = []
            for file_row in file_cursor.fetchall():
                file_version = FileVersion(
                    id=file_row[0],
                    file_path=file_row[1],
                    content=file_row[2],
                    content_hash=file_row[3],
                    timestamp=datetime.fromisoformat(file_row[4]),
                    modification_id=file_row[5],
                    parent_version_id=file_row[6],
                    status=file_row[7],
                    metadata=json.loads(file_row[8]) if file_row[8] else {}
                )
                file_versions.append(file_version)
            
            # Create system version
            system_version = SystemVersion(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                description=row[2],
                file_versions=file_versions,
                performance_metrics=performance_metrics,
                modification_ids=json.loads(row[5]),
                status=row[3],
                tags=set(json.loads(row[6])),
                metadata=json.loads(row[7]) if row[7] else {}
            )
            
            # Cache the version
            self.version_cache[version_id] = system_version
            return system_version
    
    def get_version_history(self, limit: int = 50) -> List[SystemVersion]:
        """Get version history ordered by timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id FROM system_versions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            version_ids = [row[0] for row in cursor.fetchall()]
            
        return [self.get_version(vid) for vid in version_ids if self.get_version(vid)]
    
    def compare_versions(self, from_version_id: str, to_version_id: str) -> Optional[VersionDiff]:
        """Compare two versions and generate a diff."""
        from_version = self.get_version(from_version_id)
        to_version = self.get_version(to_version_id)
        
        if not from_version or not to_version:
            return None
        
        # Create file maps
        from_files = {fv.file_path: fv for fv in from_version.file_versions}
        to_files = {fv.file_path: fv for fv in to_version.file_versions}
        
        # Find changes
        added_files = list(set(to_files.keys()) - set(from_files.keys()))
        removed_files = list(set(from_files.keys()) - set(to_files.keys()))
        
        modified_files = []
        file_changes = []
        
        for file_path in set(from_files.keys()) & set(to_files.keys()):
            from_file = from_files[file_path]
            to_file = to_files[file_path]
            
            if from_file.content_hash != to_file.content_hash:
                modified_files.append(file_path)
                file_changes.append({
                    "file_path": file_path,
                    "from_hash": from_file.content_hash,
                    "to_hash": to_file.content_hash,
                    "from_size": len(from_file.content),
                    "to_size": len(to_file.content)
                })
        
        # Calculate performance delta
        performance_delta = None
        if from_version.performance_metrics and to_version.performance_metrics:
            performance_delta = {
                "response_time": to_version.performance_metrics.response_time - from_version.performance_metrics.response_time,
                "accuracy_score": to_version.performance_metrics.accuracy_score - from_version.performance_metrics.accuracy_score,
                "memory_usage": to_version.performance_metrics.memory_usage - from_version.performance_metrics.memory_usage,
                "cpu_usage": to_version.performance_metrics.cpu_usage - from_version.performance_metrics.cpu_usage,
                "error_rate": to_version.performance_metrics.error_rate - from_version.performance_metrics.error_rate,
                "user_satisfaction": to_version.performance_metrics.user_satisfaction - from_version.performance_metrics.user_satisfaction
            }
        
        return VersionDiff(
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            file_changes=file_changes,
            added_files=added_files,
            removed_files=removed_files,
            modified_files=modified_files,
            performance_delta=performance_delta
        )
    
    def restore_from_backup(self, backup_id: str) -> bool:
        """Restore files from a backup."""
        self.logger.info(f"Restoring from backup {backup_id}")
        
        try:
            # Get backup info
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                self.logger.error(f"Backup {backup_id} not found")
                return False
            
            backup_dir = Path(backup_info.backup_path)
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Verify backup integrity
            current_checksum = self._calculate_backup_checksum(backup_dir)
            if current_checksum != backup_info.checksum:
                self.logger.error(f"Backup {backup_id} integrity check failed")
                return False
            
            # Restore files (simulated for now)
            restored_files = {}
            for backup_file in backup_dir.rglob("*"):
                if backup_file.is_file():
                    content = backup_file.read_text()
                    restored_files[str(backup_file.name)] = content
            
            # Create restoration version
            restore_version = self.create_version_snapshot(
                restored_files,
                f"Restored from backup {backup_id}",
                tags={"restore", "backup_restore"}
            )
            
            self.logger.info(f"Successfully restored from backup {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_id}: {e}")
            return False
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup information by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, backup_path, original_files, backup_size, checksum, metadata
                FROM backups WHERE id = ?
            """, (backup_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return BackupInfo(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                backup_path=row[2],
                original_files=json.loads(row[3]),
                backup_size=row[4],
                checksum=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            )
    
    def cleanup_old_versions(self, keep_count: int = 100):
        """Clean up old versions and backups to save space."""
        self.logger.info(f"Cleaning up old versions, keeping {keep_count} most recent")
        
        with sqlite3.connect(self.db_path) as conn:
            # Get versions to delete (SQLite compatible syntax)
            cursor = conn.execute("""
                SELECT id FROM system_versions 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ? 
            """, (keep_count,))
            
            versions_to_delete = [row[0] for row in cursor.fetchall()]
            
            # Delete old versions
            for version_id in versions_to_delete:
                conn.execute("DELETE FROM file_versions WHERE system_version_id = ?", (version_id,))
                conn.execute("DELETE FROM system_versions WHERE id = ?", (version_id,))
                
                # Remove from cache
                if version_id in self.version_cache:
                    del self.version_cache[version_id]
            
            # Clean up old backups (keep last 50)
            cursor = conn.execute("""
                SELECT id, backup_path FROM backups 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET 50
            """)
            
            backups_to_delete = cursor.fetchall()
            for backup_id, backup_path in backups_to_delete:
                # Remove backup directory
                backup_dir = Path(backup_path)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                
                # Remove from database
                conn.execute("DELETE FROM backups WHERE id = ?", (backup_id,))
            
            self.logger.info(f"Cleaned up {len(versions_to_delete)} versions and {len(backups_to_delete)} backups")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version management statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count versions
            version_count = conn.execute("SELECT COUNT(*) FROM system_versions").fetchone()[0]
            
            # Count backups
            backup_count = conn.execute("SELECT COUNT(*) FROM backups").fetchone()[0]
            
            # Calculate total backup size
            total_backup_size = conn.execute("SELECT SUM(backup_size) FROM backups").fetchone()[0] or 0
            
            # Get version status distribution
            status_cursor = conn.execute("""
                SELECT status, COUNT(*) FROM system_versions GROUP BY status
            """)
            status_distribution = dict(status_cursor.fetchall())
            
            # Get recent activity
            recent_versions = conn.execute("""
                SELECT COUNT(*) FROM system_versions 
                WHERE timestamp > datetime('now', '-7 days')
            """).fetchone()[0]
            
        return {
            "total_versions": version_count,
            "total_backups": backup_count,
            "total_backup_size_bytes": total_backup_size,
            "status_distribution": status_distribution,
            "recent_versions_7_days": recent_versions,
            "current_version_id": self.current_version_id,
            "cache_size": len(self.version_cache)
        }