"""
Tests for Version Management and Rollback System
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from version_manager import (
    VersionManager,
    FileVersion,
    SystemVersion,
    VersionDiff,
    BackupInfo,
    VersionStatus
)
from darwin_godel_model import (
    CodeModification,
    ImprovementType,
    PerformanceMetrics
)


class TestVersionManager:
    """Test the version manager."""
    
    def setup_method(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "versions"
        self.backup_path = Path(self.temp_dir) / "backups"
        
        self.version_manager = VersionManager(
            storage_path=str(self.storage_path),
            backup_path=str(self.backup_path)
        )
    
    def teardown_method(self):
        # Close version manager to release database connections
        if hasattr(self, 'version_manager'):
            self.version_manager.close()
        
        # Clean up temporary directories
        if Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, sometimes files are still locked
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    # If still locked, just pass - temp files will be cleaned up eventually
                    pass
    
    def test_initialization(self):
        """Test version manager initialization."""
        assert self.storage_path.exists()
        assert self.backup_path.exists()
        assert (self.storage_path / "versions.db").exists()
        assert self.version_manager.current_version_id is None
        assert len(self.version_manager.version_cache) == 0
    
    def test_create_backup(self):
        """Test creating a backup of files."""
        # Create test files
        test_file1 = Path(self.temp_dir) / "test1.py"
        test_file2 = Path(self.temp_dir) / "test2.py"
        
        test_file1.write_text("print('test1')")
        test_file2.write_text("print('test2')")
        
        # Create backup
        files_to_backup = [str(test_file1), str(test_file2)]
        backup_info = self.version_manager.create_backup(files_to_backup, "Test backup")
        
        assert backup_info.id is not None
        assert len(backup_info.original_files) == 2
        assert backup_info.backup_size > 0
        assert backup_info.checksum is not None
        
        # Verify backup directory exists
        backup_dir = Path(backup_info.backup_path)
        assert backup_dir.exists()
        assert (backup_dir / "test1.py").exists()
        assert (backup_dir / "test2.py").exists()
    
    def test_create_backup_nonexistent_file(self):
        """Test creating backup with non-existent files."""
        files_to_backup = ["nonexistent.py"]
        backup_info = self.version_manager.create_backup(files_to_backup, "Test backup")
        
        # Should still create backup but with empty file list
        assert backup_info.id is not None
        assert len(backup_info.original_files) == 0
    
    def test_create_version_snapshot(self):
        """Test creating a version snapshot."""
        files = {
            "file1.py": "print('hello')",
            "file2.py": "print('world')"
        }
        
        performance_metrics = PerformanceMetrics(
            response_time=0.5,
            accuracy_score=0.9,
            memory_usage=100,
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        
        version = self.version_manager.create_version_snapshot(
            files=files,
            description="Test snapshot",
            modification_ids=["mod1", "mod2"],
            performance_metrics=performance_metrics,
            tags={"test", "snapshot"}
        )
        
        assert version.id is not None
        assert version.description == "Test snapshot"
        assert len(version.file_versions) == 2
        assert version.performance_metrics is not None
        assert version.modification_ids == ["mod1", "mod2"]
        assert "test" in version.tags
        assert version.status == VersionStatus.ACTIVE
        
        # Should update current version
        assert self.version_manager.current_version_id == version.id
    
    def test_apply_modification_with_backup(self):
        """Test applying a modification with automatic backup."""
        # Create test modification
        modification = CodeModification(
            id="test-mod-1",
            opportunity_id="opp-1",
            original_code="print('original')",
            modified_code="print('modified')",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.CODE_QUALITY,
            rationale="Test modification",
            estimated_impact=0.3,
            safety_score=0.9
        )
        
        current_files = {
            "test.py": "print('original')",
            "other.py": "print('other')"
        }
        
        # Mock file system operations
        with patch.object(self.version_manager, '_get_current_file_contents', return_value=current_files):
            success, backup_info = self.version_manager.apply_modification_with_backup(
                modification, current_files
            )
        
        assert success is True
        assert backup_info is not None
        assert backup_info.id is not None
        
        # Should have created two versions (pre and post modification)
        history = self.version_manager.get_version_history()
        assert len(history) >= 2
    
    def test_rollback_to_version(self):
        """Test rolling back to a previous version."""
        # Create initial version
        initial_files = {"test.py": "print('initial')"}
        initial_version = self.version_manager.create_version_snapshot(
            initial_files, "Initial version"
        )
        
        # Create modified version
        modified_files = {"test.py": "print('modified')"}
        modified_version = self.version_manager.create_version_snapshot(
            modified_files, "Modified version"
        )
        
        # Mock file system operations
        with patch.object(self.version_manager, '_get_current_file_contents', return_value=modified_files):
            success = self.version_manager.rollback_to_version(initial_version.id)
        
        assert success is True
        
        # Should have created a rollback version
        history = self.version_manager.get_version_history()
        rollback_version = history[0]  # Most recent
        assert "rollback" in rollback_version.tags
    
    def test_rollback_to_nonexistent_version(self):
        """Test rolling back to a non-existent version."""
        success = self.version_manager.rollback_to_version("nonexistent-id")
        assert success is False
    
    def test_get_version(self):
        """Test retrieving a specific version."""
        files = {"test.py": "print('test')"}
        created_version = self.version_manager.create_version_snapshot(files, "Test version")
        
        retrieved_version = self.version_manager.get_version(created_version.id)
        
        assert retrieved_version is not None
        assert retrieved_version.id == created_version.id
        assert retrieved_version.description == "Test version"
        assert len(retrieved_version.file_versions) == 1
        
        # Should be cached
        assert created_version.id in self.version_manager.version_cache
    
    def test_get_nonexistent_version(self):
        """Test retrieving a non-existent version."""
        version = self.version_manager.get_version("nonexistent-id")
        assert version is None
    
    def test_get_version_history(self):
        """Test getting version history."""
        # Create multiple versions
        for i in range(5):
            files = {f"test{i}.py": f"print('test{i}')"}
            self.version_manager.create_version_snapshot(files, f"Version {i}")
        
        history = self.version_manager.get_version_history(limit=3)
        
        assert len(history) == 3
        # Should be ordered by timestamp (most recent first)
        assert history[0].description == "Version 4"
        assert history[1].description == "Version 3"
        assert history[2].description == "Version 2"
    
    def test_compare_versions(self):
        """Test comparing two versions."""
        # Create first version
        files1 = {
            "file1.py": "print('version1')",
            "file2.py": "print('common')"
        }
        version1 = self.version_manager.create_version_snapshot(files1, "Version 1")
        
        # Create second version with changes
        files2 = {
            "file1.py": "print('version2')",  # Modified
            "file3.py": "print('new')"        # Added
            # file2.py removed
        }
        version2 = self.version_manager.create_version_snapshot(files2, "Version 2")
        
        diff = self.version_manager.compare_versions(version1.id, version2.id)
        
        assert diff is not None
        assert diff.from_version_id == version1.id
        assert diff.to_version_id == version2.id
        assert "file3.py" in diff.added_files
        assert "file2.py" in diff.removed_files
        assert "file1.py" in diff.modified_files
        assert len(diff.file_changes) == 1  # Only file1.py was modified
    
    def test_compare_versions_with_performance_metrics(self):
        """Test comparing versions with performance metrics."""
        metrics1 = PerformanceMetrics(
            response_time=0.5, accuracy_score=0.9, memory_usage=100,
            cpu_usage=0.3, error_rate=0.01, user_satisfaction=4.5
        )
        metrics2 = PerformanceMetrics(
            response_time=0.4, accuracy_score=0.95, memory_usage=90,
            cpu_usage=0.25, error_rate=0.005, user_satisfaction=4.7
        )
        
        files = {"test.py": "print('test')"}
        version1 = self.version_manager.create_version_snapshot(
            files, "Version 1", performance_metrics=metrics1
        )
        version2 = self.version_manager.create_version_snapshot(
            files, "Version 2", performance_metrics=metrics2
        )
        
        diff = self.version_manager.compare_versions(version1.id, version2.id)
        
        assert diff.performance_delta is not None
        assert abs(diff.performance_delta["response_time"] - (-0.1)) < 0.001  # Improved
        assert abs(diff.performance_delta["accuracy_score"] - 0.05) < 0.001  # Improved
        assert diff.performance_delta["memory_usage"] == -10  # Improved
    
    def test_restore_from_backup(self):
        """Test restoring from a backup."""
        # Create test files
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('original')")
        
        # Create backup
        backup_info = self.version_manager.create_backup([str(test_file)], "Test backup")
        
        # Mock file system operations for restore
        with patch.object(self.version_manager, '_get_current_file_contents', return_value={}):
            success = self.version_manager.restore_from_backup(backup_info.id)
        
        assert success is True
        
        # Should have created a restore version
        history = self.version_manager.get_version_history()
        restore_version = history[0]
        assert "restore" in restore_version.tags
    
    def test_restore_from_nonexistent_backup(self):
        """Test restoring from a non-existent backup."""
        success = self.version_manager.restore_from_backup("nonexistent-id")
        assert success is False
    
    def test_get_backup_info(self):
        """Test getting backup information."""
        # Create test file and backup
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('test')")
        
        created_backup = self.version_manager.create_backup([str(test_file)], "Test backup")
        retrieved_backup = self.version_manager.get_backup_info(created_backup.id)
        
        assert retrieved_backup is not None
        assert retrieved_backup.id == created_backup.id
        assert retrieved_backup.backup_size == created_backup.backup_size
        assert retrieved_backup.checksum == created_backup.checksum
    
    def test_cleanup_old_versions(self):
        """Test cleaning up old versions."""
        # Create many versions
        for i in range(10):
            files = {f"test{i}.py": f"print('test{i}')"}
            self.version_manager.create_version_snapshot(files, f"Version {i}")
        
        # Clean up, keeping only 5
        self.version_manager.cleanup_old_versions(keep_count=5)
        
        history = self.version_manager.get_version_history()
        assert len(history) == 5
        
        # Should keep the most recent ones
        assert history[0].description == "Version 9"
        assert history[4].description == "Version 5"
    
    def test_get_statistics(self):
        """Test getting version management statistics."""
        # Create some versions and backups
        files = {"test.py": "print('test')"}
        self.version_manager.create_version_snapshot(files, "Test version")
        
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('test')")
        self.version_manager.create_backup([str(test_file)], "Test backup")
        
        stats = self.version_manager.get_statistics()
        
        assert "total_versions" in stats
        assert "total_backups" in stats
        assert "total_backup_size_bytes" in stats
        assert "status_distribution" in stats
        assert "recent_versions_7_days" in stats
        assert "current_version_id" in stats
        assert "cache_size" in stats
        
        assert stats["total_versions"] >= 1
        assert stats["total_backups"] >= 1
        assert stats["current_version_id"] is not None


class TestFileVersion:
    """Test the FileVersion dataclass."""
    
    def test_file_version_creation(self):
        """Test creating a file version."""
        file_version = FileVersion(
            id="test-id",
            file_path="test.py",
            content="print('test')",
            content_hash="abc123",
            timestamp=datetime.now(),
            modification_id="mod-1",
            status=VersionStatus.ACTIVE
        )
        
        assert file_version.id == "test-id"
        assert file_version.file_path == "test.py"
        assert file_version.content == "print('test')"
        assert file_version.status == VersionStatus.ACTIVE


class TestSystemVersion:
    """Test the SystemVersion dataclass."""
    
    def test_system_version_creation(self):
        """Test creating a system version."""
        file_version = FileVersion(
            id="file-1",
            file_path="test.py",
            content="print('test')",
            content_hash="abc123",
            timestamp=datetime.now()
        )
        
        system_version = SystemVersion(
            id="sys-1",
            timestamp=datetime.now(),
            description="Test version",
            file_versions=[file_version],
            modification_ids=["mod-1"],
            tags={"test", "example"}
        )
        
        assert system_version.id == "sys-1"
        assert system_version.description == "Test version"
        assert len(system_version.file_versions) == 1
        assert "test" in system_version.tags
        assert system_version.modification_ids == ["mod-1"]


class TestVersionDiff:
    """Test the VersionDiff dataclass."""
    
    def test_version_diff_creation(self):
        """Test creating a version diff."""
        diff = VersionDiff(
            from_version_id="v1",
            to_version_id="v2",
            file_changes=[{"file": "test.py", "change": "modified"}],
            added_files=["new.py"],
            removed_files=["old.py"],
            modified_files=["test.py"],
            performance_delta={"response_time": -0.1}
        )
        
        assert diff.from_version_id == "v1"
        assert diff.to_version_id == "v2"
        assert len(diff.file_changes) == 1
        assert "new.py" in diff.added_files
        assert "old.py" in diff.removed_files
        assert "test.py" in diff.modified_files
        assert diff.performance_delta["response_time"] == -0.1


class TestIntegration:
    """Integration tests for version management."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = VersionManager(
            storage_path=str(Path(self.temp_dir) / "versions"),
            backup_path=str(Path(self.temp_dir) / "backups")
        )
    
    def teardown_method(self):
        # Close version manager to release database connections
        if hasattr(self, 'version_manager'):
            self.version_manager.close()
            
        if Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, sometimes files are still locked
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    # If still locked, just pass - temp files will be cleaned up eventually
                    pass
    
    def test_complete_modification_cycle(self):
        """Test a complete modification cycle with backup and rollback."""
        # 1. Create initial version
        initial_files = {
            "main.py": "def main():\n    print('Hello, World!')",
            "utils.py": "def helper():\n    return 'help'"
        }
        
        initial_version = self.version_manager.create_version_snapshot(
            initial_files, "Initial implementation"
        )
        
        # 2. Apply modification with backup
        modification = CodeModification(
            id="improve-1",
            opportunity_id="opp-1",
            original_code="def main():\n    print('Hello, World!')",
            modified_code="def main():\n    print('Hello, Improved World!')",
            file_path="main.py",
            line_start=1,
            line_end=2,
            modification_type=ImprovementType.CODE_QUALITY,
            rationale="Improve greeting message",
            estimated_impact=0.1,
            safety_score=0.9
        )
        
        with patch.object(self.version_manager, '_get_current_file_contents', return_value=initial_files):
            success, backup_info = self.version_manager.apply_modification_with_backup(
                modification, initial_files
            )
        
        assert success is True
        assert backup_info is not None
        
        # 3. Verify version history
        history = self.version_manager.get_version_history()
        assert len(history) >= 2  # Pre and post modification versions
        
        # 4. Compare versions
        post_mod_version = history[0]  # Most recent
        pre_mod_version = None
        for version in history:
            if "pre_modification" in version.tags:
                pre_mod_version = version
                break
        
        assert pre_mod_version is not None
        diff = self.version_manager.compare_versions(pre_mod_version.id, post_mod_version.id)
        assert diff is not None
        assert "main.py" in diff.modified_files
        
        # 5. Rollback to initial version
        with patch.object(self.version_manager, '_get_current_file_contents', return_value={}):
            rollback_success = self.version_manager.rollback_to_version(initial_version.id)
        
        assert rollback_success is True
        
        # 6. Verify rollback created new version
        final_history = self.version_manager.get_version_history()
        rollback_version = final_history[0]
        assert "rollback" in rollback_version.tags
        
        # 7. Check statistics
        stats = self.version_manager.get_statistics()
        assert stats["total_versions"] >= 3
        assert stats["total_backups"] >= 1
    
    def test_performance_tracking_across_versions(self):
        """Test tracking performance metrics across versions."""
        # Create versions with different performance metrics
        metrics1 = PerformanceMetrics(
            response_time=1.0, accuracy_score=0.8, memory_usage=200,
            cpu_usage=0.5, error_rate=0.05, user_satisfaction=3.5
        )
        
        metrics2 = PerformanceMetrics(
            response_time=0.8, accuracy_score=0.85, memory_usage=180,
            cpu_usage=0.4, error_rate=0.03, user_satisfaction=4.0
        )
        
        metrics3 = PerformanceMetrics(
            response_time=0.6, accuracy_score=0.9, memory_usage=150,
            cpu_usage=0.3, error_rate=0.01, user_satisfaction=4.5
        )
        
        files = {"test.py": "print('test')"}
        
        v1 = self.version_manager.create_version_snapshot(
            files, "Version 1", performance_metrics=metrics1
        )
        v2 = self.version_manager.create_version_snapshot(
            files, "Version 2", performance_metrics=metrics2
        )
        v3 = self.version_manager.create_version_snapshot(
            files, "Version 3", performance_metrics=metrics3
        )
        
        # Compare performance improvements
        diff_1_to_2 = self.version_manager.compare_versions(v1.id, v2.id)
        diff_2_to_3 = self.version_manager.compare_versions(v2.id, v3.id)
        
        # Should show improvements
        assert diff_1_to_2.performance_delta["response_time"] < 0  # Faster
        assert diff_1_to_2.performance_delta["accuracy_score"] > 0  # More accurate
        
        assert diff_2_to_3.performance_delta["response_time"] < 0  # Even faster
        assert diff_2_to_3.performance_delta["user_satisfaction"] > 0  # Higher satisfaction


if __name__ == "__main__":
    pytest.main([__file__])