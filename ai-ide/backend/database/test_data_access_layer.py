"""
Test suite for data access layer and caching functionality.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from datetime import datetime, timedelta

from .repositories.base_repository import BaseRepository, QueryFilter, QueryOptions
from .repositories.agent_repository import AgentInteraction, ReasoningTrace, agent_interaction_repo, reasoning_trace_repo
from .repositories.search_repository import CodeEmbedding, WebSearchCache, code_embedding_repo, web_search_cache_repo
from .caching import CacheManager, CacheConfig, cache_manager
from .data_sync import DataSyncManager, SyncEvent, SyncEventType, data_sync_manager
from .data_management import DataCleanupManager, DataExportImportManager, cleanup_manager

class TestBaseRepository:
    """Test base repository functionality."""
    
    class TestEntity:
        def __init__(self, id=None, name="", value=0):
            self.id = id
            self.name = name
            self.value = value
    
    class TestRepository(BaseRepository):
        def __init__(self):
            super().__init__("test_table")
        
        def _row_to_entity(self, row):
            return TestBaseRepository.TestEntity(
                id=row.get('id'),
                name=row.get('name', ''),
                value=row.get('value', 0)
            )
        
        def _entity_to_dict(self, entity):
            return {
                'id': entity.id,
                'name': entity.name,
                'value': entity.value
            }
    
    @pytest.fixture
    def test_repo(self):
        return self.TestRepository()
    
    def test_build_where_clause(self, test_repo):
        """Test WHERE clause building."""
        filters = [
            QueryFilter("name", "=", "test"),
            QueryFilter("value", ">", 10),
            QueryFilter("status", "IN", ["active", "pending"])
        ]
        
        where_clause, params = test_repo._build_where_clause(filters)
        
        assert "WHERE" in where_clause
        assert "name = $1" in where_clause
        assert "value > $2" in where_clause
        assert "status IN ($3,$4)" in where_clause
        assert params == ["test", 10, "active", "pending"]
    
    def test_build_order_clause(self, test_repo):
        """Test ORDER BY clause building."""
        order_clause = test_repo._build_order_clause("name", "DESC")
        assert order_clause == " ORDER BY name DESC"
        
        order_clause = test_repo._build_order_clause(None, "ASC")
        assert order_clause == ""
    
    def test_build_limit_clause(self, test_repo):
        """Test LIMIT and OFFSET clause building."""
        limit_clause, params = test_repo._build_limit_clause(10, 20)
        assert "LIMIT $1" in limit_clause
        assert "OFFSET $2" in limit_clause
        assert params == [10, 20]
    
    @pytest.mark.asyncio
    async def test_create_entity(self, test_repo):
        """Test entity creation."""
        entity = self.TestEntity(name="test", value=42)
        
        with patch('ai_ide.backend.database.connection.db_manager.fetchval') as mock_fetchval:
            mock_fetchval.return_value = "test-id"
            
            entity_id = await test_repo.create(entity)
            
            assert entity_id == "test-id"
            mock_fetchval.assert_called_once()

class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_config(self):
        return CacheConfig(host="localhost", port=6379, db=1)
    
    @pytest.fixture
    def cache_mgr(self, cache_config):
        return CacheManager(cache_config)
    
    def test_make_key(self, cache_mgr):
        """Test cache key generation."""
        key = cache_mgr._make_key('embedding', 'test_file.py')
        assert key == 'emb:test_file.py'
        
        key = cache_mgr._make_key('search', 'query_hash')
        assert key == 'search:query_hash'
    
    def test_hash_key(self, cache_mgr):
        """Test key hashing."""
        hash1 = cache_mgr._hash_key({'query': 'test', 'context': 'code'})
        hash2 = cache_mgr._hash_key({'context': 'code', 'query': 'test'})  # Different order
        
        assert hash1 == hash2  # Should be same due to sorted keys
        assert len(hash1) == 32  # MD5 hash length
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache_mgr):
        """Test basic cache operations."""
        with patch.object(cache_mgr, '_redis') as mock_redis:
            mock_redis.setex = AsyncMock()
            mock_redis.get = AsyncMock(return_value=b'{"test": "data"}')
            mock_redis.delete = AsyncMock(return_value=1)
            mock_redis.exists = AsyncMock(return_value=1)
            
            # Test set
            result = await cache_mgr.set('test', 'key1', {'test': 'data'})
            assert result is True
            mock_redis.setex.assert_called_once()
            
            # Test get
            data = await cache_mgr.get('test', 'key1')
            assert data == {'test': 'data'}
            
            # Test delete
            result = await cache_mgr.delete('test', 'key1')
            assert result is True
            
            # Test exists
            exists = await cache_mgr.exists('test', 'key1')
            assert exists is True
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, cache_mgr):
        """Test embedding-specific cache operations."""
        embedding = np.random.rand(1536)
        
        with patch.object(cache_mgr, 'set') as mock_set, \
             patch.object(cache_mgr, 'get') as mock_get:
            
            mock_set.return_value = True
            mock_get.return_value = embedding
            
            # Test cache embedding
            result = await cache_mgr.cache_embedding('test.py', 'hash123', embedding)
            assert result is True
            mock_set.assert_called_with('embedding', 'test.py:hash123', embedding, None)
            
            # Test get embedding
            cached_embedding = await cache_mgr.get_embedding('test.py', 'hash123')
            assert np.array_equal(cached_embedding, embedding)

class TestAgentRepository:
    """Test agent interaction repository."""
    
    @pytest.fixture
    def sample_interaction(self):
        return AgentInteraction(
            session_id="session-123",
            agent_type="code_agent",
            interaction_type="completion",
            input_data={"query": "test"},
            output_data={"result": "success"},
            duration_ms=150
        )
    
    @pytest.fixture
    def sample_trace(self):
        return ReasoningTrace(
            interaction_id="interaction-123",
            step_number=1,
            reasoning_type="chain_of_thought",
            thought_process="Analyzing the problem...",
            confidence_score=0.8
        )
    
    def test_interaction_entity_conversion(self, sample_interaction):
        """Test interaction entity to dict conversion."""
        entity_dict = agent_interaction_repo._entity_to_dict(sample_interaction)
        
        assert entity_dict['session_id'] == "session-123"
        assert entity_dict['agent_type'] == "code_agent"
        assert entity_dict['input_data'] == {"query": "test"}
        assert entity_dict['duration_ms'] == 150
    
    def test_trace_entity_conversion(self, sample_trace):
        """Test reasoning trace entity conversion."""
        entity_dict = reasoning_trace_repo._entity_to_dict(sample_trace)
        
        assert entity_dict['interaction_id'] == "interaction-123"
        assert entity_dict['step_number'] == 1
        assert entity_dict['reasoning_type'] == "chain_of_thought"
        assert entity_dict['confidence_score'] == 0.8
    
    @pytest.mark.asyncio
    async def test_find_by_session(self, sample_interaction):
        """Test finding interactions by session."""
        with patch.object(agent_interaction_repo, 'find_all') as mock_find_all:
            mock_find_all.return_value = [sample_interaction]
            
            interactions = await agent_interaction_repo.find_by_session("session-123")
            
            assert len(interactions) == 1
            assert interactions[0].session_id == "session-123"
            mock_find_all.assert_called_once()

class TestSearchRepository:
    """Test search-related repositories."""
    
    @pytest.fixture
    def sample_embedding(self):
        return CodeEmbedding(
            file_path="test.py",
            content_hash="abc123",
            embedding=np.random.rand(1536),
            metadata={"language": "python"}
        )
    
    @pytest.fixture
    def sample_web_cache(self):
        return WebSearchCache(
            query_text="python async programming",
            search_engine="google",
            results={"items": [{"title": "Test", "url": "http://test.com"}]},
            expires_at=datetime.now() + timedelta(hours=4)
        )
    
    def test_embedding_entity_conversion(self, sample_embedding):
        """Test code embedding entity conversion."""
        entity_dict = code_embedding_repo._entity_to_dict(sample_embedding)
        
        assert entity_dict['file_path'] == "test.py"
        assert entity_dict['content_hash'] == "abc123"
        assert isinstance(entity_dict['embedding'], list)  # Should be converted to list
        assert entity_dict['metadata'] == {"language": "python"}
    
    def test_web_cache_entity_conversion(self, sample_web_cache):
        """Test web search cache entity conversion."""
        entity_dict = web_search_cache_repo._entity_to_dict(sample_web_cache)
        
        assert entity_dict['query_text'] == "python async programming"
        assert entity_dict['search_engine'] == "google"
        assert "items" in entity_dict['results']
    
    @pytest.mark.asyncio
    async def test_upsert_embedding(self, sample_embedding):
        """Test embedding upsert operation."""
        with patch('ai_ide.backend.database.connection.db_manager.fetchval') as mock_fetchval, \
             patch.object(cache_manager, 'cache_embedding') as mock_cache:
            
            mock_fetchval.return_value = "embedding-id"
            mock_cache.return_value = True
            
            entity_id = await code_embedding_repo.upsert_embedding(
                "test.py", "abc123", sample_embedding.embedding, {"language": "python"}
            )
            
            assert entity_id == "embedding-id"
            mock_fetchval.assert_called_once()
            mock_cache.assert_called_once()

class TestDataSyncManager:
    """Test data synchronization manager."""
    
    @pytest.fixture
    def sync_mgr(self):
        return DataSyncManager()
    
    @pytest.fixture
    def sample_sync_event(self):
        return SyncEvent(
            id="event-123",
            event_type=SyncEventType.CREATE,
            table_name="test_table",
            entity_id="entity-123",
            data={"field": "value"},
            timestamp=datetime.now(),
            source="backend"
        )
    
    def test_sync_event_serialization(self, sample_sync_event):
        """Test sync event serialization."""
        event_dict = sample_sync_event.to_dict()
        
        assert event_dict['id'] == "event-123"
        assert event_dict['event_type'] == "create"
        assert event_dict['table_name'] == "test_table"
        
        # Test deserialization
        restored_event = SyncEvent.from_dict(event_dict)
        assert restored_event.id == sample_sync_event.id
        assert restored_event.event_type == sample_sync_event.event_type
    
    @pytest.mark.asyncio
    async def test_queue_sync_event(self, sync_mgr, sample_sync_event):
        """Test queuing sync events."""
        with patch.object(cache_manager, 'set') as mock_set, \
             patch.object(cache_manager, '_redis') as mock_redis:
            
            mock_set.return_value = True
            mock_redis.lpush = AsyncMock(return_value=1)
            
            event_id = await sync_mgr.queue_sync_event(
                SyncEventType.CREATE, "test_table", "entity-123", {"field": "value"}
            )
            
            assert event_id is not None
            mock_set.assert_called_once()
            mock_redis.lpush.assert_called_once()
    
    def test_register_sync_handler(self, sync_mgr):
        """Test registering sync handlers."""
        def test_handler(event):
            return True
        
        sync_mgr.register_sync_handler("test_table", test_handler)
        
        assert "test_table" in sync_mgr.sync_handlers
        assert sync_mgr.sync_handlers["test_table"] == test_handler

class TestDataCleanupManager:
    """Test data cleanup and management."""
    
    @pytest.fixture
    def temp_archive_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def cleanup_mgr(self, temp_archive_dir):
        return DataCleanupManager(temp_archive_dir)
    
    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, cleanup_mgr):
        """Test cleanup dry run."""
        with patch('ai_ide.backend.database.connection.db_manager.fetchval') as mock_fetchval:
            mock_fetchval.return_value = 100  # 100 records to clean
            
            results = await cleanup_mgr.run_cleanup("agent_interactions", dry_run=True)
            
            assert "agent_interactions" in results
            assert results["agent_interactions"]["records_found"] == 100
            assert results["agent_interactions"]["records_deleted"] == 0
            assert results["agent_interactions"]["dry_run"] is True
    
    def test_cleanup_policy_configuration(self, cleanup_mgr):
        """Test cleanup policy configuration."""
        from .data_management import CleanupPolicy
        
        policy = CleanupPolicy(
            table_name="custom_table",
            retention_days=30,
            archive_before_delete=False
        )
        
        cleanup_mgr.set_cleanup_policy("custom_table", policy)
        
        assert "custom_table" in cleanup_mgr.cleanup_policies
        assert cleanup_mgr.cleanup_policies["custom_table"].retention_days == 30

class TestDataExportImport:
    """Test data export and import functionality."""
    
    @pytest.fixture
    def temp_export_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def export_mgr(self, temp_export_dir):
        return DataExportImportManager(temp_export_dir)
    
    @pytest.mark.asyncio
    async def test_export_table_data(self, export_mgr):
        """Test table data export."""
        mock_records = [
            {'id': '1', 'name': 'test1', 'created_at': datetime.now()},
            {'id': '2', 'name': 'test2', 'created_at': datetime.now()}
        ]
        
        with patch('ai_ide.backend.database.connection.db_manager.fetch') as mock_fetch:
            mock_fetch.return_value = mock_records
            
            filepath = await export_mgr.export_table_data("test_table", format='json', compress=False)
            
            assert os.path.exists(filepath)
            assert filepath.endswith('.json')
            
            # Verify file content
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert data['table_name'] == 'test_table'
                assert data['record_count'] == 2
                assert len(data['records']) == 2
    
    @pytest.mark.asyncio
    async def test_import_table_data(self, export_mgr, temp_export_dir):
        """Test table data import."""
        # Create test export file
        test_data = {
            'table_name': 'test_table',
            'record_count': 2,
            'records': [
                {'id': '1', 'name': 'test1'},
                {'id': '2', 'name': 'test2'}
            ]
        }
        
        test_file = os.path.join(temp_export_dir, 'test_import.json')
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('ai_ide.backend.database.connection.db_manager.execute') as mock_execute, \
             patch('ai_ide.backend.database.connection.db_manager.fetchval') as mock_fetchval:
            
            mock_fetchval.return_value = None  # No existing records
            mock_execute.return_value = "INSERT 0 1"
            
            result = await export_mgr.import_table_data(test_file, merge_strategy='skip')
            
            assert result['imported'] == 2
            assert result['skipped'] == 0
            assert result['errors'] == 0

if __name__ == "__main__":
    pytest.main([__file__])