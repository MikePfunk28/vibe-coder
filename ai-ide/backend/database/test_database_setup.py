"""
Test suite for database setup, connections, and migrations.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

from .connection import DatabaseManager, DatabaseConfig, db_manager
from .migrations import MigrationManager, Migration
from .knowledge_graph import KnowledgeGraphManager, KnowledgeEntity, KnowledgeRelationship

class TestDatabaseConnection:
    """Test database connection management."""
    
    @pytest.fixture
    def db_config(self):
        """Test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_ai_ide",
            username="test_user",
            password="test_pass",
            min_connections=2,
            max_connections=5
        )
    
    @pytest.fixture
    def db_manager_instance(self, db_config):
        """Test database manager instance."""
        return DatabaseManager(db_config)
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'test-host',
            'DB_PORT': '5433',
            'DB_NAME': 'test-db',
            'DB_USER': 'test-user',
            'DB_PASSWORD': 'test-password',
            'DB_MIN_CONNECTIONS': '3',
            'DB_MAX_CONNECTIONS': '10'
        }):
            manager = DatabaseManager()
            config = manager.config
            
            assert config.host == 'test-host'
            assert config.port == 5433
            assert config.database == 'test-db'
            assert config.username == 'test-user'
            assert config.password == 'test-password'
            assert config.min_connections == 3
            assert config.max_connections == 10
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, db_manager_instance):
        """Test connection pool initialization."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = MagicMock()
            mock_create_pool.return_value = mock_pool
            
            # Mock connection for extension check
            mock_conn = MagicMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            await db_manager_instance.initialize()
            
            assert db_manager_instance._initialized
            mock_create_pool.assert_called_once()
            mock_conn.execute.assert_called_with("CREATE EXTENSION IF NOT EXISTS vector")
    
    @pytest.mark.asyncio
    async def test_connection_context_manager(self, db_manager_instance):
        """Test connection context manager."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_create_pool.return_value = mock_pool
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            await db_manager_instance.initialize()
            
            async with db_manager_instance.get_connection() as conn:
                assert conn == mock_conn
    
    @pytest.mark.asyncio
    async def test_transaction_context_manager(self, db_manager_instance):
        """Test transaction context manager."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_transaction = MagicMock()
            
            mock_create_pool.return_value = mock_pool
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.transaction.return_value = mock_transaction
            
            await db_manager_instance.initialize()
            
            async with db_manager_instance.transaction() as conn:
                assert conn == mock_conn
                mock_conn.transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, db_manager_instance):
        """Test database health check."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_create_pool.return_value = mock_pool
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.fetchval.return_value = 1
            
            await db_manager_instance.initialize()
            
            health = await db_manager_instance.health_check()
            assert health is True
            mock_conn.fetchval.assert_called_with("SELECT 1")

class TestMigrationManager:
    """Test database migration management."""
    
    @pytest.fixture
    def temp_migrations_dir(self):
        """Create temporary migrations directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def migration_manager(self, temp_migrations_dir):
        """Test migration manager instance."""
        return MigrationManager(temp_migrations_dir)
    
    def test_migration_parsing(self, migration_manager, temp_migrations_dir):
        """Test parsing migration files."""
        # Create test migration file
        migration_content = """-- Test migration
-- UP
CREATE TABLE test_table (id SERIAL PRIMARY KEY);

-- DOWN
DROP TABLE test_table;
"""
        migration_file = os.path.join(temp_migrations_dir, "001_test_migration.sql")
        with open(migration_file, 'w') as f:
            f.write(migration_content)
        
        # Reload migrations
        migration_manager._load_migrations()
        
        assert len(migration_manager.migrations) == 1
        migration = migration_manager.migrations[0]
        assert migration.version == "001"
        assert "CREATE TABLE test_table" in migration.up_sql
        assert "DROP TABLE test_table" in migration.down_sql
    
    @pytest.mark.asyncio
    async def test_apply_migration(self, migration_manager):
        """Test applying a migration."""
        migration = Migration(
            version="001",
            description="Test migration",
            up_sql="CREATE TABLE test_table (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE test_table;"
        )
        
        with patch.object(db_manager, 'transaction') as mock_transaction:
            mock_conn = MagicMock()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            
            result = await migration_manager.apply_migration(migration)
            
            assert result is True
            mock_conn.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_migration_status(self, migration_manager):
        """Test getting migration status."""
        with patch.object(db_manager, 'fetch') as mock_fetch:
            mock_fetch.return_value = [{'version': '001'}]
            
            # Add test migration
            migration_manager.migrations = [
                Migration(version="001", description="Test 1", up_sql=""),
                Migration(version="002", description="Test 2", up_sql="")
            ]
            
            status = await migration_manager.get_migration_status()
            
            assert status['current_version'] == '001'
            assert status['latest_version'] == '002'
            assert status['applied_migrations'] == 1
            assert status['pending_migrations'] == 1
            assert not status['up_to_date']
    
    def test_create_migration(self, migration_manager):
        """Test creating a new migration."""
        up_sql = "CREATE TABLE new_table (id SERIAL PRIMARY KEY);"
        down_sql = "DROP TABLE new_table;"
        
        filepath = migration_manager.create_migration("Add new table", up_sql, down_sql)
        
        assert os.path.exists(filepath)
        assert "001_add_new_table.sql" in filepath
        
        with open(filepath, 'r') as f:
            content = f.read()
            assert up_sql in content
            assert down_sql in content

class TestKnowledgeGraph:
    """Test knowledge graph management."""
    
    @pytest.fixture
    def kg_manager(self):
        """Test knowledge graph manager instance."""
        return KnowledgeGraphManager()
    
    @pytest.fixture
    def sample_entity(self):
        """Sample knowledge entity."""
        return KnowledgeEntity(
            entity_type="function",
            entity_name="calculate_similarity",
            properties={"language": "python", "complexity": "medium"},
            embedding=np.random.rand(1536)
        )
    
    @pytest.fixture
    def sample_relationship(self):
        """Sample knowledge relationship."""
        return KnowledgeRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="calls",
            properties={"frequency": "high"},
            confidence_score=0.9
        )
    
    @pytest.mark.asyncio
    async def test_create_entity(self, kg_manager, sample_entity):
        """Test creating a knowledge entity."""
        with patch.object(db_manager, 'fetchval') as mock_fetchval:
            mock_fetchval.return_value = "test-entity-id"
            
            entity_id = await kg_manager.create_entity(sample_entity)
            
            assert entity_id == "test-entity-id"
            mock_fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, kg_manager, sample_relationship):
        """Test creating a knowledge relationship."""
        with patch.object(db_manager, 'fetchval') as mock_fetchval:
            mock_fetchval.return_value = "test-relationship-id"
            
            relationship_id = await kg_manager.create_relationship(sample_relationship)
            
            assert relationship_id == "test-relationship-id"
            mock_fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_entity(self, kg_manager):
        """Test getting a knowledge entity."""
        mock_row = {
            'id': 'test-id',
            'entity_type': 'function',
            'entity_name': 'test_function',
            'properties': {'language': 'python'},
            'embedding': [0.1, 0.2, 0.3],
            'created_at': None,
            'updated_at': None
        }
        
        with patch.object(db_manager, 'fetchrow') as mock_fetchrow:
            mock_fetchrow.return_value = mock_row
            
            entity = await kg_manager.get_entity("test-id")
            
            assert entity is not None
            assert entity.entity_name == "test_function"
            assert entity.entity_type == "function"
            assert isinstance(entity.embedding, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_find_similar_entities(self, kg_manager):
        """Test finding similar entities."""
        mock_rows = [
            {
                'id': 'entity-1',
                'entity_type': 'function',
                'entity_name': 'similar_function',
                'properties': {},
                'embedding': [0.1, 0.2, 0.3],
                'distance': 0.2,
                'created_at': None,
                'updated_at': None
            }
        ]
        
        with patch.object(db_manager, 'fetch') as mock_fetch:
            mock_fetch.return_value = mock_rows
            
            query_embedding = np.array([0.1, 0.2, 0.3])
            results = await kg_manager.find_similar_entities(query_embedding, limit=10)
            
            assert len(results) == 1
            entity, similarity = results[0]
            assert entity.entity_name == "similar_function"
            assert similarity == 0.8  # 1 - 0.2 distance
    
    @pytest.mark.asyncio
    async def test_get_connected_entities(self, kg_manager):
        """Test getting connected entities."""
        mock_rows = [
            {
                'id': 'connected-entity',
                'entity_type': 'class',
                'entity_name': 'ConnectedClass',
                'properties': {},
                'embedding': None,
                'created_at': None,
                'updated_at': None
            }
        ]
        
        with patch.object(db_manager, 'fetch') as mock_fetch:
            mock_fetch.return_value = mock_rows
            
            entities = await kg_manager.get_connected_entities("test-entity", max_depth=2)
            
            assert len(entities) == 1
            assert entities[0].entity_name == "ConnectedClass"
    
    @pytest.mark.asyncio
    async def test_get_graph_statistics(self, kg_manager):
        """Test getting graph statistics."""
        with patch.object(db_manager, 'fetch') as mock_fetch, \
             patch.object(db_manager, 'fetchval') as mock_fetchval:
            
            mock_fetch.side_effect = [
                [{'entity_type': 'function', 'count': 10}],  # entity counts
                [{'relationship_type': 'calls', 'count': 5}]  # relationship counts
            ]
            mock_fetchval.side_effect = [100, 50, 80]  # total entities, relationships, with embeddings
            
            stats = await kg_manager.get_graph_statistics()
            
            assert stats['entity_counts']['function'] == 10
            assert stats['relationship_counts']['calls'] == 5
            assert stats['total_entities'] == 100
            assert stats['total_relationships'] == 50
            assert stats['entities_with_embeddings'] == 80

if __name__ == "__main__":
    pytest.main([__file__])