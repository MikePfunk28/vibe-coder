"""
Database connection management with connection pooling and transaction support.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
import asyncpg
from asyncpg import Pool, Connection
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_ide"
    username: str = "postgres"
    password: str = ""
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: int = 60
    server_settings: Dict[str, str] = None
    
    def __post_init__(self):
        if self.server_settings is None:
            self.server_settings = {
                'application_name': 'ai_ide',
                'timezone': 'UTC'
            }

class DatabaseManager:
    """Manages database connections with pooling and transaction support."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._load_config_from_env()
        self._pool: Optional[Pool] = None
        self._initialized = False
    
    def _load_config_from_env(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'ai_ide'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONNECTIONS', '5')),
            max_connections=int(os.getenv('DB_MAX_CONNECTIONS', '20')),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', '60'))
        )
    
    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        if self._initialized:
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings
            )
            
            # Test connection and ensure pgvector extension is available
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("Database connection pool initialized successfully")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get a database connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """Get a database connection with transaction support."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the result."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows from a query."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row from a query."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from a query."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def executemany(self, query: str, args_list: list) -> None:
        """Execute a query multiple times with different parameters."""
        async with self.get_connection() as conn:
            await conn.executemany(query, args_list)
    
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._pool:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "size": self._pool.get_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "idle_connections": self._pool.get_idle_size()
        }

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for common operations
async def get_connection():
    """Get a database connection."""
    return db_manager.get_connection()

async def transaction():
    """Get a database transaction."""
    return db_manager.transaction()

async def execute(query: str, *args):
    """Execute a query."""
    return await db_manager.execute(query, *args)

async def fetch(query: str, *args):
    """Fetch multiple rows."""
    return await db_manager.fetch(query, *args)

async def fetchrow(query: str, *args):
    """Fetch a single row."""
    return await db_manager.fetchrow(query, *args)

async def fetchval(query: str, *args):
    """Fetch a single value."""
    return await db_manager.fetchval(query, *args)

async def initialize_database():
    """Initialize the database connection."""
    await db_manager.initialize()

async def close_database():
    """Close the database connection."""
    await db_manager.close()