"""
Redis caching system for embeddings, search results, and reasoning traces.
"""

import redis.asyncio as redis
import json
import pickle
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import numpy as np
import os

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = False  # Keep False for binary data
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=os.getenv('REDIS_PASSWORD'),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '10')),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', '5')),
            socket_connect_timeout=int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5'))
        )

class CacheManager:
    """Manages Redis caching for various AI IDE components."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig.from_env()
        self._redis: Optional[redis.Redis] = None
        self._initialized = False
        
        # Cache key prefixes for different data types
        self.prefixes = {
            'embedding': 'emb:',
            'search': 'search:',
            'reasoning': 'reason:',
            'web_search': 'web:',
            'context': 'ctx:',
            'rag': 'rag:',
            'agent': 'agent:',
            'performance': 'perf:'
        }
        
        # Default TTL values (in seconds)
        self.default_ttl = {
            'embedding': 3600 * 24,  # 24 hours
            'search': 3600 * 2,      # 2 hours
            'reasoning': 3600 * 6,   # 6 hours
            'web_search': 3600 * 4,  # 4 hours
            'context': 3600 * 1,     # 1 hour
            'rag': 3600 * 12,        # 12 hours
            'agent': 3600 * 8,       # 8 hours
            'performance': 3600 * 24 # 24 hours
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        try:
            self._redis = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses
            )
            
            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info("Redis cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._initialized = False
            logger.info("Redis cache manager closed")
    
    def _make_key(self, cache_type: str, key: str) -> str:
        """Create a cache key with appropriate prefix."""
        prefix = self.prefixes.get(cache_type, 'misc:')
        return f"{prefix}{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Create a hash key from data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def set(self, cache_type: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            ttl = ttl or self.default_ttl.get(cache_type, 3600)
            
            # Serialize value based on type
            if isinstance(value, np.ndarray):
                serialized_value = pickle.dumps(value)
            elif isinstance(value, (dict, list)):
                serialized_value = json.dumps(value).encode()
            else:
                serialized_value = pickle.dumps(value)
            
            await self._redis.setex(cache_key, ttl, serialized_value)
            logger.debug(f"Cached {cache_type} key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache for {cache_type}:{key}: {e}")
            return False
    
    async def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            serialized_value = await self._redis.get(cache_key)
            
            if serialized_value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(serialized_value.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(serialized_value)
                
        except Exception as e:
            logger.error(f"Failed to get cache for {cache_type}:{key}: {e}")
            return None
    
    async def delete(self, cache_type: str, key: str) -> bool:
        """Delete a value from cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            result = await self._redis.delete(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache for {cache_type}:{key}: {e}")
            return False
    
    async def exists(self, cache_type: str, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            result = await self._redis.exists(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check cache existence for {cache_type}:{key}: {e}")
            return False
    
    async def expire(self, cache_type: str, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            result = await self._redis.expire(cache_key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Failed to set expiration for {cache_type}:{key}: {e}")
            return False
    
    async def get_ttl(self, cache_type: str, key: str) -> int:
        """Get remaining TTL for a key."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(cache_type, key)
            ttl = await self._redis.ttl(cache_key)
            return ttl
            
        except Exception as e:
            logger.error(f"Failed to get TTL for {cache_type}:{key}: {e}")
            return -1
    
    # Specialized methods for different data types
    
    async def cache_embedding(self, file_path: str, content_hash: str, embedding: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache code embedding."""
        key = f"{file_path}:{content_hash}"
        return await self.set('embedding', key, embedding, ttl)
    
    async def get_embedding(self, file_path: str, content_hash: str) -> Optional[np.ndarray]:
        """Get cached code embedding."""
        key = f"{file_path}:{content_hash}"
        return await self.get('embedding', key)
    
    async def cache_search_results(self, query: str, context: Dict[str, Any], results: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache search results."""
        query_hash = self._hash_key({'query': query, 'context': context})
        return await self.set('search', query_hash, results, ttl)
    
    async def get_search_results(self, query: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        query_hash = self._hash_key({'query': query, 'context': context})
        return await self.get('search', query_hash)
    
    async def cache_reasoning_trace(self, interaction_id: str, trace: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache reasoning trace."""
        return await self.set('reasoning', interaction_id, trace, ttl)
    
    async def get_reasoning_trace(self, interaction_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached reasoning trace."""
        return await self.get('reasoning', interaction_id)
    
    async def cache_web_search(self, query: str, search_engine: str, results: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache web search results."""
        key = f"{search_engine}:{self._hash_key(query)}"
        return await self.set('web_search', key, results, ttl)
    
    async def get_web_search(self, query: str, search_engine: str) -> Optional[Dict[str, Any]]:
        """Get cached web search results."""
        key = f"{search_engine}:{self._hash_key(query)}"
        return await self.get('web_search', key)
    
    async def cache_context_window(self, session_id: str, window_id: str, content: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache context window."""
        key = f"{session_id}:{window_id}"
        return await self.set('context', key, content, ttl)
    
    async def get_context_window(self, session_id: str, window_id: str) -> Optional[Dict[str, Any]]:
        """Get cached context window."""
        key = f"{session_id}:{window_id}"
        return await self.get('context', key)
    
    async def cache_rag_results(self, query: str, document_type: str, results: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache RAG retrieval results."""
        key = f"{document_type}:{self._hash_key(query)}"
        return await self.set('rag', key, results, ttl)
    
    async def get_rag_results(self, query: str, document_type: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached RAG results."""
        key = f"{document_type}:{self._hash_key(query)}"
        return await self.get('rag', key)
    
    # Cache management operations
    
    async def clear_cache_type(self, cache_type: str) -> int:
        """Clear all keys of a specific cache type."""
        if not self._initialized:
            await self.initialize()
        
        try:
            prefix = self.prefixes.get(cache_type, 'misc:')
            pattern = f"{prefix}*"
            
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} keys for cache type: {cache_type}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear cache type {cache_type}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized:
            await self.initialize()
        
        try:
            info = await self._redis.info()
            
            # Count keys by type
            type_counts = {}
            for cache_type, prefix in self.prefixes.items():
                pattern = f"{prefix}*"
                count = 0
                async for _ in self._redis.scan_iter(match=pattern):
                    count += 1
                type_counts[cache_type] = count
            
            return {
                'redis_info': {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses')
                },
                'key_counts_by_type': type_counts,
                'total_keys': sum(type_counts.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def cleanup_expired_keys(self) -> int:
        """Manually cleanup expired keys (Redis usually handles this automatically)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            cleaned = 0
            for cache_type, prefix in self.prefixes.items():
                pattern = f"{prefix}*"
                async for key in self._redis.scan_iter(match=pattern):
                    ttl = await self._redis.ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        cleaned += 1
            
            logger.info(f"Cleaned up {cleaned} expired keys")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions
async def initialize_cache():
    """Initialize the cache manager."""
    await cache_manager.initialize()

async def close_cache():
    """Close the cache manager."""
    await cache_manager.close()

# Specialized cache functions
async def cache_embedding(file_path: str, content_hash: str, embedding: np.ndarray, ttl: Optional[int] = None) -> bool:
    """Cache code embedding."""
    return await cache_manager.cache_embedding(file_path, content_hash, embedding, ttl)

async def get_embedding(file_path: str, content_hash: str) -> Optional[np.ndarray]:
    """Get cached code embedding."""
    return await cache_manager.get_embedding(file_path, content_hash)

async def cache_search_results(query: str, context: Dict[str, Any], results: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
    """Cache search results."""
    return await cache_manager.cache_search_results(query, context, results, ttl)

async def get_search_results(query: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Get cached search results."""
    return await cache_manager.get_search_results(query, context)