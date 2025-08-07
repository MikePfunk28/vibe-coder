"""
Repository for search-related data including embeddings and web search cache.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import numpy as np

from .base_repository import BaseRepository, QueryFilter, QueryOptions
from ..caching import cache_manager
from ..connection import db_manager

@dataclass
class CodeEmbedding:
    """Code embedding entity."""
    id: Optional[str] = None
    file_path: str = ""
    content_hash: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WebSearchCache:
    """Web search cache entity."""
    id: Optional[str] = None
    query_hash: str = ""
    query_text: str = ""
    search_engine: str = ""
    results: Dict[str, Any] = None
    relevance_scores: Dict[str, Any] = None
    cached_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.relevance_scores is None:
            self.relevance_scores = {}

class CodeEmbeddingRepository(BaseRepository[CodeEmbedding]):
    """Repository for code embeddings."""
    
    def __init__(self):
        super().__init__("code_embeddings")
    
    def _row_to_entity(self, row: Dict[str, Any]) -> CodeEmbedding:
        """Convert database row to CodeEmbedding entity."""
        # Convert embedding list back to numpy array
        embedding = None
        if row['embedding']:
            embedding = np.array(row['embedding'])
        
        return CodeEmbedding(
            id=str(row['id']) if row['id'] else None,
            file_path=row['file_path'],
            content_hash=row['content_hash'],
            embedding=embedding,
            metadata=row['metadata'] or {},
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
    
    def _entity_to_dict(self, entity: CodeEmbedding) -> Dict[str, Any]:
        """Convert CodeEmbedding entity to dictionary."""
        # Convert numpy array to list for database storage
        embedding_list = None
        if entity.embedding is not None:
            embedding_list = entity.embedding.tolist() if isinstance(entity.embedding, np.ndarray) else entity.embedding
        
        return {
            'id': entity.id,
            'file_path': entity.file_path,
            'content_hash': entity.content_hash,
            'embedding': embedding_list,
            'metadata': entity.metadata,
            'created_at': entity.created_at,
            'updated_at': entity.updated_at or datetime.now()
        }
    
    async def find_by_file_path(self, file_path: str) -> Optional[CodeEmbedding]:
        """Find embedding by file path."""
        filters = [QueryFilter("file_path", "=", file_path)]
        options = QueryOptions(filters=filters, limit=1)
        results = await self.find_all(options)
        return results[0] if results else None
    
    async def find_by_content_hash(self, content_hash: str) -> Optional[CodeEmbedding]:
        """Find embedding by content hash."""
        filters = [QueryFilter("content_hash", "=", content_hash)]
        options = QueryOptions(filters=filters, limit=1)
        results = await self.find_all(options)
        return results[0] if results else None
    
    async def upsert_embedding(self, file_path: str, content_hash: str, embedding: np.ndarray, 
                              metadata: Dict[str, Any] = None) -> str:
        """Insert or update embedding for a file."""
        try:
            # Check cache first
            cached_embedding = await cache_manager.get_embedding(file_path, content_hash)
            if cached_embedding is not None:
                # If cached, just update database
                existing = await self.find_by_file_path(file_path)
                if existing and existing.content_hash == content_hash:
                    return existing.id
            
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Use ON CONFLICT to handle upsert
            query = """
                INSERT INTO code_embeddings (file_path, content_hash, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (file_path, content_hash) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
            """
            
            entity_id = await db_manager.fetchval(
                query, 
                file_path, 
                content_hash, 
                embedding_list, 
                metadata or {}
            )
            
            # Cache the embedding
            await cache_manager.cache_embedding(file_path, content_hash, embedding)
            
            return str(entity_id)
            
        except Exception as e:
            logger.error(f"Failed to upsert embedding: {e}")
            raise
    
    async def find_similar_embeddings(self, query_embedding: np.ndarray, limit: int = 10, 
                                    threshold: float = 0.7, file_paths: Optional[List[str]] = None) -> List[Tuple[CodeEmbedding, float]]:
        """Find embeddings similar to the query embedding."""
        try:
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Build query with optional file path filtering
            base_query = """
                SELECT *, (embedding <=> $1) as distance
                FROM code_embeddings
                WHERE embedding IS NOT NULL
            """
            params = [embedding_list]
            
            if file_paths:
                placeholders = ",".join([f"${i+2}" for i in range(len(file_paths))])
                base_query += f" AND file_path IN ({placeholders})"
                params.extend(file_paths)
            
            base_query += f" ORDER BY distance LIMIT ${len(params) + 1}"
            params.append(limit)
            
            rows = await db_manager.fetch(base_query, *params)
            
            results = []
            for row in rows:
                # Convert distance to similarity score (1 - distance for cosine distance)
                similarity = 1.0 - row['distance']
                
                if similarity >= threshold:
                    embedding_entity = self._row_to_entity(dict(row))
                    results.append((embedding_entity, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    async def delete_by_file_path(self, file_path: str) -> bool:
        """Delete embedding by file path."""
        try:
            # Get the embedding first to clear cache
            existing = await self.find_by_file_path(file_path)
            if existing:
                # Clear cache
                await cache_manager.delete('embedding', f"{file_path}:{existing.content_hash}")
            
            filters = [QueryFilter("file_path", "=", file_path)]
            query = f"DELETE FROM {self.table_name} WHERE file_path = $1"
            result = await db_manager.execute(query, file_path)
            return "DELETE" in result
            
        except Exception as e:
            logger.error(f"Failed to delete embedding by file path: {e}")
            return False
    
    async def cleanup_old_embeddings(self, days: int = 30) -> int:
        """Clean up embeddings older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get embeddings to delete for cache cleanup
            filters = [QueryFilter("updated_at", "<", cutoff_date)]
            old_embeddings = await self.find_all(QueryOptions(filters=filters))
            
            # Clear cache for old embeddings
            for embedding in old_embeddings:
                await cache_manager.delete('embedding', f"{embedding.file_path}:{embedding.content_hash}")
            
            # Delete from database
            query = f"DELETE FROM {self.table_name} WHERE updated_at < $1"
            result = await db_manager.execute(query, cutoff_date)
            
            deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
            logger.info(f"Cleaned up {deleted_count} old embeddings")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            return 0

class WebSearchCacheRepository(BaseRepository[WebSearchCache]):
    """Repository for web search cache."""
    
    def __init__(self):
        super().__init__("web_search_cache")
    
    def _row_to_entity(self, row: Dict[str, Any]) -> WebSearchCache:
        """Convert database row to WebSearchCache entity."""
        return WebSearchCache(
            id=str(row['id']) if row['id'] else None,
            query_hash=row['query_hash'],
            query_text=row['query_text'],
            search_engine=row['search_engine'],
            results=row['results'] or {},
            relevance_scores=row['relevance_scores'] or {},
            cached_at=row['cached_at'],
            expires_at=row['expires_at'],
            hit_count=row['hit_count'] or 0
        )
    
    def _entity_to_dict(self, entity: WebSearchCache) -> Dict[str, Any]:
        """Convert WebSearchCache entity to dictionary."""
        return {
            'id': entity.id,
            'query_hash': entity.query_hash,
            'query_text': entity.query_text,
            'search_engine': entity.search_engine,
            'results': entity.results,
            'relevance_scores': entity.relevance_scores,
            'cached_at': entity.cached_at or datetime.now(),
            'expires_at': entity.expires_at,
            'hit_count': entity.hit_count
        }
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query text."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def cache_search_results(self, query: str, search_engine: str, results: Dict[str, Any], 
                                 relevance_scores: Dict[str, Any] = None, ttl_hours: int = 4) -> str:
        """Cache web search results."""
        try:
            query_hash = self._generate_query_hash(query)
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            # Check if already cached
            existing = await self.find_cached_result(query, search_engine)
            if existing:
                # Update hit count and extend expiration
                await self.update(existing.id, {
                    'hit_count': existing.hit_count + 1,
                    'expires_at': expires_at,
                    'results': results,
                    'relevance_scores': relevance_scores or {}
                })
                
                # Update Redis cache
                await cache_manager.cache_web_search(query, search_engine, results)
                
                return existing.id
            
            # Create new cache entry
            cache_entry = WebSearchCache(
                query_hash=query_hash,
                query_text=query,
                search_engine=search_engine,
                results=results,
                relevance_scores=relevance_scores or {},
                expires_at=expires_at,
                hit_count=1
            )
            
            cache_id = await self.create(cache_entry)
            
            # Also cache in Redis for faster access
            await cache_manager.cache_web_search(query, search_engine, results)
            
            return cache_id
            
        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
            raise
    
    async def find_cached_result(self, query: str, search_engine: str) -> Optional[WebSearchCache]:
        """Find cached search result."""
        try:
            # Check Redis cache first
            cached_result = await cache_manager.get_web_search(query, search_engine)
            if cached_result:
                # Still check database for metadata
                query_hash = self._generate_query_hash(query)
                filters = [
                    QueryFilter("query_hash", "=", query_hash),
                    QueryFilter("search_engine", "=", search_engine),
                    QueryFilter("expires_at", ">", datetime.now())
                ]
                options = QueryOptions(filters=filters, limit=1)
                results = await self.find_all(options)
                
                if results:
                    # Update hit count
                    await self.update(results[0].id, {'hit_count': results[0].hit_count + 1})
                    return results[0]
            
            # Check database
            query_hash = self._generate_query_hash(query)
            filters = [
                QueryFilter("query_hash", "=", query_hash),
                QueryFilter("search_engine", "=", search_engine),
                QueryFilter("expires_at", ">", datetime.now())
            ]
            options = QueryOptions(filters=filters, limit=1)
            results = await self.find_all(options)
            
            if results:
                # Cache in Redis for next time
                await cache_manager.cache_web_search(query, search_engine, results[0].results)
                return results[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find cached result: {e}")
            return None
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            # Delete expired entries
            query = f"DELETE FROM {self.table_name} WHERE expires_at < $1"
            result = await db_manager.execute(query, datetime.now())
            
            deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
            logger.info(f"Cleaned up {deleted_count} expired search cache entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = {}
            
            # Total entries
            stats['total_entries'] = await self.count()
            
            # Entries by search engine
            engine_counts = await db_manager.fetch(
                f"SELECT search_engine, COUNT(*) as count FROM {self.table_name} GROUP BY search_engine"
            )
            stats['entries_by_engine'] = {row['search_engine']: row['count'] for row in engine_counts}
            
            # Expired entries
            expired_count = await self.count([QueryFilter("expires_at", "<", datetime.now())])
            stats['expired_entries'] = expired_count
            
            # Hit statistics
            hit_stats = await db_manager.fetchrow(
                f"SELECT AVG(hit_count) as avg_hits, MAX(hit_count) as max_hits, SUM(hit_count) as total_hits FROM {self.table_name}"
            )
            stats['hit_statistics'] = {
                'average_hits': float(hit_stats['avg_hits']) if hit_stats['avg_hits'] else 0,
                'max_hits': hit_stats['max_hits'] or 0,
                'total_hits': hit_stats['total_hits'] or 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {}

# Global repository instances
code_embedding_repo = CodeEmbeddingRepository()
web_search_cache_repo = WebSearchCacheRepository()