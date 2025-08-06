"""
Semantic Similarity Search System Integration
Combines embedding generation and search capabilities
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading

from code_embedding_generator import CodeEmbeddingGenerator, get_embedding_generator
from semantic_search_engine import SemanticSearchEngine, SearchContext, SearchResult, get_search_engine

logger = logging.getLogger('semantic_similarity_system')

class SemanticSimilaritySystem:
    """
    Main interface for semantic similarity search functionality
    Integrates embedding generation and search capabilities
    """
    
    def __init__(self, workspace_path: str = None, model_name: str = "microsoft/codebert-base"):
        self.workspace_path = workspace_path or os.getcwd()
        self.model_name = model_name
        
        # Initialize components
        self.embedding_generator = get_embedding_generator(self.workspace_path, model_name)
        self.search_engine = get_search_engine(self.workspace_path)
        
        # System state
        self.is_initialized = False
        self.is_indexing = False
        self.last_index_update = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'initialization_time': 0.0,
            'total_files_indexed': 0,
            'total_searches_performed': 0,
            'avg_search_time': 0.0,
            'system_health': 'unknown'
        }
        
        logger.info(f"Semantic Similarity System initialized for workspace: {self.workspace_path}")
    
    def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Initialize the semantic similarity system
        
        Args:
            force_reindex: Whether to force reindexing of all files
            
        Returns:
            Dictionary with initialization results
        """
        start_time = time.time()
        
        with self._lock:
            if self.is_initialized and not force_reindex:
                logger.info("System already initialized")
                return self._get_system_status()
            
            logger.info("Initializing semantic similarity system...")
            
            try:
                # Check if embedding model is available
                if not self.embedding_generator.model:
                    logger.warning("Embedding model not available - system will use text-based search only")
                    self.metrics['system_health'] = 'degraded'
                else:
                    self.metrics['system_health'] = 'healthy'
                
                # Index workspace if needed
                if force_reindex or self._needs_indexing():
                    logger.info("Starting workspace indexing...")
                    index_result = self.index_workspace()
                    self.metrics['total_files_indexed'] = index_result.get('indexed_files', 0)
                else:
                    logger.info("Workspace already indexed")
                
                self.is_initialized = True
                self.last_index_update = datetime.now()
                
                initialization_time = time.time() - start_time
                self.metrics['initialization_time'] = initialization_time
                
                logger.info(f"System initialized successfully in {initialization_time:.2f}s")
                
                return {
                    'success': True,
                    'initialization_time': initialization_time,
                    'system_health': self.metrics['system_health'],
                    'files_indexed': self.metrics['total_files_indexed'],
                    'embedding_model': self.embedding_generator.model_name,
                    'vector_store': self.embedding_generator.vector_store_type
                }
                
            except Exception as e:
                logger.error(f"System initialization failed: {e}")
                self.metrics['system_health'] = 'error'
                return {
                    'success': False,
                    'error': str(e),
                    'system_health': 'error'
                }
    
    def _needs_indexing(self) -> bool:
        """Check if workspace needs indexing"""
        # Simple check - if no chunks exist, we need indexing
        return len(self.embedding_generator.chunks) == 0
    
    def index_workspace(self, extensions: List[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """
        Index the entire workspace for semantic search
        
        Args:
            extensions: File extensions to index
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary with indexing results
        """
        with self._lock:
            if self.is_indexing:
                logger.warning("Indexing already in progress")
                return {'success': False, 'error': 'Indexing already in progress'}
            
            self.is_indexing = True
        
        try:
            logger.info("Starting workspace indexing...")
            result = self.embedding_generator.index_workspace(extensions, max_workers)
            
            # Update metrics
            self.metrics['total_files_indexed'] = result.get('indexed_files', 0)
            self.last_index_update = datetime.now()
            
            logger.info(f"Workspace indexing completed: {result}")
            return {**result, 'success': True}
            
        except Exception as e:
            logger.error(f"Workspace indexing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        finally:
            with self._lock:
                self.is_indexing = False
    
    def index_file(self, file_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index a single file
        
        Args:
            file_path: Path to the file to index
            force_reindex: Whether to force reindexing
            
        Returns:
            Dictionary with indexing results
        """
        try:
            logger.info(f"Indexing file: {file_path}")
            chunks = self.embedding_generator.index_file(file_path, force_reindex)
            
            return {
                'success': True,
                'file_path': file_path,
                'chunks_created': len(chunks) if chunks else 0,
                'chunk_types': list(set(chunk.chunk_type for chunk in chunks)) if chunks else []
            }
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return {
                'success': False,
                'file_path': file_path,
                'error': str(e)
            }
    
    def search(self, query: str, context: SearchContext = None, max_results: int = 10, **options) -> List[SearchResult]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            context: Search context for ranking
            max_results: Maximum number of results
            **options: Additional search options
            
        Returns:
            List of search results
        """
        if not self.is_initialized:
            logger.warning("System not initialized - initializing now...")
            init_result = self.initialize()
            if not init_result.get('success', False):
                logger.error("Failed to initialize system for search")
                return []
        
        start_time = time.time()
        
        try:
            # Perform search
            results = self.search_engine.search(query, context, max_results, **options)
            
            # Update metrics
            search_time = time.time() - start_time
            with self._lock:
                self.metrics['total_searches_performed'] += 1
                
                # Update average search time
                total_searches = self.metrics['total_searches_performed']
                current_avg = self.metrics['avg_search_time']
                self.metrics['avg_search_time'] = ((current_avg * (total_searches - 1)) + search_time) / total_searches
            
            logger.info(f"Search completed: '{query}' -> {len(results)} results in {search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def search_similar_code(self, file_path: str, line_start: int, line_end: int = None, 
                           max_results: int = 10) -> List[SearchResult]:
        """
        Find code similar to a specific code block
        
        Args:
            file_path: Path to the file containing the code
            line_start: Starting line number
            line_end: Ending line number (optional)
            max_results: Maximum number of results
            
        Returns:
            List of similar code results
        """
        try:
            # Find the chunk containing the specified lines
            target_chunk_id = None
            
            for chunk_id, chunk in self.embedding_generator.chunks.items():
                if (chunk.file_path == file_path and 
                    chunk.line_start <= line_start and 
                    chunk.line_end >= (line_end or line_start)):
                    target_chunk_id = chunk_id
                    break
            
            if not target_chunk_id:
                logger.warning(f"No chunk found for {file_path}:{line_start}")
                return []
            
            # Search for similar chunks
            results = self.search_engine.search_similar_to_chunk(target_chunk_id, max_results)
            
            logger.info(f"Found {len(results)} similar code blocks for {file_path}:{line_start}")
            return results
            
        except Exception as e:
            logger.error(f"Similar code search failed: {e}")
            return []
    
    def get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunk information
        """
        chunks = []
        
        if file_path in self.embedding_generator.chunk_index:
            chunk_ids = self.embedding_generator.chunk_index[file_path]
            
            for chunk_id in chunk_ids:
                if chunk_id in self.embedding_generator.chunks:
                    chunk = self.embedding_generator.chunks[chunk_id]
                    chunks.append({
                        'chunk_id': chunk_id,
                        'chunk_type': chunk.chunk_type,
                        'line_start': chunk.line_start,
                        'line_end': chunk.line_end,
                        'content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content,
                        'metadata': chunk.metadata,
                        'has_embedding': chunk.embedding is not None
                    })
        
        return chunks
    
    def get_workspace_overview(self) -> Dict[str, Any]:
        """
        Get overview of the indexed workspace
        
        Returns:
            Dictionary with workspace statistics
        """
        embedding_stats = self.embedding_generator.get_stats()
        search_stats = self.search_engine.get_stats()
        
        # Calculate chunk type distribution
        chunk_types = {}
        for chunk in self.embedding_generator.chunks.values():
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        # Calculate language distribution
        languages = {}
        for chunk in self.embedding_generator.chunks.values():
            lang = chunk.metadata.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            'workspace_path': self.workspace_path,
            'system_health': self.metrics['system_health'],
            'is_initialized': self.is_initialized,
            'last_index_update': self.last_index_update.isoformat() if self.last_index_update else None,
            
            # Embedding statistics
            'embedding_stats': embedding_stats,
            
            # Search statistics
            'search_stats': search_stats,
            
            # Content analysis
            'chunk_type_distribution': chunk_types,
            'language_distribution': languages,
            
            # System metrics
            'system_metrics': self.metrics
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_initialized': self.is_initialized,
            'is_indexing': self.is_indexing,
            'system_health': self.metrics['system_health'],
            'total_files_indexed': self.metrics['total_files_indexed'],
            'total_searches_performed': self.metrics['total_searches_performed'],
            'last_index_update': self.last_index_update.isoformat() if self.last_index_update else None
        }
    
    def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize system performance
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Optimizing semantic similarity system...")
            
            # Optimize search engine
            self.search_engine.optimize()
            
            # Clean up old embeddings if needed
            # (This could be expanded to remove embeddings for deleted files)
            
            # Update system health
            if self.embedding_generator.model:
                self.metrics['system_health'] = 'healthy'
            else:
                self.metrics['system_health'] = 'degraded'
            
            logger.info("System optimization completed")
            
            return {
                'success': True,
                'optimizations_applied': [
                    'search_cache_cleared',
                    'statistics_cleaned',
                    'system_health_updated'
                ]
            }
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down semantic similarity system...")
        
        with self._lock:
            # Save any pending data
            self.embedding_generator._save_metadata()
            
            # Clear caches
            self.search_engine.clear_cache()
            
            # Update state
            self.is_initialized = False
            
        logger.info("System shutdown completed")

# Global instance
_semantic_system = None

def get_semantic_system(workspace_path: str = None, model_name: str = "microsoft/codebert-base") -> SemanticSimilaritySystem:
    """Get or create semantic similarity system"""
    global _semantic_system
    
    if _semantic_system is None or (workspace_path and _semantic_system.workspace_path != workspace_path):
        if workspace_path is None:
            workspace_path = os.getcwd()
        _semantic_system = SemanticSimilaritySystem(workspace_path, model_name)
    
    return _semantic_system

# Convenience functions for easy access
def initialize_semantic_search(workspace_path: str = None, force_reindex: bool = False) -> Dict[str, Any]:
    """Initialize semantic search system"""
    system = get_semantic_system(workspace_path)
    return system.initialize(force_reindex)

def semantic_search(query: str, context: SearchContext = None, max_results: int = 10, **options) -> List[SearchResult]:
    """Perform semantic search"""
    system = get_semantic_system()
    return system.search(query, context, max_results, **options)

def find_similar_code(file_path: str, line_start: int, line_end: int = None, max_results: int = 10) -> List[SearchResult]:
    """Find similar code blocks"""
    system = get_semantic_system()
    return system.search_similar_code(file_path, line_start, line_end, max_results)

def get_workspace_overview() -> Dict[str, Any]:
    """Get workspace overview"""
    system = get_semantic_system()
    return system.get_workspace_overview()