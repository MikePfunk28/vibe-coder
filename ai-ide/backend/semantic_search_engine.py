"""
Semantic Search and Ranking System
Advanced semantic search with context-aware ranking and caching
"""

import os
import json
import logging
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import re

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from code_embedding_generator import CodeEmbeddingGenerator, CodeChunk, get_embedding_generator

logger = logging.getLogger('semantic_search_engine')

@dataclass
class SearchResult:
    """Represents a semantic search result"""
    chunk_id: str
    file_path: str
    content: str
    chunk_type: str
    line_start: int
    line_end: int
    similarity_score: float
    context_relevance: float
    final_score: float
    metadata: Dict[str, Any]
    snippet: str
    highlights: List[Tuple[int, int]]  # Character positions for highlighting
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SearchContext:
    """Represents the current development context for search ranking"""
    current_file: Optional[str] = None
    current_language: Optional[str] = None
    open_files: List[str] = None
    recent_files: List[str] = None
    cursor_position: Optional[Tuple[int, int]] = None  # (line, column)
    selected_text: Optional[str] = None
    project_type: Optional[str] = None
    recent_searches: List[str] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.open_files is None:
            self.open_files = []
        if self.recent_files is None:
            self.recent_files = []
        if self.recent_searches is None:
            self.recent_searches = []
        if self.user_preferences is None:
            self.user_preferences = {}

class SearchCache:
    """Caches search results for performance optimization"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, query: str, context: SearchContext, options: Dict[str, Any]) -> str:
        """Generate cache key from search parameters"""
        key_data = {
            'query': query.lower().strip(),
            'current_file': context.current_file,
            'current_language': context.current_language,
            'max_results': options.get('max_results', 10),
            'chunk_types': sorted(options.get('chunk_types', [])),
            'file_extensions': sorted(options.get('file_extensions', []))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, context: SearchContext, options: Dict[str, Any]) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        with self._lock:
            key = self._generate_key(query, context, options)
            
            if key in self.cache:
                results, timestamp = self.cache[key]
                
                # Check if cache entry is still valid
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return results
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            return None
    
    def put(self, query: str, context: SearchContext, options: Dict[str, Any], results: List[SearchResult]):
        """Cache search results"""
        with self._lock:
            key = self._generate_key(query, context, options)
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (results, datetime.now())
    
    def clear(self):
        """Clear all cached results"""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            now = datetime.now()
            valid_entries = sum(1 for _, timestamp in self.cache.values() 
                              if now - timestamp < timedelta(seconds=self.ttl_seconds))
            
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'expired_entries': len(self.cache) - valid_entries,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }

class ContextAwareRanker:
    """Ranks search results based on development context"""
    
    def __init__(self):
        self.language_weights = {
            'python': {'python': 1.0, 'javascript': 0.3, 'typescript': 0.3},
            'javascript': {'javascript': 1.0, 'typescript': 0.8, 'python': 0.3},
            'typescript': {'typescript': 1.0, 'javascript': 0.8, 'python': 0.3},
            'java': {'java': 1.0, 'cpp': 0.4, 'c': 0.3},
            'cpp': {'cpp': 1.0, 'c': 0.8, 'java': 0.4},
            'c': {'c': 1.0, 'cpp': 0.8, 'java': 0.3}
        }
        
        self.chunk_type_weights = {
            'function': 1.0,
            'class': 0.9,
            'file': 0.7,
            'comment': 0.5
        }
        
        self.recency_decay = 0.1  # How much recency affects scoring
    
    def calculate_context_relevance(self, result: SearchResult, context: SearchContext) -> float:
        """Calculate context relevance score for a search result"""
        relevance = 0.0
        
        # File proximity scoring
        if context.current_file:
            if result.file_path == context.current_file:
                relevance += 0.3  # Same file gets high relevance
            elif result.file_path in context.open_files:
                relevance += 0.2  # Open files get medium relevance
            elif result.file_path in context.recent_files:
                # Recent files get decreasing relevance based on recency
                try:
                    recency_index = context.recent_files.index(result.file_path)
                    recency_score = max(0, 0.15 - (recency_index * 0.02))
                    relevance += recency_score
                except ValueError:
                    pass
        
        # Language similarity scoring
        if context.current_language and 'language' in result.metadata:
            result_language = result.metadata['language']
            if context.current_language in self.language_weights:
                lang_weights = self.language_weights[context.current_language]
                relevance += lang_weights.get(result_language, 0.1) * 0.2
        
        # Chunk type scoring
        chunk_weight = self.chunk_type_weights.get(result.chunk_type, 0.5)
        relevance += chunk_weight * 0.1
        
        # Project type relevance
        if context.project_type and 'project_patterns' in result.metadata:
            patterns = result.metadata['project_patterns']
            if context.project_type in patterns:
                relevance += 0.1
        
        # Selected text similarity
        if context.selected_text and len(context.selected_text.strip()) > 3:
            selected_lower = context.selected_text.lower()
            content_lower = result.content.lower()
            
            # Simple substring matching (could be enhanced with fuzzy matching)
            if selected_lower in content_lower:
                relevance += 0.15
            
            # Check for similar variable/function names
            selected_words = set(re.findall(r'\w+', selected_lower))
            content_words = set(re.findall(r'\w+', content_lower))
            
            if selected_words and content_words:
                word_overlap = len(selected_words.intersection(content_words))
                word_similarity = word_overlap / len(selected_words.union(content_words))
                relevance += word_similarity * 0.1
        
        # Recent search patterns
        if context.recent_searches:
            for recent_query in context.recent_searches[-5:]:  # Last 5 searches
                if recent_query.lower() in result.content.lower():
                    relevance += 0.05
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def rank_results(self, results: List[SearchResult], context: SearchContext) -> List[SearchResult]:
        """Rank search results based on context"""
        for result in results:
            context_relevance = self.calculate_context_relevance(result, context)
            result.context_relevance = context_relevance
            
            # Combine similarity and context scores
            # Weighted combination: 70% similarity, 30% context
            result.final_score = (0.7 * result.similarity_score) + (0.3 * context_relevance)
        
        # Sort by final score
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

class SemanticSearchEngine:
    """Main semantic search engine with similarity calculation and ranking"""
    
    def __init__(self, workspace_path: str = None, embedding_generator: CodeEmbeddingGenerator = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.embedding_generator = embedding_generator or get_embedding_generator(self.workspace_path)
        
        # Search components
        self.ranker = ContextAwareRanker()
        self.cache = SearchCache()
        
        # Search statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_search_time': 0.0,
            'search_times': [],
            'popular_queries': defaultdict(int),
            'last_search': None
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Query preprocessing
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
    
    def preprocess_query(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Preprocess search query and extract search hints"""
        original_query = query
        query = query.strip()
        
        # Extract search modifiers
        modifiers = {}
        
        # Language filter: lang:python
        lang_match = re.search(r'lang:(\w+)', query)
        if lang_match:
            modifiers['language'] = lang_match.group(1)
            query = re.sub(r'lang:\w+', '', query).strip()
        
        # File type filter: type:function
        type_match = re.search(r'type:(\w+)', query)
        if type_match:
            modifiers['chunk_type'] = type_match.group(1)
            query = re.sub(r'type:\w+', '', query).strip()
        
        # File filter: file:*.py
        file_match = re.search(r'file:([^\s]+)', query)
        if file_match:
            modifiers['file_pattern'] = file_match.group(1)
            query = re.sub(r'file:[^\s]+', '', query).strip()
        
        # Extract keywords
        words = re.findall(r'\w+', query.lower())
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return query, keywords, modifiers
    
    def search_vector_store(self, query_embedding: np.ndarray, max_results: int = 50) -> List[Tuple[str, float]]:
        """Search vector store for similar embeddings"""
        if not self.embedding_generator.vector_store:
            return []
        
        try:
            if self.embedding_generator.vector_store_type == "faiss" and FAISS_AVAILABLE:
                return self._search_faiss(query_embedding, max_results)
            elif self.embedding_generator.vector_store_type == "chroma" and CHROMADB_AVAILABLE:
                return self._search_chroma(query_embedding, max_results)
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
        
        return []
    
    def _search_faiss(self, query_embedding: np.ndarray, max_results: int) -> List[Tuple[str, float]]:
        """Search FAISS index"""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_norm)
        
        # Search
        scores, indices = self.embedding_generator.vector_store.search(query_norm, max_results)
        
        # Convert indices to chunk IDs
        results = []
        chunk_ids = list(self.embedding_generator.chunks.keys())
        
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(chunk_ids):
                chunk_id = chunk_ids[idx]
                results.append((chunk_id, float(score)))
        
        return results
    
    def _search_chroma(self, query_embedding: np.ndarray, max_results: int) -> List[Tuple[str, float]]:
        """Search ChromaDB collection"""
        results = self.embedding_generator.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max_results
        )
        
        chunk_results = []
        if results['ids'] and results['distances']:
            for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
                # Convert distance to similarity (ChromaDB returns distances)
                similarity = 1.0 / (1.0 + distance)
                chunk_results.append((chunk_id, similarity))
        
        return chunk_results
    
    def search_text_fallback(self, query: str, keywords: List[str], max_results: int = 50) -> List[Tuple[str, float]]:
        """Fallback text-based search when vector search is not available"""
        results = []
        query_lower = query.lower()
        
        for chunk_id, chunk in self.embedding_generator.chunks.items():
            content_lower = chunk.content.lower()
            score = 0.0
            
            # Direct query match
            if query_lower in content_lower:
                score += 0.5
            
            # Keyword matching
            for keyword in keywords:
                if keyword in content_lower:
                    # Count occurrences
                    occurrences = content_lower.count(keyword)
                    score += min(occurrences * 0.1, 0.3)
            
            # Metadata matching
            if 'name' in chunk.metadata:
                name_lower = chunk.metadata['name'].lower()
                if query_lower in name_lower:
                    score += 0.3
                for keyword in keywords:
                    if keyword in name_lower:
                        score += 0.2
            
            if score > 0:
                results.append((chunk_id, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def create_search_results(self, chunk_matches: List[Tuple[str, float]], 
                            query: str, keywords: List[str]) -> List[SearchResult]:
        """Create SearchResult objects from chunk matches"""
        results = []
        
        for chunk_id, similarity_score in chunk_matches:
            if chunk_id not in self.embedding_generator.chunks:
                continue
            
            chunk = self.embedding_generator.chunks[chunk_id]
            
            # Create snippet with highlights
            snippet, highlights = self._create_snippet(chunk.content, query, keywords)
            
            result = SearchResult(
                chunk_id=chunk_id,
                file_path=chunk.file_path,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                similarity_score=similarity_score,
                context_relevance=0.0,  # Will be calculated by ranker
                final_score=similarity_score,  # Will be updated by ranker
                metadata=chunk.metadata,
                snippet=snippet,
                highlights=highlights
            )
            
            results.append(result)
        
        return results
    
    def _create_snippet(self, content: str, query: str, keywords: List[str], 
                       max_length: int = 200) -> Tuple[str, List[Tuple[int, int]]]:
        """Create a snippet with highlighted search terms"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find the best position for the snippet
        best_pos = 0
        best_score = 0
        
        # Look for query matches
        query_pos = content_lower.find(query_lower)
        if query_pos != -1:
            best_pos = max(0, query_pos - 50)
            best_score = 10
        
        # Look for keyword matches
        for keyword in keywords:
            keyword_pos = content_lower.find(keyword)
            if keyword_pos != -1:
                score = 5
                if score > best_score:
                    best_pos = max(0, keyword_pos - 50)
                    best_score = score
        
        # Create snippet
        snippet_start = best_pos
        snippet_end = min(len(content), snippet_start + max_length)
        snippet = content[snippet_start:snippet_end]
        
        # Add ellipsis if needed
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."
        
        # Find highlights in snippet
        highlights = []
        snippet_lower = snippet.lower()
        
        # Highlight query
        if query_lower in snippet_lower:
            start = snippet_lower.find(query_lower)
            if start != -1:
                highlights.append((start, start + len(query_lower)))
        
        # Highlight keywords
        for keyword in keywords:
            start = 0
            while True:
                pos = snippet_lower.find(keyword, start)
                if pos == -1:
                    break
                highlights.append((pos, pos + len(keyword)))
                start = pos + 1
        
        # Remove overlapping highlights
        highlights = self._merge_highlights(highlights)
        
        return snippet, highlights
    
    def _merge_highlights(self, highlights: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping highlight ranges"""
        if not highlights:
            return []
        
        # Sort by start position
        highlights.sort()
        merged = [highlights[0]]
        
        for start, end in highlights[1:]:
            last_start, last_end = merged[-1]
            
            if start <= last_end:
                # Overlapping, merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping, add
                merged.append((start, end))
        
        return merged
    
    def filter_results(self, results: List[SearchResult], modifiers: Dict[str, Any], 
                      options: Dict[str, Any]) -> List[SearchResult]:
        """Filter search results based on modifiers and options"""
        filtered = results
        
        # Language filter
        if 'language' in modifiers:
            target_lang = modifiers['language']
            filtered = [r for r in filtered if r.metadata.get('language') == target_lang]
        
        # Chunk type filter
        if 'chunk_type' in modifiers:
            target_type = modifiers['chunk_type']
            filtered = [r for r in filtered if r.chunk_type == target_type]
        
        # File pattern filter
        if 'file_pattern' in modifiers:
            pattern = modifiers['file_pattern']
            if '*' in pattern:
                # Simple glob pattern
                import fnmatch
                filtered = [r for r in filtered if fnmatch.fnmatch(os.path.basename(r.file_path), pattern)]
            else:
                # Exact match
                filtered = [r for r in filtered if pattern in r.file_path]
        
        # Options filters
        if 'chunk_types' in options:
            allowed_types = set(options['chunk_types'])
            filtered = [r for r in filtered if r.chunk_type in allowed_types]
        
        if 'file_extensions' in options:
            allowed_exts = set(options['file_extensions'])
            filtered = [r for r in filtered if any(r.file_path.endswith(ext) for ext in allowed_exts)]
        
        if 'min_score' in options:
            min_score = options['min_score']
            filtered = [r for r in filtered if r.similarity_score >= min_score]
        
        return filtered
    
    def search(self, query: str, context: SearchContext = None, 
               max_results: int = 10, **options) -> List[SearchResult]:
        """Main search method"""
        start_time = time.time()
        
        if context is None:
            context = SearchContext()
        
        # Update statistics
        with self._lock:
            self.stats['total_searches'] += 1
            self.stats['popular_queries'][query.lower()] += 1
            self.stats['last_search'] = datetime.now().isoformat()
        
        # Check cache first
        cached_results = self.cache.get(query, context, options)
        if cached_results:
            with self._lock:
                self.stats['cache_hits'] += 1
            return cached_results[:max_results]
        
        with self._lock:
            self.stats['cache_misses'] += 1
        
        try:
            # Preprocess query
            processed_query, keywords, modifiers = self.preprocess_query(query)
            
            if not processed_query and not keywords:
                return []
            
            # Generate query embedding if possible
            chunk_matches = []
            
            if self.embedding_generator.model and processed_query:
                try:
                    query_embedding = self.embedding_generator.model.encode([processed_query])[0]
                    chunk_matches = self.search_vector_store(query_embedding, max_results * 3)
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to text search: {e}")
            
            # Fallback to text search if vector search failed or unavailable
            if not chunk_matches:
                chunk_matches = self.search_text_fallback(processed_query, keywords, max_results * 3)
            
            # Create search results
            results = self.create_search_results(chunk_matches, processed_query, keywords)
            
            # Filter results
            results = self.filter_results(results, modifiers, options)
            
            # Rank results with context
            results = self.ranker.rank_results(results, context)
            
            # Limit results
            final_results = results[:max_results]
            
            # Cache results
            self.cache.put(query, context, options, final_results)
            
            # Update statistics
            search_time = time.time() - start_time
            with self._lock:
                self.stats['search_times'].append(search_time)
                if len(self.stats['search_times']) > 100:
                    self.stats['search_times'] = self.stats['search_times'][-100:]
                self.stats['avg_search_time'] = np.mean(self.stats['search_times'])
            
            logger.info(f"Search completed: '{query}' -> {len(final_results)} results in {search_time:.3f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def search_similar_to_chunk(self, chunk_id: str, max_results: int = 10) -> List[SearchResult]:
        """Find chunks similar to a given chunk"""
        if chunk_id not in self.embedding_generator.chunks:
            return []
        
        chunk = self.embedding_generator.chunks[chunk_id]
        
        if chunk.embedding is None:
            return []
        
        # Search for similar embeddings
        chunk_matches = self.search_vector_store(chunk.embedding, max_results + 1)
        
        # Remove the original chunk from results
        chunk_matches = [(cid, score) for cid, score in chunk_matches if cid != chunk_id]
        
        # Create results
        results = self.create_search_results(chunk_matches[:max_results], chunk.content, [])
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        with self._lock:
            cache_stats = self.cache.get_stats()
            
            return {
                **self.stats,
                'cache_stats': cache_stats,
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_searches']),
                'top_queries': dict(sorted(self.stats['popular_queries'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10])
            }
    
    def clear_cache(self):
        """Clear search cache"""
        self.cache.clear()
    
    def optimize(self):
        """Optimize search engine performance"""
        # Clear old cache entries
        self.cache.clear()
        
        # Reset statistics (keep only recent data)
        with self._lock:
            if len(self.stats['search_times']) > 50:
                self.stats['search_times'] = self.stats['search_times'][-50:]
            
            # Keep only top 100 popular queries
            if len(self.stats['popular_queries']) > 100:
                top_queries = dict(sorted(self.stats['popular_queries'].items(), 
                                        key=lambda x: x[1], reverse=True)[:100])
                self.stats['popular_queries'] = defaultdict(int, top_queries)

# Global instance
_search_engine = None

def get_search_engine(workspace_path: str = None) -> SemanticSearchEngine:
    """Get or create semantic search engine"""
    global _search_engine
    
    if _search_engine is None or (workspace_path and _search_engine.workspace_path != workspace_path):
        if workspace_path is None:
            workspace_path = os.getcwd()
        _search_engine = SemanticSearchEngine(workspace_path)
    
    return _search_engine