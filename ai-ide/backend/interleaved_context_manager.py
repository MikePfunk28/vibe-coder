"""
Interleaved Context Management System

Implementation of Apple's interleaved context sliding windows technique
for efficient context management in AI-powered IDE.

Based on Apple's research on interleaved context processing and sliding window optimization.
"""

import time
import heapq
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Represents a context window with semantic content and metadata."""
    id: str
    content: str
    semantic_embedding: Optional[np.ndarray]
    relevance_score: float
    priority: int
    last_accessed: datetime
    window_type: str  # 'code', 'documentation', 'conversation', 'search'
    source_file: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    tokens: int = 0
    compression_ratio: float = 1.0
    
    def __post_init__(self):
        if self.tokens == 0:
            # Rough token estimation (4 chars per token average)
            self.tokens = len(self.content) // 4


@dataclass
class InterleaveSlot:
    """Represents a slot in the interleaved context structure."""
    slot_id: str
    windows: List[ContextWindow] = field(default_factory=list)
    max_tokens: int = 2048
    current_tokens: int = 0
    slot_type: str = 'general'  # 'immediate', 'recent', 'background', 'semantic'
    priority_weight: float = 1.0
    
    def add_window(self, window: ContextWindow) -> bool:
        """Add a window to this slot if there's space."""
        if self.current_tokens + window.tokens <= self.max_tokens:
            self.windows.append(window)
            self.current_tokens += window.tokens
            return True
        return False
    
    def remove_window(self, window_id: str) -> bool:
        """Remove a window from this slot."""
        for i, window in enumerate(self.windows):
            if window.id == window_id:
                self.current_tokens -= window.tokens
                del self.windows[i]
                return True
        return False


class InterleaveContextManager:
    """
    Manages context using Apple's interleaved sliding window technique.
    
    Key features:
    - Dynamic window sizing based on task complexity
    - Semantic prioritization for context relevance
    - Memory optimization and context compression
    - Interleaved processing of multiple context streams
    """
    
    def __init__(
        self,
        max_context_length: int = 32768,
        similarity_threshold: float = 0.7,
        max_windows: int = 50,
        compression_enabled: bool = True
    ):
        self.max_context_length = max_context_length
        self.similarity_threshold = similarity_threshold
        self.max_windows = max_windows
        self.compression_enabled = compression_enabled
        
        # Interleaved slots for different types of context
        self.slots: Dict[str, InterleaveSlot] = {
            'immediate': InterleaveSlot('immediate', max_tokens=4096, slot_type='immediate', priority_weight=3.0),
            'recent': InterleaveSlot('recent', max_tokens=8192, slot_type='recent', priority_weight=2.0),
            'semantic': InterleaveSlot('semantic', max_tokens=12288, slot_type='semantic', priority_weight=1.5),
            'background': InterleaveSlot('background', max_tokens=8192, slot_type='background', priority_weight=1.0)
        }
        
        # Context management state
        self.context_cache: Dict[str, ContextWindow] = {}
        self.access_history: List[Tuple[str, datetime]] = []
        self.semantic_clusters: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            'total_windows': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compressions': 0,
            'evictions': 0
        }
        
        logger.info(f"InterleaveContextManager initialized with max_length={max_context_length}")
    
    def add_context(
        self,
        content: str,
        context_type: str = 'code',
        source_file: Optional[str] = None,
        line_range: Optional[Tuple[int, int]] = None,
        priority: int = 1,
        semantic_embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Add new context to the interleaved management system.
        
        Args:
            content: The context content
            context_type: Type of context ('code', 'documentation', etc.)
            source_file: Source file path if applicable
            line_range: Line range in source file
            priority: Priority level (1-5, higher is more important)
            semantic_embedding: Pre-computed semantic embedding
            
        Returns:
            Context window ID
        """
        # Generate unique ID for this context
        context_id = self._generate_context_id(content, source_file, line_range)
        
        # Check if context already exists
        if context_id in self.context_cache:
            self._update_access_time(context_id)
            self.stats['cache_hits'] += 1
            return context_id
        
        self.stats['cache_misses'] += 1
        
        # Create context window
        window = ContextWindow(
            id=context_id,
            content=content,
            semantic_embedding=semantic_embedding,
            relevance_score=self._calculate_initial_relevance(content, context_type),
            priority=priority,
            last_accessed=datetime.now(),
            window_type=context_type,
            source_file=source_file,
            line_range=line_range
        )
        
        # Add to cache
        self.context_cache[context_id] = window
        self.stats['total_windows'] += 1
        
        # Determine appropriate slot based on priority and type
        slot_name = self._determine_slot(window)
        
        # Try to add to the determined slot
        if not self.slots[slot_name].add_window(window):
            # If slot is full, try compression or eviction
            if self.compression_enabled:
                self._compress_slot(slot_name)
                if not self.slots[slot_name].add_window(window):
                    self._evict_from_slot(slot_name, window.tokens)
                    self.slots[slot_name].add_window(window)
            else:
                self._evict_from_slot(slot_name, window.tokens)
                self.slots[slot_name].add_window(window)
        
        # Update semantic clusters
        if semantic_embedding is not None:
            self._update_semantic_clusters(context_id, semantic_embedding)
        
        # Maintain overall context limits
        self._maintain_context_limits()
        
        logger.debug(f"Added context {context_id} to slot {slot_name}")
        return context_id
    
    def get_relevant_context(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        max_tokens: Optional[int] = None,
        context_types: Optional[List[str]] = None
    ) -> List[ContextWindow]:
        """
        Retrieve relevant context using interleaved sliding window approach.
        
        Args:
            query: Query string for context retrieval
            query_embedding: Pre-computed query embedding
            max_tokens: Maximum tokens to return
            context_types: Filter by context types
            
        Returns:
            List of relevant context windows in interleaved order
        """
        if max_tokens is None:
            max_tokens = self.max_context_length // 2
        
        # Get candidates from all slots with interleaved sampling
        candidates = self._get_interleaved_candidates(context_types)
        
        # Calculate relevance scores
        scored_candidates = []
        for window in candidates:
            relevance = self._calculate_relevance(window, query, query_embedding)
            if relevance >= self.similarity_threshold:
                scored_candidates.append((relevance, window))
        
        # Sort by relevance and priority
        scored_candidates.sort(key=lambda x: (x[0], x[1].priority), reverse=True)
        
        # Select windows within token limit using interleaved approach
        selected_windows = []
        total_tokens = 0
        
        # Interleave selection from different slots
        slot_iterators = {
            slot_name: iter([(r, w) for r, w in scored_candidates if w.id in [win.id for win in slot.windows]])
            for slot_name, slot in self.slots.items()
        }
        
        # Round-robin selection from slots
        while total_tokens < max_tokens and any(slot_iterators.values()):
            for slot_name in ['immediate', 'recent', 'semantic', 'background']:
                if slot_name in slot_iterators and slot_iterators[slot_name]:
                    try:
                        relevance, window = next(slot_iterators[slot_name])
                        if total_tokens + window.tokens <= max_tokens:
                            selected_windows.append(window)
                            total_tokens += window.tokens
                            self._update_access_time(window.id)
                        
                        if total_tokens >= max_tokens:
                            break
                    except StopIteration:
                        del slot_iterators[slot_name]
        
        logger.debug(f"Retrieved {len(selected_windows)} context windows ({total_tokens} tokens)")
        return selected_windows
    
    def update_context_relevance(self, context_id: str, relevance_delta: float) -> None:
        """Update the relevance score of a context window based on feedback."""
        if context_id in self.context_cache:
            window = self.context_cache[context_id]
            window.relevance_score = max(0.0, min(1.0, window.relevance_score + relevance_delta))
            self._update_access_time(context_id)
            logger.debug(f"Updated relevance for {context_id}: {window.relevance_score}")
    
    def optimize_windows(self) -> None:
        """Optimize context windows by reorganizing slots and compressing content."""
        logger.info("Starting context window optimization")
        
        # Reorganize windows based on current relevance and access patterns
        all_windows = []
        for slot in self.slots.values():
            all_windows.extend(slot.windows)
            slot.windows.clear()
            slot.current_tokens = 0
        
        # Re-assign windows to optimal slots
        for window in all_windows:
            slot_name = self._determine_slot(window)
            if not self.slots[slot_name].add_window(window):
                # Try compression if slot is full
                if self.compression_enabled:
                    self._compress_slot(slot_name)
                    if not self.slots[slot_name].add_window(window):
                        # Find alternative slot or evict
                        self._find_alternative_slot_or_evict(window)
                else:
                    self._find_alternative_slot_or_evict(window)
        
        # Compress underutilized slots
        for slot_name, slot in self.slots.items():
            if slot.current_tokens < slot.max_tokens * 0.5 and len(slot.windows) > 1:
                self._compress_slot(slot_name)
        
        logger.info("Context window optimization completed")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context management."""
        slot_stats = {}
        for slot_name, slot in self.slots.items():
            slot_stats[slot_name] = {
                'windows': len(slot.windows),
                'tokens': slot.current_tokens,
                'utilization': slot.current_tokens / slot.max_tokens
            }
        
        return {
            'total_windows': len(self.context_cache),
            'total_tokens': sum(slot.current_tokens for slot in self.slots.values()),
            'max_tokens': sum(slot.max_tokens for slot in self.slots.values()),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'compressions': self.stats['compressions'],
            'evictions': self.stats['evictions'],
            'slots': slot_stats
        }
    
    def clear_context(self, context_types: Optional[List[str]] = None) -> None:
        """Clear context windows, optionally filtered by type."""
        if context_types is None:
            # Clear all context
            self.context_cache.clear()
            for slot in self.slots.values():
                slot.windows.clear()
                slot.current_tokens = 0
            self.semantic_clusters.clear()
            logger.info("Cleared all context")
        else:
            # Clear specific types
            to_remove = []
            for context_id, window in self.context_cache.items():
                if window.window_type in context_types:
                    to_remove.append(context_id)
            
            for context_id in to_remove:
                self._remove_context(context_id)
            
            logger.info(f"Cleared context types: {context_types}")
    
    # Private methods
    
    def _generate_context_id(self, content: str, source_file: Optional[str], line_range: Optional[Tuple[int, int]]) -> str:
        """Generate unique ID for context content."""
        hash_input = content
        if source_file:
            hash_input += f"|{source_file}"
        if line_range:
            hash_input += f"|{line_range[0]}-{line_range[1]}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _calculate_initial_relevance(self, content: str, context_type: str) -> float:
        """Calculate initial relevance score for new context."""
        base_score = 0.5
        
        # Boost score based on context type
        type_boosts = {
            'code': 0.2,
            'documentation': 0.1,
            'conversation': 0.15,
            'search': 0.1
        }
        
        base_score += type_boosts.get(context_type, 0.0)
        
        # Boost based on content characteristics
        if any(keyword in content.lower() for keyword in ['function', 'class', 'def', 'import']):
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _determine_slot(self, window: ContextWindow) -> str:
        """Determine the appropriate slot for a context window."""
        age = datetime.now() - window.last_accessed
        
        if window.priority >= 4 or age < timedelta(minutes=5):
            return 'immediate'
        elif window.priority >= 3 or age < timedelta(hours=1):
            return 'recent'
        elif window.semantic_embedding is not None and window.relevance_score > 0.6:
            return 'semantic'
        else:
            return 'background'
    
    def _calculate_relevance(
        self,
        window: ContextWindow,
        query: str,
        query_embedding: Optional[np.ndarray]
    ) -> float:
        """Calculate relevance score between window and query."""
        # Base relevance from window
        relevance = window.relevance_score
        
        # Text-based similarity (simple keyword matching)
        query_words = set(query.lower().split())
        content_words = set(window.content.lower().split())
        text_similarity = len(query_words & content_words) / max(1, len(query_words | content_words))
        relevance += text_similarity * 0.3
        
        # Semantic similarity if embeddings available
        if query_embedding is not None and window.semantic_embedding is not None:
            semantic_similarity = np.dot(query_embedding, window.semantic_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(window.semantic_embedding)
            )
            relevance += semantic_similarity * 0.4
        
        # Recency boost
        age = datetime.now() - window.last_accessed
        recency_boost = max(0, 0.2 - age.total_seconds() / 3600 * 0.01)  # Decay over hours
        relevance += recency_boost
        
        # Priority boost
        priority_boost = (window.priority - 1) * 0.1
        relevance += priority_boost
        
        return min(1.0, relevance)
    
    def _get_interleaved_candidates(self, context_types: Optional[List[str]]) -> List[ContextWindow]:
        """Get candidates from all slots using interleaved sampling."""
        candidates = []
        
        # Collect from each slot proportionally
        for slot_name, slot in self.slots.items():
            slot_candidates = slot.windows.copy()
            
            # Filter by context types if specified
            if context_types:
                slot_candidates = [w for w in slot_candidates if w.window_type in context_types]
            
            # Sort by relevance within slot
            slot_candidates.sort(key=lambda w: w.relevance_score, reverse=True)
            
            # Take proportional sample based on slot priority
            sample_size = min(len(slot_candidates), int(len(slot_candidates) * slot.priority_weight))
            candidates.extend(slot_candidates[:sample_size])
        
        return candidates
    
    def _update_access_time(self, context_id: str) -> None:
        """Update access time for a context window."""
        if context_id in self.context_cache:
            self.context_cache[context_id].last_accessed = datetime.now()
            self.access_history.append((context_id, datetime.now()))
            
            # Trim access history
            if len(self.access_history) > 1000:
                self.access_history = self.access_history[-500:]
    
    def _update_semantic_clusters(self, context_id: str, embedding: np.ndarray) -> None:
        """Update semantic clusters for efficient similarity search."""
        # Simple clustering based on embedding similarity
        cluster_id = None
        max_similarity = 0.0
        
        for cluster, members in self.semantic_clusters.items():
            if members:
                # Get representative embedding (first member)
                rep_id = members[0]
                if rep_id in self.context_cache:
                    rep_embedding = self.context_cache[rep_id].semantic_embedding
                    if rep_embedding is not None:
                        similarity = np.dot(embedding, rep_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(rep_embedding)
                        )
                        if similarity > max_similarity and similarity > 0.8:
                            max_similarity = similarity
                            cluster_id = cluster
        
        if cluster_id is None:
            # Create new cluster
            cluster_id = f"cluster_{len(self.semantic_clusters)}"
        
        self.semantic_clusters[cluster_id].append(context_id)
    
    def _compress_slot(self, slot_name: str) -> None:
        """Compress content in a slot to make room for new windows."""
        slot = self.slots[slot_name]
        if not slot.windows:
            return
        
        # Find windows that can be compressed
        compressible = [w for w in slot.windows if w.compression_ratio == 1.0 and len(w.content) > 500]
        
        if not compressible:
            return
        
        # Compress least relevant windows first
        compressible.sort(key=lambda w: w.relevance_score)
        
        for window in compressible[:max(1, len(compressible)//2)]:  # Compress at least 1, up to half
            original_length = len(window.content)
            # Simple compression: keep first and last parts, summarize middle
            if original_length > 1000:
                compressed_content = (
                    window.content[:300] + 
                    f"\n... [compressed {original_length - 600} chars] ...\n" +
                    window.content[-300:]
                )
            else:
                # For smaller content, just truncate middle
                compressed_content = (
                    window.content[:200] + 
                    f"\n... [compressed {original_length - 400} chars] ...\n" +
                    window.content[-200:]
                )
            
            tokens_saved = window.tokens - len(compressed_content) // 4
            window.content = compressed_content
            window.tokens = len(compressed_content) // 4
            window.compression_ratio = len(compressed_content) / original_length
            slot.current_tokens -= tokens_saved
            
            self.stats['compressions'] += 1
            logger.debug(f"Compressed window {window.id}, saved {tokens_saved} tokens")
    
    def _evict_from_slot(self, slot_name: str, tokens_needed: int) -> None:
        """Evict windows from a slot to make room."""
        slot = self.slots[slot_name]
        
        # Sort by relevance and access time (least relevant and oldest first)
        eviction_candidates = sorted(
            slot.windows,
            key=lambda w: (w.relevance_score, w.last_accessed)
        )
        
        tokens_freed = 0
        for window in eviction_candidates:
            if tokens_freed >= tokens_needed:
                break
            
            slot.remove_window(window.id)
            del self.context_cache[window.id]
            tokens_freed += window.tokens
            self.stats['evictions'] += 1
            logger.debug(f"Evicted window {window.id} from slot {slot_name}")
    
    def _find_alternative_slot_or_evict(self, window: ContextWindow) -> None:
        """Find alternative slot for window or evict if necessary."""
        # Try other slots in order of preference
        slot_order = ['background', 'semantic', 'recent', 'immediate']
        
        for slot_name in slot_order:
            if self.slots[slot_name].add_window(window):
                logger.debug(f"Moved window {window.id} to alternative slot {slot_name}")
                return
        
        # If no slot can accommodate, try compression and eviction
        for slot_name in slot_order:
            if self.compression_enabled:
                self._compress_slot(slot_name)
                if self.slots[slot_name].add_window(window):
                    return
            
            self._evict_from_slot(slot_name, window.tokens)
            if self.slots[slot_name].add_window(window):
                return
        
        # If still can't fit, don't add the window
        logger.warning(f"Could not find slot for window {window.id}")
    
    def _maintain_context_limits(self) -> None:
        """Maintain overall context limits across all slots."""
        # Check window count limit
        if len(self.context_cache) > self.max_windows:
            excess_windows = len(self.context_cache) - self.max_windows
            # Evict oldest, least relevant windows
            all_windows = list(self.context_cache.values())
            all_windows.sort(key=lambda w: (w.relevance_score, w.last_accessed))
            
            for window in all_windows[:excess_windows]:
                self._remove_context(window.id)
        
        # Check token limit
        total_tokens = sum(slot.current_tokens for slot in self.slots.values())
        
        if total_tokens > self.max_context_length:
            # Need to free up space
            excess_tokens = total_tokens - self.max_context_length
            
            # Try compression first
            if self.compression_enabled:
                for slot_name in ['background', 'semantic', 'recent']:
                    if excess_tokens <= 0:
                        break
                    self._compress_slot(slot_name)
                    new_total = sum(slot.current_tokens for slot in self.slots.values())
                    excess_tokens = new_total - self.max_context_length
            
            # If still over limit, evict from background slot first, then others
            if excess_tokens > 0:
                for slot_name in ['background', 'semantic', 'recent']:
                    if excess_tokens <= 0:
                        break
                    
                    slot = self.slots[slot_name]
                    tokens_to_free = min(excess_tokens, max(1, slot.current_tokens // 2))
                    
                    if tokens_to_free > 0:
                        self._evict_from_slot(slot_name, tokens_to_free)
                        new_total = sum(slot.current_tokens for slot in self.slots.values())
                        excess_tokens = new_total - self.max_context_length
    
    def _remove_context(self, context_id: str) -> None:
        """Remove a context window completely."""
        if context_id not in self.context_cache:
            return
        
        # Remove from cache
        del self.context_cache[context_id]
        
        # Remove from slots
        for slot in self.slots.values():
            slot.remove_window(context_id)
        
        # Remove from semantic clusters
        for cluster_members in self.semantic_clusters.values():
            if context_id in cluster_members:
                cluster_members.remove(context_id)