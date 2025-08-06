"""
Tests for InterleaveContextManager

Tests the Apple-inspired interleaved context sliding windows implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from interleaved_context_manager import (
    InterleaveContextManager,
    ContextWindow,
    InterleaveSlot
)


class TestContextWindow:
    """Test ContextWindow dataclass functionality."""
    
    def test_context_window_creation(self):
        """Test basic context window creation."""
        window = ContextWindow(
            id="test_id",
            content="def hello(): pass",
            semantic_embedding=None,
            relevance_score=0.8,
            priority=2,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        assert window.id == "test_id"
        assert window.content == "def hello(): pass"
        assert window.relevance_score == 0.8
        assert window.priority == 2
        assert window.window_type == "code"
        assert window.tokens > 0  # Should auto-calculate tokens
    
    def test_token_calculation(self):
        """Test automatic token calculation."""
        content = "a" * 400  # 400 characters
        window = ContextWindow(
            id="test",
            content=content,
            semantic_embedding=None,
            relevance_score=0.5,
            priority=1,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        # Should be approximately 100 tokens (400 chars / 4)
        assert window.tokens == 100


class TestInterleaveSlot:
    """Test InterleaveSlot functionality."""
    
    def test_slot_creation(self):
        """Test basic slot creation."""
        slot = InterleaveSlot(
            slot_id="test_slot",
            max_tokens=1000,
            slot_type="immediate"
        )
        
        assert slot.slot_id == "test_slot"
        assert slot.max_tokens == 1000
        assert slot.current_tokens == 0
        assert slot.slot_type == "immediate"
        assert len(slot.windows) == 0
    
    def test_add_window_success(self):
        """Test successfully adding a window to slot."""
        slot = InterleaveSlot("test", max_tokens=1000)
        window = ContextWindow(
            id="w1",
            content="test content",
            semantic_embedding=None,
            relevance_score=0.5,
            priority=1,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        result = slot.add_window(window)
        
        assert result is True
        assert len(slot.windows) == 1
        assert slot.current_tokens == window.tokens
    
    def test_add_window_exceeds_limit(self):
        """Test adding window that exceeds token limit."""
        slot = InterleaveSlot("test", max_tokens=10)
        window = ContextWindow(
            id="w1",
            content="a" * 100,  # Will be ~25 tokens
            semantic_embedding=None,
            relevance_score=0.5,
            priority=1,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        result = slot.add_window(window)
        
        assert result is False
        assert len(slot.windows) == 0
        assert slot.current_tokens == 0
    
    def test_remove_window(self):
        """Test removing a window from slot."""
        slot = InterleaveSlot("test", max_tokens=1000)
        window = ContextWindow(
            id="w1",
            content="test",
            semantic_embedding=None,
            relevance_score=0.5,
            priority=1,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        slot.add_window(window)
        result = slot.remove_window("w1")
        
        assert result is True
        assert len(slot.windows) == 0
        assert slot.current_tokens == 0


class TestInterleaveContextManager:
    """Test InterleaveContextManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = InterleaveContextManager(
            max_context_length=10000,
            similarity_threshold=0.7,
            max_windows=20
        )
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.max_context_length == 10000
        assert self.manager.similarity_threshold == 0.7
        assert self.manager.max_windows == 20
        
        # Check slots are created
        assert 'immediate' in self.manager.slots
        assert 'recent' in self.manager.slots
        assert 'semantic' in self.manager.slots
        assert 'background' in self.manager.slots
    
    def test_add_context_basic(self):
        """Test basic context addition."""
        content = "def hello_world(): print('Hello, World!')"
        context_id = self.manager.add_context(
            content=content,
            context_type="code",
            priority=2
        )
        
        assert context_id is not None
        assert context_id in self.manager.context_cache
        
        window = self.manager.context_cache[context_id]
        assert window.content == content
        assert window.window_type == "code"
        assert window.priority == 2
    
    def test_add_context_with_embedding(self):
        """Test adding context with semantic embedding."""
        content = "function calculateSum(a, b) { return a + b; }"
        embedding = np.random.rand(384)  # Mock embedding
        
        context_id = self.manager.add_context(
            content=content,
            context_type="code",
            semantic_embedding=embedding
        )
        
        window = self.manager.context_cache[context_id]
        assert np.array_equal(window.semantic_embedding, embedding)
    
    def test_add_duplicate_context(self):
        """Test adding duplicate context returns same ID."""
        content = "def test(): pass"
        
        id1 = self.manager.add_context(content, "code")
        id2 = self.manager.add_context(content, "code")
        
        assert id1 == id2
        assert self.manager.stats['cache_hits'] > 0
    
    def test_get_relevant_context_empty(self):
        """Test getting relevant context when no context exists."""
        results = self.manager.get_relevant_context("test query")
        assert len(results) == 0
    
    def test_get_relevant_context_with_content(self):
        """Test getting relevant context with existing content."""
        # Add some context
        self.manager.add_context("def hello(): print('hello')", "code", priority=3)
        self.manager.add_context("class TestClass: pass", "code", priority=2)
        self.manager.add_context("# This is documentation", "documentation", priority=1)
        
        results = self.manager.get_relevant_context("hello function")
        
        assert len(results) > 0
        # Should prioritize based on relevance and priority
        assert any("hello" in window.content for window in results)
    
    def test_get_relevant_context_with_type_filter(self):
        """Test getting relevant context filtered by type."""
        self.manager.add_context("def hello(): pass", "code")
        self.manager.add_context("# Documentation", "documentation")
        
        results = self.manager.get_relevant_context(
            "test query",
            context_types=["code"]
        )
        
        # Should only return code contexts
        for window in results:
            assert window.window_type == "code"
    
    def test_update_context_relevance(self):
        """Test updating context relevance."""
        context_id = self.manager.add_context("test content", "code")
        original_relevance = self.manager.context_cache[context_id].relevance_score
        
        self.manager.update_context_relevance(context_id, 0.2)
        
        new_relevance = self.manager.context_cache[context_id].relevance_score
        assert new_relevance > original_relevance
    
    def test_slot_determination(self):
        """Test slot determination logic."""
        # High priority should go to immediate
        window_high = ContextWindow(
            id="high",
            content="test",
            semantic_embedding=None,
            relevance_score=0.8,
            priority=5,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        slot = self.manager._determine_slot(window_high)
        assert slot == "immediate"
        
        # Old, low priority should go to background
        window_old = ContextWindow(
            id="old",
            content="test",
            semantic_embedding=None,
            relevance_score=0.3,
            priority=1,
            last_accessed=datetime.now() - timedelta(hours=5),
            window_type="code"
        )
        
        slot = self.manager._determine_slot(window_old)
        assert slot == "background"
    
    def test_context_compression(self):
        """Test context compression functionality."""
        # Add a large context that can be compressed
        large_content = "a" * 2000  # Large content
        context_id = self.manager.add_context(large_content, "code")
        
        # Force compression
        slot_name = "background"
        self.manager.slots[slot_name].windows.append(self.manager.context_cache[context_id])
        self.manager.slots[slot_name].current_tokens += self.manager.context_cache[context_id].tokens
        
        original_length = len(self.manager.context_cache[context_id].content)
        self.manager._compress_slot(slot_name)
        
        compressed_length = len(self.manager.context_cache[context_id].content)
        assert compressed_length < original_length
        assert self.manager.context_cache[context_id].compression_ratio < 1.0
    
    def test_context_eviction(self):
        """Test context eviction when limits are exceeded."""
        # Fill up a slot
        slot = self.manager.slots['background']
        original_max = slot.max_tokens
        slot.max_tokens = 100  # Reduce for testing
        
        # Add contexts that exceed limit
        for i in range(5):
            content = f"def function_{i}(): pass"
            context_id = self.manager.add_context(content, "code", priority=1)
            # Manually add to background slot for testing
            window = self.manager.context_cache[context_id]
            if slot.current_tokens + window.tokens <= slot.max_tokens:
                slot.add_window(window)
        
        initial_count = len(slot.windows)
        
        # Force eviction
        self.manager._evict_from_slot('background', 50)
        
        # Should have fewer windows
        assert len(slot.windows) < initial_count
        
        # Restore original max
        slot.max_tokens = original_max
    
    def test_semantic_clustering(self):
        """Test semantic clustering functionality."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.9, 0.1, 0.0])  # Similar to embedding1
        embedding3 = np.array([0.0, 1.0, 0.0])  # Different
        
        id1 = self.manager.add_context("code 1", "code", semantic_embedding=embedding1)
        id2 = self.manager.add_context("code 2", "code", semantic_embedding=embedding2)
        id3 = self.manager.add_context("code 3", "code", semantic_embedding=embedding3)
        
        # Check that similar embeddings are clustered together
        assert len(self.manager.semantic_clusters) > 0
    
    def test_interleaved_candidate_selection(self):
        """Test interleaved candidate selection from slots."""
        # Add contexts with different characteristics to force different slots
        contexts = []
        
        # Add high priority context (should go to immediate)
        id1 = self.manager.add_context("def high_priority(): pass", "code", priority=5)
        contexts.append(id1)
        
        # Add medium priority context (should go to recent)
        id2 = self.manager.add_context("def medium_priority(): pass", "code", priority=3)
        contexts.append(id2)
        
        # Add low priority context with semantic embedding (should go to semantic)
        embedding = np.random.rand(384)
        id3 = self.manager.add_context("def semantic_func(): pass", "code", priority=1, semantic_embedding=embedding)
        # Manually set high relevance to ensure it goes to semantic slot
        self.manager.context_cache[id3].relevance_score = 0.8
        contexts.append(id3)
        
        # Add old, low priority context (should go to background)
        id4 = self.manager.add_context("def background_func(): pass", "code", priority=1)
        # Make it old
        self.manager.context_cache[id4].last_accessed = datetime.now() - timedelta(hours=2)
        contexts.append(id4)
        
        # Manually redistribute to ensure different slots
        self.manager.optimize_windows()
        
        candidates = self.manager._get_interleaved_candidates(None)
        
        # Should get candidates from multiple slots
        assert len(candidates) > 0
        
        # Check that candidates come from different slots
        slot_types = set()
        for candidate in candidates:
            for slot_name, slot in self.manager.slots.items():
                if candidate in [w for w in slot.windows]:
                    slot_types.add(slot_name)
        
        # Should have candidates from multiple slot types (at least 2)
        assert len(slot_types) >= 2
    
    def test_context_stats(self):
        """Test context statistics reporting."""
        # Add some contexts
        for i in range(5):
            self.manager.add_context(f"content {i}", "code")
        
        stats = self.manager.get_context_stats()
        
        assert 'total_windows' in stats
        assert 'total_tokens' in stats
        assert 'cache_hit_rate' in stats
        assert 'slots' in stats
        
        assert stats['total_windows'] == 5
        assert stats['total_tokens'] > 0
    
    def test_clear_context_all(self):
        """Test clearing all context."""
        # Add some contexts
        self.manager.add_context("content 1", "code")
        self.manager.add_context("content 2", "documentation")
        
        assert len(self.manager.context_cache) == 2
        
        self.manager.clear_context()
        
        assert len(self.manager.context_cache) == 0
        for slot in self.manager.slots.values():
            assert len(slot.windows) == 0
    
    def test_clear_context_by_type(self):
        """Test clearing context by type."""
        self.manager.add_context("code content", "code")
        self.manager.add_context("doc content", "documentation")
        
        assert len(self.manager.context_cache) == 2
        
        self.manager.clear_context(["code"])
        
        # Should only have documentation left
        assert len(self.manager.context_cache) == 1
        remaining = list(self.manager.context_cache.values())[0]
        assert remaining.window_type == "documentation"
    
    def test_optimize_windows(self):
        """Test window optimization."""
        # Add various contexts
        for i in range(10):
            priority = i % 3 + 1
            relevance = 0.3 + (i % 5) * 0.1
            context_id = self.manager.add_context(f"content {i}", "code", priority=priority)
            self.manager.context_cache[context_id].relevance_score = relevance
        
        # Run optimization
        self.manager.optimize_windows()
        
        # Should still have contexts
        assert len(self.manager.context_cache) > 0
        
        # Windows should be distributed across slots
        total_windows = sum(len(slot.windows) for slot in self.manager.slots.values())
        assert total_windows > 0
    
    def test_memory_optimization(self):
        """Test memory optimization under pressure."""
        # Set very low limits to force optimization
        manager = InterleaveContextManager(
            max_context_length=1000,
            max_windows=5
        )
        
        # Add many contexts to trigger optimization
        for i in range(20):
            content = f"def function_{i}(): " + "pass " * 50  # Make content larger
            manager.add_context(content, "code", priority=i % 3 + 1)
        
        # Should not exceed limits
        total_tokens = sum(slot.current_tokens for slot in manager.slots.values())
        assert total_tokens <= manager.max_context_length
        assert len(manager.context_cache) <= manager.max_windows
    
    def test_relevance_calculation(self):
        """Test relevance calculation between window and query."""
        window = ContextWindow(
            id="test",
            content="def calculate_sum(a, b): return a + b",
            semantic_embedding=np.array([1.0, 0.0, 0.0]),
            relevance_score=0.5,
            priority=2,
            last_accessed=datetime.now(),
            window_type="code"
        )
        
        query = "sum calculation function"
        query_embedding = np.array([0.9, 0.1, 0.0])
        
        relevance = self.manager._calculate_relevance(window, query, query_embedding)
        
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # Should be higher than base relevance due to matches
    
    def test_access_time_tracking(self):
        """Test access time tracking and updates."""
        context_id = self.manager.add_context("test content", "code")
        original_time = self.manager.context_cache[context_id].last_accessed
        
        # Simulate some time passing
        import time
        time.sleep(0.01)
        
        self.manager._update_access_time(context_id)
        new_time = self.manager.context_cache[context_id].last_accessed
        
        assert new_time > original_time
        assert len(self.manager.access_history) > 0


if __name__ == "__main__":
    pytest.main([__file__])