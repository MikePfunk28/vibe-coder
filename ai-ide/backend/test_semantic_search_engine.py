"""
Test suite for Semantic Search Engine
"""

import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest decorators for basic testing
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        
        @staticmethod
        def skip(reason):
            pass
        
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator

from semantic_search_engine import (
    SemanticSearchEngine, SearchResult, SearchContext, SearchCache, 
    ContextAwareRanker, get_search_engine,
    SENTENCE_TRANSFORMERS_AVAILABLE
)
from code_embedding_generator import CodeEmbeddingGenerator

class TestSearchResult:
    """Test SearchResult functionality"""
    
    def test_search_result_creation(self):
        """Test basic search result creation"""
        result = SearchResult(
            chunk_id="test_chunk",
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            similarity_score=0.8,
            context_relevance=0.6,
            final_score=0.7,
            metadata={"name": "hello", "language": "python"},
            snippet="def hello():",
            highlights=[(0, 3)]
        )
        
        assert result.chunk_id == "test_chunk"
        assert result.similarity_score == 0.8
        assert result.context_relevance == 0.6
        assert result.final_score == 0.7
        assert result.metadata["name"] == "hello"
    
    def test_search_result_to_dict(self):
        """Test search result serialization"""
        result = SearchResult(
            chunk_id="test_chunk",
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            similarity_score=0.8,
            context_relevance=0.6,
            final_score=0.7,
            metadata={"name": "hello"},
            snippet="def hello():",
            highlights=[(0, 3)]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['chunk_id'] == "test_chunk"
        assert result_dict['similarity_score'] == 0.8
        assert result_dict['metadata']['name'] == "hello"

class TestSearchContext:
    """Test SearchContext functionality"""
    
    def test_search_context_creation(self):
        """Test basic search context creation"""
        context = SearchContext(
            current_file="main.py",
            current_language="python",
            open_files=["main.py", "utils.py"],
            recent_files=["test.py", "config.py"]
        )
        
        assert context.current_file == "main.py"
        assert context.current_language == "python"
        assert len(context.open_files) == 2
        assert len(context.recent_files) == 2
    
    def test_search_context_defaults(self):
        """Test search context default values"""
        context = SearchContext()
        
        assert context.current_file is None
        assert context.open_files == []
        assert context.recent_files == []
        assert context.recent_searches == []
        assert context.user_preferences == {}

class TestSearchCache:
    """Test SearchCache functionality"""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = SearchCache(max_size=10, ttl_seconds=60)
        
        query = "test query"
        context = SearchContext()
        options = {"max_results": 10}
        
        # Cache miss
        result = cache.get(query, context, options)
        assert result is None
        
        # Cache put and hit
        test_results = [SearchResult(
            chunk_id="test",
            file_path="test.py",
            content="test content",
            chunk_type="function",
            line_start=1,
            line_end=1,
            similarity_score=0.8,
            context_relevance=0.6,
            final_score=0.7,
            metadata={},
            snippet="test",
            highlights=[]
        )]
        
        cache.put(query, context, options, test_results)
        cached_result = cache.get(query, context, options)
        
        assert cached_result is not None
        assert len(cached_result) == 1
        assert cached_result[0].chunk_id == "test"
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        cache = SearchCache()
        
        context1 = SearchContext(current_file="test1.py")
        context2 = SearchContext(current_file="test2.py")
        options = {"max_results": 10}
        
        key1 = cache._generate_key("query", context1, options)
        key2 = cache._generate_key("query", context2, options)
        
        # Different contexts should generate different keys
        assert key1 != key2
        
        # Same parameters should generate same key
        key3 = cache._generate_key("query", context1, options)
        assert key1 == key3
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = SearchCache()
        stats = cache.get_stats()
        
        assert 'total_entries' in stats
        assert 'valid_entries' in stats
        assert 'expired_entries' in stats
        assert 'max_size' in stats
        assert 'ttl_seconds' in stats

class TestContextAwareRanker:
    """Test ContextAwareRanker functionality"""
    
    def test_ranker_creation(self):
        """Test ranker creation"""
        ranker = ContextAwareRanker()
        
        assert 'python' in ranker.language_weights
        assert 'function' in ranker.chunk_type_weights
        assert ranker.recency_decay > 0
    
    def test_context_relevance_calculation(self):
        """Test context relevance calculation"""
        ranker = ContextAwareRanker()
        
        result = SearchResult(
            chunk_id="test",
            file_path="main.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            similarity_score=0.8,
            context_relevance=0.0,
            final_score=0.8,
            metadata={"language": "python", "name": "hello"},
            snippet="def hello():",
            highlights=[]
        )
        
        # Test same file context
        context = SearchContext(
            current_file="main.py",
            current_language="python"
        )
        
        relevance = ranker.calculate_context_relevance(result, context)
        assert relevance > 0
        
        # Test different file context
        context2 = SearchContext(
            current_file="other.py",
            current_language="javascript"
        )
        
        relevance2 = ranker.calculate_context_relevance(result, context2)
        assert relevance > relevance2  # Same file should have higher relevance
    
    def test_result_ranking(self):
        """Test result ranking"""
        ranker = ContextAwareRanker()
        
        results = [
            SearchResult(
                chunk_id="test1",
                file_path="main.py",
                content="def hello():",
                chunk_type="function",
                line_start=1,
                line_end=1,
                similarity_score=0.6,
                context_relevance=0.0,
                final_score=0.6,
                metadata={"language": "python"},
                snippet="def hello():",
                highlights=[]
            ),
            SearchResult(
                chunk_id="test2",
                file_path="other.py",
                content="def world():",
                chunk_type="function",
                line_start=1,
                line_end=1,
                similarity_score=0.8,
                context_relevance=0.0,
                final_score=0.8,
                metadata={"language": "python"},
                snippet="def world():",
                highlights=[]
            )
        ]
        
        context = SearchContext(current_file="main.py")
        ranked_results = ranker.rank_results(results, context)
        
        # Results should be ranked by final score
        assert len(ranked_results) == 2
        assert ranked_results[0].final_score >= ranked_results[1].final_score

class TestSemanticSearchEngine:
    """Test SemanticSearchEngine functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_files = self._create_sample_files()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_files(self):
        """Create sample files for testing"""
        files = {}
        
        # Python file
        python_content = '''def calculate_sum(a, b):
    """Calculate the sum of two numbers"""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers"""
    return a * b

class Calculator:
    """A simple calculator class"""
    
    def add(self, x, y):
        return calculate_sum(x, y)
    
    def multiply(self, x, y):
        return calculate_product(x, y)
'''
        
        python_file = os.path.join(self.temp_dir, "calculator.py")
        with open(python_file, 'w') as f:
            f.write(python_content)
        files['python'] = python_file
        
        # JavaScript file
        js_content = '''function calculateSum(a, b) {
    // Calculate the sum of two numbers
    return a + b;
}

function calculateProduct(a, b) {
    // Calculate the product of two numbers
    return a * b;
}

class Calculator {
    add(x, y) {
        return calculateSum(x, y);
    }
    
    multiply(x, y) {
        return calculateProduct(x, y);
    }
}
'''
        
        js_file = os.path.join(self.temp_dir, "calculator.js")
        with open(js_file, 'w') as f:
            f.write(js_content)
        files['javascript'] = js_file
        
        return files
    
    def test_search_engine_creation(self):
        """Test search engine creation"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        assert engine.workspace_path == self.temp_dir
        assert engine.embedding_generator is not None
        assert engine.ranker is not None
        assert engine.cache is not None
    
    def test_query_preprocessing(self):
        """Test query preprocessing"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Test basic query
        query, keywords, modifiers = engine.preprocess_query("calculate sum function")
        assert "calculate sum function" in query
        assert "calculate" in keywords
        assert "sum" in keywords
        assert "function" in keywords
        
        # Test query with modifiers
        query, keywords, modifiers = engine.preprocess_query("lang:python type:function calculate")
        assert modifiers['language'] == 'python'
        assert modifiers['chunk_type'] == 'function'
        assert "calculate" in keywords
    
    def test_snippet_creation(self):
        """Test snippet creation with highlights"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        content = "def calculate_sum(a, b):\n    return a + b"
        query = "calculate"
        keywords = ["calculate", "sum"]
        
        snippet, highlights = engine._create_snippet(content, query, keywords)
        
        assert "calculate" in snippet
        assert len(highlights) > 0
        
        # Check that highlights are valid positions
        for start, end in highlights:
            assert 0 <= start < end <= len(snippet)
    
    def test_highlight_merging(self):
        """Test highlight range merging"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Overlapping highlights
        highlights = [(0, 5), (3, 8), (10, 15)]
        merged = engine._merge_highlights(highlights)
        
        assert len(merged) == 2  # First two should be merged
        assert merged[0] == (0, 8)
        assert merged[1] == (10, 15)
    
    def test_text_fallback_search(self):
        """Test text-based fallback search"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Index files first
        embedding_gen = engine.embedding_generator
        for file_path in self.sample_files.values():
            embedding_gen.index_file(file_path)
        
        # Test text search
        results = engine.search_text_fallback("calculate", ["calculate"], max_results=10)
        
        assert len(results) > 0
        
        # Results should be tuples of (chunk_id, score)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
            assert score > 0
    
    def test_result_filtering(self):
        """Test result filtering"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Create test results
        results = [
            SearchResult(
                chunk_id="test1",
                file_path="test.py",
                content="def hello():",
                chunk_type="function",
                line_start=1,
                line_end=1,
                similarity_score=0.8,
                context_relevance=0.6,
                final_score=0.7,
                metadata={"language": "python"},
                snippet="def hello():",
                highlights=[]
            ),
            SearchResult(
                chunk_id="test2",
                file_path="test.js",
                content="function world() {}",
                chunk_type="function",
                line_start=1,
                line_end=1,
                similarity_score=0.7,
                context_relevance=0.5,
                final_score=0.6,
                metadata={"language": "javascript"},
                snippet="function world() {}",
                highlights=[]
            )
        ]
        
        # Test language filter
        modifiers = {"language": "python"}
        filtered = engine.filter_results(results, modifiers, {})
        
        assert len(filtered) == 1
        assert filtered[0].metadata["language"] == "python"
        
        # Test chunk type filter
        modifiers = {"chunk_type": "function"}
        filtered = engine.filter_results(results, modifiers, {})
        
        assert len(filtered) == 2  # Both are functions
        
        # Test minimum score filter
        options = {"min_score": 0.75}
        filtered = engine.filter_results(results, {}, options)
        
        assert len(filtered) == 1  # Only first result has score >= 0.75
    
    def test_basic_search(self):
        """Test basic search functionality"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Index files first
        embedding_gen = engine.embedding_generator
        for file_path in self.sample_files.values():
            embedding_gen.index_file(file_path)
        
        # Test search
        context = SearchContext(current_language="python")
        results = engine.search("calculate", context, max_results=5)
        
        # Should find results even without embeddings
        assert isinstance(results, list)
        
        # If results found, they should be SearchResult objects
        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, 'chunk_id')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'final_score')
    
    def test_search_caching(self):
        """Test search result caching"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        # Index files first
        embedding_gen = engine.embedding_generator
        for file_path in self.sample_files.values():
            embedding_gen.index_file(file_path)
        
        context = SearchContext()
        
        # First search - should be cache miss
        results1 = engine.search("test query", context)
        stats1 = engine.get_stats()
        
        # Second search - should be cache hit
        results2 = engine.search("test query", context)
        stats2 = engine.get_stats()
        
        # Cache hits should increase
        assert stats2['cache_hits'] > stats1['cache_hits']
    
    def test_search_stats(self):
        """Test search statistics tracking"""
        engine = SemanticSearchEngine(workspace_path=self.temp_dir)
        
        stats = engine.get_stats()
        
        assert 'total_searches' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'avg_search_time' in stats
        assert 'cache_stats' in stats
        assert 'top_queries' in stats
    
    def test_global_instance(self):
        """Test global instance management"""
        engine1 = get_search_engine(self.temp_dir)
        engine2 = get_search_engine(self.temp_dir)
        
        # Should be the same instance
        assert engine1 is engine2
        
        # Different workspace should create new instance
        temp_dir2 = tempfile.mkdtemp()
        try:
            engine3 = get_search_engine(temp_dir2)
            assert engine3 is not engine1
        finally:
            shutil.rmtree(temp_dir2)

if __name__ == "__main__":
    # Run basic tests without pytest
    import sys
    
    print("Testing Semantic Search Engine...")
    
    # Test basic functionality
    temp_dir = tempfile.mkdtemp()
    try:
        # Test search result creation
        result = SearchResult(
            chunk_id="test",
            file_path="test.py",
            content="def hello():",
            chunk_type="function",
            line_start=1,
            line_end=1,
            similarity_score=0.8,
            context_relevance=0.6,
            final_score=0.7,
            metadata={"name": "hello"},
            snippet="def hello():",
            highlights=[(0, 3)]
        )
        print(f"✓ SearchResult creation: {result.chunk_id}")
        
        # Test search context
        context = SearchContext(current_file="main.py", current_language="python")
        print(f"✓ SearchContext creation: {context.current_file}")
        
        # Test cache
        cache = SearchCache()
        stats = cache.get_stats()
        print(f"✓ SearchCache creation: {stats['max_size']} max entries")
        
        # Test ranker
        ranker = ContextAwareRanker()
        print(f"✓ ContextAwareRanker creation: {len(ranker.language_weights)} languages")
        
        # Test search engine
        engine = SemanticSearchEngine(workspace_path=temp_dir)
        print(f"✓ SemanticSearchEngine creation: {engine.workspace_path}")
        
        # Test query preprocessing
        query, keywords, modifiers = engine.preprocess_query("lang:python calculate sum")
        print(f"✓ Query preprocessing: {len(keywords)} keywords, {len(modifiers)} modifiers")
        
        # Test snippet creation
        content = "def calculate_sum(a, b):\n    return a + b"
        snippet, highlights = engine._create_snippet(content, "calculate", ["calculate"])
        print(f"✓ Snippet creation: {len(highlights)} highlights")
        
        # Test stats
        stats = engine.get_stats()
        print(f"✓ Stats tracking: {len(stats)} metrics")
        
        print("\nAll basic tests passed!")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("sentence-transformers available - vector search enabled")
        else:
            print("sentence-transformers not available - using text search fallback")
    
    finally:
        shutil.rmtree(temp_dir)