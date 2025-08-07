"""
Test suite for Internet-Enabled Reasoning System

Tests deep reasoning capabilities with real-time information retrieval,
context-aware web search, documentation lookup, and technology trend analysis.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from internet_enabled_reasoning import (
    InternetEnabledReasoningEngine, ReasoningContext, ReasoningStep, ReasoningResult,
    DocumentationSearcher, TechnologyTrendAnalyzer, ContextAwareSearchEngine
)
from web_search_agent import WebSearchAgent, SearchResult, SearchQuery


class TestReasoningContext:
    """Test ReasoningContext dataclass"""
    
    def test_reasoning_context_creation(self):
        """Test creating a ReasoningContext"""
        context = ReasoningContext(
            query="How to handle async in Python",
            language="python",
            project_type="web_app",
            error_message="SyntaxError: invalid syntax"
        )
        
        assert context.query == "How to handle async in Python"
        assert context.language == "python"
        assert context.project_type == "web_app"
        assert context.error_message == "SyntaxError: invalid syntax"
        assert isinstance(context.timestamp, datetime)
    
    def test_reasoning_context_defaults(self):
        """Test ReasoningContext with default values"""
        context = ReasoningContext(query="test query")
        
        assert context.query == "test query"
        assert context.code_context is None
        assert context.file_path is None
        assert context.language is None
        assert context.project_type is None
        assert context.error_message is None
        assert context.user_intent is None
        assert context.timestamp is not None


class TestReasoningStep:
    """Test ReasoningStep dataclass"""
    
    def test_reasoning_step_creation(self):
        """Test creating a ReasoningStep"""
        step = ReasoningStep(
            step_type="search",
            description="Performed web search",
            input_data={"query": "test"},
            output_data={"results": 5},
            confidence=0.8
        )
        
        assert step.step_type == "search"
        assert step.description == "Performed web search"
        assert step.input_data == {"query": "test"}
        assert step.output_data == {"results": 5}
        assert step.confidence == 0.8
        assert isinstance(step.timestamp, datetime)


class TestReasoningResult:
    """Test ReasoningResult dataclass"""
    
    def test_reasoning_result_creation(self):
        """Test creating a ReasoningResult"""
        steps = [ReasoningStep("analyze", "test", {}, {}, 0.9)]
        sources = [SearchResult("Title", "https://example.com", "Snippet", "Engine", 0.8)]
        
        result = ReasoningResult(
            query="test query",
            answer="test answer",
            confidence=0.85,
            reasoning_steps=steps,
            sources=sources,
            recommendations=["rec1", "rec2"],
            code_examples=["code1"],
            related_topics=["topic1"]
        )
        
        assert result.query == "test query"
        assert result.answer == "test answer"
        assert result.confidence == 0.85
        assert len(result.reasoning_steps) == 1
        assert len(result.sources) == 1
        assert result.recommendations == ["rec1", "rec2"]
        assert result.code_examples == ["code1"]
        assert result.related_topics == ["topic1"]
        assert isinstance(result.timestamp, datetime)


class TestDocumentationSearcher:
    """Test DocumentationSearcher functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_web_search_agent = Mock(spec=WebSearchAgent)
        self.doc_searcher = DocumentationSearcher(self.mock_web_search_agent)
    
    def test_initialization(self):
        """Test DocumentationSearcher initialization"""
        assert self.doc_searcher.web_search_agent == self.mock_web_search_agent
        assert len(self.doc_searcher.doc_domains) > 0
        assert "docs.python.org" in self.doc_searcher.doc_domains
        assert "stackoverflow.com" in self.doc_searcher.doc_domains
    
    def test_enhance_documentation_query(self):
        """Test query enhancement for documentation search"""
        enhanced = self.doc_searcher._enhance_documentation_query(
            "async await", "python", "fastapi"
        )
        
        assert "async await" in enhanced
        assert "python" in enhanced
        assert "fastapi" in enhanced
        assert "documentation" in enhanced
    
    def test_prioritize_documentation_results(self):
        """Test prioritization of documentation results"""
        results = [
            SearchResult("Regular Result", "https://example.com", "snippet", "Engine", 0.5),
            SearchResult("Python Docs", "https://docs.python.org/guide", "official docs", "Engine", 0.6),
            SearchResult("Stack Overflow", "https://stackoverflow.com/questions/123", "tutorial guide", "Engine", 0.7)
        ]
        
        prioritized = self.doc_searcher._prioritize_documentation_results(results)
        
        # Should be sorted by relevance (which was boosted for doc domains)
        assert prioritized[0].url == "https://stackoverflow.com/questions/123"  # Highest after boost
        assert prioritized[1].url == "https://docs.python.org/guide"  # Boosted for doc domain
        assert prioritized[2].url == "https://example.com"  # Lowest relevance
    
    @pytest.mark.asyncio
    async def test_search_documentation(self):
        """Test documentation search"""
        mock_results = [
            SearchResult("Python Async Guide", "https://docs.python.org/async", "async guide", "Engine", 0.9)
        ]
        
        self.mock_web_search_agent.search = AsyncMock(return_value=mock_results)
        
        results = await self.doc_searcher.search_documentation("async await", "python", "fastapi")
        
        assert len(results) == 1
        assert results[0].title == "Python Async Guide"
        self.mock_web_search_agent.search.assert_called_once()


class TestTechnologyTrendAnalyzer:
    """Test TechnologyTrendAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_web_search_agent = Mock(spec=WebSearchAgent)
        self.trend_analyzer = TechnologyTrendAnalyzer(self.mock_web_search_agent)
    
    def test_initialization(self):
        """Test TechnologyTrendAnalyzer initialization"""
        assert self.trend_analyzer.web_search_agent == self.mock_web_search_agent
        assert len(self.trend_analyzer.trend_sources) > 0
        assert "github.com" in self.trend_analyzer.trend_sources
    
    def test_analyze_trend_results(self):
        """Test trend analysis from search results"""
        results = [
            SearchResult("Python 2024 Trends", "https://example.com", "new features performance", "Engine", 0.9),
            SearchResult("Latest Python Updates", "https://example.com", "security best practice", "Engine", 0.8),
            SearchResult("Python Migration Guide", "https://example.com", "migration modern", "Engine", 0.7)
        ]
        
        analysis = self.trend_analyzer._analyze_trend_results(results, "python")
        
        assert analysis["technology"] == "python"
        assert "themes" in analysis
        assert "recommendations" in analysis
        assert len(analysis["recommendations"]) > 0
        assert analysis["source_count"] == 3
    
    @pytest.mark.asyncio
    async def test_analyze_technology_trends(self):
        """Test technology trend analysis"""
        mock_results = [
            SearchResult("Python Trends 2024", "https://example.com", "performance security", "Engine", 0.9)
        ]
        
        self.mock_web_search_agent.search = AsyncMock(return_value=mock_results)
        
        analysis = await self.trend_analyzer.analyze_technology_trends("python", "async programming")
        
        assert analysis["technology"] == "python"
        assert "themes" in analysis
        assert "recommendations" in analysis
        # Should have made multiple search calls for different trend queries
        assert self.mock_web_search_agent.search.call_count >= 1


class TestContextAwareSearchEngine:
    """Test ContextAwareSearchEngine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_web_search_agent = Mock(spec=WebSearchAgent)
        self.mock_web_search_agent._merge_and_deduplicate_results = Mock(side_effect=lambda x: x)
        self.context_search = ContextAwareSearchEngine(self.mock_web_search_agent)
    
    def test_initialization(self):
        """Test ContextAwareSearchEngine initialization"""
        assert self.context_search.web_search_agent == self.mock_web_search_agent
        assert isinstance(self.context_search.documentation_searcher, DocumentationSearcher)
        assert isinstance(self.context_search.trend_analyzer, TechnologyTrendAnalyzer)
    
    def test_determine_search_strategy(self):
        """Test search strategy determination"""
        # Error resolution strategy
        error_context = ReasoningContext(
            query="fix bug",
            error_message="TypeError: unsupported operand"
        )
        assert self.context_search._determine_search_strategy(error_context) == "error_resolution"
        
        # Library usage strategy
        library_context = ReasoningContext(
            query="use requests",
            code_context="import requests\nfrom flask import Flask"
        )
        assert self.context_search._determine_search_strategy(library_context) == "library_usage"
        
        # Language specific strategy
        lang_context = ReasoningContext(
            query="async programming",
            language="python"
        )
        assert self.context_search._determine_search_strategy(lang_context) == "language_specific"
        
        # Project specific strategy
        project_context = ReasoningContext(
            query="authentication",
            project_type="web_app"
        )
        assert self.context_search._determine_search_strategy(project_context) == "project_specific"
        
        # General strategy
        general_context = ReasoningContext(query="programming concepts")
        assert self.context_search._determine_search_strategy(general_context) == "general_coding"
    
    def test_generate_contextual_queries(self):
        """Test contextual query generation"""
        context = ReasoningContext(
            query="async programming",
            language="python",
            error_message="SyntaxError"
        )
        
        # Test error resolution queries
        error_queries = self.context_search._generate_contextual_queries(context, "error_resolution")
        assert len(error_queries) <= 4
        assert any("fix" in query.lower() for query in error_queries)
        
        # Test language specific queries
        lang_queries = self.context_search._generate_contextual_queries(context, "language_specific")
        assert len(lang_queries) <= 4
        assert any("python" in query.lower() for query in lang_queries)
    
    def test_apply_contextual_filtering(self):
        """Test contextual filtering of results"""
        context = ReasoningContext(
            query="async programming",
            language="python",
            project_type="web_app"
        )
        
        results = [
            SearchResult("Python Async Guide", "https://example.com", "python async tutorial", "Engine", 0.5),
            SearchResult("JavaScript Async", "https://example.com", "javascript async", "Engine", 0.6),
            SearchResult("Web App Architecture", "https://example.com", "web_app patterns", "Engine", 0.4)
        ]
        
        filtered = self.context_search._apply_contextual_filtering(results, context)
        
        # Results should be sorted by relevance (boosted by context matches)
        assert filtered[0].title == "Python Async Guide"  # Should have highest relevance after boost
        assert all(result.relevance_score <= 1.0 for result in filtered)
    
    @pytest.mark.asyncio
    async def test_search_coding_problem(self):
        """Test coding problem search"""
        context = ReasoningContext(
            query="async programming",
            language="python"
        )
        
        mock_results = [
            SearchResult("Python Async Tutorial", "https://example.com", "async await", "Engine", 0.9)
        ]
        
        self.mock_web_search_agent.search = AsyncMock(return_value=mock_results)
        
        results = await self.context_search.search_coding_problem(context)
        
        assert len(results) >= 1
        # Should have called search multiple times for different queries
        assert self.mock_web_search_agent.search.call_count >= 1


class TestInternetEnabledReasoningEngine:
    """Test InternetEnabledReasoningEngine main functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_web_search_agent = Mock(spec=WebSearchAgent)
        self.reasoning_engine = InternetEnabledReasoningEngine(self.mock_web_search_agent)
    
    def test_initialization(self):
        """Test InternetEnabledReasoningEngine initialization"""
        assert self.reasoning_engine.web_search_agent == self.mock_web_search_agent
        assert isinstance(self.reasoning_engine.context_search_engine, ContextAwareSearchEngine)
        assert isinstance(self.reasoning_engine.documentation_searcher, DocumentationSearcher)
        assert isinstance(self.reasoning_engine.trend_analyzer, TechnologyTrendAnalyzer)
        assert self.reasoning_engine.max_reasoning_steps == 10
        assert self.reasoning_engine.confidence_threshold == 0.7
    
    def test_classify_query_type(self):
        """Test query type classification"""
        assert self.reasoning_engine._classify_query_type("fix this error") == "error_resolution"
        assert self.reasoning_engine._classify_query_type("how to learn python") == "learning"
        assert self.reasoning_engine._classify_query_type("best practice for API") == "best_practices"
        assert self.reasoning_engine._classify_query_type("optimize performance") == "optimization"
        assert self.reasoning_engine._classify_query_type("security authentication") == "security"
        assert self.reasoning_engine._classify_query_type("general question") == "general"
    
    def test_assess_query_complexity(self):
        """Test query complexity assessment"""
        assert self.reasoning_engine._assess_query_complexity("simple") == "simple"
        assert self.reasoning_engine._assess_query_complexity("moderate length query here") == "moderate"
        assert self.reasoning_engine._assess_query_complexity("this is a very long and complex query with many words") == "complex"
    
    def test_should_analyze_trends(self):
        """Test trend analysis decision"""
        trend_context = ReasoningContext(query="latest python trends 2024")
        no_trend_context = ReasoningContext(query="basic python syntax")
        
        assert self.reasoning_engine._should_analyze_trends(trend_context) is True
        assert self.reasoning_engine._should_analyze_trends(no_trend_context) is False
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation"""
        steps = [
            ReasoningStep("analyze", "test", {}, {}, 0.9),
            ReasoningStep("search", "test", {}, {}, 0.8),
            ReasoningStep("synthesize", "test", {}, {}, 0.7)  # Higher weight
        ]
        
        confidence = self.reasoning_engine._calculate_overall_confidence(steps)
        
        assert 0.0 <= confidence <= 1.0
        # Synthesize step should have higher weight
        assert confidence > 0.7  # Should be influenced by synthesis step
    
    def test_get_cache_key(self):
        """Test cache key generation"""
        context1 = ReasoningContext(query="test", language="python")
        context2 = ReasoningContext(query="test", language="javascript")
        context3 = ReasoningContext(query="different", language="python")
        
        key1 = self.reasoning_engine._get_cache_key(context1)
        key2 = self.reasoning_engine._get_cache_key(context2)
        key3 = self.reasoning_engine._get_cache_key(context3)
        
        assert key1 != key2  # Different languages
        assert key1 != key3  # Different queries
        assert len(key1) == 32  # MD5 hash length
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        context = ReasoningContext(
            query="async programming",
            language="python",
            error_message="SyntaxError"
        )
        
        sources = [
            SearchResult("Python Docs", "https://docs.python.org", "official docs", "Engine", 0.9),
            SearchResult("Stack Overflow", "https://stackoverflow.com/q/123", "community help", "Engine", 0.8),
            SearchResult("GitHub Repo", "https://github.com/user/repo", "code examples", "Engine", 0.7)
        ]
        
        steps = [ReasoningStep("analyze", "test", {}, {}, 0.8)]
        
        recommendations = self.reasoning_engine._generate_recommendations(context, sources, steps)
        
        assert len(recommendations) <= 5
        assert any("error" in rec.lower() for rec in recommendations)  # Error-related recommendation
        assert any("python" in rec.lower() for rec in recommendations)  # Language-related recommendation
        assert any("documentation" in rec.lower() for rec in recommendations)  # Doc-related recommendation
    
    def test_extract_code_examples(self):
        """Test code example extraction"""
        sources = [
            SearchResult("Code Example", "https://example.com", "```python\ndef async_func():\n    pass\n```", "Engine", 0.9),
            SearchResult("Inline Code", "https://example.com", "Use `await asyncio.sleep(1)` for delays", "Engine", 0.8),
            SearchResult("Function Def", "https://example.com", "def main(): print('hello')", "Engine", 0.7)
        ]
        
        code_examples = self.reasoning_engine._extract_code_examples(sources)
        
        assert len(code_examples) <= 3
        assert any("async_func" in example for example in code_examples)
    
    def test_identify_related_topics(self):
        """Test related topic identification"""
        context = ReasoningContext(query="async programming", language="python")
        
        sources = [
            SearchResult("Async Guide", "https://example.com", "asyncio coroutines performance", "Engine", 0.9),
            SearchResult("Python Tutorial", "https://example.com", "FastAPI framework REST API", "Engine", 0.8)
        ]
        
        topics = self.reasoning_engine._identify_related_topics(context, sources)
        
        assert len(topics) <= 8
        assert any("Performance" == topic for topic in topics) or any("performance" in t.lower() for t in topics)
        assert any("FastAPI" == topic for topic in topics)
    
    def test_cache_operations(self):
        """Test cache operations"""
        # Test cache stats
        stats = self.reasoning_engine.get_cache_stats()
        assert "cache_size" in stats
        assert "cache_keys" in stats
        
        # Test clear cache
        self.reasoning_engine.reasoning_cache["test"] = Mock()
        self.reasoning_engine.clear_cache()
        assert len(self.reasoning_engine.reasoning_cache) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_context(self):
        """Test context analysis"""
        context = ReasoningContext(
            query="fix async error",
            language="python",
            error_message="SyntaxError",
            code_context="async def test(): pass"
        )
        
        step = await self.reasoning_engine._analyze_context(context)
        
        assert step.step_type == "analyze"
        assert step.confidence > 0.0
        assert "query_type" in step.output_data
        assert "has_code_context" in step.output_data
        assert "has_error" in step.output_data
        assert step.output_data["has_error"] is True
        assert step.output_data["language"] == "python"
    
    def test_create_synthesized_answer(self):
        """Test synthesized answer creation"""
        context = ReasoningContext(
            query="async programming",
            language="python"
        )
        
        key_points = [
            "Async programming allows concurrent execution",
            "Use async/await keywords in Python",
            "asyncio is the main library for async programming"
        ]
        
        steps = [ReasoningStep("analyze", "test", {}, {}, 0.9)]
        
        answer = self.reasoning_engine._create_synthesized_answer(context, key_points, steps)
        
        assert "async programming" in answer.lower()
        assert "python" in answer.lower()
        assert len(answer) > 100  # Should be a substantial answer
        assert "reliable" in answer.lower()  # Should include confidence note


@pytest.mark.asyncio
class TestInternetEnabledReasoningEngineAsync:
    """Test async functionality of InternetEnabledReasoningEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_web_search_agent = Mock(spec=WebSearchAgent)
        self.reasoning_engine = InternetEnabledReasoningEngine(self.mock_web_search_agent)
    
    async def test_perform_contextual_search(self):
        """Test contextual search step"""
        context = ReasoningContext(query="async programming", language="python")
        
        mock_results = [
            SearchResult("Python Async", "https://example.com", "async guide", "Engine", 0.9)
        ]
        
        # Mock the context search engine
        with patch.object(self.reasoning_engine.context_search_engine, 'search_coding_problem', 
                         new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            
            step, results = await self.reasoning_engine._perform_contextual_search(context)
            
            assert step.step_type == "search"
            assert step.confidence > 0.0
            assert len(results) == 1
            assert results[0].title == "Python Async"
            mock_search.assert_called_once_with(context)
    
    async def test_search_documentation(self):
        """Test documentation search step"""
        context = ReasoningContext(query="async programming", language="python")
        
        mock_results = [
            SearchResult("Python Docs", "https://docs.python.org", "official docs", "Engine", 0.9)
        ]
        
        with patch.object(self.reasoning_engine.documentation_searcher, 'search_documentation',
                         new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            
            step, results = await self.reasoning_engine._search_documentation(context)
            
            assert step.step_type == "search"
            assert step.confidence > 0.0
            assert len(results) == 1
            assert results[0].title == "Python Docs"
            mock_search.assert_called_once_with(context.query, context.language, context.project_type)
    
    async def test_analyze_trends(self):
        """Test trend analysis step"""
        context = ReasoningContext(query="latest python features", language="python")
        
        mock_trend_data = {
            "technology": "python",
            "themes": {"performance": ["Python 3.12 improvements"]},
            "recommendations": ["Use latest Python version"]
        }
        
        with patch.object(self.reasoning_engine.trend_analyzer, 'analyze_technology_trends',
                         new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = mock_trend_data
            
            step, trend_data = await self.reasoning_engine._analyze_trends(context)
            
            assert step.step_type == "analyze"
            assert step.confidence > 0.0
            assert trend_data["technology"] == "python"
            mock_analyze.assert_called_once_with("python", context.query)
    
    async def test_synthesize_information(self):
        """Test information synthesis step"""
        context = ReasoningContext(query="async programming", language="python")
        
        sources = [
            SearchResult("Guide 1", "https://example1.com", "Async allows concurrent execution", "Engine", 0.9),
            SearchResult("Guide 2", "https://example2.com", "Use async/await keywords", "Engine", 0.8)
        ]
        
        steps = [ReasoningStep("search", "test", {}, {}, 0.8)]
        
        step, answer = await self.reasoning_engine._synthesize_information(context, sources, steps)
        
        assert step.step_type == "synthesize"
        assert step.confidence > 0.0
        assert "async programming" in answer.lower()
        assert len(answer) > 50  # Should be substantial
    
    async def test_reason_with_internet_cached(self):
        """Test reasoning with cached result"""
        context = ReasoningContext(query="test query", language="python")
        
        # Pre-populate cache with recent result
        cached_result = ReasoningResult(
            query="test query",
            answer="cached answer",
            confidence=0.9,
            reasoning_steps=[],
            sources=[],
            recommendations=[],
            code_examples=[],
            related_topics=[],
            timestamp=datetime.now()  # Recent timestamp
        )
        
        cache_key = self.reasoning_engine._get_cache_key(context)
        self.reasoning_engine.reasoning_cache[cache_key] = cached_result
        
        result = await self.reasoning_engine.reason_with_internet(context)
        
        assert result.answer == "cached answer"
        assert result.confidence == 0.9
    
    async def test_reason_with_internet_full_flow(self):
        """Test full reasoning flow"""
        context = ReasoningContext(
            query="async programming best practices",
            language="python",
            project_type="web_app"
        )
        
        # Mock all the search methods
        mock_search_results = [
            SearchResult("Async Guide", "https://example.com", "async best practices", "Engine", 0.9)
        ]
        
        mock_doc_results = [
            SearchResult("Python Docs", "https://docs.python.org", "official async docs", "Engine", 0.95)
        ]
        
        mock_trend_data = {
            "technology": "python",
            "themes": {"performance": ["async improvements"]},
            "recommendations": ["Use modern async patterns"]
        }
        
        with patch.object(self.reasoning_engine.context_search_engine, 'search_coding_problem',
                         new_callable=AsyncMock) as mock_context_search:
            mock_context_search.return_value = mock_search_results
            
            with patch.object(self.reasoning_engine.documentation_searcher, 'search_documentation',
                             new_callable=AsyncMock) as mock_doc_search:
                mock_doc_search.return_value = mock_doc_results
                
                with patch.object(self.reasoning_engine.trend_analyzer, 'analyze_technology_trends',
                                 new_callable=AsyncMock) as mock_trend_analyze:
                    mock_trend_analyze.return_value = mock_trend_data
                    
                    result = await self.reasoning_engine.reason_with_internet(context)
                    
                    assert result.query == context.query
                    assert result.confidence > 0.0
                    assert len(result.answer) > 50
                    assert len(result.reasoning_steps) >= 4  # Should have multiple steps
                    assert len(result.sources) > 0
                    assert len(result.recommendations) > 0
                    
                    # Verify all search methods were called
                    mock_context_search.assert_called_once()
                    mock_doc_search.assert_called_once()
                    mock_trend_analyze.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])