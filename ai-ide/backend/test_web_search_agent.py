"""
Test suite for WebSearchAgent

Tests web search functionality including multiple search engines,
result filtering, content extraction, and caching.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from web_search_agent import (
    WebSearchAgent, SearchQuery, SearchResult, SearchResultCache,
    DuckDuckGoSearchEngine, BingSearchEngine, GoogleSearchEngine,
    ContentExtractor
)


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult"""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="TestEngine",
            relevance_score=0.8
        )
        
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.source == "TestEngine"
        assert result.relevance_score == 0.8
        assert isinstance(result.timestamp, datetime)
    
    def test_search_result_default_timestamp(self):
        """Test that timestamp is set automatically"""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test",
            source="Test"
        )
        
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestSearchQuery:
    """Test SearchQuery dataclass"""
    
    def test_search_query_defaults(self):
        """Test SearchQuery with default values"""
        query = SearchQuery(query="test query")
        
        assert query.query == "test query"
        assert query.max_results == 10
        assert query.language == "en"
        assert query.region == "us"
        assert query.safe_search is True
        assert query.time_range is None
    
    def test_search_query_custom_values(self):
        """Test SearchQuery with custom values"""
        query = SearchQuery(
            query="custom query",
            max_results=20,
            language="fr",
            region="fr",
            safe_search=False,
            time_range="week"
        )
        
        assert query.query == "custom query"
        assert query.max_results == 20
        assert query.language == "fr"
        assert query.region == "fr"
        assert query.safe_search is False
        assert query.time_range == "week"


class TestSearchResultCache:
    """Test SearchResultCache functionality"""
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        cache = SearchResultCache()
        query = SearchQuery(query="test")
        
        key1 = cache._get_cache_key(query, "Engine1")
        key2 = cache._get_cache_key(query, "Engine2")
        key3 = cache._get_cache_key(SearchQuery(query="different"), "Engine1")
        
        assert key1 != key2  # Different engines
        assert key1 != key3  # Different queries
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_set_and_get(self):
        """Test caching and retrieving results"""
        cache = SearchResultCache()
        query = SearchQuery(query="test")
        
        results = [
            SearchResult(
                title="Test 1",
                url="https://example1.com",
                snippet="Snippet 1",
                source="TestEngine"
            ),
            SearchResult(
                title="Test 2",
                url="https://example2.com",
                snippet="Snippet 2",
                source="TestEngine"
            )
        ]
        
        # Cache results
        cache.set(query, "TestEngine", results)
        
        # Retrieve results
        cached_results = cache.get(query, "TestEngine")
        
        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0].title == "Test 1"
        assert cached_results[1].title == "Test 2"
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = SearchResultCache(cache_duration_hours=0)  # Immediate expiration
        query = SearchQuery(query="test")
        
        results = [SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            source="TestEngine"
        )]
        
        # Cache results
        cache.set(query, "TestEngine", results)
        
        # Results should be expired immediately
        cached_results = cache.get(query, "TestEngine")
        assert cached_results is None
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = SearchResultCache()
        query = SearchQuery(query="nonexistent")
        
        cached_results = cache.get(query, "TestEngine")
        assert cached_results is None


class TestDuckDuckGoSearchEngine:
    """Test DuckDuckGo search engine"""
    
    def test_initialization(self):
        """Test DuckDuckGo engine initialization"""
        engine = DuckDuckGoSearchEngine()
        
        assert engine.name == "DuckDuckGo"
        assert engine.rate_limit == 0.5
        assert engine.base_url == "https://html.duckduckgo.com/html/"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        engine = DuckDuckGoSearchEngine()
        
        start_time = asyncio.get_event_loop().time()
        await engine._rate_limit_check()
        await engine._rate_limit_check()
        end_time = asyncio.get_event_loop().time()
        
        # Second call should be delayed by rate limit
        assert end_time - start_time >= (1.0 / engine.rate_limit) - 0.1  # Allow small tolerance
    
    def test_parse_duckduckgo_results(self):
        """Test parsing DuckDuckGo HTML results"""
        engine = DuckDuckGoSearchEngine()
        query = SearchQuery(query="test", max_results=2)
        
        # Mock HTML content
        html_content = '''
        <div class="result">
            <a class="result__a" href="https://example1.com">Test Title 1</a>
            <a class="result__snippet">Test snippet 1</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://example2.com">Test Title 2</a>
            <a class="result__snippet">Test snippet 2</a>
        </div>
        '''
        
        results = engine._parse_duckduckgo_results(html_content, query)
        
        assert len(results) == 2
        assert results[0].title == "Test Title 1"
        assert results[0].url == "https://example1.com"
        assert results[0].snippet == "Test snippet 1"
        assert results[0].source == "DuckDuckGo"
        assert results[0].relevance_score == 1.0
        assert results[1].relevance_score == 0.9  # Decreasing relevance


class TestBingSearchEngine:
    """Test Bing search engine"""
    
    def test_initialization_without_api_key(self):
        """Test Bing engine initialization without API key"""
        engine = BingSearchEngine()
        
        assert engine.name == "Bing"
        assert engine.api_key is None
        assert engine.rate_limit == 3.0
    
    def test_initialization_with_api_key(self):
        """Test Bing engine initialization with API key"""
        engine = BingSearchEngine(api_key="test_key")
        
        assert engine.api_key == "test_key"
    
    def test_parse_bing_results(self):
        """Test parsing Bing API results"""
        engine = BingSearchEngine(api_key="test_key")
        query = SearchQuery(query="test", max_results=2)
        
        # Mock Bing API response
        api_response = {
            "webPages": {
                "value": [
                    {
                        "name": "Bing Title 1",
                        "url": "https://bing1.com",
                        "snippet": "Bing snippet 1"
                    },
                    {
                        "name": "Bing Title 2",
                        "url": "https://bing2.com",
                        "snippet": "Bing snippet 2"
                    }
                ]
            }
        }
        
        results = engine._parse_bing_results(api_response, query)
        
        assert len(results) == 2
        assert results[0].title == "Bing Title 1"
        assert results[0].url == "https://bing1.com"
        assert results[0].snippet == "Bing snippet 1"
        assert results[0].source == "Bing"
        assert results[0].relevance_score == 1.0
        assert results[1].relevance_score == 0.95


class TestGoogleSearchEngine:
    """Test Google search engine"""
    
    def test_initialization_without_credentials(self):
        """Test Google engine initialization without credentials"""
        engine = GoogleSearchEngine()
        
        assert engine.name == "Google"
        assert engine.api_key is None
        assert engine.search_engine_id is None
        assert engine.rate_limit == 1.0
    
    def test_initialization_with_credentials(self):
        """Test Google engine initialization with credentials"""
        engine = GoogleSearchEngine(
            api_key="test_key",
            search_engine_id="test_id"
        )
        
        assert engine.api_key == "test_key"
        assert engine.search_engine_id == "test_id"
    
    def test_parse_google_results(self):
        """Test parsing Google API results"""
        engine = GoogleSearchEngine(
            api_key="test_key",
            search_engine_id="test_id"
        )
        query = SearchQuery(query="test", max_results=2)
        
        # Mock Google API response
        api_response = {
            "items": [
                {
                    "title": "Google Title 1",
                    "link": "https://google1.com",
                    "snippet": "Google snippet 1"
                },
                {
                    "title": "Google Title 2",
                    "link": "https://google2.com",
                    "snippet": "Google snippet 2"
                }
            ]
        }
        
        results = engine._parse_google_results(api_response, query)
        
        assert len(results) == 2
        assert results[0].title == "Google Title 1"
        assert results[0].url == "https://google1.com"
        assert results[0].snippet == "Google snippet 1"
        assert results[0].source == "Google"


class TestContentExtractor:
    """Test ContentExtractor functionality"""
    
    def test_clean_html_content(self):
        """Test HTML content cleaning"""
        extractor = ContentExtractor()
        
        html_content = '''
        <html>
            <head><title>Test</title></head>
            <body>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
                <nav>Navigation</nav>
                <header>Header</header>
                <main>
                    <h1>Main Title</h1>
                    <p>This is the main content.</p>
                    <p>Another paragraph with useful information.</p>
                </main>
                <footer>Footer</footer>
            </body>
        </html>
        '''
        
        cleaned_text = extractor._clean_html_content(html_content)
        
        assert "Main Title" in cleaned_text
        assert "main content" in cleaned_text
        assert "useful information" in cleaned_text
        assert "alert('test')" not in cleaned_text
        assert "color: red" not in cleaned_text
        assert "Navigation" not in cleaned_text
        assert "Header" not in cleaned_text
        assert "Footer" not in cleaned_text
    
    def test_content_length_limiting(self):
        """Test content length limiting"""
        extractor = ContentExtractor()
        
        # Create long HTML content
        long_content = "<p>" + "A" * 10000 + "</p>"
        
        cleaned_text = extractor._clean_html_content(long_content)
        
        assert len(cleaned_text) <= 5003  # 5000 + "..."
        assert cleaned_text.endswith("...")


class TestWebSearchAgent:
    """Test WebSearchAgent main functionality"""
    
    def test_initialization_default(self):
        """Test WebSearchAgent initialization with defaults"""
        agent = WebSearchAgent()
        
        assert len(agent.engines) == 1  # Only DuckDuckGo without API keys
        assert agent.engines[0].name == "DuckDuckGo"
        assert agent.enable_content_extraction is True
        assert agent.cache is not None
    
    def test_initialization_with_api_keys(self):
        """Test WebSearchAgent initialization with API keys"""
        agent = WebSearchAgent(
            google_api_key="google_key",
            google_search_engine_id="google_id",
            bing_api_key="bing_key"
        )
        
        assert len(agent.engines) == 3  # DuckDuckGo, Google, Bing
        engine_names = [engine.name for engine in agent.engines]
        assert "DuckDuckGo" in engine_names
        assert "Google" in engine_names
        assert "Bing" in engine_names
    
    def test_merge_and_deduplicate_results(self):
        """Test merging and deduplicating search results"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult(
                title="Title 1",
                url="https://example.com",
                snippet="Snippet 1",
                source="Engine1",
                relevance_score=0.8
            ),
            SearchResult(
                title="Title 2",
                url="https://example.com/",  # Same URL with trailing slash
                snippet="Snippet 2",
                source="Engine2",
                relevance_score=0.9
            ),
            SearchResult(
                title="Title 3",
                url="https://different.com",
                snippet="Snippet 3",
                source="Engine1",
                relevance_score=0.7
            )
        ]
        
        merged = agent._merge_and_deduplicate_results(results)
        
        assert len(merged) == 2  # Duplicates removed
        assert merged[0].url == "https://example.com"
        assert merged[0].relevance_score == 0.9  # Higher score kept
        assert "Engine1, Engine2" in merged[0].source  # Sources combined
        assert merged[1].url == "https://different.com"
    
    def test_filter_results_by_relevance(self):
        """Test filtering results by relevance score"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult("Title 1", "https://example1.com", "Snippet 1", "Engine", 0.9),
            SearchResult("Title 2", "https://example2.com", "Snippet 2", "Engine", 0.7),
            SearchResult("Title 3", "https://example3.com", "Snippet 3", "Engine", 0.5),
            SearchResult("Title 4", "https://example4.com", "Snippet 4", "Engine", 0.3)
        ]
        
        filtered = agent.filter_results(results, min_relevance=0.6)
        
        assert len(filtered) == 2
        assert all(r.relevance_score >= 0.6 for r in filtered)
    
    def test_filter_results_by_domains(self):
        """Test filtering results by domains"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult("Title 1", "https://github.com/test", "Snippet 1", "Engine", 0.9),
            SearchResult("Title 2", "https://stackoverflow.com/test", "Snippet 2", "Engine", 0.8),
            SearchResult("Title 3", "https://example.com/test", "Snippet 3", "Engine", 0.7),
            SearchResult("Title 4", "https://docs.python.org/test", "Snippet 4", "Engine", 0.6)
        ]
        
        # Test include domains
        filtered_include = agent.filter_results(
            results, 
            include_domains=["github.com", "stackoverflow.com"]
        )
        assert len(filtered_include) == 2
        assert all("github.com" in r.url or "stackoverflow.com" in r.url for r in filtered_include)
        
        # Test exclude domains
        filtered_exclude = agent.filter_results(
            results,
            exclude_domains=["example.com"]
        )
        assert len(filtered_exclude) == 3
        assert all("example.com" not in r.url for r in filtered_exclude)
    
    def test_filter_results_max_results(self):
        """Test limiting maximum results"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult(f"Title {i}", f"https://example{i}.com", f"Snippet {i}", "Engine", 1.0 - i*0.1)
            for i in range(10)
        ]
        
        filtered = agent.filter_results(results, max_results=3)
        
        assert len(filtered) == 3
        assert filtered[0].title == "Title 0"
        assert filtered[2].title == "Title 2"
    
    def test_get_search_summary(self):
        """Test search summary generation"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult("Title 1", "https://example1.com", "Snippet 1", "Engine1", 0.9),
            SearchResult("Title 2", "https://example2.com", "Snippet 2", "Engine1", 0.8),
            SearchResult("Title 3", "https://example3.com", "Snippet 3", "Engine2", 0.7)
        ]
        
        summary = agent.get_search_summary(results)
        
        assert summary["total_results"] == 3
        assert summary["sources"]["Engine1"] == 2
        assert summary["sources"]["Engine2"] == 1
        assert summary["average_relevance"] == (0.9 + 0.8 + 0.7) / 3
        assert summary["top_result"]["title"] == "Title 1"
        assert summary["top_result"]["relevance"] == 0.9
    
    def test_get_search_summary_empty(self):
        """Test search summary with empty results"""
        agent = WebSearchAgent()
        
        summary = agent.get_search_summary([])
        
        assert summary["total_results"] == 0
        assert "sources" not in summary
        assert "average_relevance" not in summary
        assert "top_result" not in summary
    
    def test_cache_operations(self):
        """Test cache operations"""
        agent = WebSearchAgent()
        
        # Test clear cache
        agent.cache.cache["test"] = {"data": "test"}
        agent.clear_cache()
        assert len(agent.cache.cache) == 0
        
        # Test cleanup cache (no expired entries to clean)
        agent.cleanup_cache()  # Should not raise any errors


@pytest.mark.asyncio
class TestWebSearchAgentAsync:
    """Test async functionality of WebSearchAgent"""
    
    async def test_search_string_query(self):
        """Test search with string query"""
        agent = WebSearchAgent()
        
        # Mock the search method of engines
        mock_results = [
            SearchResult("Test Title", "https://example.com", "Test snippet", "DuckDuckGo", 0.9)
        ]
        
        with patch.object(agent.engines[0], 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            
            results = await agent.search("test query")
            
            assert len(results) == 1
            assert results[0].title == "Test Title"
            mock_search.assert_called_once()
            
            # Check that string was converted to SearchQuery
            call_args = mock_search.call_args[0][0]
            assert isinstance(call_args, SearchQuery)
            assert call_args.query == "test query"
    
    async def test_search_with_specific_engines(self):
        """Test search with specific engines"""
        agent = WebSearchAgent(
            google_api_key="test_key",
            google_search_engine_id="test_id"
        )
        
        mock_results = [
            SearchResult("Google Result", "https://google.com", "Google snippet", "Google", 0.9)
        ]
        
        # Mock only Google engine
        with patch.object(agent.engines[1], 'search', new_callable=AsyncMock) as mock_google_search:
            mock_google_search.return_value = mock_results
            
            with patch.object(agent.engines[0], 'search', new_callable=AsyncMock) as mock_ddg_search:
                mock_ddg_search.return_value = []
                
                results = await agent.search("test query", engines=["Google"])
                
                assert len(results) == 1
                assert results[0].source == "Google"
                mock_google_search.assert_called_once()
                mock_ddg_search.assert_not_called()
    
    async def test_search_with_cache(self):
        """Test search with caching"""
        agent = WebSearchAgent()
        
        query = SearchQuery(query="test query")
        mock_results = [
            SearchResult("Cached Result", "https://cached.com", "Cached snippet", "DuckDuckGo", 0.9)
        ]
        
        # Pre-populate cache
        agent.cache.set(query, "DuckDuckGo", mock_results)
        
        with patch.object(agent.engines[0], 'search', new_callable=AsyncMock) as mock_search:
            results = await agent.search(query)
            
            assert len(results) == 1
            assert results[0].title == "Cached Result"
            mock_search.assert_not_called()  # Should use cache, not call engine
    
    async def test_search_error_handling(self):
        """Test search error handling"""
        agent = WebSearchAgent()
        
        with patch.object(agent.engines[0], 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Search failed")
            
            results = await agent.search("test query")
            
            assert len(results) == 0  # Should handle error gracefully
    
    async def test_extract_content_for_results(self):
        """Test content extraction for results"""
        agent = WebSearchAgent()
        
        results = [
            SearchResult("Title 1", "https://example1.com", "Snippet 1", "Engine", 0.9),
            SearchResult("Title 2", "https://example2.com", "Snippet 2", "Engine", 0.8)
        ]
        
        mock_content = "Extracted content"
        
        with patch('web_search_agent.ContentExtractor') as mock_extractor_class:
            mock_extractor = AsyncMock()
            mock_extractor.extract_content.return_value = mock_content
            mock_extractor_class.return_value.__aenter__.return_value = mock_extractor
            
            updated_results = await agent._extract_content_for_results(results)
            
            assert len(updated_results) == 2
            assert updated_results[0].content == mock_content
            assert updated_results[1].content == mock_content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])