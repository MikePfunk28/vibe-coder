"""
Web Search Agent for AI IDE

This module provides web search capabilities with support for multiple search engines,
result filtering, content extraction, and caching.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote_plus, urljoin
import re

import aiohttp
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    snippet: str
    source: str  # search engine used
    relevance_score: float = 0.0
    content: Optional[str] = None  # extracted full content
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    query: str
    max_results: int = 10
    language: str = "en"
    region: str = "us"
    safe_search: bool = True
    time_range: Optional[str] = None  # "day", "week", "month", "year"


class SearchEngine(ABC):
    """Abstract base class for search engines"""
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        self.name = name
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0.0
    
    async def _rate_limit_check(self):
        """Ensure rate limiting is respected"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search and return results"""
        pass


class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo search engine implementation"""
    
    def __init__(self):
        super().__init__("DuckDuckGo", rate_limit=0.5)  # Conservative rate limit
        self.base_url = "https://html.duckduckgo.com/html/"
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        await self._rate_limit_check()
        
        params = {
            'q': query.query,
            'kl': f'{query.region}-{query.language}',
            's': '0',  # start index
            'dc': str(query.max_results),
            'v': 'l',  # layout
            'o': 'json',
            'api': '/d.js'
        }
        
        if not query.safe_search:
            params['safe_search'] = '-1'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._parse_duckduckgo_results(html_content, query)
                    else:
                        logger.error(f"DuckDuckGo search failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _parse_duckduckgo_results(self, html_content: str, query: SearchQuery) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results"""
        results = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find result containers
        result_containers = soup.find_all('div', class_='result')
        
        for container in result_containers[:query.max_results]:
            try:
                # Extract title and URL
                title_link = container.find('a', class_='result__a')
                if not title_link:
                    continue
                
                title = title_link.get_text(strip=True)
                url = title_link.get('href', '')
                
                # Extract snippet
                snippet_elem = container.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and url:
                    result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=self.name,
                        relevance_score=1.0 - (len(results) * 0.1)  # Simple relevance scoring
                    )
                    results.append(result)
            
            except Exception as e:
                logger.warning(f"Error parsing DuckDuckGo result: {e}")
                continue
        
        return results


class BingSearchEngine(SearchEngine):
    """Bing search engine implementation (requires API key)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Bing", rate_limit=3.0)  # Bing allows higher rate limits
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Bing Search API"""
        if not self.api_key:
            logger.warning("Bing API key not provided, skipping Bing search")
            return []
        
        await self._rate_limit_check()
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'User-Agent': 'AI-IDE-WebSearch/1.0'
        }
        
        params = {
            'q': query.query,
            'count': query.max_results,
            'mkt': f'{query.language}-{query.region}',
            'safeSearch': 'Strict' if query.safe_search else 'Off',
            'responseFilter': 'Webpages'
        }
        
        if query.time_range:
            # Map time ranges to Bing freshness parameter
            freshness_map = {
                'day': 'Day',
                'week': 'Week',
                'month': 'Month'
            }
            if query.time_range in freshness_map:
                params['freshness'] = freshness_map[query.time_range]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_bing_results(data, query)
                    else:
                        logger.error(f"Bing search failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []
    
    def _parse_bing_results(self, data: Dict, query: SearchQuery) -> List[SearchResult]:
        """Parse Bing API results"""
        results = []
        
        web_pages = data.get('webPages', {})
        web_results = web_pages.get('value', [])
        
        for i, result in enumerate(web_results[:query.max_results]):
            try:
                title = result.get('name', '')
                url = result.get('url', '')
                snippet = result.get('snippet', '')
                
                if title and url:
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=self.name,
                        relevance_score=1.0 - (i * 0.05)  # Bing results are pre-ranked
                    )
                    results.append(search_result)
            
            except Exception as e:
                logger.warning(f"Error parsing Bing result: {e}")
                continue
        
        return results


class GoogleSearchEngine(SearchEngine):
    """Google search engine implementation (requires Custom Search API)"""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        super().__init__("Google", rate_limit=1.0)  # Google has strict rate limits
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google API credentials not provided, skipping Google search")
            return []
        
        await self._rate_limit_check()
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query.query,
            'num': min(query.max_results, 10),  # Google limits to 10 per request
            'hl': query.language,
            'gl': query.region,
            'safe': 'active' if query.safe_search else 'off'
        }
        
        if query.time_range:
            # Map time ranges to Google date restrict parameter
            date_restrict_map = {
                'day': 'd1',
                'week': 'w1',
                'month': 'm1',
                'year': 'y1'
            }
            if query.time_range in date_restrict_map:
                params['dateRestrict'] = date_restrict_map[query.time_range]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_results(data, query)
                    else:
                        logger.error(f"Google search failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    def _parse_google_results(self, data: Dict, query: SearchQuery) -> List[SearchResult]:
        """Parse Google Custom Search API results"""
        results = []
        
        items = data.get('items', [])
        
        for i, item in enumerate(items[:query.max_results]):
            try:
                title = item.get('title', '')
                url = item.get('link', '')
                snippet = item.get('snippet', '')
                
                if title and url:
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=self.name,
                        relevance_score=1.0 - (i * 0.05)  # Google results are pre-ranked
                    )
                    results.append(search_result)
            
            except Exception as e:
                logger.warning(f"Error parsing Google result: {e}")
                continue
        
        return results


class ContentExtractor:
    """Extracts and cleans content from web pages"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-IDE-ContentExtractor/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract clean text content from a URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    return self._clean_html_content(html_content)
                else:
                    logger.warning(f"Failed to fetch content from {url}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract readable text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit content length
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text


class SearchResultCache:
    """Caches search results to reduce API calls"""
    
    def __init__(self, cache_duration_hours: int = 24):
        self.cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)
    
    def _get_cache_key(self, query: SearchQuery, engine_name: str) -> str:
        """Generate cache key for query and engine"""
        query_str = f"{query.query}_{query.max_results}_{query.language}_{query.region}_{engine_name}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get(self, query: SearchQuery, engine_name: str) -> Optional[List[SearchResult]]:
        """Get cached results if available and not expired"""
        cache_key = self._get_cache_key(query, engine_name)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cached_time = cached_data['timestamp']
            
            if datetime.now() - cached_time < self.cache_duration:
                # Reconstruct SearchResult objects
                results = []
                for result_data in cached_data['results']:
                    result = SearchResult(**result_data)
                    results.append(result)
                return results
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def set(self, query: SearchQuery, engine_name: str, results: List[SearchResult]):
        """Cache search results"""
        cache_key = self._get_cache_key(query, engine_name)
        
        # Convert SearchResult objects to dictionaries for JSON serialization
        results_data = []
        for result in results:
            result_dict = asdict(result)
            # Convert datetime to string for JSON serialization
            result_dict['timestamp'] = result.timestamp.isoformat()
            results_data.append(result_dict)
        
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'results': results_data
        }
    
    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, data in self.cache.items():
            if current_time - data['timestamp'] >= self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]


class WebSearchAgent:
    """Main web search agent that coordinates multiple search engines"""
    
    def __init__(self, 
                 google_api_key: Optional[str] = None,
                 google_search_engine_id: Optional[str] = None,
                 bing_api_key: Optional[str] = None,
                 enable_content_extraction: bool = True,
                 cache_duration_hours: int = 24):
        
        # Initialize search engines
        self.engines: List[SearchEngine] = []
        
        # Always include DuckDuckGo as it doesn't require API keys
        self.engines.append(DuckDuckGoSearchEngine())
        
        # Add Google if credentials provided
        if google_api_key and google_search_engine_id:
            self.engines.append(GoogleSearchEngine(google_api_key, google_search_engine_id))
        
        # Add Bing if API key provided
        if bing_api_key:
            self.engines.append(BingSearchEngine(bing_api_key))
        
        self.enable_content_extraction = enable_content_extraction
        self.cache = SearchResultCache(cache_duration_hours)
        
        logger.info(f"WebSearchAgent initialized with {len(self.engines)} search engines: "
                   f"{[engine.name for engine in self.engines]}")
    
    async def search(self, 
                    query: Union[str, SearchQuery], 
                    engines: Optional[List[str]] = None,
                    merge_results: bool = True,
                    extract_content: bool = False) -> List[SearchResult]:
        """
        Perform web search using specified engines
        
        Args:
            query: Search query string or SearchQuery object
            engines: List of engine names to use (None = use all)
            merge_results: Whether to merge and deduplicate results
            extract_content: Whether to extract full content from URLs
        
        Returns:
            List of SearchResult objects
        """
        
        # Convert string query to SearchQuery object
        if isinstance(query, str):
            query = SearchQuery(query=query)
        
        # Filter engines if specified
        active_engines = self.engines
        if engines:
            active_engines = [e for e in self.engines if e.name in engines]
        
        if not active_engines:
            logger.warning("No active search engines available")
            return []
        
        # Search with each engine
        all_results = []
        
        for engine in active_engines:
            try:
                # Check cache first
                cached_results = self.cache.get(query, engine.name)
                if cached_results:
                    logger.info(f"Using cached results for {engine.name}")
                    all_results.extend(cached_results)
                    continue
                
                # Perform search
                logger.info(f"Searching with {engine.name}: {query.query}")
                results = await engine.search(query)
                
                if results:
                    # Cache results
                    self.cache.set(query, engine.name, results)
                    all_results.extend(results)
                    logger.info(f"{engine.name} returned {len(results)} results")
                else:
                    logger.warning(f"{engine.name} returned no results")
                
            except Exception as e:
                logger.error(f"Error searching with {engine.name}: {e}")
                continue
        
        # Extract content if requested
        if extract_content and all_results:
            all_results = await self._extract_content_for_results(all_results)
        
        # Merge and deduplicate results if requested
        if merge_results:
            all_results = self._merge_and_deduplicate_results(all_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Search completed. Total results: {len(all_results)}")
        return all_results
    
    async def _extract_content_for_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract content for search results"""
        if not self.enable_content_extraction:
            return results
        
        async with ContentExtractor() as extractor:
            for result in results:
                try:
                    content = await extractor.extract_content(result.url)
                    if content:
                        result.content = content
                except Exception as e:
                    logger.warning(f"Failed to extract content from {result.url}: {e}")
        
        return results
    
    def _merge_and_deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Merge results from different engines and remove duplicates"""
        seen_urls = set()
        merged_results = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = result.url.lower().rstrip('/')
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                merged_results.append(result)
            else:
                # If duplicate, update relevance score if higher
                for existing_result in merged_results:
                    if existing_result.url.lower().rstrip('/') == normalized_url:
                        if result.relevance_score > existing_result.relevance_score:
                            existing_result.relevance_score = result.relevance_score
                        # Combine sources
                        if result.source not in existing_result.source:
                            existing_result.source += f", {result.source}"
                        break
        
        return merged_results
    
    def filter_results(self, 
                      results: List[SearchResult], 
                      min_relevance: float = 0.0,
                      exclude_domains: Optional[List[str]] = None,
                      include_domains: Optional[List[str]] = None,
                      max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Filter search results based on various criteria
        
        Args:
            results: List of search results to filter
            min_relevance: Minimum relevance score
            exclude_domains: Domains to exclude
            include_domains: Only include these domains
            max_results: Maximum number of results to return
        
        Returns:
            Filtered list of search results
        """
        filtered_results = results.copy()
        
        # Filter by relevance
        if min_relevance > 0:
            filtered_results = [r for r in filtered_results if r.relevance_score >= min_relevance]
        
        # Filter by domains
        if exclude_domains:
            exclude_domains = [d.lower() for d in exclude_domains]
            filtered_results = [r for r in filtered_results 
                              if not any(domain in r.url.lower() for domain in exclude_domains)]
        
        if include_domains:
            include_domains = [d.lower() for d in include_domains]
            filtered_results = [r for r in filtered_results 
                              if any(domain in r.url.lower() for domain in include_domains)]
        
        # Limit results
        if max_results:
            filtered_results = filtered_results[:max_results]
        
        return filtered_results
    
    def get_search_summary(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate summary statistics for search results"""
        if not results:
            return {"total_results": 0}
        
        sources = {}
        total_relevance = 0
        
        for result in results:
            # Count by source
            if result.source in sources:
                sources[result.source] += 1
            else:
                sources[result.source] = 1
            
            total_relevance += result.relevance_score
        
        return {
            "total_results": len(results),
            "sources": sources,
            "average_relevance": total_relevance / len(results),
            "top_result": {
                "title": results[0].title,
                "url": results[0].url,
                "relevance": results[0].relevance_score
            } if results else None
        }
    
    def clear_cache(self):
        """Clear the search result cache"""
        self.cache.cache.clear()
        logger.info("Search cache cleared")
    
    def cleanup_cache(self):
        """Remove expired cache entries"""
        self.cache.clear_expired()
        logger.info("Expired cache entries removed")


# Example usage and testing
if __name__ == "__main__":
    async def test_web_search():
        """Test the web search functionality"""
        
        # Initialize search agent (only DuckDuckGo without API keys)
        search_agent = WebSearchAgent()
        
        # Test basic search
        query = "Python async programming best practices"
        results = await search_agent.search(query, extract_content=False)
        
        print(f"Search results for: {query}")
        print(f"Total results: {len(results)}")
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Snippet: {result.snippet[:100]}...")
        
        # Test search summary
        summary = search_agent.get_search_summary(results)
        print(f"\nSearch Summary: {summary}")
        
        # Test filtering
        filtered_results = search_agent.filter_results(
            results, 
            min_relevance=0.5, 
            max_results=3
        )
        print(f"\nFiltered results: {len(filtered_results)}")
    
    # Run test
    asyncio.run(test_web_search())