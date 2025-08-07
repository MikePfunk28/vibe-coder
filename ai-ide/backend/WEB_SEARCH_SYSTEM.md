# Web Search and Internet-Enabled Reasoning System

## Overview

The Web Search and Internet-Enabled Reasoning System provides comprehensive web search capabilities and deep reasoning with real-time information retrieval for the AI IDE. This system enables the IDE to search the internet for coding solutions, documentation, and technology trends, then synthesize this information into actionable insights.

## Architecture

### Core Components

1. **WebSearchAgent** - Multi-engine web search with caching and rate limiting
2. **InternetEnabledReasoningEngine** - Deep reasoning with real-time information retrieval
3. **DocumentationSearcher** - Specialized searcher for official documentation
4. **TechnologyTrendAnalyzer** - Analyzes current technology trends
5. **ContextAwareSearchEngine** - Context-aware search for coding problems

### Component Relationships

```
InternetEnabledReasoningEngine
├── WebSearchAgent
│   ├── DuckDuckGoSearchEngine
│   ├── BingSearchEngine (optional)
│   └── GoogleSearchEngine (optional)
├── ContextAwareSearchEngine
├── DocumentationSearcher
└── TechnologyTrendAnalyzer
```

## Features

### Web Search Agent

#### Multi-Engine Support
- **DuckDuckGo**: No API key required, privacy-focused
- **Bing Search API**: High-quality results with API key
- **Google Custom Search**: Comprehensive results with API credentials

#### Advanced Features
- **Rate Limiting**: Respects search engine rate limits
- **Result Caching**: Reduces API calls and improves performance
- **Content Extraction**: Extracts clean text from web pages
- **Result Deduplication**: Merges and deduplicates results from multiple engines
- **Relevance Filtering**: Filters results by relevance, domains, and other criteria

### Internet-Enabled Reasoning

#### Deep Reasoning Capabilities
- **Context Analysis**: Analyzes query context and determines reasoning strategy
- **Multi-Step Reasoning**: Performs structured reasoning with multiple steps
- **Information Synthesis**: Combines information from multiple sources
- **Confidence Assessment**: Calculates confidence scores for reasoning steps

#### Specialized Search Types
- **Documentation Search**: Prioritizes official documentation and API references
- **Technology Trend Analysis**: Analyzes current trends and best practices
- **Context-Aware Search**: Adapts search strategy based on coding context
- **Error Resolution**: Specialized search for debugging and error resolution

## Usage Examples

### Basic Web Search

```python
from web_search_agent import WebSearchAgent, SearchQuery

# Initialize search agent
search_agent = WebSearchAgent()

# Simple search
results = await search_agent.search("Python async programming")

# Advanced search with options
query = SearchQuery(
    query="Python async best practices",
    max_results=15,
    language="en",
    time_range="month"
)
results = await search_agent.search(query, extract_content=True)

# Filter results
filtered = search_agent.filter_results(
    results,
    min_relevance=0.7,
    include_domains=["docs.python.org", "stackoverflow.com"],
    max_results=5
)
```

### Internet-Enabled Reasoning

```python
from internet_enabled_reasoning import InternetEnabledReasoningEngine, ReasoningContext
from web_search_agent import WebSearchAgent

# Initialize components
web_search = WebSearchAgent()
reasoning_engine = InternetEnabledReasoningEngine(web_search)

# Create reasoning context
context = ReasoningContext(
    query="How to handle async/await errors in Python",
    language="python",
    project_type="web_application",
    error_message="RuntimeError: This event loop is already running",
    user_intent="debugging"
)

# Perform reasoning
result = await reasoning_engine.reason_with_internet(context)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Recommendations: {result.recommendations}")
print(f"Code Examples: {result.code_examples}")
```

### Documentation Search

```python
from internet_enabled_reasoning import DocumentationSearcher
from web_search_agent import WebSearchAgent

# Initialize components
web_search = WebSearchAgent()
doc_searcher = DocumentationSearcher(web_search)

# Search for documentation
results = await doc_searcher.search_documentation(
    query="async context managers",
    language="python",
    framework="asyncio"
)

# Results are automatically prioritized by documentation relevance
for result in results[:3]:
    print(f"{result.title}: {result.url}")
```

### Technology Trend Analysis

```python
from internet_enabled_reasoning import TechnologyTrendAnalyzer
from web_search_agent import WebSearchAgent

# Initialize components
web_search = WebSearchAgent()
trend_analyzer = TechnologyTrendAnalyzer(web_search)

# Analyze technology trends
analysis = await trend_analyzer.analyze_technology_trends(
    technology="python",
    context="web development"
)

print(f"Technology: {analysis['technology']}")
print(f"Themes: {analysis['themes']}")
print(f"Recommendations: {analysis['recommendations']}")
```

## Configuration

### API Keys (Optional)

For enhanced search capabilities, configure API keys:

```python
# With API keys for better results
search_agent = WebSearchAgent(
    google_api_key="your_google_api_key",
    google_search_engine_id="your_search_engine_id",
    bing_api_key="your_bing_api_key"
)

# Without API keys (DuckDuckGo only)
search_agent = WebSearchAgent()
```

### Reasoning Engine Configuration

```python
reasoning_engine = InternetEnabledReasoningEngine(
    web_search_agent=web_search,
    max_reasoning_steps=10,
    confidence_threshold=0.7
)
```

### Cache Configuration

```python
search_agent = WebSearchAgent(
    cache_duration_hours=24,  # Cache results for 24 hours
    enable_content_extraction=True
)
```

## Data Models

### Core Data Structures

```python
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    content: Optional[str] = None
    timestamp: datetime = None

@dataclass
class ReasoningContext:
    query: str
    code_context: Optional[str] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    project_type: Optional[str] = None
    error_message: Optional[str] = None
    user_intent: Optional[str] = None

@dataclass
class ReasoningResult:
    query: str
    answer: str
    confidence: float
    reasoning_steps: List[ReasoningStep]
    sources: List[SearchResult]
    recommendations: List[str]
    code_examples: List[str]
    related_topics: List[str]
```

## Performance Considerations

### Rate Limiting
- DuckDuckGo: 0.5 requests/second
- Bing API: 3.0 requests/second
- Google API: 1.0 requests/second

### Caching Strategy
- Search results cached for 24 hours by default
- Reasoning results cached for 1 hour
- Cache keys based on query content and parameters

### Content Extraction
- Limited to 5000 characters per page
- Removes scripts, styles, and navigation elements
- Timeout of 30 seconds per page

## Error Handling

### Search Engine Failures
- Graceful fallback between search engines
- Continues with available engines if one fails
- Returns empty results rather than throwing exceptions

### Network Issues
- Timeout handling for web requests
- Retry mechanisms with exponential backoff
- Circuit breaker pattern for repeated failures

### Content Extraction Errors
- Skips pages that fail to load
- Continues processing other results
- Logs warnings for debugging

## Testing

### Unit Tests
- Comprehensive test coverage for all components
- Mock external API calls for reliable testing
- Test error conditions and edge cases

### Integration Tests
- Test full reasoning workflows
- Verify search engine integration
- Test caching and performance optimizations

### Running Tests

```bash
# Run all web search tests
python -m pytest test_web_search_agent.py -v

# Run all reasoning tests
python -m pytest test_internet_enabled_reasoning.py -v

# Run specific test
python -m pytest test_web_search_agent.py::TestWebSearchAgent::test_search -v
```

## Integration with AI IDE

### VSCodium Extension Integration
The web search system integrates with the VSCodium extension through the backend API:

```typescript
// TypeScript extension code
const searchResults = await fetch('/api/search', {
    method: 'POST',
    body: JSON.stringify({
        query: 'Python async programming',
        context: {
            language: 'python',
            file_path: 'src/main.py',
            error_message: currentError
        }
    })
});
```

### PocketFlow Integration
The reasoning engine can be used as a PocketFlow node:

```python
class InternetReasoningNode(Node):
    def __init__(self):
        super().__init__("internet_reasoning")
        self.reasoning_engine = InternetEnabledReasoningEngine(WebSearchAgent())
    
    async def execute(self, context):
        reasoning_context = ReasoningContext(
            query=context.get('query'),
            language=context.get('language'),
            error_message=context.get('error')
        )
        
        result = await self.reasoning_engine.reason_with_internet(reasoning_context)
        return {
            'answer': result.answer,
            'confidence': result.confidence,
            'sources': result.sources,
            'recommendations': result.recommendations
        }
```

## Future Enhancements

### Planned Features
1. **Semantic Search Integration**: Combine with semantic similarity search
2. **Multi-Language Support**: Enhanced support for non-English queries
3. **Real-Time Updates**: WebSocket-based real-time search updates
4. **Advanced Filtering**: Machine learning-based result filtering
5. **Custom Search Engines**: Support for domain-specific search engines

### Performance Improvements
1. **Parallel Search**: Concurrent searches across multiple engines
2. **Intelligent Caching**: ML-based cache invalidation
3. **Result Prefetching**: Predictive result caching
4. **Compression**: Result compression for faster transmission

## Troubleshooting

### Common Issues

#### No Search Results
- Check internet connectivity
- Verify API keys if using Bing/Google
- Check rate limiting status
- Review query formatting

#### Low Relevance Scores
- Refine search queries
- Use more specific context information
- Adjust filtering parameters
- Check domain inclusion/exclusion settings

#### Slow Performance
- Enable result caching
- Reduce max_results parameter
- Disable content extraction if not needed
- Check network latency

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('web_search_agent')
logger.setLevel(logging.DEBUG)

# Search with debug info
results = await search_agent.search("query", debug=True)
```

## Security Considerations

### API Key Management
- Store API keys in environment variables
- Use secure key rotation practices
- Monitor API usage and quotas

### Content Filtering
- Safe search enabled by default
- Content validation before processing
- Malicious URL detection and blocking

### Privacy Protection
- No storage of personal information
- Anonymized search queries
- Respect for robots.txt and rate limits

## Dependencies

### Required Packages
```
aiohttp>=3.8.0
beautifulsoup4>=4.12.0
requests>=2.31.0
```

### Optional Packages
```
# For enhanced performance
uvloop>=0.17.0  # Linux/macOS only

# For advanced content processing
lxml>=4.9.0
html5lib>=1.1
```

## License and Attribution

This system respects the terms of service of all integrated search engines and follows best practices for web scraping and API usage. Users are responsible for complying with the terms of service of any external APIs they configure.