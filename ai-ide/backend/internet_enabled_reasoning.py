"""
Internet-Enabled Special Reasoning System for AI IDE

This module provides deep reasoning capabilities with real-time information retrieval,
context-aware web search for coding problems, documentation lookup, and technology
trend analysis.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse
import hashlib

from web_search_agent import WebSearchAgent, SearchQuery, SearchResult


logger = logging.getLogger(__name__)


@dataclass
class ReasoningContext:
    """Context for reasoning operations"""
    query: str
    code_context: Optional[str] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    project_type: Optional[str] = None
    error_message: Optional[str] = None
    user_intent: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_type: str  # "search", "analyze", "synthesize", "validate"
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReasoningResult:
    """Result of internet-enabled reasoning"""
    query: str
    answer: str
    confidence: float
    reasoning_steps: List[ReasoningStep]
    sources: List[SearchResult]
    recommendations: List[str]
    code_examples: List[str]
    related_topics: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DocumentationSearcher:
    """Specialized searcher for documentation and API references"""
    
    def __init__(self, web_search_agent: WebSearchAgent):
        self.web_search_agent = web_search_agent
        
        # Documentation domains with high priority
        self.doc_domains = [
            "docs.python.org",
            "developer.mozilla.org",
            "docs.microsoft.com",
            "docs.oracle.com",
            "kubernetes.io",
            "docker.com",
            "reactjs.org",
            "vuejs.org",
            "angular.io",
            "nodejs.org",
            "expressjs.com",
            "flask.palletsprojects.com",
            "django.readthedocs.io",
            "fastapi.tiangolo.com",
            "spring.io",
            "hibernate.org",
            "maven.apache.org",
            "gradle.org",
            "npmjs.com",
            "pypi.org",
            "github.com",
            "stackoverflow.com",
            "medium.com",
            "dev.to"
        ]
    
    async def search_documentation(self, 
                                 query: str, 
                                 language: Optional[str] = None,
                                 framework: Optional[str] = None) -> List[SearchResult]:
        """Search for documentation and API references"""
        
        # Enhance query for documentation search
        doc_query = self._enhance_documentation_query(query, language, framework)
        
        # Search with documentation-focused terms
        search_query = SearchQuery(
            query=doc_query,
            max_results=15,
            safe_search=True
        )
        
        results = await self.web_search_agent.search(search_query)
        
        # Filter and prioritize documentation results
        doc_results = self._prioritize_documentation_results(results)
        
        return doc_results
    
    def _enhance_documentation_query(self, 
                                   query: str, 
                                   language: Optional[str] = None,
                                   framework: Optional[str] = None) -> str:
        """Enhance query with documentation-specific terms"""
        enhanced_terms = []
        
        # Add language-specific terms
        if language:
            enhanced_terms.append(f"{language} documentation")
            enhanced_terms.append(f"{language} API reference")
        
        # Add framework-specific terms
        if framework:
            enhanced_terms.append(f"{framework} docs")
            enhanced_terms.append(f"{framework} guide")
        
        # Add general documentation terms
        enhanced_terms.extend([
            "official documentation",
            "API reference",
            "tutorial",
            "guide",
            "examples"
        ])
        
        # Combine original query with enhanced terms
        enhanced_query = f"{query} {' OR '.join(enhanced_terms[:3])}"
        
        return enhanced_query
    
    def _prioritize_documentation_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Prioritize documentation results based on domain and content"""
        
        for result in results:
            domain = urlparse(result.url).netloc.lower()
            
            # Boost relevance for documentation domains
            if any(doc_domain in domain for doc_domain in self.doc_domains):
                result.relevance_score += 0.3
            
            # Boost relevance for documentation-related content
            doc_keywords = ["documentation", "docs", "api", "reference", "tutorial", "guide"]
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()
            
            doc_keyword_count = sum(1 for keyword in doc_keywords 
                                  if keyword in title_lower or keyword in snippet_lower)
            result.relevance_score += doc_keyword_count * 0.1
            
            # Ensure relevance score doesn't exceed 1.0
            result.relevance_score = min(result.relevance_score, 1.0)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results


class TechnologyTrendAnalyzer:
    """Analyzes technology trends and provides recommendations"""
    
    def __init__(self, web_search_agent: WebSearchAgent):
        self.web_search_agent = web_search_agent
        
        # Technology trend sources
        self.trend_sources = [
            "github.com",
            "stackoverflow.com",
            "hackernews.ycombinator.com",
            "reddit.com/r/programming",
            "dev.to",
            "medium.com",
            "techcrunch.com",
            "arstechnica.com",
            "infoq.com",
            "martinfowler.com"
        ]
    
    async def analyze_technology_trends(self, 
                                      technology: str,
                                      context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze current trends for a specific technology"""
        
        # Search for recent trends
        trend_queries = [
            f"{technology} trends 2024 2025",
            f"{technology} best practices latest",
            f"{technology} new features updates",
            f"{technology} vs alternatives comparison"
        ]
        
        all_results = []
        
        for query in trend_queries:
            search_query = SearchQuery(
                query=query,
                max_results=10,
                time_range="month",  # Focus on recent content
                safe_search=True
            )
            
            results = await self.web_search_agent.search(search_query)
            all_results.extend(results)
        
        # Analyze trends from results
        trend_analysis = self._analyze_trend_results(all_results, technology)
        
        return trend_analysis
    
    def _analyze_trend_results(self, results: List[SearchResult], technology: str) -> Dict[str, Any]:
        """Analyze search results to extract technology trends"""
        
        # Extract key themes and patterns
        themes = {}
        recommendations = []
        emerging_patterns = []
        
        for result in results:
            content = f"{result.title} {result.snippet}".lower()
            
            # Look for trend indicators
            trend_keywords = [
                "new", "latest", "2024", "2025", "trending", "popular",
                "best practice", "recommended", "modern", "updated",
                "performance", "security", "scalability", "migration"
            ]
            
            for keyword in trend_keywords:
                if keyword in content:
                    if keyword not in themes:
                        themes[keyword] = []
                    themes[keyword].append(result.title)
        
        # Generate recommendations based on themes
        if "performance" in themes:
            recommendations.append(f"Consider performance optimizations for {technology}")
        
        if "security" in themes:
            recommendations.append(f"Review security best practices for {technology}")
        
        if "migration" in themes:
            recommendations.append(f"Evaluate migration paths for {technology}")
        
        if any(year in themes for year in ["2024", "2025"]):
            recommendations.append(f"Stay updated with latest {technology} developments")
        
        return {
            "technology": technology,
            "themes": themes,
            "recommendations": recommendations,
            "emerging_patterns": emerging_patterns,
            "source_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }


class ContextAwareSearchEngine:
    """Context-aware search engine for coding problems"""
    
    def __init__(self, web_search_agent: WebSearchAgent):
        self.web_search_agent = web_search_agent
        self.documentation_searcher = DocumentationSearcher(web_search_agent)
        self.trend_analyzer = TechnologyTrendAnalyzer(web_search_agent)
    
    async def search_coding_problem(self, context: ReasoningContext) -> List[SearchResult]:
        """Search for solutions to coding problems with context awareness"""
        
        # Analyze the context to determine search strategy
        search_strategy = self._determine_search_strategy(context)
        
        # Generate context-aware search queries
        queries = self._generate_contextual_queries(context, search_strategy)
        
        all_results = []
        
        for query in queries:
            search_query = SearchQuery(
                query=query,
                max_results=8,
                safe_search=True
            )
            
            results = await self.web_search_agent.search(search_query)
            all_results.extend(results)
        
        # Deduplicate and rank results
        final_results = self.web_search_agent._merge_and_deduplicate_results(all_results)
        
        # Apply context-aware filtering
        filtered_results = self._apply_contextual_filtering(final_results, context)
        
        return filtered_results
    
    def _determine_search_strategy(self, context: ReasoningContext) -> str:
        """Determine the best search strategy based on context"""
        
        if context.error_message:
            return "error_resolution"
        elif context.code_context and "import" in context.code_context:
            return "library_usage"
        elif context.language:
            return "language_specific"
        elif context.project_type:
            return "project_specific"
        else:
            return "general_coding"
    
    def _generate_contextual_queries(self, context: ReasoningContext, strategy: str) -> List[str]:
        """Generate search queries based on context and strategy"""
        
        base_query = context.query
        queries = [base_query]
        
        if strategy == "error_resolution" and context.error_message:
            # Focus on error resolution
            error_query = f"{base_query} {context.error_message}"
            queries.extend([
                f"how to fix {error_query}",
                f"{error_query} solution",
                f"{error_query} troubleshooting"
            ])
        
        elif strategy == "library_usage" and context.language:
            # Focus on library and API usage
            queries.extend([
                f"{base_query} {context.language} example",
                f"{base_query} {context.language} tutorial",
                f"{base_query} {context.language} best practices"
            ])
        
        elif strategy == "language_specific" and context.language:
            # Language-specific searches
            queries.extend([
                f"{base_query} {context.language}",
                f"{context.language} {base_query} implementation",
                f"{context.language} {base_query} code example"
            ])
        
        elif strategy == "project_specific" and context.project_type:
            # Project type specific searches
            queries.extend([
                f"{base_query} {context.project_type}",
                f"{context.project_type} {base_query} pattern",
                f"{context.project_type} {base_query} architecture"
            ])
        
        else:
            # General coding searches
            queries.extend([
                f"{base_query} programming",
                f"{base_query} code example",
                f"{base_query} implementation guide"
            ])
        
        return queries[:4]  # Limit to 4 queries to avoid overwhelming
    
    def _apply_contextual_filtering(self, 
                                  results: List[SearchResult], 
                                  context: ReasoningContext) -> List[SearchResult]:
        """Apply context-aware filtering to search results"""
        
        for result in results:
            content = f"{result.title} {result.snippet}".lower()
            
            # Boost relevance based on context matches
            if context.language and context.language.lower() in content:
                result.relevance_score += 0.2
            
            if context.project_type and context.project_type.lower() in content:
                result.relevance_score += 0.15
            
            if context.error_message:
                error_keywords = context.error_message.lower().split()[:3]  # First 3 words
                for keyword in error_keywords:
                    if keyword in content:
                        result.relevance_score += 0.1
            
            # Boost for code-related content
            code_indicators = ["code", "example", "implementation", "tutorial", "guide"]
            for indicator in code_indicators:
                if indicator in content:
                    result.relevance_score += 0.05
            
            # Ensure relevance doesn't exceed 1.0
            result.relevance_score = min(result.relevance_score, 1.0)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results


class InternetEnabledReasoningEngine:
    """Main reasoning engine with internet-enabled capabilities"""
    
    def __init__(self, 
                 web_search_agent: WebSearchAgent,
                 max_reasoning_steps: int = 10,
                 confidence_threshold: float = 0.7):
        
        self.web_search_agent = web_search_agent
        self.context_search_engine = ContextAwareSearchEngine(web_search_agent)
        self.documentation_searcher = DocumentationSearcher(web_search_agent)
        self.trend_analyzer = TechnologyTrendAnalyzer(web_search_agent)
        
        self.max_reasoning_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        
        # Cache for reasoning results
        self.reasoning_cache: Dict[str, ReasoningResult] = {}
    
    async def reason_with_internet(self, context: ReasoningContext) -> ReasoningResult:
        """Perform deep reasoning with real-time information retrieval"""
        
        # Check cache first
        cache_key = self._get_cache_key(context)
        if cache_key in self.reasoning_cache:
            cached_result = self.reasoning_cache[cache_key]
            # Return cached result if it's recent (within 1 hour)
            if datetime.now() - cached_result.timestamp < timedelta(hours=1):
                logger.info(f"Returning cached reasoning result for: {context.query}")
                return cached_result
        
        logger.info(f"Starting internet-enabled reasoning for: {context.query}")
        
        reasoning_steps = []
        all_sources = []
        
        # Step 1: Initial context analysis
        analysis_step = await self._analyze_context(context)
        reasoning_steps.append(analysis_step)
        
        # Step 2: Context-aware search
        search_step, search_results = await self._perform_contextual_search(context)
        reasoning_steps.append(search_step)
        all_sources.extend(search_results)
        
        # Step 3: Documentation search (if applicable)
        if context.language or context.project_type:
            doc_step, doc_results = await self._search_documentation(context)
            reasoning_steps.append(doc_step)
            all_sources.extend(doc_results)
        
        # Step 4: Technology trend analysis (if applicable)
        if self._should_analyze_trends(context):
            trend_step, trend_data = await self._analyze_trends(context)
            reasoning_steps.append(trend_step)
        
        # Step 5: Synthesize information
        synthesis_step, answer = await self._synthesize_information(context, all_sources, reasoning_steps)
        reasoning_steps.append(synthesis_step)
        
        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(context, all_sources, reasoning_steps)
        
        # Step 7: Extract code examples
        code_examples = self._extract_code_examples(all_sources)
        
        # Step 8: Identify related topics
        related_topics = self._identify_related_topics(context, all_sources)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_steps)
        
        # Create reasoning result
        result = ReasoningResult(
            query=context.query,
            answer=answer,
            confidence=overall_confidence,
            reasoning_steps=reasoning_steps,
            sources=all_sources[:10],  # Limit sources
            recommendations=recommendations,
            code_examples=code_examples,
            related_topics=related_topics
        )
        
        # Cache the result
        self.reasoning_cache[cache_key] = result
        
        logger.info(f"Completed reasoning with confidence: {overall_confidence:.2f}")
        return result
    
    async def _analyze_context(self, context: ReasoningContext) -> ReasoningStep:
        """Analyze the reasoning context"""
        
        analysis = {
            "query_type": self._classify_query_type(context.query),
            "has_code_context": context.code_context is not None,
            "has_error": context.error_message is not None,
            "language": context.language,
            "project_type": context.project_type,
            "complexity": self._assess_query_complexity(context.query)
        }
        
        return ReasoningStep(
            step_type="analyze",
            description="Analyzed reasoning context and query characteristics",
            input_data={"context": asdict(context)},
            output_data=analysis,
            confidence=0.9
        )
    
    async def _perform_contextual_search(self, context: ReasoningContext) -> Tuple[ReasoningStep, List[SearchResult]]:
        """Perform context-aware search"""
        
        search_results = await self.context_search_engine.search_coding_problem(context)
        
        step = ReasoningStep(
            step_type="search",
            description=f"Performed contextual search and found {len(search_results)} relevant results",
            input_data={"query": context.query, "context_type": "coding_problem"},
            output_data={"result_count": len(search_results), "top_sources": [r.url for r in search_results[:3]]},
            confidence=0.8 if search_results else 0.3
        )
        
        return step, search_results
    
    async def _search_documentation(self, context: ReasoningContext) -> Tuple[ReasoningStep, List[SearchResult]]:
        """Search for documentation"""
        
        doc_results = await self.documentation_searcher.search_documentation(
            context.query,
            context.language,
            context.project_type
        )
        
        step = ReasoningStep(
            step_type="search",
            description=f"Searched documentation and found {len(doc_results)} relevant documents",
            input_data={"query": context.query, "language": context.language},
            output_data={"doc_count": len(doc_results), "doc_sources": [r.url for r in doc_results[:3]]},
            confidence=0.85 if doc_results else 0.4
        )
        
        return step, doc_results
    
    async def _analyze_trends(self, context: ReasoningContext) -> Tuple[ReasoningStep, Dict[str, Any]]:
        """Analyze technology trends"""
        
        technology = context.language or context.project_type or "programming"
        trend_data = await self.trend_analyzer.analyze_technology_trends(technology, context.query)
        
        step = ReasoningStep(
            step_type="analyze",
            description=f"Analyzed technology trends for {technology}",
            input_data={"technology": technology, "context": context.query},
            output_data=trend_data,
            confidence=0.7
        )
        
        return step, trend_data
    
    async def _synthesize_information(self, 
                                    context: ReasoningContext, 
                                    sources: List[SearchResult],
                                    reasoning_steps: List[ReasoningStep]) -> Tuple[ReasoningStep, str]:
        """Synthesize information from all sources"""
        
        # Extract key information from sources
        key_points = []
        for source in sources[:5]:  # Top 5 sources
            if source.snippet:
                key_points.append(source.snippet)
        
        # Create synthesized answer
        answer = self._create_synthesized_answer(context, key_points, reasoning_steps)
        
        step = ReasoningStep(
            step_type="synthesize",
            description="Synthesized information from multiple sources into coherent answer",
            input_data={"source_count": len(sources), "key_points_count": len(key_points)},
            output_data={"answer_length": len(answer), "synthesis_method": "multi_source_aggregation"},
            confidence=0.8 if key_points else 0.5
        )
        
        return step, answer
    
    def _create_synthesized_answer(self, 
                                 context: ReasoningContext, 
                                 key_points: List[str],
                                 reasoning_steps: List[ReasoningStep]) -> str:
        """Create a synthesized answer from key points"""
        
        if not key_points:
            return f"I found limited information about '{context.query}'. Consider refining your query or checking official documentation."
        
        # Start with context acknowledgment
        answer_parts = [f"Based on my analysis of '{context.query}'"]
        
        # Add context-specific information
        if context.language:
            answer_parts.append(f"in {context.language}")
        
        if context.error_message:
            answer_parts.append("regarding the error you're experiencing")
        
        answer_parts.append(":\n\n")
        
        # Add synthesized information
        answer_parts.append("Here's what I found from multiple sources:\n\n")
        
        # Process key points
        processed_points = []
        for i, point in enumerate(key_points[:3], 1):
            # Clean and truncate point
            clean_point = point.strip()
            if len(clean_point) > 200:
                clean_point = clean_point[:200] + "..."
            processed_points.append(f"{i}. {clean_point}")
        
        answer_parts.extend(processed_points)
        
        # Add confidence note
        confidence_levels = [step.confidence for step in reasoning_steps]
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0.5
        
        if avg_confidence > 0.8:
            answer_parts.append("\n\nThis information appears to be highly reliable based on multiple authoritative sources.")
        elif avg_confidence > 0.6:
            answer_parts.append("\n\nThis information is based on several sources and appears to be reliable.")
        else:
            answer_parts.append("\n\nPlease verify this information with official documentation as confidence is moderate.")
        
        return "\n".join(answer_parts)
    
    def _generate_recommendations(self, 
                                context: ReasoningContext, 
                                sources: List[SearchResult],
                                reasoning_steps: List[ReasoningStep]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Context-based recommendations
        if context.error_message:
            recommendations.append("Debug the error step by step using the suggested solutions")
            recommendations.append("Check for common causes of this error in your environment")
        
        if context.language:
            recommendations.append(f"Review the official {context.language} documentation for best practices")
            recommendations.append(f"Consider using {context.language}-specific tools and libraries")
        
        # Source-based recommendations
        doc_sources = [s for s in sources if any(domain in s.url for domain in 
                      ["docs.", "documentation", "api", "reference"])]
        
        if doc_sources:
            recommendations.append("Refer to the official documentation links provided")
        
        stackoverflow_sources = [s for s in sources if "stackoverflow.com" in s.url]
        if stackoverflow_sources:
            recommendations.append("Check the Stack Overflow discussions for community insights")
        
        github_sources = [s for s in sources if "github.com" in s.url]
        if github_sources:
            recommendations.append("Examine the GitHub repositories for code examples and issues")
        
        # General recommendations
        recommendations.extend([
            "Test the solution in a development environment first",
            "Consider the performance and security implications",
            "Keep your dependencies and tools up to date"
        ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _extract_code_examples(self, sources: List[SearchResult]) -> List[str]:
        """Extract code examples from sources"""
        
        code_examples = []
        
        for source in sources:
            content = f"{source.title} {source.snippet}"
            
            # Look for code patterns
            code_patterns = [
                r'```[\s\S]*?```',  # Markdown code blocks
                r'`[^`]+`',         # Inline code
                r'def\s+\w+\(',     # Python functions
                r'function\s+\w+\(',# JavaScript functions
                r'class\s+\w+',     # Class definitions
                r'import\s+\w+',    # Import statements
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) > 10 and match not in code_examples:
                        code_examples.append(match.strip())
        
        return code_examples[:3]  # Limit to 3 examples
    
    def _identify_related_topics(self, context: ReasoningContext, sources: List[SearchResult]) -> List[str]:
        """Identify related topics from sources"""
        
        related_topics = set()
        
        # Extract topics from titles and snippets
        all_text = " ".join([f"{s.title} {s.snippet}" for s in sources])
        
        # Common programming topics
        topic_keywords = [
            "algorithm", "data structure", "design pattern", "framework",
            "library", "API", "database", "testing", "debugging", "performance",
            "security", "deployment", "architecture", "microservices", "REST",
            "GraphQL", "authentication", "authorization", "caching", "logging"
        ]
        
        for keyword in topic_keywords:
            if keyword.lower() in all_text.lower():
                related_topics.add(keyword.title())
        
        # Language-specific topics
        if context.language:
            lang_topics = {
                "python": ["Django", "Flask", "FastAPI", "NumPy", "Pandas", "pytest"],
                "javascript": ["React", "Vue", "Angular", "Node.js", "Express", "Jest"],
                "java": ["Spring", "Hibernate", "Maven", "Gradle", "JUnit"],
                "csharp": [".NET", "ASP.NET", "Entity Framework", "NUnit"],
                "go": ["Gin", "Echo", "GORM", "Testify"],
                "rust": ["Cargo", "Serde", "Tokio", "Actix"]
            }
            
            lang_key = context.language.lower()
            if lang_key in lang_topics:
                for topic in lang_topics[lang_key]:
                    if topic.lower() in all_text.lower():
                        related_topics.add(topic)
        
        return list(related_topics)[:8]  # Limit to 8 topics
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["error", "exception", "bug", "fix", "broken"]):
            return "error_resolution"
        elif any(word in query_lower for word in ["how to", "tutorial", "guide", "learn"]):
            return "learning"
        elif any(word in query_lower for word in ["best practice", "pattern", "architecture"]):
            return "best_practices"
        elif any(word in query_lower for word in ["performance", "optimize", "speed", "memory"]):
            return "optimization"
        elif any(word in query_lower for word in ["security", "authentication", "authorization"]):
            return "security"
        else:
            return "general"
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the query"""
        
        word_count = len(query.split())
        
        if word_count <= 3:
            return "simple"
        elif word_count <= 8:
            return "moderate"
        else:
            return "complex"
    
    def _should_analyze_trends(self, context: ReasoningContext) -> bool:
        """Determine if trend analysis is needed"""
        
        trend_indicators = [
            "latest", "new", "modern", "current", "trend", "popular",
            "2024", "2025", "best", "recommended", "comparison"
        ]
        
        query_lower = context.query.lower()
        return any(indicator in query_lower for indicator in trend_indicators)
    
    def _calculate_overall_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from reasoning steps"""
        
        if not reasoning_steps:
            return 0.5
        
        confidences = [step.confidence for step in reasoning_steps]
        
        # Weighted average with higher weight for synthesis steps
        weighted_sum = 0
        total_weight = 0
        
        for step in reasoning_steps:
            weight = 2.0 if step.step_type == "synthesize" else 1.0
            weighted_sum += step.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _get_cache_key(self, context: ReasoningContext) -> str:
        """Generate cache key for reasoning context"""
        
        key_data = {
            "query": context.query,
            "language": context.language,
            "project_type": context.project_type,
            "error_message": context.error_message
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the reasoning cache"""
        self.reasoning_cache.clear()
        logger.info("Reasoning cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.reasoning_cache),
            "cache_keys": list(self.reasoning_cache.keys())
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_internet_reasoning():
        """Test the internet-enabled reasoning functionality"""
        
        # Initialize components
        web_search_agent = WebSearchAgent()
        reasoning_engine = InternetEnabledReasoningEngine(web_search_agent)
        
        # Test context
        context = ReasoningContext(
            query="How to handle async/await in Python with error handling",
            language="python",
            project_type="web_application",
            user_intent="learning"
        )
        
        # Perform reasoning
        result = await reasoning_engine.reason_with_internet(context)
        
        print(f"Query: {result.query}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Answer: {result.answer}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\nSources ({len(result.sources)}):")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"  {i}. {source.title} - {source.url}")
        
        print(f"\nReasoning Steps ({len(result.reasoning_steps)}):")
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"  {i}. {step.step_type}: {step.description} (confidence: {step.confidence:.2f})")
    
    # Run test
    asyncio.run(test_internet_reasoning())