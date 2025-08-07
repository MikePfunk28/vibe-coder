"""
Context-Aware RAG Pipeline

This module implements a context-aware RAG pipeline with query expansion,
re-ranking, context fusion, and quality assessment capabilities.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_system import (
    ChunkType,
    DocumentChunk,
    KnowledgeBaseRetrieval,
    EmbeddingModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context information for a query."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_file: Optional[str] = None
    project_context: Optional[str] = None
    recent_queries: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    task_type: Optional[str] = None  # e.g., "code_completion", "debugging", "documentation"
    programming_language: Optional[str] = None
    framework: Optional[str] = None
    domain: Optional[str] = None  # e.g., "web_development", "data_science", "machine_learning"


@dataclass
class ExpandedQuery:
    """An expanded query with additional terms and context."""
    original_query: str
    expanded_terms: List[str]
    synonyms: List[str]
    related_concepts: List[str]
    context_keywords: List[str]
    final_query: str
    expansion_confidence: float = 0.0


@dataclass
class RankedResult:
    """A search result with ranking information."""
    chunk: DocumentChunk
    original_score: float
    rerank_score: float
    final_score: float
    ranking_features: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None


@dataclass
class RAGResponse:
    """Complete RAG response with context and quality metrics."""
    query: str
    expanded_query: ExpandedQuery
    retrieved_chunks: List[RankedResult]
    synthesized_answer: str
    confidence_score: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    feedback_score: Optional[float] = None


class QueryExpander(ABC):
    """Abstract base class for query expansion."""
    
    @abstractmethod
    def expand_query(self, query: str, context: QueryContext) -> ExpandedQuery:
        """Expand a query with additional terms and context."""
        pass


class SemanticQueryExpander(QueryExpander):
    """Semantic query expander using embeddings and knowledge base."""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBaseRetrieval,
                 embedding_model: EmbeddingModel,
                 expansion_ratio: float = 0.3):
        self.knowledge_base = knowledge_base
        self.embedding_model = embedding_model
        self.expansion_ratio = expansion_ratio
        
        # Pre-built vocabulary for common programming concepts
        self.programming_synonyms = {
            'function': ['method', 'procedure', 'routine', 'callable'],
            'variable': ['var', 'identifier', 'symbol', 'name'],
            'class': ['object', 'type', 'constructor', 'blueprint'],
            'error': ['exception', 'bug', 'issue', 'problem', 'failure'],
            'test': ['testing', 'unittest', 'spec', 'validation'],
            'debug': ['debugging', 'troubleshoot', 'diagnose', 'fix'],
            'optimize': ['optimization', 'performance', 'efficiency', 'speed'],
            'api': ['interface', 'endpoint', 'service', 'rest'],
            'database': ['db', 'storage', 'persistence', 'data'],
            'framework': ['library', 'toolkit', 'platform', 'stack']
        }
        
        self.domain_keywords = {
            'web_development': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'node', 'express'],
            'data_science': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'analysis'],
            'machine_learning': ['tensorflow', 'pytorch', 'sklearn', 'model', 'training', 'neural'],
            'backend': ['server', 'api', 'database', 'microservice', 'rest', 'graphql'],
            'frontend': ['ui', 'ux', 'component', 'state', 'routing', 'styling']
        }
    
    def expand_query(self, query: str, context: QueryContext) -> ExpandedQuery:
        """Expand query using semantic similarity and context."""
        expanded_terms = []
        synonyms = []
        related_concepts = []
        context_keywords = []
        
        # Extract key terms from original query
        query_terms = self._extract_key_terms(query)
        
        # Add synonyms for programming terms
        for term in query_terms:
            term_lower = term.lower()
            if term_lower in self.programming_synonyms:
                synonyms.extend(self.programming_synonyms[term_lower])
        
        # Add domain-specific keywords based on context
        if context.domain and context.domain in self.domain_keywords:
            context_keywords.extend(self.domain_keywords[context.domain])
        
        if context.programming_language:
            context_keywords.append(context.programming_language.lower())
        
        if context.framework:
            context_keywords.append(context.framework.lower())
        
        # Find semantically similar terms using knowledge base
        semantic_terms = self._find_semantic_terms(query, context)
        related_concepts.extend(semantic_terms)
        
        # Add terms from recent queries for session continuity
        if context.recent_queries:
            recent_terms = []
            for recent_query in context.recent_queries[-3:]:  # Last 3 queries
                recent_terms.extend(self._extract_key_terms(recent_query))
            
            # Add unique terms that might be related
            for term in recent_terms:
                if (term.lower() not in query.lower() and 
                    len(term) > 2 and 
                    term not in expanded_terms):
                    expanded_terms.append(term)
        
        # Combine all expansion terms
        all_expansion_terms = list(set(synonyms + related_concepts + context_keywords + expanded_terms))
        
        # Limit expansion terms based on ratio
        max_terms = max(1, int(len(query_terms) * self.expansion_ratio))
        if len(all_expansion_terms) > max_terms:
            # Prioritize terms based on relevance
            all_expansion_terms = self._prioritize_terms(query, all_expansion_terms, max_terms)
        
        # Create final expanded query
        final_query = query
        if all_expansion_terms:
            final_query += " " + " ".join(all_expansion_terms[:max_terms])
        
        # Calculate expansion confidence
        confidence = min(1.0, len(all_expansion_terms) / max(1, max_terms))
        
        return ExpandedQuery(
            original_query=query,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            related_concepts=related_concepts,
            context_keywords=context_keywords,
            final_query=final_query,
            expansion_confidence=confidence
        )
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple term extraction - can be enhanced with NLP
        terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        key_terms = [term for term in terms if term.lower() not in stop_words and len(term) > 2]
        return list(set(key_terms))  # Remove duplicates
    
    def _find_semantic_terms(self, query: str, context: QueryContext) -> List[str]:
        """Find semantically similar terms using the knowledge base."""
        semantic_terms = []
        
        try:
            # Search for similar chunks in knowledge base
            similar_chunks = self.knowledge_base.search(query, top_k=10)
            
            # Extract terms from similar chunks
            for chunk in similar_chunks:
                if chunk.relevance_score > 0.7:  # High similarity threshold
                    chunk_terms = self._extract_key_terms(chunk.content)
                    semantic_terms.extend(chunk_terms[:3])  # Top 3 terms per chunk
            
            # Remove duplicates and terms already in query
            query_lower = query.lower()
            semantic_terms = [
                term for term in set(semantic_terms) 
                if term.lower() not in query_lower and len(term) > 2
            ]
        
        except Exception as e:
            logger.warning(f"Error finding semantic terms: {e}")
        
        return semantic_terms[:5]  # Limit to top 5 semantic terms
    
    def _prioritize_terms(self, query: str, terms: List[str], max_terms: int) -> List[str]:
        """Prioritize expansion terms based on relevance to query."""
        if not terms:
            return []
        
        try:
            # Use TF-IDF to score term relevance
            documents = [query] + terms
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarity between query and each term
            query_vector = tfidf_matrix[0:1]
            term_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, term_vectors)[0]
            
            # Sort terms by similarity
            term_scores = list(zip(terms, similarities))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [term for term, score in term_scores[:max_terms]]
        
        except Exception as e:
            logger.warning(f"Error prioritizing terms: {e}")
            return terms[:max_terms]


class CrossEncoderReranker:
    """Re-ranker using cross-encoder models for better relevance scoring."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Initialized cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback re-ranking")
            self.model = None
        except Exception as e:
            logger.warning(f"Error initializing cross-encoder model: {e}")
            self.model = None
    
    def rerank(self, 
               query: str, 
               chunks: List[DocumentChunk], 
               context: QueryContext,
               top_k: Optional[int] = None) -> List[RankedResult]:
        """Re-rank chunks using cross-encoder model."""
        if not chunks:
            return []
        
        ranked_results = []
        
        if self.model is not None:
            # Use cross-encoder model
            ranked_results = self._rerank_with_model(query, chunks, context)
        else:
            # Fallback to heuristic re-ranking
            ranked_results = self._rerank_heuristic(query, chunks, context)
        
        # Apply top-k filtering if specified
        if top_k is not None:
            ranked_results = ranked_results[:top_k]
        
        return ranked_results
    
    def _rerank_with_model(self, 
                          query: str, 
                          chunks: List[DocumentChunk], 
                          context: QueryContext) -> List[RankedResult]:
        """Re-rank using cross-encoder model."""
        try:
            # Prepare query-document pairs
            pairs = []
            for chunk in chunks:
                # Combine query with context for better scoring
                enhanced_query = self._enhance_query_with_context(query, context)
                pairs.append([enhanced_query, chunk.content])
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Create ranked results
            ranked_results = []
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                original_score = getattr(chunk, 'relevance_score', 0.0)
                
                # Combine original and rerank scores
                final_score = 0.3 * original_score + 0.7 * float(score)
                
                ranking_features = {
                    'cross_encoder_score': float(score),
                    'original_score': original_score,
                    'context_boost': self._calculate_context_boost(chunk, context),
                    'chunk_type_boost': self._calculate_chunk_type_boost(chunk, context),
                    'recency_boost': self._calculate_recency_boost(chunk)
                }
                
                # Apply additional boosts
                for feature, boost in ranking_features.items():
                    if feature != 'cross_encoder_score' and feature != 'original_score':
                        final_score += boost
                
                ranked_result = RankedResult(
                    chunk=chunk,
                    original_score=original_score,
                    rerank_score=float(score),
                    final_score=final_score,
                    ranking_features=ranking_features,
                    explanation=self._generate_ranking_explanation(ranking_features)
                )
                ranked_results.append(ranked_result)
            
            # Sort by final score
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in model re-ranking: {e}")
            # Fallback to heuristic re-ranking
            ranked_results = self._rerank_heuristic(query, chunks, context)
        
        return ranked_results
    
    def _rerank_heuristic(self, 
                         query: str, 
                         chunks: List[DocumentChunk], 
                         context: QueryContext) -> List[RankedResult]:
        """Fallback heuristic re-ranking."""
        ranked_results = []
        
        for chunk in chunks:
            original_score = getattr(chunk, 'relevance_score', 0.0)
            
            # Calculate heuristic features
            ranking_features = {
                'text_similarity': self._calculate_text_similarity(query, chunk.content),
                'context_boost': self._calculate_context_boost(chunk, context),
                'chunk_type_boost': self._calculate_chunk_type_boost(chunk, context),
                'recency_boost': self._calculate_recency_boost(chunk),
                'length_penalty': self._calculate_length_penalty(chunk.content)
            }
            
            # Combine features for final score
            final_score = original_score
            for feature, value in ranking_features.items():
                final_score += value
            
            ranked_result = RankedResult(
                chunk=chunk,
                original_score=original_score,
                rerank_score=ranking_features['text_similarity'],
                final_score=final_score,
                ranking_features=ranking_features,
                explanation=self._generate_ranking_explanation(ranking_features)
            )
            ranked_results.append(ranked_result)
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_results
    
    def _enhance_query_with_context(self, query: str, context: QueryContext) -> str:
        """Enhance query with context information."""
        enhanced_query = query
        
        if context.programming_language:
            enhanced_query += f" {context.programming_language}"
        
        if context.framework:
            enhanced_query += f" {context.framework}"
        
        if context.task_type:
            enhanced_query += f" {context.task_type.replace('_', ' ')}"
        
        return enhanced_query
    
    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """Calculate text similarity using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([query, content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity) * 0.2  # Weight for text similarity
        except:
            return 0.0
    
    def _calculate_context_boost(self, chunk: DocumentChunk, context: QueryContext) -> float:
        """Calculate boost based on context matching."""
        boost = 0.0
        content_lower = chunk.content.lower()
        
        # Programming language boost
        if context.programming_language:
            lang = context.programming_language.lower()
            if lang in content_lower:
                boost += 0.1
        
        # Framework boost
        if context.framework:
            framework = context.framework.lower()
            if framework in content_lower:
                boost += 0.1
        
        # Domain boost
        if context.domain:
            domain_keywords = {
                'web_development': ['html', 'css', 'javascript', 'web', 'browser'],
                'data_science': ['data', 'analysis', 'pandas', 'numpy', 'visualization'],
                'machine_learning': ['model', 'training', 'neural', 'algorithm', 'prediction']
            }
            
            if context.domain in domain_keywords:
                for keyword in domain_keywords[context.domain]:
                    if keyword in content_lower:
                        boost += 0.05
                        break
        
        return boost
    
    def _calculate_chunk_type_boost(self, chunk: DocumentChunk, context: QueryContext) -> float:
        """Calculate boost based on chunk type relevance."""
        boost = 0.0
        
        # Task-specific chunk type preferences
        if context.task_type == 'code_completion':
            if chunk.chunk_type == ChunkType.CODE_BLOCK:
                boost += 0.15
            elif chunk.chunk_type == ChunkType.PARAGRAPH:
                boost += 0.05
        
        elif context.task_type == 'documentation':
            if chunk.chunk_type == ChunkType.SECTION:
                boost += 0.1
            elif chunk.chunk_type == ChunkType.PARAGRAPH:
                boost += 0.15
        
        elif context.task_type == 'debugging':
            if chunk.chunk_type == ChunkType.CODE_BLOCK:
                boost += 0.2
        
        return boost
    
    def _calculate_recency_boost(self, chunk: DocumentChunk) -> float:
        """Calculate boost based on document recency."""
        if not chunk.metadata.updated_at:
            return 0.0
        
        try:
            days_old = (datetime.now() - chunk.metadata.updated_at).days
            
            # Boost recent documents
            if days_old < 30:
                return 0.1
            elif days_old < 90:
                return 0.05
            elif days_old > 365:
                return -0.05  # Slight penalty for very old content
            
        except:
            pass
        
        return 0.0
    
    def _calculate_length_penalty(self, content: str) -> float:
        """Calculate penalty for very short or very long content."""
        length = len(content)
        
        if length < 50:
            return -0.1  # Penalty for very short content
        elif length > 2000:
            return -0.05  # Small penalty for very long content
        
        return 0.0
    
    def _generate_ranking_explanation(self, features: Dict[str, float]) -> str:
        """Generate human-readable explanation for ranking."""
        explanations = []
        
        for feature, value in features.items():
            if value > 0.1:
                if feature == 'cross_encoder_score':
                    explanations.append("high semantic relevance")
                elif feature == 'context_boost':
                    explanations.append("matches context")
                elif feature == 'chunk_type_boost':
                    explanations.append("appropriate content type")
                elif feature == 'recency_boost':
                    explanations.append("recent content")
        
        if not explanations:
            return "baseline relevance"
        
        return ", ".join(explanations)


class ContextFusion:
    """Context fusion for combining multiple information sources."""
    
    def __init__(self):
        self.fusion_strategies = {
            'weighted_average': self._weighted_average_fusion,
            'max_score': self._max_score_fusion,
            'rank_fusion': self._rank_fusion,
            'adaptive': self._adaptive_fusion
        }
    
    def fuse_results(self, 
                    results_list: List[List[RankedResult]], 
                    strategy: str = 'adaptive',
                    context: Optional[QueryContext] = None) -> List[RankedResult]:
        """Fuse results from multiple sources."""
        if not results_list or not any(results_list):
            return []
        
        if len(results_list) == 1:
            return results_list[0]
        
        fusion_func = self.fusion_strategies.get(strategy, self._adaptive_fusion)
        return fusion_func(results_list, context)
    
    def _weighted_average_fusion(self, 
                                results_list: List[List[RankedResult]], 
                                context: Optional[QueryContext] = None) -> List[RankedResult]:
        """Fuse results using weighted average of scores."""
        # Collect all unique chunks
        chunk_scores = {}
        chunk_objects = {}
        
        # Default weights (can be adjusted based on source reliability)
        weights = [1.0 / len(results_list)] * len(results_list)
        
        for i, results in enumerate(results_list):
            weight = weights[i]
            for result in results:
                chunk_id = result.chunk.id
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0.0
                    chunk_objects[chunk_id] = result
                
                chunk_scores[chunk_id] += result.final_score * weight
        
        # Create fused results
        fused_results = []
        for chunk_id, score in chunk_scores.items():
            result = chunk_objects[chunk_id]
            fused_result = RankedResult(
                chunk=result.chunk,
                original_score=result.original_score,
                rerank_score=result.rerank_score,
                final_score=score,
                ranking_features=result.ranking_features,
                explanation=f"fused score from {len(results_list)} sources"
            )
            fused_results.append(fused_result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _max_score_fusion(self, 
                         results_list: List[List[RankedResult]], 
                         context: Optional[QueryContext] = None) -> List[RankedResult]:
        """Fuse results using maximum score for each chunk."""
        chunk_best = {}
        
        for results in results_list:
            for result in results:
                chunk_id = result.chunk.id
                
                if (chunk_id not in chunk_best or 
                    result.final_score > chunk_best[chunk_id].final_score):
                    chunk_best[chunk_id] = result
        
        fused_results = list(chunk_best.values())
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _rank_fusion(self, 
                    results_list: List[List[RankedResult]], 
                    context: Optional[QueryContext] = None) -> List[RankedResult]:
        """Fuse results using reciprocal rank fusion."""
        chunk_rank_scores = {}
        chunk_objects = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                chunk_id = result.chunk.id
                
                if chunk_id not in chunk_rank_scores:
                    chunk_rank_scores[chunk_id] = 0.0
                    chunk_objects[chunk_id] = result
                
                # Reciprocal rank fusion formula
                chunk_rank_scores[chunk_id] += 1.0 / (rank + 1)
        
        # Create fused results
        fused_results = []
        for chunk_id, rank_score in chunk_rank_scores.items():
            result = chunk_objects[chunk_id]
            fused_result = RankedResult(
                chunk=result.chunk,
                original_score=result.original_score,
                rerank_score=result.rerank_score,
                final_score=rank_score,
                ranking_features=result.ranking_features,
                explanation="rank fusion score"
            )
            fused_results.append(fused_result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    def _adaptive_fusion(self, 
                        results_list: List[List[RankedResult]], 
                        context: Optional[QueryContext] = None) -> List[RankedResult]:
        """Adaptive fusion based on context and result quality."""
        # Analyze result quality to determine best fusion strategy
        avg_scores = [
            sum(r.final_score for r in results) / len(results) if results else 0.0
            for results in results_list
        ]
        
        score_variance = np.var(avg_scores) if avg_scores else 0.0
        
        # Choose fusion strategy based on score distribution
        if score_variance > 0.1:
            # High variance - use max score to favor best results
            return self._max_score_fusion(results_list, context)
        else:
            # Low variance - use weighted average for stability
            return self._weighted_average_fusion(results_list, context)


class AnswerSynthesizer:
    """Synthesize answers from retrieved chunks."""
    
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length
    
    def synthesize_answer(self, 
                         query: str, 
                         ranked_results: List[RankedResult],
                         context: QueryContext) -> str:
        """Synthesize an answer from ranked results."""
        if not ranked_results:
            return "I couldn't find relevant information to answer your query."
        
        # Select top chunks for synthesis
        top_chunks = ranked_results[:5]  # Use top 5 chunks
        
        # Extract and combine content
        combined_content = self._combine_chunk_content(top_chunks)
        
        # Generate answer based on task type
        if context.task_type == 'code_completion':
            return self._synthesize_code_answer(query, combined_content, top_chunks)
        elif context.task_type == 'debugging':
            return self._synthesize_debug_answer(query, combined_content, top_chunks)
        elif context.task_type == 'documentation':
            return self._synthesize_doc_answer(query, combined_content, top_chunks)
        else:
            return self._synthesize_general_answer(query, combined_content, top_chunks)
    
    def _combine_chunk_content(self, ranked_results: List[RankedResult]) -> str:
        """Combine content from multiple chunks."""
        combined = []
        current_length = 0
        
        for result in ranked_results:
            content = result.chunk.content
            
            # Add source information
            source_info = f"[Source: {result.chunk.metadata.title}]"
            content_with_source = f"{source_info}\n{content}\n"
            
            if current_length + len(content_with_source) > self.max_context_length:
                # Truncate if too long
                remaining_length = self.max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful content can fit
                    truncated_content = content_with_source[:remaining_length] + "..."
                    combined.append(truncated_content)
                break
            
            combined.append(content_with_source)
            current_length += len(content_with_source)
        
        return "\n\n".join(combined)
    
    def _synthesize_code_answer(self, 
                               query: str, 
                               content: str, 
                               chunks: List[RankedResult]) -> str:
        """Synthesize answer for code completion tasks."""
        code_blocks = []
        explanations = []
        
        for result in chunks:
            if result.chunk.chunk_type == ChunkType.CODE_BLOCK:
                code_blocks.append(result.chunk.content)
            else:
                explanations.append(result.chunk.content)
        
        answer_parts = []
        
        if code_blocks:
            answer_parts.append("Here are relevant code examples:")
            for i, code in enumerate(code_blocks[:3]):  # Top 3 code blocks
                answer_parts.append(f"\n**Example {i+1}:**\n```\n{code}\n```")
        
        if explanations:
            answer_parts.append("\n**Explanation:**")
            answer_parts.append(explanations[0][:500] + "..." if len(explanations[0]) > 500 else explanations[0])
        
        return "\n".join(answer_parts)
    
    def _synthesize_debug_answer(self, 
                                query: str, 
                                content: str, 
                                chunks: List[RankedResult]) -> str:
        """Synthesize answer for debugging tasks."""
        answer_parts = ["Based on the available information, here are potential solutions:"]
        
        solutions = []
        for result in chunks[:3]:  # Top 3 results
            chunk_content = result.chunk.content
            if len(chunk_content) > 300:
                chunk_content = chunk_content[:300] + "..."
            
            solutions.append(f"• {chunk_content}")
        
        answer_parts.extend(solutions)
        
        # Add source references
        sources = [result.chunk.metadata.source_url for result in chunks[:3]]
        if sources:
            answer_parts.append("\n**Sources:**")
            for source in sources:
                answer_parts.append(f"- {source}")
        
        return "\n".join(answer_parts)
    
    def _synthesize_doc_answer(self, 
                              query: str, 
                              content: str, 
                              chunks: List[RankedResult]) -> str:
        """Synthesize answer for documentation tasks."""
        # Prioritize section and paragraph chunks for documentation
        doc_chunks = [
            result for result in chunks 
            if result.chunk.chunk_type in [ChunkType.SECTION, ChunkType.PARAGRAPH]
        ]
        
        if not doc_chunks:
            doc_chunks = chunks[:3]
        
        answer_parts = []
        
        for i, result in enumerate(doc_chunks[:3]):
            content = result.chunk.content
            if len(content) > 400:
                content = content[:400] + "..."
            
            title = result.chunk.metadata.title
            answer_parts.append(f"**From {title}:**\n{content}")
        
        return "\n\n".join(answer_parts)
    
    def _synthesize_general_answer(self, 
                                  query: str, 
                                  content: str, 
                                  chunks: List[RankedResult]) -> str:
        """Synthesize general answer."""
        # Extract key information from top chunks
        key_info = []
        
        for result in chunks[:3]:
            content = result.chunk.content
            
            # Extract first few sentences or up to 200 characters
            sentences = content.split('. ')
            if len(sentences) > 0:
                summary = sentences[0]
                if len(summary) < 150 and len(sentences) > 1:
                    summary += '. ' + sentences[1]
                
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                
                key_info.append(summary)
        
        if key_info:
            answer = "Based on the available information:\n\n"
            answer += "\n\n".join(f"• {info}" for info in key_info)
            return answer
        
        return "I found some relevant information, but couldn't synthesize a clear answer. Please check the source documents for more details."


class RAGQualityAssessment:
    """Quality assessment and feedback system for RAG responses."""
    
    def __init__(self):
        self.quality_metrics = [
            'relevance',
            'completeness',
            'accuracy',
            'clarity',
            'timeliness'
        ]
    
    def assess_quality(self, 
                      query: str, 
                      response: RAGResponse, 
                      context: QueryContext) -> Dict[str, float]:
        """Assess the quality of a RAG response."""
        metrics = {}
        
        # Relevance assessment
        metrics['relevance'] = self._assess_relevance(query, response)
        
        # Completeness assessment
        metrics['completeness'] = self._assess_completeness(query, response, context)
        
        # Accuracy assessment (based on source quality)
        metrics['accuracy'] = self._assess_accuracy(response)
        
        # Clarity assessment
        metrics['clarity'] = self._assess_clarity(response)
        
        # Timeliness assessment
        metrics['timeliness'] = self._assess_timeliness(response)
        
        # Overall quality score
        weights = {
            'relevance': 0.3,
            'completeness': 0.25,
            'accuracy': 0.25,
            'clarity': 0.1,
            'timeliness': 0.1
        }
        
        overall_score = sum(metrics[metric] * weights[metric] for metric in metrics)
        metrics['overall'] = overall_score
        
        return metrics
    
    def _assess_relevance(self, query: str, response: RAGResponse) -> float:
        """Assess relevance of response to query."""
        if not response.retrieved_chunks:
            return 0.0
        
        # Average relevance score of retrieved chunks
        avg_relevance = sum(
            result.final_score for result in response.retrieved_chunks
        ) / len(response.retrieved_chunks)
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, avg_relevance))
    
    def _assess_completeness(self, 
                           query: str, 
                           response: RAGResponse, 
                           context: QueryContext) -> float:
        """Assess completeness of response."""
        completeness_score = 0.0
        
        # Check if response has content
        if response.synthesized_answer and len(response.synthesized_answer) > 50:
            completeness_score += 0.4
        
        # Check if multiple sources were used
        if len(response.retrieved_chunks) >= 3:
            completeness_score += 0.3
        
        # Check if response addresses the query type
        if context.task_type:
            if context.task_type == 'code_completion' and 'code' in response.synthesized_answer.lower():
                completeness_score += 0.2
            elif context.task_type == 'debugging' and any(word in response.synthesized_answer.lower() 
                                                        for word in ['solution', 'fix', 'error', 'problem']):
                completeness_score += 0.2
            elif context.task_type == 'documentation' and len(response.synthesized_answer) > 200:
                completeness_score += 0.2
            else:
                completeness_score += 0.1
        
        # Check if sources are provided
        if response.sources:
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
    
    def _assess_accuracy(self, response: RAGResponse) -> float:
        """Assess accuracy based on source quality."""
        if not response.retrieved_chunks:
            return 0.0
        
        accuracy_score = 0.0
        
        # Source type quality weights
        source_weights = {
            'documentation': 0.9,
            'github': 0.8,
            'stackoverflow': 0.7,
            'web_page': 0.6,
            'local_file': 0.8
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in response.retrieved_chunks:
            source_type = result.chunk.metadata.source_type.value
            weight = source_weights.get(source_type, 0.5)
            
            # Factor in the result score
            score = result.final_score * weight
            
            weighted_score += score
            total_weight += weight
        
        if total_weight > 0:
            accuracy_score = weighted_score / total_weight
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _assess_clarity(self, response: RAGResponse) -> float:
        """Assess clarity of the synthesized answer."""
        answer = response.synthesized_answer
        
        if not answer:
            return 0.0
        
        clarity_score = 0.0
        
        # Length check (not too short, not too long)
        length = len(answer)
        if 100 <= length <= 1000:
            clarity_score += 0.3
        elif 50 <= length < 100 or 1000 < length <= 2000:
            clarity_score += 0.2
        elif length > 50:
            clarity_score += 0.1
        
        # Structure check (has paragraphs or bullet points)
        if '\n\n' in answer or '•' in answer or '*' in answer:
            clarity_score += 0.3
        
        # Code formatting check
        if '```' in answer or '`' in answer:
            clarity_score += 0.2
        
        # Source references check
        if '[Source:' in answer or 'Source:' in answer:
            clarity_score += 0.2
        
        return min(1.0, clarity_score)
    
    def _assess_timeliness(self, response: RAGResponse) -> float:
        """Assess timeliness based on source recency."""
        if not response.retrieved_chunks:
            return 0.5  # Neutral score if no chunks
        
        timeliness_scores = []
        
        for result in response.retrieved_chunks:
            if result.chunk.metadata.updated_at:
                days_old = (datetime.now() - result.chunk.metadata.updated_at).days
                
                if days_old <= 30:
                    timeliness_scores.append(1.0)
                elif days_old <= 90:
                    timeliness_scores.append(0.8)
                elif days_old <= 365:
                    timeliness_scores.append(0.6)
                elif days_old <= 730:
                    timeliness_scores.append(0.4)
                else:
                    timeliness_scores.append(0.2)
            else:
                timeliness_scores.append(0.5)  # Unknown age
        
        return sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.5
    
    def collect_feedback(self, 
                        response_id: str, 
                        user_feedback: Dict[str, Any]) -> None:
        """Collect user feedback for continuous improvement."""
        # This would typically store feedback in a database
        # For now, we'll just log it
        logger.info(f"Feedback collected for response {response_id}: {user_feedback}")
        
        # Extract useful feedback metrics
        if 'rating' in user_feedback:
            rating = user_feedback['rating']
            logger.info(f"User rating: {rating}/5")
        
        if 'helpful' in user_feedback:
            helpful = user_feedback['helpful']
            logger.info(f"Response helpful: {helpful}")
        
        if 'comments' in user_feedback:
            comments = user_feedback['comments']
            logger.info(f"User comments: {comments}")


class ContextAwareRAGPipeline:
    """Main context-aware RAG pipeline."""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBaseRetrieval,
                 embedding_model: EmbeddingModel):
        self.knowledge_base = knowledge_base
        self.embedding_model = embedding_model
        
        # Initialize components
        self.query_expander = SemanticQueryExpander(knowledge_base, embedding_model)
        self.reranker = CrossEncoderReranker()
        self.context_fusion = ContextFusion()
        self.answer_synthesizer = AnswerSynthesizer()
        self.quality_assessor = RAGQualityAssessment()
        
        # Performance tracking
        self.response_history = []
    
    async def process_query(self, 
                           query: str, 
                           context: QueryContext,
                           top_k: int = 10) -> RAGResponse:
        """Process a query through the complete RAG pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Query expansion
            expanded_query = self.query_expander.expand_query(query, context)
            logger.info(f"Expanded query: {expanded_query.final_query}")
            
            # Step 2: Retrieval with multiple strategies
            retrieval_results = await self._multi_strategy_retrieval(
                expanded_query, context, top_k
            )
            
            # Step 3: Re-ranking
            reranked_results = self.reranker.rerank(
                query, [r.chunk for r in retrieval_results], context, top_k
            )
            
            # Step 4: Answer synthesis
            synthesized_answer = self.answer_synthesizer.synthesize_answer(
                query, reranked_results, context
            )
            
            # Step 5: Create response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = RAGResponse(
                query=query,
                expanded_query=expanded_query,
                retrieved_chunks=reranked_results,
                synthesized_answer=synthesized_answer,
                confidence_score=self._calculate_confidence(reranked_results),
                sources=[r.chunk.metadata.source_url for r in reranked_results[:5]],
                context_used=[context.programming_language, context.framework, context.domain],
                processing_time=processing_time
            )
            
            # Step 6: Quality assessment
            quality_metrics = self.quality_assessor.assess_quality(query, response, context)
            response.quality_metrics = quality_metrics
            
            # Store response for learning
            self.response_history.append(response)
            
            logger.info(f"RAG pipeline completed in {processing_time:.2f}s")
            logger.info(f"Quality score: {quality_metrics.get('overall', 0.0):.2f}")
            
            return response
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            
            # Return error response
            return RAGResponse(
                query=query,
                expanded_query=ExpandedQuery(
                    original_query=query,
                    expanded_terms=[],
                    synonyms=[],
                    related_concepts=[],
                    context_keywords=[],
                    final_query=query
                ),
                retrieved_chunks=[],
                synthesized_answer=f"I encountered an error while processing your query: {str(e)}",
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _multi_strategy_retrieval(self, 
                                      expanded_query: ExpandedQuery, 
                                      context: QueryContext,
                                      top_k: int) -> List[RankedResult]:
        """Retrieve using multiple strategies and fuse results."""
        retrieval_strategies = []
        
        # Strategy 1: Original query
        original_results = self.knowledge_base.search(
            expanded_query.original_query, top_k=top_k
        )
        if original_results:
            strategy1_results = [
                RankedResult(
                    chunk=chunk,
                    original_score=chunk.relevance_score,
                    rerank_score=chunk.relevance_score,
                    final_score=chunk.relevance_score
                ) for chunk in original_results
            ]
            retrieval_strategies.append(strategy1_results)
        
        # Strategy 2: Expanded query
        if expanded_query.final_query != expanded_query.original_query:
            expanded_results = self.knowledge_base.search(
                expanded_query.final_query, top_k=top_k
            )
            if expanded_results:
                strategy2_results = [
                    RankedResult(
                        chunk=chunk,
                        original_score=chunk.relevance_score,
                        rerank_score=chunk.relevance_score,
                        final_score=chunk.relevance_score * 0.9  # Slight penalty for expansion
                    ) for chunk in expanded_results
                ]
                retrieval_strategies.append(strategy2_results)
        
        # Strategy 3: Context-specific search
        if context.programming_language or context.framework:
            context_query = expanded_query.original_query
            if context.programming_language:
                context_query += f" {context.programming_language}"
            if context.framework:
                context_query += f" {context.framework}"
            
            context_results = self.knowledge_base.search(context_query, top_k=top_k)
            if context_results:
                strategy3_results = [
                    RankedResult(
                        chunk=chunk,
                        original_score=chunk.relevance_score,
                        rerank_score=chunk.relevance_score,
                        final_score=chunk.relevance_score * 1.1  # Boost for context match
                    ) for chunk in context_results
                ]
                retrieval_strategies.append(strategy3_results)
        
        # Fuse results from all strategies
        if retrieval_strategies:
            fused_results = self.context_fusion.fuse_results(
                retrieval_strategies, strategy='adaptive', context=context
            )
            return fused_results[:top_k]
        
        return []
    
    def _calculate_confidence(self, results: List[RankedResult]) -> float:
        """Calculate confidence score for the response."""
        if not results:
            return 0.0
        
        # Base confidence on top result scores
        top_scores = [r.final_score for r in results[:3]]
        avg_top_score = sum(top_scores) / len(top_scores)
        
        # Factor in number of results
        result_count_factor = min(1.0, len(results) / 5.0)
        
        # Factor in score consistency
        if len(top_scores) > 1:
            score_variance = np.var(top_scores)
            consistency_factor = max(0.5, 1.0 - score_variance)
        else:
            consistency_factor = 1.0
        
        confidence = avg_top_score * result_count_factor * consistency_factor
        return min(1.0, max(0.0, confidence))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the RAG pipeline."""
        if not self.response_history:
            return {}
        
        recent_responses = self.response_history[-100:]  # Last 100 responses
        
        metrics = {
            'total_queries': len(self.response_history),
            'avg_processing_time': sum(r.processing_time for r in recent_responses) / len(recent_responses),
            'avg_confidence': sum(r.confidence_score for r in recent_responses) / len(recent_responses),
            'avg_quality': sum(r.quality_metrics.get('overall', 0.0) for r in recent_responses) / len(recent_responses),
            'success_rate': sum(1 for r in recent_responses if r.confidence_score > 0.5) / len(recent_responses)
        }
        
        return metrics