"""
Tests for the Context-Aware RAG Pipeline
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from rag_pipeline import (
    AnswerSynthesizer,
    ContextAwareRAGPipeline,
    ContextFusion,
    CrossEncoderReranker,
    ExpandedQuery,
    QueryContext,
    QueryExpander,
    RAGQualityAssessment,
    RAGResponse,
    RankedResult,
    SemanticQueryExpander,
)
from rag_system import (
    ChunkType,
    DocumentChunk,
    DocumentMetadata,
    KnowledgeBaseRetrieval,
    SourceType,
)


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def encode(self, texts):
        """Return random embeddings."""
        return np.random.rand(len(texts), 384)
    
    def get_dimension(self):
        """Return embedding dimension."""
        return 384


class TestQueryContext:
    """Test QueryContext class."""
    
    def test_create_context(self):
        """Test creating query context."""
        context = QueryContext(
            user_id="user123",
            session_id="session456",
            current_file="main.py",
            programming_language="python",
            framework="django",
            domain="web_development",
            task_type="code_completion"
        )
        
        assert context.user_id == "user123"
        assert context.programming_language == "python"
        assert context.framework == "django"
        assert context.domain == "web_development"
        assert context.task_type == "code_completion"


class TestSemanticQueryExpander:
    """Test SemanticQueryExpander class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_kb = Mock(spec=KnowledgeBaseRetrieval)
        self.mock_embedding = MockEmbeddingModel()
        self.expander = SemanticQueryExpander(self.mock_kb, self.mock_embedding)
    
    def test_expand_query_basic(self):
        """Test basic query expansion."""
        self.setUp()
        
        query = "how to create a function in python"
        context = QueryContext(
            programming_language="python",
            domain="web_development",
            task_type="code_completion"
        )
        
        # Mock knowledge base search
        mock_chunk = Mock()
        mock_chunk.content = "def example_function(): pass"
        mock_chunk.relevance_score = 0.8
        self.mock_kb.search.return_value = [mock_chunk]
        
        expanded = self.expander.expand_query(query, context)
        
        assert expanded.original_query == query
        assert len(expanded.synonyms) > 0
        assert len(expanded.context_keywords) > 0
        assert "python" in expanded.final_query.lower()
    
    def test_extract_key_terms(self):
        """Test extracting key terms from text."""
        self.setUp()
        
        text = "How to create a function in Python for web development"
        terms = self.expander._extract_key_terms(text)
        
        expected_terms = ["create", "function", "Python", "web", "development"]
        for term in expected_terms:
            assert term in terms
        
        # Should not include stop words
        assert "to" not in terms
        assert "a" not in terms
    
    def test_programming_synonyms(self):
        """Test programming term synonyms."""
        self.setUp()
        
        query = "function error test"
        context = QueryContext()
        
        self.mock_kb.search.return_value = []
        
        expanded = self.expander.expand_query(query, context)
        
        # Should include synonyms for programming terms
        synonyms_text = " ".join(expanded.synonyms).lower()
        assert any(syn in synonyms_text for syn in ["method", "procedure"])
        assert any(syn in synonyms_text for syn in ["exception", "bug"])
        assert any(syn in synonyms_text for syn in ["testing", "unittest"])


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reranker = CrossEncoderReranker()
        
        # Create test chunks
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document",
            updated_at=datetime.now()
        )
        
        self.test_chunks = [
            DocumentChunk(
                id="chunk1",
                content="Python function definition with def keyword",
                chunk_type=ChunkType.CODE_BLOCK,
                metadata=metadata
            ),
            DocumentChunk(
                id="chunk2",
                content="JavaScript function definition with function keyword",
                chunk_type=ChunkType.CODE_BLOCK,
                metadata=metadata
            ),
            DocumentChunk(
                id="chunk3",
                content="General programming concepts and best practices",
                chunk_type=ChunkType.PARAGRAPH,
                metadata=metadata
            )
        ]
        
        # Set relevance scores
        for i, chunk in enumerate(self.test_chunks):
            chunk.relevance_score = 0.8 - (i * 0.1)
    
    def test_rerank_heuristic(self):
        """Test heuristic re-ranking."""
        self.setUp()
        
        query = "python function definition"
        context = QueryContext(
            programming_language="python",
            task_type="code_completion"
        )
        
        # Force heuristic re-ranking by setting model to None
        self.reranker.model = None
        
        results = self.reranker.rerank(query, self.test_chunks, context)
        
        assert len(results) == 3
        assert all(isinstance(r, RankedResult) for r in results)
        
        # Python-related chunk should be ranked higher
        python_result = next(r for r in results if "Python" in r.chunk.content)
        assert python_result.final_score > 0.5
    
    def test_calculate_context_boost(self):
        """Test context boost calculation."""
        self.setUp()
        
        context = QueryContext(
            programming_language="python",
            framework="django",
            domain="web_development"
        )
        
        # Test with Python content
        python_chunk = self.test_chunks[0]  # Contains "Python"
        boost = self.reranker._calculate_context_boost(python_chunk, context)
        assert boost > 0.0
        
        # Test with non-Python content
        js_chunk = self.test_chunks[1]  # Contains "JavaScript"
        boost = self.reranker._calculate_context_boost(js_chunk, context)
        assert boost >= 0.0  # Should be lower or zero
    
    def test_calculate_chunk_type_boost(self):
        """Test chunk type boost calculation."""
        self.setUp()
        
        context = QueryContext(task_type="code_completion")
        
        # Code block should get boost for code completion
        code_chunk = self.test_chunks[0]
        boost = self.reranker._calculate_chunk_type_boost(code_chunk, context)
        assert boost > 0.0
        
        # Paragraph should get less boost
        para_chunk = self.test_chunks[2]
        boost = self.reranker._calculate_chunk_type_boost(para_chunk, context)
        assert boost >= 0.0
    
    def test_calculate_recency_boost(self):
        """Test recency boost calculation."""
        self.setUp()
        
        # Recent document
        recent_metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Recent Document",
            updated_at=datetime.now() - timedelta(days=10)
        )
        recent_chunk = DocumentChunk(
            id="recent",
            content="Recent content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=recent_metadata
        )
        
        boost = self.reranker._calculate_recency_boost(recent_chunk)
        assert boost > 0.0
        
        # Old document
        old_metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Old Document",
            updated_at=datetime.now() - timedelta(days=400)
        )
        old_chunk = DocumentChunk(
            id="old",
            content="Old content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=old_metadata
        )
        
        boost = self.reranker._calculate_recency_boost(old_chunk)
        assert boost <= 0.0


class TestContextFusion:
    """Test ContextFusion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion = ContextFusion()
        
        # Create test results
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunk1 = DocumentChunk(
            id="chunk1",
            content="Content 1",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )
        chunk2 = DocumentChunk(
            id="chunk2",
            content="Content 2",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )
        
        self.results_list = [
            [
                RankedResult(chunk=chunk1, original_score=0.8, rerank_score=0.8, final_score=0.8),
                RankedResult(chunk=chunk2, original_score=0.6, rerank_score=0.6, final_score=0.6)
            ],
            [
                RankedResult(chunk=chunk1, original_score=0.7, rerank_score=0.7, final_score=0.7),
                RankedResult(chunk=chunk2, original_score=0.9, rerank_score=0.9, final_score=0.9)
            ]
        ]
    
    def test_weighted_average_fusion(self):
        """Test weighted average fusion."""
        self.setUp()
        
        fused = self.fusion._weighted_average_fusion(self.results_list)
        
        assert len(fused) == 2  # Two unique chunks
        
        # Check that scores are averaged
        chunk1_result = next(r for r in fused if r.chunk.id == "chunk1")
        chunk2_result = next(r for r in fused if r.chunk.id == "chunk2")
        
        # chunk1: (0.8 + 0.7) / 2 = 0.75
        assert abs(chunk1_result.final_score - 0.75) < 0.01
        
        # chunk2: (0.6 + 0.9) / 2 = 0.75
        assert abs(chunk2_result.final_score - 0.75) < 0.01
    
    def test_max_score_fusion(self):
        """Test max score fusion."""
        self.setUp()
        
        fused = self.fusion._max_score_fusion(self.results_list)
        
        assert len(fused) == 2
        
        # Check that max scores are used
        chunk1_result = next(r for r in fused if r.chunk.id == "chunk1")
        chunk2_result = next(r for r in fused if r.chunk.id == "chunk2")
        
        assert chunk1_result.final_score == 0.8  # max(0.8, 0.7)
        assert chunk2_result.final_score == 0.9  # max(0.6, 0.9)
    
    def test_rank_fusion(self):
        """Test reciprocal rank fusion."""
        self.setUp()
        
        fused = self.fusion._rank_fusion(self.results_list)
        
        assert len(fused) == 2
        
        # Check that results are sorted by rank fusion score
        assert fused[0].final_score >= fused[1].final_score
    
    def test_adaptive_fusion(self):
        """Test adaptive fusion strategy selection."""
        self.setUp()
        
        fused = self.fusion._adaptive_fusion(self.results_list)
        
        assert len(fused) == 2
        assert all(isinstance(r, RankedResult) for r in fused)


class TestAnswerSynthesizer:
    """Test AnswerSynthesizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = AnswerSynthesizer()
        
        # Create test ranked results
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Python Documentation"
        )
        
        code_chunk = DocumentChunk(
            id="code1",
            content="def hello_world():\n    print('Hello, World!')",
            chunk_type=ChunkType.CODE_BLOCK,
            metadata=metadata
        )
        
        text_chunk = DocumentChunk(
            id="text1",
            content="Functions in Python are defined using the def keyword.",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )
        
        self.ranked_results = [
            RankedResult(
                chunk=code_chunk,
                original_score=0.9,
                rerank_score=0.9,
                final_score=0.9
            ),
            RankedResult(
                chunk=text_chunk,
                original_score=0.8,
                rerank_score=0.8,
                final_score=0.8
            )
        ]
    
    def test_synthesize_code_answer(self):
        """Test synthesizing code completion answer."""
        self.setUp()
        
        query = "how to define a function in python"
        context = QueryContext(task_type="code_completion")
        
        answer = self.synthesizer._synthesize_code_answer(
            query, "", self.ranked_results
        )
        
        assert "code examples" in answer.lower()
        assert "def hello_world" in answer
        assert "explanation" in answer.lower()
    
    def test_synthesize_debug_answer(self):
        """Test synthesizing debugging answer."""
        self.setUp()
        
        query = "python function not working"
        context = QueryContext(task_type="debugging")
        
        answer = self.synthesizer._synthesize_debug_answer(
            query, "", self.ranked_results
        )
        
        assert "solutions" in answer.lower()
        assert "sources" in answer.lower()
        assert len(answer) > 50
    
    def test_synthesize_doc_answer(self):
        """Test synthesizing documentation answer."""
        self.setUp()
        
        query = "python function documentation"
        context = QueryContext(task_type="documentation")
        
        answer = self.synthesizer._synthesize_doc_answer(
            query, "", self.ranked_results
        )
        
        assert "Python Documentation" in answer
        assert len(answer) > 50
    
    def test_combine_chunk_content(self):
        """Test combining content from multiple chunks."""
        self.setUp()
        
        combined = self.synthesizer._combine_chunk_content(self.ranked_results)
        
        assert "[Source: Python Documentation]" in combined
        assert "def hello_world" in combined
        assert "Functions in Python" in combined


class TestRAGQualityAssessment:
    """Test RAGQualityAssessment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = RAGQualityAssessment()
        
        # Create test response
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document",
            updated_at=datetime.now() - timedelta(days=10)
        )
        
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content for quality assessment",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )
        
        ranked_result = RankedResult(
            chunk=chunk,
            original_score=0.8,
            rerank_score=0.8,
            final_score=0.8
        )
        
        self.test_response = RAGResponse(
            query="test query",
            expanded_query=Mock(),
            retrieved_chunks=[ranked_result],
            synthesized_answer="This is a test answer with sufficient content for quality assessment.",
            confidence_score=0.8,
            sources=["https://example.com/docs"]
        )
    
    def test_assess_relevance(self):
        """Test relevance assessment."""
        self.setUp()
        
        relevance = self.assessor._assess_relevance("test query", self.test_response)
        
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # Should be reasonably relevant
    
    def test_assess_completeness(self):
        """Test completeness assessment."""
        self.setUp()
        
        context = QueryContext(task_type="documentation")
        completeness = self.assessor._assess_completeness(
            "test query", self.test_response, context
        )
        
        assert 0.0 <= completeness <= 1.0
        assert completeness > 0.3  # Should have some completeness
    
    def test_assess_accuracy(self):
        """Test accuracy assessment."""
        self.setUp()
        
        accuracy = self.assessor._assess_accuracy(self.test_response)
        
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.5  # Documentation source should have good accuracy
    
    def test_assess_clarity(self):
        """Test clarity assessment."""
        self.setUp()
        
        clarity = self.assessor._assess_clarity(self.test_response)
        
        assert 0.0 <= clarity <= 1.0
        # Should have reasonable clarity for a proper answer
        assert clarity > 0.2
    
    def test_assess_timeliness(self):
        """Test timeliness assessment."""
        self.setUp()
        
        timeliness = self.assessor._assess_timeliness(self.test_response)
        
        assert 0.0 <= timeliness <= 1.0
        # Recent content should have good timeliness
        assert timeliness > 0.8
    
    def test_assess_quality_overall(self):
        """Test overall quality assessment."""
        self.setUp()
        
        context = QueryContext(task_type="documentation")
        metrics = self.assessor.assess_quality("test query", self.test_response, context)
        
        required_metrics = ['relevance', 'completeness', 'accuracy', 'clarity', 'timeliness', 'overall']
        for metric in required_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
    
    def test_collect_feedback(self):
        """Test feedback collection."""
        self.setUp()
        
        feedback = {
            'rating': 4,
            'helpful': True,
            'comments': 'Good answer but could be more detailed'
        }
        
        # Should not raise an exception
        self.assessor.collect_feedback("response_123", feedback)


class TestContextAwareRAGPipeline:
    """Test ContextAwareRAGPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_kb = Mock(spec=KnowledgeBaseRetrieval)
        self.mock_embedding = MockEmbeddingModel()
        self.pipeline = ContextAwareRAGPipeline(self.mock_kb, self.mock_embedding)
        
        # Mock knowledge base search
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        mock_chunk = DocumentChunk(
            id="test_chunk",
            content="Test content for pipeline",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )
        mock_chunk.relevance_score = 0.8
        
        self.mock_kb.search.return_value = [mock_chunk]
    
    @pytest.mark.asyncio
    async def test_process_query(self):
        """Test processing a complete query."""
        self.setUp()
        
        query = "how to create a function"
        context = QueryContext(
            programming_language="python",
            task_type="code_completion"
        )
        
        response = await self.pipeline.process_query(query, context)
        
        assert isinstance(response, RAGResponse)
        assert response.query == query
        assert response.synthesized_answer is not None
        assert len(response.synthesized_answer) > 0
        assert 0.0 <= response.confidence_score <= 1.0
        assert response.processing_time > 0.0
        assert 'overall' in response.quality_metrics
    
    @pytest.mark.asyncio
    async def test_multi_strategy_retrieval(self):
        """Test multi-strategy retrieval."""
        self.setUp()
        
        expanded_query = Mock()
        expanded_query.original_query = "test query"
        expanded_query.final_query = "test query expanded"
        
        context = QueryContext(
            programming_language="python",
            framework="django"
        )
        
        results = await self.pipeline._multi_strategy_retrieval(
            expanded_query, context, top_k=5
        )
        
        assert isinstance(results, list)
        # Should have called search multiple times for different strategies
        assert self.mock_kb.search.call_count >= 2
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        self.setUp()
        
        # High confidence results
        high_results = [
            Mock(final_score=0.9),
            Mock(final_score=0.8),
            Mock(final_score=0.85)
        ]
        
        confidence = self.pipeline._calculate_confidence(high_results)
        assert confidence > 0.7
        
        # Low confidence results
        low_results = [
            Mock(final_score=0.3),
            Mock(final_score=0.2)
        ]
        
        confidence = self.pipeline._calculate_confidence(low_results)
        assert confidence < 0.5
        
        # Empty results
        confidence = self.pipeline._calculate_confidence([])
        assert confidence == 0.0
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        self.setUp()
        
        # Add some mock responses to history
        mock_response = Mock()
        mock_response.processing_time = 1.5
        mock_response.confidence_score = 0.8
        mock_response.quality_metrics = {'overall': 0.75}
        
        self.pipeline.response_history = [mock_response] * 10
        
        metrics = self.pipeline.get_performance_metrics()
        
        assert 'total_queries' in metrics
        assert 'avg_processing_time' in metrics
        assert 'avg_confidence' in metrics
        assert 'avg_quality' in metrics
        assert 'success_rate' in metrics
        
        assert metrics['total_queries'] == 10
        assert metrics['avg_processing_time'] == 1.5
        assert metrics['avg_confidence'] == 0.8


if __name__ == "__main__":
    # Run tests
    unittest.main()