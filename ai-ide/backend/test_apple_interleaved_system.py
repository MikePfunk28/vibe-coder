"""
Tests for AppleInterleaveSystem

Tests the integrated Apple-inspired interleaved context and reasoning system.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from apple_interleaved_system import (
    AppleInterleaveSystem,
    AppleInterleaveConfig
)
from interleaved_reasoning_engine import ReasoningMode


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = [
            "Here's how to implement a function in Python",
            "Let me analyze this code structure",
            "Based on the context, I recommend this approach",
            "The implementation should follow these steps"
        ]
    
    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Mock generate method."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate processing time
        
        response_index = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[response_index]


class TestAppleInterleaveConfig:
    """Test AppleInterleaveConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AppleInterleaveConfig()
        
        assert config.max_context_length == 32768
        assert config.similarity_threshold == 0.7
        assert config.max_windows == 50
        assert config.compression_enabled is True
        assert config.max_reasoning_steps == 10
        assert config.ttft_target_ms == 200
        assert config.enable_progressive is True
        assert config.auto_context_refresh is True
        assert config.context_relevance_feedback is True
        assert config.adaptive_reasoning_mode is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AppleInterleaveConfig(
            max_context_length=16384,
            similarity_threshold=0.8,
            max_reasoning_steps=5,
            ttft_target_ms=100
        )
        
        assert config.max_context_length == 16384
        assert config.similarity_threshold == 0.8
        assert config.max_reasoning_steps == 5
        assert config.ttft_target_ms == 100


class TestAppleInterleaveSystem:
    """Test AppleInterleaveSystem functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_client = MockLLMClient()
        self.config = AppleInterleaveConfig(
            max_context_length=10000,
            max_windows=10,
            max_reasoning_steps=3
        )
        
        self.system = AppleInterleaveSystem(
            llm_client=self.mock_llm_client,
            config=self.config
        )
    
    def test_system_initialization(self):
        """Test system initialization."""
        assert self.system.config == self.config
        assert self.system.llm_client == self.mock_llm_client
        assert self.system.context_manager is not None
        assert self.system.reasoning_engine is not None
        assert len(self.system.active_sessions) == 0
        assert 'total_requests' in self.system.performance_metrics
    
    @pytest.mark.asyncio
    async def test_add_code_context(self):
        """Test adding code context to the system."""
        content = "def hello_world(): print('Hello, World!')"
        
        context_id = await self.system.add_code_context(
            content=content,
            context_type="code",
            source_file="test.py",
            line_range=(1, 1),
            priority=2
        )
        
        assert context_id is not None
        assert len(context_id) > 0
        
        # Check that context was added to context manager
        assert context_id in self.system.context_manager.context_cache
        window = self.system.context_manager.context_cache[context_id]
        assert window.content == content
        assert window.window_type == "code"
        assert window.source_file == "test.py"
        assert window.priority == 2
    
    @pytest.mark.asyncio
    async def test_search_semantic_context(self):
        """Test semantic context search."""
        # Add some contexts first
        await self.system.add_code_context("def calculate_sum(a, b): return a + b", "code")
        await self.system.add_code_context("class Calculator: pass", "code")
        await self.system.add_code_context("# Documentation about math functions", "documentation")
        
        # Search for relevant context
        results = await self.system.search_semantic_context(
            query="sum calculation",
            max_results=5
        )
        
        assert len(results) > 0
        assert all('id' in result for result in results)
        assert all('content' in result for result in results)
        assert all('relevance_score' in result for result in results)
        assert all('context_type' in result for result in results)
        
        # Should find the sum function
        sum_found = any("calculate_sum" in result['content'] for result in results)
        assert sum_found
    
    @pytest.mark.asyncio
    async def test_search_semantic_context_with_type_filter(self):
        """Test semantic context search with type filtering."""
        await self.system.add_code_context("def test_function(): pass", "code")
        await self.system.add_code_context("# This is documentation", "documentation")
        
        # Search only for code contexts
        results = await self.system.search_semantic_context(
            query="function",
            context_types=["code"],
            max_results=5
        )
        
        # Should only return code contexts
        for result in results:
            assert result['context_type'] == "code"
    
    def test_update_context_relevance(self):
        """Test updating context relevance."""
        # Add a context first
        context_id = self.system.context_manager.add_context(
            "test content", "code"
        )
        
        original_relevance = self.system.context_manager.context_cache[context_id].relevance_score
        
        # Update relevance
        self.system.update_context_relevance(context_id, 0.2, "user")
        
        new_relevance = self.system.context_manager.context_cache[context_id].relevance_score
        assert new_relevance > original_relevance
    
    @pytest.mark.asyncio
    async def test_process_code_assistance_request_simple(self):
        """Test processing a simple code assistance request."""
        query = "How do I create a Python function?"
        code_context = {
            'current_file': {
                'content': 'print("hello")',
                'path': 'test.py'
            }
        }
        
        response = await self.system.process_code_assistance_request(
            query=query,
            code_context=code_context,
            stream_response=False
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert self.system.performance_metrics['total_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_process_code_assistance_request_with_session(self):
        """Test processing request with session tracking."""
        query = "Explain this code"
        code_context = {'current_file': {'content': 'x = 5', 'path': 'test.py'}}
        session_id = "test_session_123"
        
        response = await self.system.process_code_assistance_request(
            query=query,
            code_context=code_context,
            session_id=session_id,
            reasoning_mode=ReasoningMode.FAST,
            stream_response=False
        )
        
        assert isinstance(response, str)
        assert session_id in self.system.active_sessions
        
        session = self.system.active_sessions[session_id]
        assert query in session['queries']
        assert ReasoningMode.FAST.value in session['reasoning_modes']
    
    @pytest.mark.asyncio
    async def test_process_code_assistance_request_streaming(self):
        """Test processing request with streaming response."""
        query = "Implement a sorting algorithm"
        code_context = {}
        
        response_generator = await self.system.process_code_assistance_request(
            query=query,
            code_context=code_context,
            reasoning_mode=ReasoningMode.PROGRESSIVE,
            stream_response=True
        )
        
        # Collect streamed responses
        responses = []
        async for chunk in response_generator:
            responses.append(chunk)
        
        assert len(responses) > 0
        full_response = "".join(responses)
        assert len(full_response) > 0
    
    def test_determine_optimal_reasoning_mode(self):
        """Test automatic reasoning mode determination."""
        # Fast mode for simple queries
        fast_query = "What is a variable?"
        mode = self.system._determine_optimal_reasoning_mode(fast_query, {})
        assert mode == ReasoningMode.FAST
        
        # Deep mode for complex analysis
        deep_query = "Analyze the performance of this algorithm"
        mode = self.system._determine_optimal_reasoning_mode(deep_query, {})
        assert mode == ReasoningMode.DEEP
        
        # Progressive mode for implementation tasks
        progressive_query = "Implement a binary search tree step by step"
        mode = self.system._determine_optimal_reasoning_mode(progressive_query, {})
        assert mode == ReasoningMode.PROGRESSIVE
        
        # Balanced mode for general queries
        balanced_query = "How can I improve this code?"
        mode = self.system._determine_optimal_reasoning_mode(balanced_query, {})
        assert mode == ReasoningMode.BALANCED
    
    def test_optimize_system_performance(self):
        """Test system performance optimization."""
        # Add some contexts and process some requests
        self.system.context_manager.add_context("test content 1", "code")
        self.system.context_manager.add_context("test content 2", "code")
        self.system.performance_metrics['total_requests'] = 5
        
        results = self.system.optimize_system_performance()
        
        assert 'context_optimization' in results
        assert 'reasoning_optimization' in results
        assert 'system_metrics' in results
        
        context_opt = results['context_optimization']
        assert 'windows_optimized' in context_opt
        assert 'memory_utilization' in context_opt
        assert 'cache_hit_rate' in context_opt
        
        reasoning_opt = results['reasoning_optimization']
        assert 'total_queries' in reasoning_opt
        assert 'avg_ttft' in reasoning_opt
        assert 'mode_usage' in reasoning_opt
    
    def test_get_system_status(self):
        """Test getting system status."""
        # Add some context and simulate usage
        self.system.context_manager.add_context("test content", "code")
        self.system.performance_metrics['total_requests'] = 3
        
        status = self.system.get_system_status()
        
        assert status['status'] == 'healthy'
        assert 'context_manager' in status
        assert 'reasoning_engine' in status
        assert 'system_performance' in status
        assert 'configuration' in status
        
        # Check context manager status
        cm_status = status['context_manager']
        assert 'total_windows' in cm_status
        assert 'memory_usage' in cm_status
        assert 'cache_performance' in cm_status
        assert 'active_slots' in cm_status
        
        # Check reasoning engine status
        re_status = status['reasoning_engine']
        assert 'total_queries' in re_status
        assert 'performance' in re_status
        assert 'mode_distribution' in re_status
        
        # Check configuration
        config_status = status['configuration']
        assert 'max_context_length' in config_status
        assert 'ttft_target' in config_status
        assert 'adaptive_reasoning' in config_status
    
    def test_clear_context(self):
        """Test clearing context."""
        # Add some contexts
        self.system.context_manager.add_context("code content", "code")
        self.system.context_manager.add_context("doc content", "documentation")
        
        assert len(self.system.context_manager.context_cache) == 2
        
        # Clear only code contexts
        self.system.clear_context(context_types=["code"])
        
        # Should have only documentation left
        remaining_contexts = list(self.system.context_manager.context_cache.values())
        assert len(remaining_contexts) == 1
        assert remaining_contexts[0].window_type == "documentation"
        
        # Clear all
        self.system.clear_context()
        assert len(self.system.context_manager.context_cache) == 0
    
    def test_export_context_data(self):
        """Test exporting context data."""
        # Add some context and simulate usage
        self.system.context_manager.add_context("test content", "code")
        self.system.performance_metrics['total_requests'] = 2
        
        export_data = self.system.export_context_data()
        
        assert 'timestamp' in export_data
        assert 'statistics' in export_data
        assert 'configuration' in export_data
        assert 'performance_metrics' in export_data
        
        # Check configuration export
        config = export_data['configuration']
        assert config['max_context_length'] == self.config.max_context_length
        assert config['similarity_threshold'] == self.config.similarity_threshold
        assert config['max_windows'] == self.config.max_windows
        
        # Check performance metrics
        assert export_data['performance_metrics']['total_requests'] == 2
    
    @pytest.mark.asyncio
    async def test_update_context_from_code(self):
        """Test updating context from code context."""
        code_context = {
            'current_file': {
                'content': 'def main(): print("hello")',
                'path': 'main.py'
            },
            'selected_text': 'print("hello")',
            'project_files': [
                {'content': 'import os', 'path': 'utils.py'},
                {'content': 'class Config: pass', 'path': 'config.py'}
            ]
        }
        
        initial_count = len(self.system.context_manager.context_cache)
        
        await self.system._update_context_from_code(code_context)
        
        # Should have added contexts for current file, selected text, and project files
        final_count = len(self.system.context_manager.context_cache)
        assert final_count > initial_count
        
        # Check that different priority levels were assigned
        contexts = list(self.system.context_manager.context_cache.values())
        priorities = [ctx.priority for ctx in contexts]
        
        # Should have high priority for selected text and current file
        assert 4 in priorities  # Selected text
        assert 3 in priorities  # Current file
        assert 1 in priorities  # Project files
    
    @pytest.mark.asyncio
    async def test_update_context_relevance_feedback(self):
        """Test updating context relevance based on feedback."""
        # Add some contexts
        context_id1 = self.system.context_manager.add_context("function definition", "code")
        context_id2 = self.system.context_manager.add_context("unrelated content", "code")
        
        query = "how to define a function"
        response = "You can define a function using the def keyword"
        
        original_relevance1 = self.system.context_manager.context_cache[context_id1].relevance_score
        original_relevance2 = self.system.context_manager.context_cache[context_id2].relevance_score
        
        await self.system._update_context_relevance_feedback(query, response)
        
        new_relevance1 = self.system.context_manager.context_cache[context_id1].relevance_score
        new_relevance2 = self.system.context_manager.context_cache[context_id2].relevance_score
        
        # Context with relevant content should have increased relevance
        assert new_relevance1 >= original_relevance1
        # Unrelated context should remain the same or increase less
        assert new_relevance2 >= original_relevance2
    
    @pytest.mark.asyncio
    async def test_session_state_management(self):
        """Test session state management."""
        session_id = "test_session"
        query = "How to implement a function?"
        response = "Use the def keyword"
        reasoning_mode = ReasoningMode.BALANCED
        
        # Initially no sessions
        assert session_id not in self.system.active_sessions
        
        await self.system._update_session_state(session_id, query, response, reasoning_mode)
        
        # Session should be created
        assert session_id in self.system.active_sessions
        session = self.system.active_sessions[session_id]
        
        assert query in session['queries']
        assert reasoning_mode.value in session['reasoning_modes']
        assert 'context_usage' in session
        assert 'created_at' in session
        
        # Add more queries to test limit
        for i in range(12):
            await self.system._update_session_state(
                session_id, f"query {i}", "response", ReasoningMode.FAST
            )
        
        # Should keep only last 10 queries
        assert len(session['queries']) == 10
        assert len(session['reasoning_modes']) == 10
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        queries = [
            "What is a function?",
            "How to use loops?",
            "Explain classes",
            "What are variables?"
        ]
        
        # Process multiple requests concurrently
        tasks = []
        for i, query in enumerate(queries):
            task = asyncio.create_task(
                self.system.process_code_assistance_request(
                    query=query,
                    code_context={'session': f'session_{i}'},
                    stream_response=False
                )
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 4
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Should have processed all requests
        assert self.system.performance_metrics['total_requests'] == 4
    
    @pytest.mark.asyncio
    async def test_system_with_semantic_embeddings(self):
        """Test system functionality with semantic embeddings."""
        # Add context with semantic embeddings
        embedding1 = np.random.rand(384)
        embedding2 = np.random.rand(384)
        
        context_id1 = await self.system.add_code_context(
            content="def calculate_sum(a, b): return a + b",
            context_type="code",
            semantic_embedding=embedding1
        )
        
        context_id2 = await self.system.add_code_context(
            content="def calculate_product(a, b): return a * b",
            context_type="code",
            semantic_embedding=embedding2
        )
        
        # Search should work with embeddings
        results = await self.system.search_semantic_context("calculation functions")
        
        assert len(results) == 2
        assert all('relevance_score' in result for result in results)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        initial_requests = self.system.performance_metrics['total_requests']
        initial_hits = self.system.performance_metrics['context_hits']
        
        # Simulate some operations
        self.system.performance_metrics['total_requests'] += 3
        
        # Add context and search (should increase context hits)
        self.system.context_manager.add_context("test", "code")
        
        # The search_semantic_context method should update context_hits
        # We'll test this indirectly through the system status
        status = self.system.get_system_status()
        assert status['system_performance']['total_requests'] == initial_requests + 3


if __name__ == "__main__":
    pytest.main([__file__])