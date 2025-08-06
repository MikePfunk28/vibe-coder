"""
Tests for InterleaveReasoningEngine

Tests the Apple-inspired interleaved reasoning patterns implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from interleaved_reasoning_engine import (
    InterleaveReasoningEngine,
    ReasoningMode,
    ReasoningStep,
    ReasoningTrace,
    IntermediateSignal
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = [
            "Initial quick response",
            "Thinking about the problem step by step",
            "Based on analysis, here's the answer",
            "Let me verify this solution",
            "Final comprehensive response"
        ]
    
    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Mock generate method."""
        self.call_count += 1
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Return different responses based on call count
        response_index = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[response_index]


class MockContextManager:
    """Mock context manager for testing."""
    
    def __init__(self):
        self.contexts = [
            Mock(id="ctx1", content="def hello(): print('hello')", window_type="code"),
            Mock(id="ctx2", content="class TestClass: pass", window_type="code"),
            Mock(id="ctx3", content="# Documentation about functions", window_type="documentation")
        ]
    
    def get_relevant_context(self, query: str, max_tokens: int = 2048):
        """Mock get_relevant_context method."""
        # Return first 2 contexts for simplicity
        return self.contexts[:2]


class TestReasoningStep:
    """Test ReasoningStep dataclass."""
    
    def test_reasoning_step_creation(self):
        """Test basic reasoning step creation."""
        step = ReasoningStep(
            step_id="test_step",
            step_type="think",
            content="This is a thinking step",
            confidence=0.8,
            timestamp=datetime.now(),
            processing_time=0.1
        )
        
        assert step.step_id == "test_step"
        assert step.step_type == "think"
        assert step.content == "This is a thinking step"
        assert step.confidence == 0.8
        assert step.processing_time == 0.1


class TestReasoningTrace:
    """Test ReasoningTrace dataclass."""
    
    def test_reasoning_trace_creation(self):
        """Test basic reasoning trace creation."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="How do I write a function?",
            mode=ReasoningMode.BALANCED
        )
        
        assert trace.trace_id == "test_trace"
        assert trace.query == "How do I write a function?"
        assert trace.mode == ReasoningMode.BALANCED
        assert len(trace.steps) == 0
        assert trace.final_answer is None


class TestIntermediateSignal:
    """Test IntermediateSignal dataclass."""
    
    def test_intermediate_signal_creation(self):
        """Test basic intermediate signal creation."""
        signal = IntermediateSignal(
            signal_type="confidence",
            value=0.7,
            description="Confidence level is good"
        )
        
        assert signal.signal_type == "confidence"
        assert signal.value == 0.7
        assert signal.description == "Confidence level is good"


class TestInterleaveReasoningEngine:
    """Test InterleaveReasoningEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context_manager = MockContextManager()
        self.mock_llm_client = MockLLMClient()
        
        self.engine = InterleaveReasoningEngine(
            context_manager=self.mock_context_manager,
            llm_client=self.mock_llm_client,
            max_reasoning_steps=5,
            ttft_target_ms=200
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.context_manager == self.mock_context_manager
        assert self.engine.llm_client == self.mock_llm_client
        assert self.engine.max_reasoning_steps == 5
        assert self.engine.ttft_target_ms == 200
        assert len(self.engine.active_traces) == 0
        assert len(self.engine.reasoning_history) == 0
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_fast_mode(self):
        """Test complete reasoning in fast mode."""
        query = "How do I create a Python function?"
        context = {"file_type": "python"}
        
        response = await self.engine.reason_and_respond(
            query=query,
            context=context,
            mode=ReasoningMode.FAST,
            stream=False
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert self.mock_llm_client.call_count >= 1
        
        # Check that trace was created and moved to history
        assert len(self.engine.reasoning_history) == 1
        trace = self.engine.reasoning_history[0]
        assert trace.query == query
        assert trace.mode == ReasoningMode.FAST
        assert trace.final_answer == response
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_balanced_mode(self):
        """Test complete reasoning in balanced mode."""
        query = "Explain object-oriented programming"
        context = {"topic": "programming"}
        
        response = await self.engine.reason_and_respond(
            query=query,
            context=context,
            mode=ReasoningMode.BALANCED,
            stream=False
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check trace
        trace = self.engine.reasoning_history[0]
        assert trace.mode == ReasoningMode.BALANCED
        assert len(trace.steps) >= 1
        assert trace.total_time > 0
    
    @pytest.mark.asyncio
    async def test_progressive_reasoning_stream(self):
        """Test progressive reasoning with streaming."""
        query = "How to implement a binary search?"
        context = {"algorithm": "search"}
        
        response_generator = await self.engine.reason_and_respond(
            query=query,
            context=context,
            mode=ReasoningMode.DEEP,
            stream=True
        )
        
        # Collect all streamed responses
        responses = []
        async for chunk in response_generator:
            responses.append(chunk)
        
        assert len(responses) > 0
        
        # Should have multiple chunks for deep reasoning
        full_response = "".join(responses)
        assert len(full_response) > 0
        
        # Check that thinking and answer steps are present
        assert "🤔" in full_response or "💭" in full_response or "💡" in full_response
    
    @pytest.mark.asyncio
    async def test_initial_response_generation(self):
        """Test initial response generation for TTFT optimization."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="What is a variable?",
            mode=ReasoningMode.BALANCED
        )
        
        start_time = time.time()
        initial_step = await self.engine._generate_initial_response(
            trace, "What is a variable?", {}
        )
        ttft = (time.time() - start_time) * 1000
        
        assert initial_step is not None
        assert initial_step.step_type == "initial"
        assert len(initial_step.content) > 0
        assert initial_step.confidence > 0
        assert ttft < 1000  # Should be reasonably fast
    
    @pytest.mark.asyncio
    async def test_think_step_generation(self):
        """Test thinking step generation."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="How to sort an array?",
            mode=ReasoningMode.DEEP
        )
        
        think_step = await self.engine._generate_think_step(
            trace, "How to sort an array?", {}, 0
        )
        
        assert think_step is not None
        assert think_step.step_type == "think"
        assert len(think_step.content) > 0
        assert think_step.confidence > 0
        assert think_step.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_answer_step_generation(self):
        """Test answer step generation."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="Explain recursion",
            mode=ReasoningMode.BALANCED
        )
        
        # Add a thinking step first
        think_step = ReasoningStep(
            step_id="think_1",
            step_type="think",
            content="Need to explain recursion with examples",
            confidence=0.7,
            timestamp=datetime.now(),
            processing_time=0.1
        )
        trace.steps.append(think_step)
        
        answer_step = await self.engine._generate_answer_step(
            trace, "Explain recursion", {}, 0
        )
        
        assert answer_step is not None
        assert answer_step.step_type == "answer"
        assert len(answer_step.content) > 0
        assert answer_step.confidence > 0
    
    @pytest.mark.asyncio
    async def test_verification_step_generation(self):
        """Test verification step generation."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="What is a loop?",
            mode=ReasoningMode.BALANCED
        )
        trace.final_answer = "A loop is a control structure that repeats code"
        
        verification_step = await self.engine._generate_verification_step(
            trace, "What is a loop?", {}
        )
        
        assert verification_step is not None
        assert verification_step.step_type == "verification"
        assert len(verification_step.content) > 0
        assert verification_step.confidence > 0
    
    @pytest.mark.asyncio
    async def test_intermediate_signals_generation(self):
        """Test intermediate signals generation."""
        trace = ReasoningTrace(
            trace_id="test_trace",
            query="Complex algorithm question",
            mode=ReasoningMode.DEEP
        )
        
        step = ReasoningStep(
            step_id="test_step",
            step_type="think",
            content="This is a complex problem that needs more analysis",
            confidence=0.6,
            timestamp=datetime.now(),
            processing_time=0.1
        )
        
        signals = await self.engine._generate_intermediate_signals(trace, step)
        
        assert len(signals) == 4  # confidence, direction, context_need, complexity
        
        signal_types = [s.signal_type for s in signals]
        assert "confidence" in signal_types
        assert "direction" in signal_types
        assert "context_need" in signal_types
        assert "complexity" in signal_types
        
        # Check signal values are in valid range
        for signal in signals:
            assert 0.0 <= signal.value <= 1.0
    
    def test_reasoning_direction_analysis(self):
        """Test reasoning direction analysis."""
        # Clear direction
        clear_content = "First we need to do this, then we can proceed to the next step"
        direction_score = self.engine._analyze_reasoning_direction(clear_content)
        assert direction_score > 0.5
        
        # Unclear direction
        unclear_content = "What should we do? How can we approach this? Why is this happening?"
        direction_score = self.engine._analyze_reasoning_direction(unclear_content)
        assert direction_score < 0.7
    
    def test_context_need_analysis(self):
        """Test context need analysis."""
        trace = ReasoningTrace("test", "query", ReasoningMode.BALANCED)
        
        # High context need
        high_need_step = ReasoningStep(
            "step1", "think", "We need more information and require additional context",
            0.5, datetime.now(), 0.1
        )
        need_score = self.engine._analyze_context_need(trace, high_need_step)
        assert need_score > 0.5
        
        # Low context need
        low_need_step = ReasoningStep(
            "step2", "think", "The given information is clear and available",
            0.8, datetime.now(), 0.1
        )
        need_score = self.engine._analyze_context_need(trace, low_need_step)
        assert need_score < 0.7
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        # High complexity
        complex_content = "This is a complex problem with multiple considerations and various trade-offs to evaluate"
        complexity_score = self.engine._analyze_complexity(complex_content)
        assert complexity_score > 0.5
        
        # Low complexity
        simple_content = "This is simple and straightforward, easy to understand"
        complexity_score = self.engine._analyze_complexity(simple_content)
        assert complexity_score < 0.7
    
    def test_confidence_calculation(self):
        """Test step confidence calculation."""
        trace = ReasoningTrace("test", "query", ReasoningMode.BALANCED)
        
        # High confidence response
        detailed_response = "Because of the clear requirements, we can implement this function using a for loop"
        confidence = self.engine._calculate_step_confidence(trace, detailed_response)
        assert confidence > 0.7
        
        # Low confidence response
        uncertain_response = "Maybe we could try this approach, but it's unclear if it will work"
        confidence = self.engine._calculate_step_confidence(trace, uncertain_response)
        assert confidence < 0.7
    
    def test_reasoning_prompt_building(self):
        """Test reasoning prompt building for different modes."""
        query = "How to implement a hash table?"
        contexts = [Mock(content="Hash table implementation details")]
        
        # Fast mode
        fast_prompt = self.engine._build_reasoning_prompt(query, contexts, ReasoningMode.FAST)
        assert "quick" in fast_prompt.lower()
        assert len(fast_prompt) < 1000
        
        # Deep mode
        deep_prompt = self.engine._build_reasoning_prompt(query, contexts, ReasoningMode.DEEP)
        assert "analyze" in deep_prompt.lower()
        assert "thoroughly" in deep_prompt.lower()
        assert len(deep_prompt) > len(fast_prompt)
    
    @pytest.mark.asyncio
    async def test_signal_processing(self):
        """Test intermediate signal processing."""
        trace = ReasoningTrace("test", "query", ReasoningMode.BALANCED)
        
        # Test confidence signal processing
        confidence_signal = IntermediateSignal("confidence", 0.3, "Low confidence")
        await self.engine._process_signal(confidence_signal, trace)
        
        # Test direction signal processing
        direction_signal = IntermediateSignal("direction", 0.2, "Unclear direction")
        await self.engine._process_signal(direction_signal, trace)
        
        # Test context need signal processing
        context_signal = IntermediateSignal("context_need", 0.8, "High context need")
        await self.engine._process_signal(context_signal, trace)
        
        # Test complexity signal processing
        complexity_signal = IntermediateSignal("complexity", 0.9, "High complexity")
        await self.engine._process_signal(complexity_signal, trace)
        
        # Should not raise any exceptions
        assert True
    
    def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
        # Create a trace with some steps
        trace = ReasoningTrace("test", "query", ReasoningMode.BALANCED)
        trace.total_time = 1.5
        trace.confidence_score = 0.8
        
        step = ReasoningStep(
            "step1", "answer", "Response", 0.8, datetime.now(), 0.2
        )
        trace.steps.append(step)
        
        initial_queries = self.engine.performance_stats['total_queries']
        
        # Update stats
        self.engine._update_performance_stats(trace)
        
        # Check stats were updated
        assert self.engine.performance_stats['total_queries'] == initial_queries + 1
        assert self.engine.performance_stats['mode_usage'][ReasoningMode.BALANCED.value] == 1
        assert len(self.engine.performance_stats['confidence_scores']) == 1
        assert self.engine.performance_stats['avg_reasoning_time'] > 0
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Add some confidence scores
        self.engine.performance_stats['confidence_scores'] = [0.7, 0.8, 0.9]
        
        stats = self.engine.get_performance_stats()
        
        assert 'total_queries' in stats
        assert 'avg_ttft' in stats
        assert 'avg_reasoning_time' in stats
        assert 'avg_confidence' in stats
        assert 'confidence_std' in stats
        assert 'mode_usage' in stats
        
        assert abs(stats['avg_confidence'] - 0.8) < 0.001  # Mean of [0.7, 0.8, 0.9]
    
    def test_reasoning_trace_retrieval(self):
        """Test reasoning trace retrieval."""
        # Create and add a trace to history
        trace = ReasoningTrace("test_trace_123", "query", ReasoningMode.FAST)
        self.engine.reasoning_history.append(trace)
        
        # Test retrieval
        retrieved = self.engine.get_reasoning_trace("test_trace_123")
        assert retrieved is not None
        assert retrieved.trace_id == "test_trace_123"
        
        # Test non-existent trace
        not_found = self.engine.get_reasoning_trace("non_existent")
        assert not_found is None
    
    def test_history_clearing(self):
        """Test reasoning history clearing."""
        # Add multiple traces
        for i in range(15):
            trace = ReasoningTrace(f"trace_{i}", "query", ReasoningMode.FAST)
            self.engine.reasoning_history.append(trace)
        
        assert len(self.engine.reasoning_history) == 15
        
        # Clear keeping recent 5
        self.engine.clear_history(keep_recent=5)
        assert len(self.engine.reasoning_history) == 5
        
        # Check that the most recent ones were kept
        assert self.engine.reasoning_history[0].trace_id == "trace_10"
        assert self.engine.reasoning_history[-1].trace_id == "trace_14"
        
        # Clear all
        self.engine.clear_history(keep_recent=0)
        assert len(self.engine.reasoning_history) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_reasoning(self):
        """Test handling multiple concurrent reasoning requests."""
        queries = [
            "What is a function?",
            "How to use loops?",
            "Explain classes"
        ]
        
        # Start multiple reasoning tasks concurrently
        tasks = []
        for query in queries:
            task = asyncio.create_task(
                self.engine.reason_and_respond(
                    query=query,
                    context={},
                    mode=ReasoningMode.FAST,
                    stream=False
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Should have 3 traces in history
        assert len(self.engine.reasoning_history) == 3
    
    @pytest.mark.asyncio
    async def test_reasoning_mode_differences(self):
        """Test that different reasoning modes produce different behaviors."""
        query = "Implement a sorting algorithm"
        context = {}
        
        # Test all modes
        responses = {}
        for mode in ReasoningMode:
            response = await self.engine.reason_and_respond(
                query=query,
                context=context,
                mode=mode,
                stream=False
            )
            responses[mode] = response
        
        # All should produce responses
        for mode, response in responses.items():
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Check that traces have different characteristics
        traces = self.engine.reasoning_history[-4:]  # Last 4 traces
        
        for trace in traces:
            assert trace.mode in ReasoningMode
            assert trace.total_time > 0


if __name__ == "__main__":
    pytest.main([__file__])