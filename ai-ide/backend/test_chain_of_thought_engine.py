"""
Tests for Chain-of-Thought Reasoning Engine

Tests the step-by-step problem decomposition, reasoning trace generation,
quality assessment, and debugging capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from chain_of_thought_engine import (
    ChainOfThoughtEngine,
    CoTStep,
    CoTTrace,
    CoTStepType,
    CoTComplexity
)


class TestChainOfThoughtEngine:
    """Test suite for ChainOfThoughtEngine."""
    
    @pytest.fixture
    def mock_context_manager(self):
        """Mock context manager."""
        mock = Mock()
        mock.get_relevant_context.return_value = [
            Mock(id="ctx1", content="Sample context 1"),
            Mock(id="ctx2", content="Sample context 2")
        ]
        return mock
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        mock = Mock()
        mock.generate = AsyncMock()
        return mock
    
    @pytest.fixture
    def cot_engine(self, mock_context_manager, mock_llm_client):
        """Create CoT engine instance."""
        return ChainOfThoughtEngine(
            context_manager=mock_context_manager,
            llm_client=mock_llm_client,
            max_steps=10,
            enable_visualization=True,
            quality_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_basic_reasoning_flow(self, cot_engine, mock_llm_client):
        """Test basic reasoning flow through problem."""
        # Setup mock responses
        mock_responses = [
            "This is a coding problem that requires implementing a function.",
            "We need to break this into: 1) Input validation 2) Core logic 3) Output formatting",
            "We'll use a simple algorithm approach with error handling.",
            "def solve_problem(input_data): return processed_result",
            "The solution handles all edge cases and returns correct output."
        ]
        
        mock_llm_client.generate.side_effect = mock_responses
        
        # Execute reasoning
        trace = await cot_engine.reason_through_problem(
            problem="Implement a function to process user input",
            problem_type="coding_problem",
            complexity=CoTComplexity.MODERATE
        )
        
        # Verify trace structure
        assert trace.problem_statement == "Implement a function to process user input"
        assert trace.complexity == CoTComplexity.MODERATE
        assert len(trace.steps) == 5
        assert trace.final_solution is not None
        assert trace.confidence_score > 0
        assert trace.quality_score > 0
    
    @pytest.mark.asyncio
    async def test_step_generation(self, cot_engine, mock_llm_client):
        """Test individual step generation."""
        mock_llm_client.generate.return_value = "Detailed analysis of the problem requirements."
        
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.SIMPLE
        )
        
        step = await cot_engine._generate_reasoning_step(
            trace=trace,
            step_type=CoTStepType.PROBLEM_ANALYSIS,
            step_number=1,
            context={}
        )
        
        assert step is not None
        assert step.step_type == CoTStepType.PROBLEM_ANALYSIS
        assert step.step_number == 1
        assert step.content == "Detailed analysis of the problem requirements."
        assert 0 < step.confidence <= 1
    
    def test_step_prompt_building(self, cot_engine):
        """Test step-specific prompt building."""
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.MODERATE
        )
        
        # Add a previous step
        trace.steps.append(CoTStep(
            step_id="step1",
            step_number=1,
            step_type=CoTStepType.PROBLEM_ANALYSIS,
            title="Analysis",
            content="Problem analysis content",
            reasoning="Analysis reasoning",
            confidence=0.8
        ))
        
        prompt = cot_engine._build_step_prompt(
            trace=trace,
            step_type=CoTStepType.DECOMPOSITION,
            step_number=2,
            context={"key": "value"}
        )
        
        assert "Test problem" in prompt
        assert "Problem analysis content" in prompt
        assert "Decomposition" in prompt
        assert "Step 2" in prompt
    
    def test_response_parsing(self, cot_engine):
        """Test parsing of LLM responses."""
        response = """
        This is the main content of the step.
        It contains multiple lines.
        
        Reasoning: This is because we need to handle the complexity.
        The rationale is based on best practices.
        """
        
        content, reasoning = cot_engine._parse_step_response(response, CoTStepType.PROBLEM_ANALYSIS)
        
        assert "main content" in content
        assert "multiple lines" in content
        assert "because we need" in reasoning
        assert "best practices" in reasoning
    
    def test_confidence_calculation(self, cot_engine):
        """Test step confidence calculation."""
        # High confidence content
        high_conf_content = "def solve_problem(data): return process_data(data) because this approach is optimal"
        confidence = cot_engine._calculate_step_confidence(high_conf_content, CoTStepType.IMPLEMENTATION)
        assert confidence > 0.8
        
        # Low confidence content
        low_conf_content = "Maybe we could try something, but I'm not sure what would work"
        confidence = cot_engine._calculate_step_confidence(low_conf_content, CoTStepType.PROBLEM_ANALYSIS)
        assert confidence < 0.6
    
    def test_logical_flow_assessment(self, cot_engine):
        """Test logical flow assessment between steps."""
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.MODERATE
        )
        
        # Add connected steps
        trace.steps.extend([
            CoTStep(
                step_id="step1",
                step_number=1,
                step_type=CoTStepType.PROBLEM_ANALYSIS,
                title="Analysis",
                content="We need to solve X",
                reasoning="",
                confidence=0.8
            ),
            CoTStep(
                step_id="step2",
                step_number=2,
                step_type=CoTStepType.DECOMPOSITION,
                title="Decomposition",
                content="Based on the analysis above, we break X into parts",
                reasoning="",
                confidence=0.8
            )
        ])
        
        flow_score = cot_engine._assess_logical_flow(trace)
        assert flow_score > 0.5
    
    def test_completeness_assessment(self, cot_engine):
        """Test completeness assessment."""
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.COMPLEX
        )
        
        # Add comprehensive steps
        step_types = [
            CoTStepType.PROBLEM_ANALYSIS,
            CoTStepType.DECOMPOSITION,
            CoTStepType.SOLUTION_PLANNING,
            CoTStepType.IMPLEMENTATION,
            CoTStepType.VERIFICATION
        ]
        
        for i, step_type in enumerate(step_types):
            trace.steps.append(CoTStep(
                step_id=f"step{i+1}",
                step_number=i+1,
                step_type=step_type,
                title=f"Step {i+1}",
                content=f"Content for {step_type.value}",
                reasoning="",
                confidence=0.8
            ))
        
        completeness = cot_engine._assess_completeness(trace)
        assert completeness > 0.8
    
    def test_quality_assessment(self, cot_engine):
        """Test overall quality assessment."""
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.MODERATE
        )
        
        # Add high-quality steps
        trace.steps.extend([
            CoTStep(
                step_id="step1",
                step_number=1,
                step_type=CoTStepType.PROBLEM_ANALYSIS,
                title="Analysis",
                content="1. Identify requirements 2. Analyze constraints because we need clarity",
                reasoning="Clear analysis",
                confidence=0.9
            ),
            CoTStep(
                step_id="step2",
                step_number=2,
                step_type=CoTStepType.IMPLEMENTATION,
                title="Implementation",
                content="def solve(): return result",
                reasoning="Simple implementation",
                confidence=0.8
            )
        ])
        
        quality_score = asyncio.run(cot_engine._assess_reasoning_quality(trace))
        assert quality_score > 0.5
    
    def test_visualization_generation(self, cot_engine):
        """Test reasoning trace visualization."""
        trace = CoTTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            complexity=CoTComplexity.SIMPLE,
            confidence_score=0.85,
            quality_score=0.78,
            total_time=2.5
        )
        
        trace.steps.append(CoTStep(
            step_id="step1",
            step_number=1,
            step_type=CoTStepType.PROBLEM_ANALYSIS,
            title="Step 1: Analyze Problem",
            content="This is the analysis content",
            reasoning="",
            confidence=0.8,
            processing_time=1.2
        ))
        
        visualization = cot_engine.visualize_reasoning_trace(trace)
        
        assert "Chain-of-Thought Reasoning Trace" in visualization
        assert "Test problem" in visualization
        assert "Confidence: 0.85" in visualization
        assert "Quality: 0.78" in visualization
        assert "🔍 Step 1: Analyze Problem" in visualization
    
    def test_debugging_info(self, cot_engine):
        """Test debugging information generation."""
        trace = CoTTrace(
            trace_id="debug_trace",
            problem_statement="Debug test",
            complexity=CoTComplexity.MODERATE,
            confidence_score=0.75,
            quality_score=0.68,
            total_time=3.2
        )
        
        trace.steps.append(CoTStep(
            step_id="step1",
            step_number=1,
            step_type=CoTStepType.PROBLEM_ANALYSIS,
            title="Analysis",
            content="Analysis content",
            reasoning="Analysis reasoning",
            confidence=0.8,
            processing_time=1.5
        ))
        
        # Add to completed traces
        cot_engine.completed_traces.append(trace)
        
        debug_info = cot_engine.get_debugging_info("debug_trace")
        
        assert debug_info["trace_id"] == "debug_trace"
        assert debug_info["problem"] == "Debug test"
        assert debug_info["complexity"] == "moderate"
        assert debug_info["total_steps"] == 1
        assert debug_info["confidence_score"] == 0.75
        assert len(debug_info["steps"]) == 1
        assert debug_info["steps"][0]["step_type"] == "problem_analysis"
    
    def test_quality_validation(self, cot_engine):
        """Test reasoning quality validation."""
        # Create a trace with issues
        trace = CoTTrace(
            trace_id="validation_trace",
            problem_statement="Validation test",
            complexity=CoTComplexity.COMPLEX,
            quality_score=0.5
        )
        
        # Add only one low-confidence step (should trigger issues)
        trace.steps.append(CoTStep(
            step_id="step1",
            step_number=1,
            step_type=CoTStepType.SOLUTION_PLANNING,  # Missing problem analysis
            title="Planning",
            content="Maybe we could try something",
            reasoning="",
            confidence=0.3  # Low confidence
        ))
        
        validation = cot_engine.validate_reasoning_quality(trace)
        
        assert not validation["meets_threshold"]
        assert len(validation["issues"]) > 0
        assert len(validation["suggestions"]) > 0
        assert "Too few reasoning steps" in validation["issues"]
        assert "Missing problem analysis step" in validation["issues"]
    
    def test_performance_stats(self, cot_engine):
        """Test performance statistics generation."""
        # Add some completed traces
        for i in range(3):
            trace = CoTTrace(
                trace_id=f"trace_{i}",
                problem_statement=f"Problem {i}",
                complexity=CoTComplexity.MODERATE,
                confidence_score=0.8 + i * 0.05,
                quality_score=0.7 + i * 0.1,
                total_time=2.0 + i
            )
            
            trace.steps.append(CoTStep(
                step_id=f"step_{i}",
                step_number=1,
                step_type=CoTStepType.PROBLEM_ANALYSIS,
                title="Analysis",
                content="Content",
                reasoning="",
                confidence=0.8
            ))
            
            cot_engine.completed_traces.append(trace)
        
        stats = cot_engine.get_performance_stats()
        
        assert stats["total_traces"] == 3
        assert stats["avg_steps"] == 1.0
        assert stats["avg_confidence"] > 0.8
        assert stats["avg_quality"] > 0.7
        assert "complexity_distribution" in stats
        assert "step_type_usage" in stats
    
    def test_additional_steps_generation(self, cot_engine):
        """Test generation of additional steps when needed."""
        trace = CoTTrace(
            trace_id="additional_test",
            problem_statement="Complex problem",
            complexity=CoTComplexity.COMPLEX
        )
        
        # Create a low-confidence decomposition step
        low_conf_step = CoTStep(
            step_id="step1",
            step_number=1,
            step_type=CoTStepType.DECOMPOSITION,
            title="Decomposition",
            content="This is complex and has multiple parts",
            reasoning="",
            confidence=0.5  # Low confidence should trigger additional steps
        )
        
        should_add = asyncio.run(cot_engine._should_add_additional_steps(trace, low_conf_step))
        assert should_add
    
    def test_context_token_limits(self, cot_engine):
        """Test context token limits for different complexities."""
        simple_tokens = cot_engine._get_context_tokens_for_complexity(CoTComplexity.SIMPLE)
        expert_tokens = cot_engine._get_context_tokens_for_complexity(CoTComplexity.EXPERT)
        
        assert simple_tokens < expert_tokens
        assert simple_tokens == 1024
        assert expert_tokens == 6144
    
    def test_step_token_limits(self, cot_engine):
        """Test token limits for different step types."""
        analysis_tokens = cot_engine._get_max_tokens_for_step(CoTStepType.PROBLEM_ANALYSIS)
        implementation_tokens = cot_engine._get_max_tokens_for_step(CoTStepType.IMPLEMENTATION)
        
        assert implementation_tokens > analysis_tokens
        assert analysis_tokens == 400
        assert implementation_tokens == 800


if __name__ == "__main__":
    pytest.main([__file__])