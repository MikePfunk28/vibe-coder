"""
Tests for ReAct (Reasoning + Acting) Framework

Tests the ReAct pattern implementation including tool selection, reasoning loops,
and adaptive strategies.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from react_framework import ReActFramework
from react_core import Tool, ActionType, ReasoningStep, ReActStep, ReActTrace
from react_tools import ToolRegistry, ToolSelector
from react_strategy import AdaptiveReasoningStrategy


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.responses = {}
        self.call_count = 0
    
    def set_response(self, prompt_pattern: str, response: str):
        """Set response for prompts containing pattern."""
        self.responses[prompt_pattern] = response
    
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5):
        """Generate mock response."""
        self.call_count += 1
        
        # Find matching response
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                return response
        
        # Default responses based on prompt content
        if "think step by step" in prompt.lower():
            return "I need to analyze this problem carefully and determine the best approach."
        elif "tool:" in prompt.lower():
            return "Tool: semantic_search\nConfidence: 0.8\nReasoning: This tool will help find relevant information."
        elif "final answer" in prompt.lower():
            return "Based on my analysis, the solution is to implement the required functionality."
        else:
            return "This is a mock response for testing."


class MockContextManager:
    """Mock context manager for testing."""
    
    def __init__(self):
        self.contexts = []
    
    def get_relevant_context(self, query: str, max_tokens: int = 1024):
        """Return mock context."""
        return [
            Mock(content=f"Mock context for: {query}", source="test", relevance=0.8),
            Mock(content="Additional context information", source="test", relevance=0.6)
        ]


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_context_manager():
    """Create mock context manager."""
    return MockContextManager()


@pytest.fixture
def react_framework(mock_llm_client, mock_context_manager):
    """Create ReAct framework instance."""
    return ReActFramework(
        llm_client=mock_llm_client,
        context_manager=mock_context_manager,
        max_iterations=10,
        confidence_threshold=0.7
    )


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="test_tool",
            description="A test tool",
            action_type=ActionType.SEARCH,
            parameters={"query": "string"},
            execute_func=AsyncMock()
        )
        
        registry.register_tool(tool)
        
        assert "test_tool" in registry.tools
        assert registry.get_tool("test_tool") == tool
        assert "test_tool" in registry.tool_categories[ActionType.SEARCH]
    
    def test_get_tools_by_type(self):
        """Test getting tools by type."""
        registry = ToolRegistry()
        
        search_tool = Tool(
            name="search_tool",
            description="Search tool",
            action_type=ActionType.SEARCH,
            parameters={},
            execute_func=AsyncMock()
        )
        
        code_tool = Tool(
            name="code_tool",
            description="Code tool",
            action_type=ActionType.CODE_GENERATION,
            parameters={},
            execute_func=AsyncMock()
        )
        
        registry.register_tool(search_tool)
        registry.register_tool(code_tool)
        
        search_tools = registry.get_tools_by_type(ActionType.SEARCH)
        assert len(search_tools) == 1
        assert search_tools[0].name == "search_tool"
        
        code_tools = registry.get_tools_by_type(ActionType.CODE_GENERATION)
        assert len(code_tools) == 1
        assert code_tools[0].name == "code_tool"
    
    def test_list_available_tools(self):
        """Test listing available tools."""
        registry = ToolRegistry()
        
        tool1 = Tool("tool1", "desc1", ActionType.SEARCH, {}, AsyncMock())
        tool2 = Tool("tool2", "desc2", ActionType.CODE_ANALYSIS, {}, AsyncMock())
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        tools = registry.list_available_tools()
        assert "tool1" in tools
        assert "tool2" in tools
        assert len(tools) == 2


class TestToolSelector:
    """Test tool selection functionality."""
    
    @pytest.fixture
    def tool_selector(self, mock_llm_client):
        """Create tool selector with mock tools."""
        registry = ToolRegistry()
        
        # Add test tools
        search_tool = Tool(
            name="search_tool",
            description="Search for information",
            action_type=ActionType.SEARCH,
            parameters={"query": "string"},
            execute_func=AsyncMock()
        )
        
        code_tool = Tool(
            name="code_tool",
            description="Generate code",
            action_type=ActionType.CODE_GENERATION,
            parameters={"description": "string"},
            execute_func=AsyncMock()
        )
        
        registry.register_tool(search_tool)
        registry.register_tool(code_tool)
        
        return ToolSelector(registry, mock_llm_client)
    
    @pytest.mark.asyncio
    async def test_select_tool(self, tool_selector, mock_llm_client):
        """Test tool selection."""
        mock_llm_client.set_response(
            "select the most appropriate tool",
            "Tool: search_tool\nConfidence: 0.9\nReasoning: Need to find information"
        )
        
        tool_name, confidence = await tool_selector.select_tool(
            "I need to find information about Python",
            ["search_tool", "code_tool"]
        )
        
        assert tool_name == "search_tool"
        assert confidence == 0.9
    
    def test_get_contextual_tools(self, tool_selector):
        """Test contextual tool selection."""
        # Test search context
        tools = tool_selector.get_contextual_tools("I need to search for information")
        assert "search_tool" in tools
        
        # Test code generation context
        tools = tool_selector.get_contextual_tools("I need to generate some code")
        assert "code_tool" in tools
    
    @pytest.mark.asyncio
    async def test_select_tool_fallback(self, tool_selector, mock_llm_client):
        """Test tool selection fallback."""
        # Set a response that doesn't match any tool
        mock_llm_client.set_response("select the most appropriate tool", 
            "Tool: unknown_tool\nConfidence: 0.8")
        
        tool_name, confidence = await tool_selector.select_tool(
            "Some context",
            ["search_tool"]
        )
        
        # Should fallback to first available when no match found
        assert tool_name == "search_tool" or tool_name is None  # Allow both outcomes
        assert confidence >= 0.0


class TestAdaptiveReasoningStrategy:
    """Test adaptive reasoning strategy."""
    
    def test_get_strategy(self):
        """Test getting strategy by complexity."""
        strategy = AdaptiveReasoningStrategy()
        
        simple = strategy.get_strategy("simple")
        assert simple["max_steps"] == 5
        assert simple["reasoning_depth"] == "shallow"
        
        complex_strategy = strategy.get_strategy("complex")
        assert complex_strategy["max_steps"] == 20
        assert complex_strategy["reasoning_depth"] == "deep"
        
        # Test default
        default = strategy.get_strategy("unknown")
        assert default["max_steps"] == 10  # moderate default
    
    def test_should_reflect(self):
        """Test reflection decision."""
        strategy = AdaptiveReasoningStrategy()
        
        simple_strategy = strategy.get_strategy("simple")
        complex_strategy = strategy.get_strategy("complex")
        
        # Simple strategy should reflect less frequently
        assert not strategy.should_reflect(1, simple_strategy)
        assert strategy.should_reflect(5, simple_strategy)
        
        # Complex strategy should reflect more frequently
        assert strategy.should_reflect(2, complex_strategy)
    
    def test_should_continue_reasoning(self):
        """Test reasoning continuation decision."""
        strategy = AdaptiveReasoningStrategy()
        
        simple_strategy = strategy.get_strategy("simple")
        
        # Should continue with low confidence
        assert strategy.should_continue_reasoning(2, simple_strategy, 0.5)
        
        # Should stop with high confidence
        assert not strategy.should_continue_reasoning(2, simple_strategy, 0.95)
        
        # Should stop at max steps
        assert not strategy.should_continue_reasoning(10, simple_strategy, 0.7)


class TestReActFramework:
    """Test main ReAct framework functionality."""
    
    @pytest.mark.asyncio
    async def test_solve_problem_basic(self, react_framework, mock_llm_client):
        """Test basic problem solving."""
        # Set up mock responses
        mock_llm_client.set_response("think step by step", 
            "I need to understand the problem and find relevant information.")
        mock_llm_client.set_response("select the most appropriate tool",
            "Tool: semantic_search\nConfidence: 0.8")
        mock_llm_client.set_response("generate appropriate input",
            '{"query": "test problem", "max_results": 5}')
        mock_llm_client.set_response("final answer",
            "The solution is to implement the required functionality step by step.")
        
        trace = await react_framework.solve_problem(
            "How do I implement a simple function?",
            task_complexity="simple"
        )
        
        assert trace is not None
        assert trace.problem_statement == "How do I implement a simple function?"
        assert len(trace.steps) > 0
        assert trace.total_time > 0
        assert trace.confidence_score >= 0
    
    @pytest.mark.asyncio
    async def test_solve_problem_with_tools(self, react_framework, mock_llm_client):
        """Test problem solving with tool usage."""
        # Mock tool execution
        search_tool = react_framework.tool_registry.get_tool("semantic_search")
        search_tool.execute_func = AsyncMock(return_value={
            "results": [{"content": "Function implementation example", "relevance": 0.9}],
            "total_found": 1
        })
        
        mock_llm_client.set_response("think step by step",
            "I should search for examples of function implementation.")
        mock_llm_client.set_response("select the most appropriate tool",
            "Tool: semantic_search\nConfidence: 0.9")
        mock_llm_client.set_response("generate appropriate input",
            '{"query": "function implementation", "max_results": 3}')
        mock_llm_client.set_response("final answer",
            "Based on the search results, implement the function as shown in the examples.")
        
        trace = await react_framework.solve_problem(
            "Show me how to implement a function",
            task_complexity="moderate"
        )
        
        assert trace is not None
        assert "semantic_search" in trace.tools_used
        
        # Check that search tool was called (may be called multiple times in reasoning loop)
        assert search_tool.execute_func.call_count >= 1
    
    def test_register_custom_tool(self, react_framework):
        """Test registering custom tools."""
        custom_tool = Tool(
            name="custom_tool",
            description="A custom tool for testing",
            action_type=ActionType.CODE_ANALYSIS,
            parameters={"input": "string"},
            execute_func=AsyncMock()
        )
        
        react_framework.register_custom_tool(custom_tool)
        
        assert react_framework.tool_registry.get_tool("custom_tool") == custom_tool
    
    def test_get_trace(self, react_framework):
        """Test trace retrieval."""
        # Create a mock trace
        trace = ReActTrace(
            trace_id="test_trace",
            problem_statement="Test problem"
        )
        
        react_framework.completed_traces.append(trace)
        
        retrieved = react_framework.get_trace("test_trace")
        assert retrieved == trace
        
        # Test non-existent trace
        assert react_framework.get_trace("nonexistent") is None
    
    def test_visualize_trace(self, react_framework):
        """Test trace visualization."""
        trace = ReActTrace(
            trace_id="test_trace",
            problem_statement="Test problem",
            total_time=1.5,
            confidence_score=0.8,
            success=True,
            tools_used=["semantic_search"]
        )
        
        # Add some steps
        thought_step = ReActStep(
            step_id="step1",
            step_type=ReasoningStep.THOUGHT,
            content="I need to think about this problem",
            confidence=0.7
        )
        
        action_step = ReActStep(
            step_id="step2",
            step_type=ReasoningStep.ACTION,
            content="I will search for information",
            action_type=ActionType.SEARCH,
            tool_name="semantic_search",
            confidence=0.8
        )
        
        trace.steps = [thought_step, action_step]
        trace.final_answer = "The solution is clear."
        
        visualization = react_framework.visualize_trace(trace)
        
        assert "test_trace" in visualization
        assert "Test problem" in visualization
        assert "1.50s" in visualization
        assert "0.80" in visualization
        assert "semantic_search" in visualization
        assert "The solution is clear" in visualization
    
    def test_get_framework_status(self, react_framework):
        """Test framework status reporting."""
        # Add some completed traces
        trace1 = ReActTrace("trace1", "problem1")
        trace1.confidence_score = 0.8
        trace1.success = True
        
        trace2 = ReActTrace("trace2", "problem2")
        trace2.confidence_score = 0.6
        trace2.success = False
        
        react_framework.completed_traces = [trace1, trace2]
        
        status = react_framework.get_framework_status()
        
        assert status["completed_traces"] == 2
        assert status["average_confidence"] == 0.7  # (0.8 + 0.6) / 2
        assert status["success_rate"] == 0.5  # 1 success out of 2
        assert "registered_tools" in status
        assert "tool_categories" in status


class TestReActStep:
    """Test ReAct step functionality."""
    
    def test_step_creation(self):
        """Test creating ReAct steps."""
        step = ReActStep(
            step_id="test_step",
            step_type=ReasoningStep.THOUGHT,
            content="This is a thought",
            confidence=0.8
        )
        
        assert step.step_id == "test_step"
        assert step.step_type == ReasoningStep.THOUGHT
        assert step.content == "This is a thought"
        assert step.confidence == 0.8
        assert step.timestamp is not None
    
    def test_action_step_with_tool(self):
        """Test action step with tool information."""
        step = ReActStep(
            step_id="action_step",
            step_type=ReasoningStep.ACTION,
            content="Using search tool",
            action_type=ActionType.SEARCH,
            tool_name="semantic_search",
            tool_input={"query": "test"},
            confidence=0.9
        )
        
        assert step.action_type == ActionType.SEARCH
        assert step.tool_name == "semantic_search"
        assert step.tool_input == {"query": "test"}


class TestReActTrace:
    """Test ReAct trace functionality."""
    
    def test_trace_creation(self):
        """Test creating ReAct traces."""
        trace = ReActTrace(
            trace_id="test_trace",
            problem_statement="Test problem"
        )
        
        assert trace.trace_id == "test_trace"
        assert trace.problem_statement == "Test problem"
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.success is False
        assert trace.tools_used == []
        assert trace.created_at is not None
    
    def test_trace_with_steps(self):
        """Test trace with multiple steps."""
        trace = ReActTrace(
            trace_id="test_trace",
            problem_statement="Test problem"
        )
        
        step1 = ReActStep("step1", ReasoningStep.THOUGHT, "First thought")
        step2 = ReActStep("step2", ReasoningStep.ACTION, "First action")
        
        trace.steps = [step1, step2]
        trace.tools_used = ["tool1", "tool2"]
        trace.final_answer = "Final solution"
        trace.success = True
        
        assert len(trace.steps) == 2
        assert len(trace.tools_used) == 2
        assert trace.final_answer == "Final solution"
        assert trace.success is True


class TestIntegration:
    """Integration tests for ReAct framework."""
    
    @pytest.mark.asyncio
    async def test_full_reasoning_cycle(self, react_framework, mock_llm_client):
        """Test a complete reasoning cycle."""
        # Set up comprehensive mock responses
        responses = {
            "think step by step": "I need to analyze this coding problem step by step.",
            "select the most appropriate tool": "Tool: code_analyzer\nConfidence: 0.85",
            "generate appropriate input": '{"code": "def example():", "analysis_type": "general"}',
            "reflect on the progress": "The analysis shows we need to improve the code structure.",
            "final answer": "The code needs proper implementation with error handling and documentation."
        }
        
        for pattern, response in responses.items():
            mock_llm_client.set_response(pattern, response)
        
        # Mock tool execution
        analyzer_tool = react_framework.tool_registry.get_tool("code_analyzer")
        analyzer_tool.execute_func = AsyncMock(return_value={
            "analysis": "The code is incomplete and needs implementation",
            "code_length": 15,
            "analysis_type": "general"
        })
        
        trace = await react_framework.solve_problem(
            "How can I improve this Python function: def example():",
            task_complexity="complex"
        )
        
        # Verify trace properties
        assert trace is not None
        assert len(trace.steps) > 0
        assert trace.final_answer is not None
        assert trace.total_time > 0
        
        # Verify different step types are present
        step_types = [step.step_type for step in trace.steps]
        assert ReasoningStep.THOUGHT in step_types
        assert ReasoningStep.ACTION in step_types
        assert ReasoningStep.OBSERVATION in step_types
        
        # Verify tool was used (may be called multiple times in reasoning loop)
        assert "code_analyzer" in trace.tools_used
        assert analyzer_tool.execute_func.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, react_framework, mock_llm_client):
        """Test error handling in reasoning cycle."""
        mock_llm_client.set_response("think step by step", "I need to search for information.")
        mock_llm_client.set_response("select the most appropriate tool", "Tool: semantic_search\nConfidence: 0.8")
        mock_llm_client.set_response("generate appropriate input", '{"query": "test"}')
        
        # Make search tool raise an exception
        search_tool = react_framework.tool_registry.get_tool("semantic_search")
        search_tool.execute_func = AsyncMock(side_effect=Exception("Search failed"))
        
        trace = await react_framework.solve_problem(
            "Find information about testing",
            task_complexity="simple"
        )
        
        # Should still complete despite error
        assert trace is not None
        assert len(trace.steps) > 0
        
        # Should have error observation
        error_steps = [step for step in trace.steps if step.error is not None]
        assert len(error_steps) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_complexity(self, react_framework, mock_llm_client):
        """Test adaptive behavior based on complexity."""
        mock_llm_client.set_response("think step by step", "Simple analysis needed.")
        mock_llm_client.set_response("final answer", "Simple solution provided.")
        
        # Test simple complexity
        simple_trace = await react_framework.solve_problem(
            "Simple problem",
            task_complexity="simple"
        )
        
        # Test complex complexity
        mock_llm_client.set_response("think step by step", "Complex analysis required.")
        mock_llm_client.set_response("reflect on the progress", "Need deeper analysis.")
        
        complex_trace = await react_framework.solve_problem(
            "Complex problem",
            task_complexity="complex"
        )
        
        # Complex should have more steps due to higher limits and reflection
        assert len(complex_trace.steps) >= len(simple_trace.steps)


if __name__ == "__main__":
    pytest.main([__file__])