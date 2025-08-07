"""
Tests for ReAct Framework Integration

Tests the integration between ReAct framework and multi-agent system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from react_integration import ReActAgentIntegration, ReActIntegrationConfig, ContextAwareToolSelector
from react_framework import ReActFramework, Tool, ActionType
from multi_agent_system import MultiAgentSystem, AgentTask, TaskStatus, TaskPriority
from chain_of_thought_engine import ChainOfThoughtEngine


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_type: str, agent_id: str = None):
        self.agent_type = agent_type
        self.agent_id = agent_id or f"mock_{agent_type}_agent"
        self.active_tasks = {}
        self.max_concurrent_tasks = 3
        self.task_results = {}
    
    async def assign_task(self, task: AgentTask) -> bool:
        """Mock task assignment."""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return False
        
        task.status = TaskStatus.ASSIGNED
        self.active_tasks[task.task_id] = task
        return True
    
    async def process_tasks(self) -> None:
        """Mock task processing."""
        for task_id, task in list(self.active_tasks.items()):
            task.status = TaskStatus.COMPLETED
            
            # Set mock result based on task type
            if task.task_type == "code_generation":
                task.result = {
                    "generated_code": "def example(): pass",
                    "language": "python"
                }
            elif task.task_type == "semantic_search":
                task.result = {
                    "results": [{"content": "Mock search result", "relevance": 0.8}],
                    "total_found": 1
                }
            elif task.task_type == "logical_analysis":
                task.result = {
                    "analysis": "Mock logical analysis result",
                    "confidence": 0.8
                }
            elif task.task_type == "test_generation":
                task.result = {
                    "test_code": "def test_example(): assert True",
                    "test_count": 1
                }
            else:
                task.result = {"mock_result": "success"}
            
            del self.active_tasks[task_id]


class MockMultiAgentSystem:
    """Mock multi-agent system for testing."""
    
    def __init__(self):
        self.agents = {
            "code": MockAgent("code"),
            "search": MockAgent("search"),
            "reasoning": MockAgent("reasoning"),
            "test": MockAgent("test")
        }
    
    def get_agent_by_type(self, agent_type: str):
        """Get agent by type."""
        return self.agents.get(agent_type)
    
    def get_system_status(self) -> dict:
        """Get system status."""
        return {
            "total_agents": len(self.agents),
            "active_agents": len(self.agents),
            "agent_types": list(self.agents.keys())
        }


class MockCoTEngine:
    """Mock chain-of-thought engine for testing."""
    
    async def reason_through_problem(self, problem: str, problem_type: str = "coding_problem", complexity=None):
        """Mock reasoning."""
        mock_trace = Mock()
        mock_trace.trace_id = "mock_trace_123"
        mock_trace.steps = [Mock(), Mock()]  # Mock steps
        mock_trace.final_solution = f"Mock solution for: {problem}"
        mock_trace.confidence_score = 0.85
        mock_trace.quality_score = 0.8
        mock_trace.total_time = 1.5
        return mock_trace


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5):
        """Generate mock response."""
        if "think step by step" in prompt.lower():
            return "I need to analyze this problem and use the appropriate tools."
        elif "tool:" in prompt.lower():
            return "Tool: code_agent\nConfidence: 0.8\nReasoning: This tool will help with code generation."
        elif "final answer" in prompt.lower():
            return "Based on my analysis using various tools, the solution is implemented."
        else:
            return "Mock response for testing."


class MockContextManager:
    """Mock context manager for testing."""
    
    def get_relevant_context(self, query: str, max_tokens: int = 1024):
        """Return mock context."""
        return [Mock(content=f"Mock context for: {query}", source="test", relevance=0.8)]


@pytest.fixture
def mock_multi_agent_system():
    """Create mock multi-agent system."""
    return MockMultiAgentSystem()


@pytest.fixture
def mock_cot_engine():
    """Create mock CoT engine."""
    return MockCoTEngine()


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
        max_iterations=5,
        confidence_threshold=0.7
    )


@pytest.fixture
def react_integration(react_framework, mock_multi_agent_system, mock_cot_engine):
    """Create ReAct integration instance."""
    return ReActAgentIntegration(
        react_framework=react_framework,
        multi_agent_system=mock_multi_agent_system,
        cot_engine=mock_cot_engine,
        config=ReActIntegrationConfig()
    )


class TestReActIntegrationConfig:
    """Test ReAct integration configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReActIntegrationConfig()
        
        assert config.enable_multi_agent_tools is True
        assert config.enable_cot_reasoning_tool is True
        assert config.enable_context_aware_selection is True
        assert config.max_reasoning_depth == 5
        assert config.tool_timeout == 30
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReActIntegrationConfig(
            enable_multi_agent_tools=False,
            enable_cot_reasoning_tool=False,
            max_reasoning_depth=10,
            tool_timeout=60
        )
        
        assert config.enable_multi_agent_tools is False
        assert config.enable_cot_reasoning_tool is False
        assert config.max_reasoning_depth == 10
        assert config.tool_timeout == 60


class TestReActAgentIntegration:
    """Test ReAct agent integration functionality."""
    
    def test_initialization(self, react_integration):
        """Test integration initialization."""
        assert react_integration.react_framework is not None
        assert react_integration.multi_agent_system is not None
        assert react_integration.cot_engine is not None
        assert react_integration.config is not None
        
        # Check that agent tools are registered
        tool_names = react_integration.react_framework.tool_registry.list_available_tools()
        assert "code_agent" in tool_names
        assert "search_agent" in tool_names
        assert "reasoning_agent" in tool_names
        assert "test_agent" in tool_names
        assert "chain_of_thought" in tool_names
    
    @pytest.mark.asyncio
    async def test_code_agent_tool(self, react_integration):
        """Test code agent tool execution."""
        tool = react_integration.react_framework.tool_registry.get_tool("code_agent")
        assert tool is not None
        
        input_data = {
            "task_type": "code_generation",
            "input_data": {"description": "Create a simple function"},
            "priority": "normal"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "result" in result
        assert result["status"] == "completed"
        assert "agent_id" in result
        assert result["result"]["generated_code"] == "def example(): pass"
    
    @pytest.mark.asyncio
    async def test_search_agent_tool(self, react_integration):
        """Test search agent tool execution."""
        tool = react_integration.react_framework.tool_registry.get_tool("search_agent")
        assert tool is not None
        
        input_data = {
            "task_type": "semantic_search",
            "input_data": {"query": "test search"},
            "priority": "high"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "result" in result
        assert result["status"] == "completed"
        assert "agent_id" in result
        assert len(result["result"]["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_reasoning_agent_tool(self, react_integration):
        """Test reasoning agent tool execution."""
        tool = react_integration.react_framework.tool_registry.get_tool("reasoning_agent")
        assert tool is not None
        
        input_data = {
            "task_type": "logical_analysis",
            "input_data": {"statement": "Test logical statement"},
            "priority": "normal"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "result" in result
        assert result["status"] == "completed"
        assert "agent_id" in result
        assert result["result"]["analysis"] == "Mock logical analysis result"
    
    @pytest.mark.asyncio
    async def test_test_agent_tool(self, react_integration):
        """Test test agent tool execution."""
        tool = react_integration.react_framework.tool_registry.get_tool("test_agent")
        assert tool is not None
        
        input_data = {
            "task_type": "test_generation",
            "input_data": {"code": "def example(): pass"},
            "priority": "low"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "result" in result
        assert result["status"] == "completed"
        assert "agent_id" in result
        assert result["result"]["test_code"] == "def test_example(): assert True"
    
    @pytest.mark.asyncio
    async def test_cot_tool(self, react_integration):
        """Test chain-of-thought tool execution."""
        tool = react_integration.react_framework.tool_registry.get_tool("chain_of_thought")
        assert tool is not None
        
        input_data = {
            "problem": "How to implement a sorting algorithm?",
            "complexity": "moderate",
            "problem_type": "coding_problem"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "reasoning_trace" in result
        assert result["success"] is True
        assert result["reasoning_trace"]["confidence"] == 0.85
        assert "Mock solution for:" in result["reasoning_trace"]["final_solution"]
    
    @pytest.mark.asyncio
    async def test_agent_unavailable(self, react_integration):
        """Test behavior when agent is unavailable."""
        # Remove code agent to simulate unavailability
        react_integration.multi_agent_system.agents.pop("code", None)
        
        tool = react_integration.react_framework.tool_registry.get_tool("code_agent")
        input_data = {
            "task_type": "code_generation",
            "input_data": {"description": "Create a function"},
            "priority": "normal"
        }
        
        result = await tool.execute_func(input_data)
        
        assert "error" in result
        assert "Code agent not available" in result["error"]
    
    @pytest.mark.asyncio
    async def test_solve_with_integrated_reasoning(self, react_integration):
        """Test integrated reasoning solution."""
        result = await react_integration.solve_with_integrated_reasoning(
            problem="Create a simple Python function",
            task_complexity="simple",
            use_agents=True,
            use_cot=True
        )
        
        assert "react_trace" in result
        assert "integration_stats" in result
        assert result["success"] is not None
        assert result["confidence"] >= 0.0
        assert result["final_answer"] is not None
        
        # Check integration stats
        stats = result["integration_stats"]
        assert "agent_tools_used" in stats
        assert "cot_tools_used" in stats
        assert stats["agents_enabled"] is True
        assert stats["cot_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_solve_without_agents(self, react_integration):
        """Test solving without agent tools."""
        result = await react_integration.solve_with_integrated_reasoning(
            problem="Analyze this code",
            task_complexity="moderate",
            use_agents=False,
            use_cot=True
        )
        
        assert result["integration_stats"]["agents_enabled"] is False
        assert result["integration_stats"]["cot_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_solve_without_cot(self, react_integration):
        """Test solving without CoT tools."""
        result = await react_integration.solve_with_integrated_reasoning(
            problem="Generate test cases",
            task_complexity="moderate",
            use_agents=True,
            use_cot=False
        )
        
        assert result["integration_stats"]["agents_enabled"] is True
        assert result["integration_stats"]["cot_enabled"] is False
    
    def test_get_integration_status(self, react_integration):
        """Test integration status reporting."""
        status = react_integration.get_integration_status()
        
        assert "react_framework_status" in status
        assert "multi_agent_status" in status
        assert "integration_config" in status
        assert "available_integrated_tools" in status
        
        # Check config status
        config = status["integration_config"]
        assert config["multi_agent_tools_enabled"] is True
        assert config["cot_reasoning_enabled"] is True
        
        # Check available tools
        tools = status["available_integrated_tools"]
        assert "code_agent" in tools
        assert "search_agent" in tools
        assert "reasoning_agent" in tools
        assert "test_agent" in tools
        assert "chain_of_thought" in tools


class TestContextAwareToolSelector:
    """Test context-aware tool selector."""
    
    @pytest.fixture
    def tool_selector(self, react_integration):
        """Create context-aware tool selector."""
        return ContextAwareToolSelector(react_integration)
    
    @pytest.mark.asyncio
    async def test_select_optimal_tool_basic(self, tool_selector):
        """Test basic optimal tool selection."""
        tool_name, confidence, metadata = await tool_selector.select_optimal_tool(
            reasoning_context="I need to generate some code",
            available_tools=["code_agent", "search_agent"],
            consider_agent_load=False
        )
        
        assert tool_name is not None
        assert confidence > 0.0
        assert "reason" in metadata
    
    @pytest.mark.asyncio
    async def test_select_with_load_balancing(self, tool_selector):
        """Test tool selection with load balancing."""
        # Simulate high load on code agent
        code_agent = tool_selector.react_integration.multi_agent_system.get_agent_by_type("code")
        code_agent.active_tasks = {"task1": Mock(), "task2": Mock(), "task3": Mock()}  # Full load
        
        tool_name, confidence, metadata = await tool_selector.select_optimal_tool(
            reasoning_context="I need to generate some code",
            available_tools=["code_agent", "search_agent"],
            consider_agent_load=True
        )
        
        assert tool_name is not None
        assert confidence > 0.0
        assert "reason" in metadata
        
        # If load balancing occurred, metadata should indicate it
        if metadata["reason"] == "load_balancing":
            assert "original_tool" in metadata
            assert "selected_load" in metadata
    
    @pytest.mark.asyncio
    async def test_select_no_tools_available(self, tool_selector):
        """Test selection when no tools are available."""
        tool_name, confidence, metadata = await tool_selector.select_optimal_tool(
            reasoning_context="Some context",
            available_tools=[],
            consider_agent_load=True
        )
        
        assert tool_name is None
        assert confidence == 0.0
        assert metadata["reason"] == "no_tools_available"
    
    def test_get_selection_history(self, tool_selector):
        """Test selection history tracking."""
        history = tool_selector.get_selection_history()
        assert isinstance(history, list)


class TestIntegrationScenarios:
    """Test various integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complex_problem_solving(self, react_integration):
        """Test solving a complex problem using multiple integrated tools."""
        problem = """
        I need to create a Python function that sorts a list of dictionaries 
        by multiple keys, write tests for it, and analyze the code quality.
        """
        
        result = await react_integration.solve_with_integrated_reasoning(
            problem=problem,
            task_complexity="complex",
            use_agents=True,
            use_cot=True
        )
        
        assert result["success"] is not None
        assert result["confidence"] >= 0.0
        assert result["final_answer"] is not None
        
        # Check that the framework completed successfully
        stats = result["integration_stats"]
        # Tools may or may not be used depending on the reasoning path
        assert stats["total_tools_used"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_integration(self, react_integration):
        """Test error handling in integrated reasoning."""
        # Simulate an error by removing all agents
        react_integration.multi_agent_system.agents.clear()
        
        result = await react_integration.solve_with_integrated_reasoning(
            problem="Generate code using agents",
            task_complexity="simple",
            use_agents=True,
            use_cot=False
        )
        
        # Should still complete despite agent unavailability
        assert "react_trace" in result or "error" in result
        assert result["success"] is not None
    
    @pytest.mark.asyncio
    async def test_configuration_changes(self, react_integration):
        """Test dynamic configuration changes."""
        # Test with agents disabled
        config = ReActIntegrationConfig(enable_multi_agent_tools=False)
        react_integration.config = config
        
        result = await react_integration.solve_with_integrated_reasoning(
            problem="Test problem",
            task_complexity="simple",
            use_agents=False,
            use_cot=True
        )
        
        assert result["integration_stats"]["agents_enabled"] is False
        
        # Re-enable agents
        config.enable_multi_agent_tools = True
        
        result2 = await react_integration.solve_with_integrated_reasoning(
            problem="Test problem",
            task_complexity="simple",
            use_agents=True,
            use_cot=True
        )
        
        assert result2["integration_stats"]["agents_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__])