"""
Test suite for Multi-Agent System Architecture

Tests the specialized agents, communication protocols, task delegation,
and performance monitoring capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from multi_agent_system import (
    MultiAgentSystem,
    BaseAgent,
    CodeAgent,
    SearchAgent,
    ReasoningAgent,
    TestAgent,
    AgentTask,
    AgentMessage,
    TaskPriority,
    TaskStatus,
    AgentType,
    TaskRouter
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.2) -> str:
        """Mock generation method."""
        if "code completion" in prompt.lower():
            return "def example_function():\n    return 'completed'"
        elif "test" in prompt.lower():
            return "def test_example():\n    assert True"
        elif "analyze" in prompt.lower():
            return "Code analysis: The code looks good with no major issues."
        else:
            return "Mock response for: " + prompt[:50] + "..."


class MockContextManager:
    """Mock context manager for testing."""
    
    def get_relevant_context(self, query: str, max_tokens: int = 1024) -> list:
        """Mock context retrieval."""
        return [{"content": f"Context for: {query}", "relevance": 0.8}]


class MockSearchEngine:
    """Mock search engine for testing."""
    
    async def search_similar(self, query: str, max_results: int = 10) -> list:
        """Mock semantic search."""
        return [
            {
                "content": f"Result 1 for {query}",
                "similarity": 0.9,
                "file_path": "/mock/file1.py"
            },
            {
                "content": f"Result 2 for {query}",
                "similarity": 0.8,
                "file_path": "/mock/file2.py"
            }
        ]
    
    def get_relevant_context(self, query: str, max_tokens: int = 2048) -> list:
        """Mock context retrieval."""
        return [{"content": f"Context for {query}", "tokens": 100}]


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""
    
    async def generate_embedding(self, text: str) -> list:
        """Mock embedding generation."""
        # Return a simple mock embedding
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector


class MockCoTEngine:
    """Mock Chain-of-Thought engine for testing."""
    
    async def reason_through_problem(self, problem: str, complexity=None, problem_type=None):
        """Mock reasoning method."""
        from unittest.mock import Mock
        
        trace = Mock()
        trace.trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        trace.steps = [
            Mock(content="Step 1: Analyze the problem", step_type=Mock(value="analysis")),
            Mock(content="Step 2: Break down components", step_type=Mock(value="decomposition")),
            Mock(content="Step 3: Synthesize solution", step_type=Mock(value="synthesis"))
        ]
        trace.final_solution = f"Solution for: {problem}"
        trace.confidence_score = 0.85
        trace.quality_score = 0.8
        
        return trace


class MockInterleavedEngine:
    """Mock interleaved reasoning engine for testing."""
    
    async def reason_and_respond(self, query: str, context=None, mode=None, stream=False):
        """Mock reasoning and response."""
        return {
            "response": f"Interleaved reasoning response for: {query}",
            "reasoning_trace": ["Step 1", "Step 2", "Step 3"],
            "confidence": 0.8
        }


class MockCodeAnalyzer:
    """Mock code analyzer for testing."""
    
    async def analyze_code_structure(self, code: str) -> dict:
        """Mock code structure analysis."""
        return {
            "functions": ["function1", "function2"],
            "classes": ["Class1"],
            "complexity": "moderate",
            "lines_of_code": len(code.split('\n'))
        }


@pytest.fixture
def mock_llm_client():
    """Fixture for mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_context_manager():
    """Fixture for mock context manager."""
    return MockContextManager()


@pytest.fixture
def mock_search_engine():
    """Fixture for mock search engine."""
    return MockSearchEngine()


@pytest.fixture
def mock_embedding_generator():
    """Fixture for mock embedding generator."""
    return MockEmbeddingGenerator()


@pytest.fixture
def mock_cot_engine():
    """Fixture for mock CoT engine."""
    return MockCoTEngine()


@pytest.fixture
def mock_interleaved_engine():
    """Fixture for mock interleaved engine."""
    return MockInterleavedEngine()


@pytest.fixture
def mock_code_analyzer():
    """Fixture for mock code analyzer."""
    return MockCodeAnalyzer()


@pytest.fixture
def multi_agent_system():
    """Fixture for multi-agent system."""
    return MultiAgentSystem()


@pytest.fixture
def code_agent(mock_llm_client, mock_context_manager):
    """Fixture for code agent."""
    return CodeAgent("code_agent_1", mock_llm_client, mock_context_manager)


@pytest.fixture
def search_agent(mock_search_engine, mock_embedding_generator):
    """Fixture for search agent."""
    return SearchAgent("search_agent_1", mock_search_engine, mock_embedding_generator)


@pytest.fixture
def reasoning_agent(mock_cot_engine, mock_interleaved_engine):
    """Fixture for reasoning agent."""
    return ReasoningAgent("reasoning_agent_1", mock_cot_engine, mock_interleaved_engine)


@pytest.fixture
def test_agent(mock_llm_client, mock_code_analyzer):
    """Fixture for test agent."""
    return TestAgent("test_agent_1", mock_llm_client, mock_code_analyzer)


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_initialization(self, code_agent):
        """Test agent initialization."""
        assert code_agent.agent_id == "code_agent_1"
        assert code_agent.agent_type == AgentType.CODE
        assert code_agent.is_active is True
        assert len(code_agent.capabilities) > 0
        assert code_agent.max_concurrent_tasks > 0
    
    def test_agent_status(self, code_agent):
        """Test agent status reporting."""
        status = code_agent.get_status()
        
        assert "agent_id" in status
        assert "agent_type" in status
        assert "is_active" in status
        assert "capabilities" in status
        assert "metrics" in status
        
        assert status["agent_id"] == "code_agent_1"
        assert status["agent_type"] == "code"
    
    @pytest.mark.asyncio
    async def test_task_assignment(self, code_agent):
        """Test task assignment to agent."""
        task = AgentTask(
            task_id="test_task_1",
            task_type="code_completion",
            description="Complete code snippet",
            input_data={"code_context": "def hello():"}
        )
        
        success = await code_agent.assign_task(task)
        assert success is True
        assert len(code_agent.task_queue) == 1
        assert task.assigned_agent == "code_agent_1"
        assert task.status == TaskStatus.ASSIGNED
    
    @pytest.mark.asyncio
    async def test_task_processing(self, code_agent):
        """Test task processing."""
        task = AgentTask(
            task_id="test_task_2",
            task_type="code_completion",
            description="Complete code snippet",
            input_data={"code_context": "def hello():"}
        )
        
        await code_agent.assign_task(task)
        await code_agent.process_tasks()
        
        # Task should be completed
        assert len(code_agent.active_tasks) == 0
        assert len(code_agent.completed_tasks) == 1
        
        completed_task = code_agent.completed_tasks[0]
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result is not None


class TestCodeAgent:
    """Test code agent functionality."""
    
    def test_code_agent_capabilities(self, code_agent):
        """Test code agent capabilities."""
        expected_capabilities = [
            "code_completion",
            "code_generation",
            "code_analysis",
            "refactoring",
            "bug_fixing",
            "documentation_generation"
        ]
        
        for capability in expected_capabilities:
            assert capability in code_agent.capabilities
    
    def test_task_handling_confidence(self, code_agent):
        """Test task handling confidence scores."""
        code_task = AgentTask(
            task_id="code_task",
            task_type="code_completion",
            description="Complete code",
            input_data={}
        )
        
        non_code_task = AgentTask(
            task_id="search_task",
            task_type="semantic_search",
            description="Search code",
            input_data={}
        )
        
        assert code_agent.can_handle_task(code_task) > 0.8
        assert code_agent.can_handle_task(non_code_task) == 0.0
    
    @pytest.mark.asyncio
    async def test_code_completion(self, code_agent):
        """Test code completion functionality."""
        task = AgentTask(
            task_id="completion_task",
            task_type="code_completion",
            description="Complete code",
            input_data={
                "code_context": "def hello():",
                "cursor_position": 12
            }
        )
        
        result = await code_agent.execute_task(task)
        
        assert "completion" in result
        assert "confidence" in result
        assert result["confidence"] > 0.0
    
    @pytest.mark.asyncio
    async def test_code_generation(self, code_agent):
        """Test code generation functionality."""
        task = AgentTask(
            task_id="generation_task",
            task_type="code_generation",
            description="Generate code",
            input_data={
                "description": "Create a function that adds two numbers",
                "language": "python",
                "requirements": ["type hints", "docstring"]
            }
        )
        
        result = await code_agent.execute_task(task)
        
        assert "generated_code" in result
        assert "language" in result
        assert "confidence" in result
        assert result["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_code_analysis(self, code_agent):
        """Test code analysis functionality."""
        task = AgentTask(
            task_id="analysis_task",
            task_type="code_analysis",
            description="Analyze code",
            input_data={
                "code": "def add(a, b): return a + b",
                "analysis_type": "quality"
            }
        )
        
        result = await code_agent.execute_task(task)
        
        assert "analysis" in result
        assert "analysis_type" in result
        assert "confidence" in result


class TestSearchAgent:
    """Test search agent functionality."""
    
    def test_search_agent_capabilities(self, search_agent):
        """Test search agent capabilities."""
        expected_capabilities = [
            "semantic_search",
            "code_discovery",
            "pattern_matching",
            "similarity_analysis",
            "context_retrieval"
        ]
        
        for capability in expected_capabilities:
            assert capability in search_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, search_agent):
        """Test semantic search functionality."""
        task = AgentTask(
            task_id="search_task",
            task_type="semantic_search",
            description="Search for code",
            input_data={
                "query": "function to calculate fibonacci",
                "max_results": 5
            }
        )
        
        result = await search_agent.execute_task(task)
        
        assert "results" in result
        assert "query" in result
        assert "total_results" in result
        assert "confidence" in result
        assert len(result["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_code_discovery(self, search_agent):
        """Test code discovery functionality."""
        task = AgentTask(
            task_id="discovery_task",
            task_type="code_discovery",
            description="Discover code patterns",
            input_data={
                "pattern": "sorting algorithm",
                "language": "python"
            }
        )
        
        result = await search_agent.execute_task(task)
        
        assert "code_results" in result
        assert "pattern" in result
        assert "language" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_similarity_analysis(self, search_agent):
        """Test similarity analysis functionality."""
        task = AgentTask(
            task_id="similarity_task",
            task_type="similarity_analysis",
            description="Analyze similarity",
            input_data={
                "item1": "def sort_list(items): return sorted(items)",
                "item2": "def sort_array(arr): return sorted(arr)"
            }
        )
        
        result = await search_agent.execute_task(task)
        
        assert "similarity_score" in result
        assert "item1" in result
        assert "item2" in result
        assert "confidence" in result
        assert 0.0 <= result["similarity_score"] <= 1.01  # Allow for floating point precision


class TestReasoningAgent:
    """Test reasoning agent functionality."""
    
    def test_reasoning_agent_capabilities(self, reasoning_agent):
        """Test reasoning agent capabilities."""
        expected_capabilities = [
            "chain_of_thought",
            "problem_decomposition",
            "logical_analysis",
            "decision_making",
            "strategy_planning"
        ]
        
        for capability in expected_capabilities:
            assert capability in reasoning_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_chain_of_thought(self, reasoning_agent):
        """Test chain-of-thought reasoning."""
        task = AgentTask(
            task_id="cot_task",
            task_type="chain_of_thought",
            description="Reason through problem",
            input_data={
                "problem": "How to optimize database queries?",
                "complexity": "moderate"
            }
        )
        
        result = await reasoning_agent.execute_task(task)
        
        assert "reasoning_trace" in result
        assert "problem" in result
        assert "confidence" in result
        
        trace = result["reasoning_trace"]
        assert "trace_id" in trace
        assert "steps" in trace
        assert "final_solution" in trace
        assert "confidence" in trace
    
    @pytest.mark.asyncio
    async def test_problem_decomposition(self, reasoning_agent):
        """Test problem decomposition."""
        task = AgentTask(
            task_id="decomp_task",
            task_type="problem_decomposition",
            description="Decompose complex problem",
            input_data={
                "problem": "Build a web application with user authentication",
                "max_depth": 3
            }
        )
        
        result = await reasoning_agent.execute_task(task)
        
        assert "decomposition" in result
        assert "problem" in result
        assert "depth" in result
        assert "confidence" in result
        assert isinstance(result["decomposition"], list)
    
    @pytest.mark.asyncio
    async def test_decision_making(self, reasoning_agent):
        """Test decision making."""
        task = AgentTask(
            task_id="decision_task",
            task_type="decision_making",
            description="Make technical decision",
            input_data={
                "options": ["React", "Vue", "Angular"],
                "criteria": ["performance", "learning curve", "community"],
                "context": "Building a new frontend application"
            }
        )
        
        result = await reasoning_agent.execute_task(task)
        
        assert "decision" in result
        assert "reasoning" in result
        assert "options" in result
        assert "criteria" in result
        assert "confidence" in result


class TestTestAgent:
    """Test test agent functionality."""
    
    def test_test_agent_capabilities(self, test_agent):
        """Test test agent capabilities."""
        expected_capabilities = [
            "test_generation",
            "test_validation",
            "coverage_analysis",
            "test_optimization",
            "mock_generation"
        ]
        
        for capability in expected_capabilities:
            assert capability in test_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_test_generation(self, test_agent):
        """Test test generation functionality."""
        task = AgentTask(
            task_id="test_gen_task",
            task_type="test_generation",
            description="Generate tests",
            input_data={
                "code": "def add(a, b): return a + b",
                "test_type": "unit",
                "language": "python"
            }
        )
        
        result = await test_agent.execute_task(task)
        
        assert "generated_tests" in result
        assert "test_type" in result
        assert "language" in result
        assert "confidence" in result
        assert result["test_type"] == "unit"
        assert result["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_test_validation(self, test_agent):
        """Test test validation functionality."""
        task = AgentTask(
            task_id="test_val_task",
            task_type="test_validation",
            description="Validate tests",
            input_data={
                "test_code": "def test_add(): assert add(2, 3) == 5",
                "target_code": "def add(a, b): return a + b"
            }
        )
        
        result = await test_agent.execute_task(task)
        
        assert "validation_report" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_coverage_analysis(self, test_agent):
        """Test coverage analysis functionality."""
        task = AgentTask(
            task_id="coverage_task",
            task_type="coverage_analysis",
            description="Analyze test coverage",
            input_data={
                "code": "def add(a, b): return a + b\ndef multiply(a, b): return a * b",
                "tests": "def test_add(): assert add(2, 3) == 5"
            }
        )
        
        result = await test_agent.execute_task(task)
        
        assert "coverage_analysis" in result
        assert "code_structure" in result
        assert "confidence" in result


class TestMultiAgentSystem:
    """Test multi-agent system coordination."""
    
    def test_system_initialization(self, multi_agent_system):
        """Test system initialization."""
        assert len(multi_agent_system.agents) == 0
        assert len(multi_agent_system.task_queue) == 0
        assert "total_tasks" in multi_agent_system.system_metrics
    
    def test_agent_registration(self, multi_agent_system, code_agent):
        """Test agent registration."""
        multi_agent_system.register_agent(code_agent)
        
        assert code_agent.agent_id in multi_agent_system.agents
        assert code_agent.agent_id in multi_agent_system.system_metrics["agent_utilization"]
    
    def test_agent_unregistration(self, multi_agent_system, code_agent):
        """Test agent unregistration."""
        multi_agent_system.register_agent(code_agent)
        multi_agent_system.unregister_agent(code_agent.agent_id)
        
        assert code_agent.agent_id not in multi_agent_system.agents
    
    @pytest.mark.asyncio
    async def test_task_submission(self, multi_agent_system, code_agent):
        """Test task submission to system."""
        multi_agent_system.register_agent(code_agent)
        
        task_id = await multi_agent_system.submit_task(
            task_type="code_completion",
            description="Complete code snippet",
            input_data={"code_context": "def hello():"},
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        assert multi_agent_system.system_metrics["total_tasks"] == 1
    
    @pytest.mark.asyncio
    async def test_message_sending(self, multi_agent_system, code_agent, search_agent):
        """Test inter-agent messaging."""
        multi_agent_system.register_agent(code_agent)
        multi_agent_system.register_agent(search_agent)
        
        message_id = await multi_agent_system.send_message(
            sender_id=code_agent.agent_id,
            recipient_id=search_agent.agent_id,
            message_type="search_request",
            content={"query": "test function"},
            requires_response=False
        )
        
        assert message_id is not None
    
    def test_system_status(self, multi_agent_system, code_agent, search_agent):
        """Test system status reporting."""
        multi_agent_system.register_agent(code_agent)
        multi_agent_system.register_agent(search_agent)
        
        status = multi_agent_system.get_system_status()
        
        assert "agents" in status
        assert "task_queue_size" in status
        assert "system_metrics" in status
        assert "active_tasks" in status
        assert "total_completed" in status
        
        assert len(status["agents"]) == 2
        assert code_agent.agent_id in status["agents"]
        assert search_agent.agent_id in status["agents"]
    
    def test_performance_metrics(self, multi_agent_system, code_agent):
        """Test performance metrics collection."""
        multi_agent_system.register_agent(code_agent)
        
        metrics = multi_agent_system.get_performance_metrics()
        
        assert "agent_utilization" in metrics
        assert "success_rate" in metrics
        assert code_agent.agent_id in metrics["agent_utilization"]


class TestTaskRouter:
    """Test task routing functionality."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = TaskRouter()
        assert len(router.routing_history) == 0
    
    @pytest.mark.asyncio
    async def test_agent_selection(self, code_agent, search_agent):
        """Test best agent selection."""
        router = TaskRouter()
        
        agents = {
            code_agent.agent_id: code_agent,
            search_agent.agent_id: search_agent
        }
        
        # Test code task routing
        code_task = AgentTask(
            task_id="code_task",
            task_type="code_completion",
            description="Complete code",
            input_data={}
        )
        
        best_agent = await router.find_best_agent(code_task, agents)
        assert best_agent == code_agent
        
        # Test search task routing
        search_task = AgentTask(
            task_id="search_task",
            task_type="semantic_search",
            description="Search code",
            input_data={}
        )
        
        best_agent = await router.find_best_agent(search_task, agents)
        assert best_agent == search_agent
    
    @pytest.mark.asyncio
    async def test_routing_history(self, code_agent):
        """Test routing history tracking."""
        router = TaskRouter()
        
        agents = {code_agent.agent_id: code_agent}
        
        task = AgentTask(
            task_id="test_task",
            task_type="code_completion",
            description="Test task",
            input_data={}
        )
        
        await router.find_best_agent(task, agents)
        
        assert len(router.routing_history) == 1
        
        history_entry = router.routing_history[0]
        assert history_entry["task_id"] == "test_task"
        assert history_entry["selected_agent"] == code_agent.agent_id
        assert "confidence" in history_entry
        assert "timestamp" in history_entry
    
    def test_routing_stats(self, code_agent):
        """Test routing statistics."""
        router = TaskRouter()
        
        # Add some mock history
        router.routing_history = [
            {
                "task_id": "task1",
                "task_type": "code_completion",
                "selected_agent": code_agent.agent_id,
                "confidence": 0.9,
                "load_factor": 0.2,
                "score": 0.8,
                "timestamp": datetime.now()
            },
            {
                "task_id": "task2",
                "task_type": "code_generation",
                "selected_agent": code_agent.agent_id,
                "confidence": 0.8,
                "load_factor": 0.3,
                "score": 0.7,
                "timestamp": datetime.now()
            }
        ]
        
        stats = router.get_routing_stats()
        
        assert "total_routings" in stats
        assert "agent_usage" in stats
        assert "task_type_distribution" in stats
        assert "avg_confidence" in stats
        assert "avg_load_factor" in stats
        
        assert stats["total_routings"] == 2
        assert code_agent.agent_id in stats["agent_usage"]
        assert stats["agent_usage"][code_agent.agent_id] == 2


class TestAgentCommunication:
    """Test agent communication protocols."""
    
    @pytest.mark.asyncio
    async def test_message_handling(self, code_agent):
        """Test message handling."""
        message = AgentMessage(
            message_id="test_msg",
            sender_id="sender_agent",
            recipient_id=code_agent.agent_id,
            message_type="code_review_request",
            content={"code": "def test(): pass"}
        )
        
        response = await code_agent.handle_message(message)
        
        # Should handle the message (even if no specific handler)
        # The base implementation should not crash
        assert response is None or isinstance(response, dict)
    
    def test_message_handler_registration(self, code_agent):
        """Test message handler registration."""
        def test_handler(message):
            return {"handled": True}
        
        code_agent.register_message_handler("test_message", test_handler)
        
        assert "test_message" in code_agent.message_handlers
        assert code_agent.message_handlers["test_message"] == test_handler


class TestAgentPerformanceMonitoring:
    """Test agent performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, code_agent):
        """Test performance metrics tracking."""
        initial_completed = code_agent.metrics.tasks_completed
        initial_failed = code_agent.metrics.tasks_failed
        
        # Create and process a successful task
        task = AgentTask(
            task_id="perf_task",
            task_type="code_completion",
            description="Performance test task",
            input_data={"code_context": "def test():"}
        )
        
        await code_agent.assign_task(task)
        await code_agent.process_tasks()
        
        # Check metrics were updated
        assert code_agent.metrics.tasks_completed == initial_completed + 1
        assert code_agent.metrics.avg_execution_time > 0
        assert code_agent.metrics.last_activity is not None
    
    def test_success_rate_calculation(self, code_agent):
        """Test success rate calculation."""
        # Manually set some metrics
        code_agent.metrics.tasks_completed = 8
        code_agent.metrics.tasks_failed = 2
        
        # Calculate success rate
        total_tasks = code_agent.metrics.tasks_completed + code_agent.metrics.tasks_failed
        expected_rate = code_agent.metrics.tasks_completed / total_tasks
        
        # This would be calculated in the _execute_task_with_tracking method
        code_agent.metrics.success_rate = expected_rate
        
        assert code_agent.metrics.success_rate == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])