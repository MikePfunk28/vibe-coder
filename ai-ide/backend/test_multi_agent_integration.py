"""
Integration tests for Multi-Agent System

Tests the complete multi-agent system working together with
agent communication, task delegation, and coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from multi_agent_system import (
    MultiAgentSystem,
    CodeAgent,
    SearchAgent,
    ReasoningAgent,
    TestAgent,
    TaskPriority
)
from test_multi_agent_system import (
    MockLLMClient,
    MockContextManager,
    MockSearchEngine,
    MockEmbeddingGenerator,
    MockCoTEngine,
    MockInterleavedEngine,
    MockCodeAnalyzer
)


@pytest.fixture
def integrated_system():
    """Fixture for fully integrated multi-agent system."""
    system = MultiAgentSystem()
    
    # Create mock dependencies
    llm_client = MockLLMClient()
    context_manager = MockContextManager()
    search_engine = MockSearchEngine()
    embedding_generator = MockEmbeddingGenerator()
    cot_engine = MockCoTEngine()
    interleaved_engine = MockInterleavedEngine()
    code_analyzer = MockCodeAnalyzer()
    
    # Create and register agents
    code_agent = CodeAgent("code_agent", llm_client, context_manager)
    search_agent = SearchAgent("search_agent", search_engine, embedding_generator)
    reasoning_agent = ReasoningAgent("reasoning_agent", cot_engine, interleaved_engine)
    test_agent = TestAgent("test_agent", llm_client, code_analyzer)
    
    system.register_agent(code_agent)
    system.register_agent(search_agent)
    system.register_agent(reasoning_agent)
    system.register_agent(test_agent)
    
    return system


class TestMultiAgentIntegration:
    """Test multi-agent system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_development_workflow(self, integrated_system):
        """Test a complete development workflow using multiple agents."""
        
        # Step 1: Use reasoning agent to decompose a complex problem
        reasoning_task_id = await integrated_system.submit_task(
            task_type="problem_decomposition",
            description="Decompose building a user authentication system",
            input_data={
                "problem": "Build a secure user authentication system with JWT tokens",
                "max_depth": 3
            },
            priority=TaskPriority.HIGH
        )
        
        # Step 2: Use search agent to find relevant code examples
        search_task_id = await integrated_system.submit_task(
            task_type="semantic_search",
            description="Find authentication examples",
            input_data={
                "query": "JWT authentication implementation",
                "max_results": 5
            },
            priority=TaskPriority.NORMAL
        )
        
        # Step 3: Use code agent to generate authentication code
        code_task_id = await integrated_system.submit_task(
            task_type="code_generation",
            description="Generate JWT authentication code",
            input_data={
                "description": "Create JWT authentication middleware",
                "language": "python",
                "requirements": ["security", "error handling", "type hints"]
            },
            priority=TaskPriority.NORMAL
        )
        
        # Step 4: Use test agent to generate tests for the code
        test_task_id = await integrated_system.submit_task(
            task_type="test_generation",
            description="Generate tests for authentication",
            input_data={
                "code": "def authenticate_user(token): pass",
                "test_type": "unit",
                "language": "python"
            },
            priority=TaskPriority.NORMAL
        )
        
        # Wait for all tasks to complete
        await asyncio.sleep(0.5)  # Allow time for processing
        
        # Verify all tasks were processed
        reasoning_status = integrated_system.get_task_status(reasoning_task_id)
        search_status = integrated_system.get_task_status(search_task_id)
        code_status = integrated_system.get_task_status(code_task_id)
        test_status = integrated_system.get_task_status(test_task_id)
        
        assert reasoning_status["status"] == "completed"
        assert search_status["status"] == "completed"
        assert code_status["status"] == "completed"
        assert test_status["status"] == "completed"
        
        # Verify results contain expected data
        assert reasoning_status["assigned_agent"] == "reasoning_agent"
        assert search_status["assigned_agent"] == "search_agent"
        assert code_status["assigned_agent"] == "code_agent"
        assert test_status["assigned_agent"] == "test_agent"
    
    @pytest.mark.asyncio
    async def test_agent_communication_workflow(self, integrated_system):
        """Test agents communicating with each other."""
        
        # Get agents for direct communication testing
        code_agent = integrated_system.agents["code_agent"]
        search_agent = integrated_system.agents["search_agent"]
        
        # Code agent requests search from search agent
        message_id = await integrated_system.send_message(
            sender_id="code_agent",
            recipient_id="search_agent",
            message_type="search_request",
            content={
                "query": "error handling patterns",
                "max_results": 3
            },
            requires_response=True
        )
        
        assert message_id is not None
        
        # Allow time for message processing
        await asyncio.sleep(0.1)
        
        # Verify message was processed (would be in real implementation)
        # In this test, we just verify the system can handle the communication
        assert len(integrated_system.message_queue) >= 1
    
    @pytest.mark.asyncio
    async def test_load_balancing_and_routing(self, integrated_system):
        """Test task routing and load balancing across agents."""
        
        # Submit multiple code tasks to test load balancing
        task_ids = []
        for i in range(5):
            task_id = await integrated_system.submit_task(
                task_type="code_completion",
                description=f"Complete code snippet {i}",
                input_data={"code_context": f"def function_{i}():"},
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify all tasks were assigned to the code agent
        for task_id in task_ids:
            status = integrated_system.get_task_status(task_id)
            assert status["assigned_agent"] == "code_agent"
            assert status["status"] == "completed"
        
        # Check system metrics
        metrics = integrated_system.get_performance_metrics()
        assert metrics["agent_utilization"]["code_agent"] >= 0.0
        assert metrics["success_rate"] > 0.0
    
    @pytest.mark.asyncio
    async def test_priority_task_handling(self, integrated_system):
        """Test priority-based task handling."""
        
        # Submit low priority task first
        low_priority_task = await integrated_system.submit_task(
            task_type="code_analysis",
            description="Analyze code quality",
            input_data={"code": "def slow_function(): pass"},
            priority=TaskPriority.LOW
        )
        
        # Submit high priority task second
        high_priority_task = await integrated_system.submit_task(
            task_type="code_completion",
            description="Complete urgent code",
            input_data={"code_context": "def urgent():"},
            priority=TaskPriority.URGENT
        )
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Verify both tasks completed
        low_status = integrated_system.get_task_status(low_priority_task)
        high_status = integrated_system.get_task_status(high_priority_task)
        
        assert low_status["status"] == "completed"
        assert high_status["status"] == "completed"
        
        # High priority task should have been processed
        assert high_status["priority"] == TaskPriority.URGENT.value
        assert low_status["priority"] == TaskPriority.LOW.value
    
    @pytest.mark.asyncio
    async def test_agent_specialization_routing(self, integrated_system):
        """Test that tasks are routed to the most specialized agents."""
        
        # Submit different types of tasks
        tasks = [
            ("code_completion", "code_agent"),
            ("semantic_search", "search_agent"),
            ("chain_of_thought", "reasoning_agent"),
            ("test_generation", "test_agent")
        ]
        
        task_ids = []
        for task_type, expected_agent in tasks:
            task_id = await integrated_system.submit_task(
                task_type=task_type,
                description=f"Test {task_type}",
                input_data={"test": "data"},
                priority=TaskPriority.NORMAL
            )
            task_ids.append((task_id, expected_agent))
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify correct agent assignment
        for task_id, expected_agent in task_ids:
            status = integrated_system.get_task_status(task_id)
            assert status["assigned_agent"] == expected_agent
            assert status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_system_performance_monitoring(self, integrated_system):
        """Test system performance monitoring and metrics."""
        
        # Submit various tasks to generate metrics
        for i in range(3):
            await integrated_system.submit_task(
                task_type="code_completion",
                description=f"Performance test {i}",
                input_data={"code_context": f"def test_{i}():"},
                priority=TaskPriority.NORMAL
            )
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Check system status
        status = integrated_system.get_system_status()
        
        assert "agents" in status
        assert "system_metrics" in status
        assert "active_tasks" in status
        assert "total_completed" in status
        
        assert len(status["agents"]) == 4  # All 4 agents registered
        assert status["total_completed"] >= 3  # At least 3 tasks completed
        
        # Check performance metrics
        metrics = integrated_system.get_performance_metrics()
        
        assert "agent_utilization" in metrics
        assert "success_rate" in metrics
        assert len(metrics["agent_utilization"]) == 4
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integrated_system):
        """Test error handling and system recovery."""
        
        # Submit a task with invalid data to trigger error handling
        task_id = await integrated_system.submit_task(
            task_type="invalid_task_type",
            description="This should fail gracefully",
            input_data={},
            priority=TaskPriority.NORMAL
        )
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # The task should either be rejected or handled gracefully
        status = integrated_system.get_task_status(task_id)
        
        # Task should still be in queue (no agent can handle it)
        # or have been assigned but failed
        assert status is not None
        
        # System should still be functional
        system_status = integrated_system.get_system_status()
        assert len(system_status["agents"]) == 4
        
        # Submit a valid task to verify system recovery
        recovery_task = await integrated_system.submit_task(
            task_type="code_completion",
            description="Recovery test",
            input_data={"code_context": "def recovery():"},
            priority=TaskPriority.NORMAL
        )
        
        await asyncio.sleep(0.2)
        
        recovery_status = integrated_system.get_task_status(recovery_task)
        assert recovery_status["status"] == "completed"
    
    def test_system_shutdown(self, integrated_system):
        """Test graceful system shutdown."""
        
        # Verify system is active
        status = integrated_system.get_system_status()
        assert len(status["agents"]) == 4
        
        # Test shutdown (in real implementation this would be async)
        # For now, just verify the method exists and can be called
        assert hasattr(integrated_system, 'shutdown')
        
        # Verify agents are still registered (shutdown not called)
        assert len(integrated_system.agents) == 4


class TestAgentCoordination:
    """Test agent coordination and collaboration."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, integrated_system):
        """Test multiple agents working together on a complex task."""
        
        # Simulate a complex development task requiring multiple agents
        
        # 1. Reasoning agent analyzes the problem
        analysis_task = await integrated_system.submit_task(
            task_type="logical_analysis",
            description="Analyze system architecture requirements",
            input_data={
                "statement": "Design a microservices architecture for e-commerce",
                "analysis_type": "feasibility"
            }
        )
        
        # 2. Search agent finds relevant patterns
        pattern_task = await integrated_system.submit_task(
            task_type="pattern_matching",
            description="Find microservices patterns",
            input_data={
                "target_pattern": "microservices e-commerce architecture",
                "search_space": ["pattern1", "pattern2", "pattern3"]
            }
        )
        
        # 3. Code agent generates implementation
        implementation_task = await integrated_system.submit_task(
            task_type="code_generation",
            description="Generate microservice code",
            input_data={
                "description": "Create user service microservice",
                "language": "python",
                "requirements": ["FastAPI", "async", "database"]
            }
        )
        
        # 4. Test agent creates tests
        testing_task = await integrated_system.submit_task(
            task_type="test_generation",
            description="Generate microservice tests",
            input_data={
                "code": "class UserService: pass",
                "test_type": "integration",
                "language": "python"
            }
        )
        
        # Wait for all tasks to complete
        await asyncio.sleep(0.8)
        
        # Verify all tasks completed successfully
        tasks = [analysis_task, pattern_task, implementation_task, testing_task]
        expected_agents = ["reasoning_agent", "search_agent", "code_agent", "test_agent"]
        
        for task_id, expected_agent in zip(tasks, expected_agents):
            status = integrated_system.get_task_status(task_id)
            assert status["status"] == "completed"
            assert status["assigned_agent"] == expected_agent
    
    @pytest.mark.asyncio
    async def test_task_dependency_handling(self, integrated_system):
        """Test handling of task dependencies (conceptual)."""
        
        # In a real implementation, tasks could have dependencies
        # For now, we test sequential task processing
        
        # Submit tasks in sequence that would logically depend on each other
        design_task = await integrated_system.submit_task(
            task_type="strategy_planning",
            description="Plan API design",
            input_data={
                "goal": "Design REST API for user management",
                "constraints": ["RESTful", "secure", "scalable"],
                "resources": ["Python", "FastAPI", "PostgreSQL"]
            }
        )
        
        await asyncio.sleep(0.2)  # Wait for design to complete
        
        implementation_task = await integrated_system.submit_task(
            task_type="code_generation",
            description="Implement API based on design",
            input_data={
                "description": "Implement user management API",
                "language": "python",
                "requirements": ["FastAPI", "Pydantic", "SQLAlchemy"]
            }
        )
        
        await asyncio.sleep(0.2)  # Wait for implementation
        
        test_task = await integrated_system.submit_task(
            task_type="test_generation",
            description="Test the implemented API",
            input_data={
                "code": "class UserAPI: pass",
                "test_type": "integration",
                "language": "python"
            }
        )
        
        await asyncio.sleep(0.3)  # Wait for testing
        
        # Verify all tasks completed in order
        design_status = integrated_system.get_task_status(design_task)
        impl_status = integrated_system.get_task_status(implementation_task)
        test_status = integrated_system.get_task_status(test_task)
        
        assert design_status["status"] == "completed"
        assert impl_status["status"] == "completed"
        assert test_status["status"] == "completed"
        
        # Verify correct agent assignment
        assert design_status["assigned_agent"] == "reasoning_agent"
        assert impl_status["assigned_agent"] == "code_agent"
        assert test_status["assigned_agent"] == "test_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])