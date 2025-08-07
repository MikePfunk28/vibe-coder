"""
Tests for LangChain Orchestrator
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from langchain_orchestrator import (
    WorkflowManager, TaskType, ModelType, WorkflowStep, Workflow,
    ToolChain, AIIDECallbackHandler, get_workflow_manager
)
from tool_chain_builder import ToolChainBuilder, ToolDefinition, ToolType as BuilderToolType
from context_router import ContextRouter, RoutingContext, ModelCapability, TaskComplexity

class TestWorkflowManager:
    """Test cases for WorkflowManager"""
    
    @pytest.fixture
    async def workflow_manager(self):
        """Create a workflow manager for testing"""
        manager = WorkflowManager()
        # Mock the initialization to avoid external dependencies
        with patch.object(manager, 'initialize', new_callable=AsyncMock):
            await manager.initialize()
        return manager
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing"""
        return {
            "prompt": "Generate a Python function to calculate fibonacci numbers",
            "language": "python",
            "max_tokens": 2048,
            "temperature": 0.3
        }
    
    def test_workflow_manager_initialization(self):
        """Test WorkflowManager initialization"""
        manager = WorkflowManager()
        
        assert manager.workflows == {}
        assert manager.active_workflows == {}
        assert isinstance(manager.callback_handler, AIIDECallbackHandler)
        assert len(manager.tools) > 0
        
        # Check default tools are registered
        expected_tools = [
            "code_generation", "semantic_search", "web_search",
            "reasoning", "multi_agent", "react", "file_operation", "analysis"
        ]
        for tool in expected_tools:
            assert tool in manager.tools
    
    def test_create_code_generation_workflow(self, workflow_manager, sample_context):
        """Test creating a code generation workflow"""
        workflow = workflow_manager.create_workflow(TaskType.CODE_GENERATION, sample_context)
        
        assert workflow.id.startswith("workflow_")
        assert workflow.name == "Code Generation Workflow"
        assert len(workflow.steps) == 3
        assert workflow.context == sample_context
        
        # Check step sequence
        step_names = [step.name for step in workflow.steps]
        expected_names = ["Analyze Code Request", "Generate Code", "Validate Generated Code"]
        assert step_names == expected_names
        
        # Check dependencies
        assert workflow.steps[0].dependencies == []
        assert workflow.steps[1].dependencies == ["analyze_request"]
        assert workflow.steps[2].dependencies == ["generate_code"]
    
    def test_create_semantic_search_workflow(self, workflow_manager, sample_context):
        """Test creating a semantic search workflow"""
        search_context = {**sample_context, "query": "find similar functions"}
        workflow = workflow_manager.create_workflow(TaskType.SEMANTIC_SEARCH, search_context)
        
        assert workflow.name == "Semantic Search Workflow"
        assert len(workflow.steps) == 2
        
        step_tools = [step.tool for step in workflow.steps]
        assert "semantic_search" in step_tools
        assert "analysis" in step_tools
    
    def test_create_multi_step_workflow(self, workflow_manager, sample_context):
        """Test creating a complex multi-step workflow"""
        workflow = workflow_manager.create_workflow(TaskType.MULTI_STEP, sample_context)
        
        assert workflow.name == "Multi-Step Complex Workflow"
        assert len(workflow.steps) == 5
        
        # Check that steps have proper dependencies
        step_ids = [step.id for step in workflow.steps]
        assert "analyze_task" in step_ids
        assert "search_context" in step_ids
        assert "web_research" in step_ids
        assert "reasoning" in step_ids
        assert "generate_solution" in step_ids
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow_manager, sample_context):
        """Test successful workflow execution"""
        # Mock tool execution methods
        workflow_manager._execute_code_generation = AsyncMock(return_value={
            "success": True, "code": "def fibonacci(n): return n", "confidence": 0.9
        })
        workflow_manager._execute_analysis = AsyncMock(return_value={
            "success": True, "analysis": "Code generation request analyzed"
        })
        
        workflow = workflow_manager.create_workflow(TaskType.CODE_GENERATION, sample_context)
        result = await workflow_manager.execute_workflow(workflow)
        
        assert result["success"] is True
        assert "workflow_id" in result
        assert "results" in result
        assert "execution_time" in result
        assert workflow.status == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_workflow_failure(self, workflow_manager, sample_context):
        """Test workflow execution with failure"""
        # Mock tool execution to raise an exception
        workflow_manager._execute_analysis = AsyncMock(side_effect=Exception("Tool failed"))
        
        workflow = workflow_manager.create_workflow(TaskType.CODE_GENERATION, sample_context)
        result = await workflow_manager.execute_workflow(workflow)
        
        assert result["success"] is False
        assert "error" in result
        assert workflow.status == "failed"
    
    @pytest.mark.asyncio
    async def test_execute_code_generation_tool(self, workflow_manager):
        """Test code generation tool execution"""
        # Mock Qwen Coder agent
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.code = "def test(): pass"
        mock_response.language = "python"
        mock_response.confidence = 0.9
        mock_response.explanation = "Generated test function"
        mock_response.execution_time = 2.5
        
        mock_agent.generate_code = AsyncMock(return_value=mock_response)
        workflow_manager.qwen_coder_agent = mock_agent
        
        inputs = {
            "prompt": "Generate a test function",
            "language": "python",
            "max_tokens": 1024,
            "temperature": 0.3
        }
        
        result = await workflow_manager._execute_code_generation(inputs)
        
        assert result["success"] is True
        assert result["code"] == "def test(): pass"
        assert result["language"] == "python"
        assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_execute_semantic_search_tool(self, workflow_manager):
        """Test semantic search tool execution"""
        # Mock semantic search engine
        mock_search = Mock()
        mock_search.search_async = AsyncMock(return_value=[
            {"file": "test.py", "line": 10, "content": "def test():", "score": 0.9}
        ])
        workflow_manager.semantic_search = mock_search
        
        inputs = {"query": "test function", "max_results": 5}
        result = await workflow_manager._execute_semantic_search(inputs)
        
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["query"] == "test function"
    
    def test_get_workflow_status(self, workflow_manager, sample_context):
        """Test getting workflow status"""
        workflow = workflow_manager.create_workflow(TaskType.CODE_GENERATION, sample_context)
        status = workflow_manager.get_workflow_status(workflow.id)
        
        assert status is not None
        assert status["id"] == workflow.id
        assert status["name"] == workflow.name
        assert status["status"] == "created"
        assert status["steps_total"] == 3
        assert status["steps_completed"] == 0
    
    def test_list_workflows(self, workflow_manager, sample_context):
        """Test listing workflows"""
        workflow1 = workflow_manager.create_workflow(TaskType.CODE_GENERATION, sample_context)
        workflow2 = workflow_manager.create_workflow(TaskType.SEMANTIC_SEARCH, sample_context)
        
        workflows = workflow_manager.list_workflows()
        
        assert len(workflows) == 2
        workflow_ids = [w["id"] for w in workflows]
        assert workflow1.id in workflow_ids
        assert workflow2.id in workflow_ids

class TestAIIDECallbackHandler:
    """Test cases for AIIDECallbackHandler"""
    
    def test_callback_handler_initialization(self):
        """Test callback handler initialization"""
        handler = AIIDECallbackHandler()
        
        assert handler.events == []
        assert handler.current_step is None
    
    def test_on_chain_start(self):
        """Test chain start callback"""
        handler = AIIDECallbackHandler()
        serialized = {"name": "test_chain"}
        inputs = {"input": "test"}
        
        handler.on_chain_start(serialized, inputs)
        
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event["type"] == "chain_start"
        assert event["inputs"] == inputs
        assert event["serialized"] == serialized
    
    def test_on_chain_end(self):
        """Test chain end callback"""
        handler = AIIDECallbackHandler()
        outputs = {"output": "result"}
        
        handler.on_chain_end(outputs)
        
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event["type"] == "chain_end"
        assert event["outputs"] == outputs
    
    def test_on_chain_error(self):
        """Test chain error callback"""
        handler = AIIDECallbackHandler()
        error = Exception("Test error")
        
        handler.on_chain_error(error)
        
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event["type"] == "chain_error"
        assert event["error"] == "Test error"
    
    def test_on_tool_start(self):
        """Test tool start callback"""
        handler = AIIDECallbackHandler()
        serialized = {"name": "test_tool"}
        input_str = "test input"
        
        handler.on_tool_start(serialized, input_str)
        
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event["type"] == "tool_start"
        assert event["tool"] == "test_tool"
        assert event["input"] == input_str

class TestIntegration:
    """Integration tests for the complete orchestration system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_code_generation(self):
        """Test end-to-end code generation workflow"""
        # This test would require actual model integrations
        # For now, we'll test the workflow structure
        
        manager = WorkflowManager()
        context = {
            "prompt": "Create a Python function to sort a list",
            "language": "python"
        }
        
        workflow = manager.create_workflow(TaskType.CODE_GENERATION, context)
        
        # Verify workflow structure
        assert len(workflow.steps) == 3
        assert workflow.steps[0].tool == "analysis"
        assert workflow.steps[1].tool == "code_generation"
        assert workflow.steps[2].tool == "analysis"
        
        # Verify dependencies
        assert workflow.steps[1].dependencies == ["analyze_request"]
        assert workflow.steps[2].dependencies == ["generate_code"]
    
    @pytest.mark.asyncio
    async def test_workflow_with_tool_chain_builder(self):
        """Test integration with tool chain builder"""
        manager = WorkflowManager()
        builder = ToolChainBuilder()
        
        # Create a tool chain
        tool_names = ["semantic_code_search", "qwen_coder_generation", "code_validator"]
        context = {"prompt": "Generate code", "language": "python"}
        
        chain = builder.build_chain(tool_names, context)
        
        # Verify chain structure
        assert len(chain.steps) == 3
        assert chain.total_estimated_time > 0
        
        # The workflow manager should be able to use similar tools
        workflow = manager.create_workflow(TaskType.CODE_GENERATION, context)
        assert len(workflow.steps) > 0

if __name__ == "__main__":
    pytest.main([__file__])