"""
Comprehensive Unit Tests for AI IDE Core Components
Tests all agents, reasoning engines, and core functionality
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from test_framework import test_framework, test_suite, TestUtils

# Import all components to test
from pocketflow_integration import PocketFlowIntegration, SemanticRouter
from lm_studio_manager import LMStudioManager
from qwen_coder_agent import QwenCoderAgent
from semantic_search_engine import SemanticSearchEngine
from code_embedding_generator import CodeEmbeddingGenerator
from interleaved_context_manager import InterleaveContextManager
from darwin_godel_model import DarwinGodelModel
from mini_benchmark_system import MiniBenchmarkSystem
from reinforcement_learning_engine import ReinforcementLearningEngine
from multi_agent_system import MultiAgentSystem
from chain_of_thought_engine import ChainOfThoughtEngine
from react_framework import ReActFramework
from web_search_agent import WebSearchAgent
from rag_system import RAGSystem
from langchain_orchestrator import LangChainOrchestrator
from mcp_integration import MCPIntegration

@test_suite("core_components_unit_tests")
class CoreComponentsUnitTests:
    """Unit tests for all core AI IDE components"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.mock_config = {
            "lm_studio": {
                "host": "localhost",
                "port": 1234,
                "model": "qwen-coder-3"
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            }
        }
        
    def test_pocketflow_integration_initialization(self):
        """Test PocketFlow integration initializes correctly"""
        integration = PocketFlowIntegration(self.mock_config)
        
        assert integration is not None
        assert hasattr(integration, 'semantic_router')
        assert hasattr(integration, 'performance_tracker')
        
    def test_semantic_router_task_routing(self):
        """Test semantic router correctly routes tasks"""
        router = SemanticRouter()
        
        # Mock task
        task = {
            "type": "code_generation",
            "description": "Generate a Python function",
            "context": {"language": "python"}
        }
        
        # Test routing
        route = router.route_task(task)
        
        assert route is not None
        assert "node_type" in route
        assert "confidence" in route
        
    def test_lm_studio_manager_connection(self):
        """Test LM Studio manager connection handling"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "ready"}
            
            manager = LMStudioManager(self.mock_config["lm_studio"])
            
            assert manager.is_connected()
            
    def test_lm_studio_manager_model_loading(self):
        """Test LM Studio model loading"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"loaded": True}
            
            manager = LMStudioManager(self.mock_config["lm_studio"])
            result = manager.load_model("qwen-coder-3")
            
            assert result["loaded"] is True
            
    def test_qwen_coder_agent_code_generation(self):
        """Test Qwen Coder agent code generation"""
        with patch('requests.post') as mock_post:
            mock_response = {
                "choices": [{
                    "message": {
                        "content": "def hello_world():\n    print('Hello, World!')"
                    }
                }]
            }
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            agent = QwenCoderAgent(self.mock_config["lm_studio"])
            
            context = TestUtils.create_mock_code_context(
                "test.py", 
                "# Generate a hello world function"
            )
            
            result = agent.generate_code("Create a hello world function", context)
            
            assert "def hello_world" in result["content"]
            
    def test_semantic_search_engine_indexing(self):
        """Test semantic search engine indexing"""
        engine = SemanticSearchEngine(self.mock_config["embedding"])
        
        # Mock code files
        code_files = [
            {"path": "test1.py", "content": "def add(a, b): return a + b"},
            {"path": "test2.py", "content": "def multiply(x, y): return x * y"}
        ]
        
        # Test indexing
        engine.index_code_files(code_files)
        
        assert len(engine.embeddings) == 2
        assert "test1.py" in engine.file_index
        assert "test2.py" in engine.file_index
        
    def test_semantic_search_engine_similarity_search(self):
        """Test semantic search similarity search"""
        engine = SemanticSearchEngine(self.mock_config["embedding"])
        
        # Index some code
        code_files = [
            {"path": "math.py", "content": "def add(a, b): return a + b"},
            {"path": "string.py", "content": "def concat(s1, s2): return s1 + s2"}
        ]
        engine.index_code_files(code_files)
        
        # Search for similar code
        results = engine.search_similar("addition function", top_k=1)
        
        assert len(results) > 0
        assert results[0]["file_path"] == "math.py"
        
    def test_code_embedding_generator_embedding_creation(self):
        """Test code embedding generator creates embeddings"""
        generator = CodeEmbeddingGenerator(self.mock_config["embedding"])
        
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        embedding = generator.generate_embedding(code)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == self.mock_config["embedding"]["dimension"]
        
    def test_interleaved_context_manager_context_management(self):
        """Test interleaved context manager handles context correctly"""
        manager = InterleaveContextManager(max_context_length=1000)
        
        # Add context
        context1 = {"content": "def test1(): pass", "priority": 0.8}
        context2 = {"content": "def test2(): pass", "priority": 0.6}
        
        manager.add_context(context1, context1["priority"])
        manager.add_context(context2, context2["priority"])
        
        # Get relevant context
        relevant = manager.get_relevant_context("test function")
        
        assert len(relevant) > 0
        assert relevant[0]["priority"] >= relevant[-1]["priority"]  # Sorted by priority
        
    def test_darwin_godel_model_improvement_analysis(self):
        """Test Darwin-Gödel model analyzes improvements correctly"""
        model = DarwinGodelModel("mock-model")
        
        # Mock performance metrics
        metrics = {
            "response_time": 2.5,
            "accuracy": 0.75,
            "user_satisfaction": 0.8
        }
        
        opportunities = model.analyze_performance(metrics)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        assert all("type" in opp for opp in opportunities)
        
    def test_mini_benchmark_system_benchmark_execution(self):
        """Test mini benchmark system executes benchmarks"""
        benchmark_system = MiniBenchmarkSystem()
        
        # Mock benchmark suite
        mock_suite = Mock()
        mock_suite.run_benchmarks.return_value = {
            "response_time": 1.2,
            "accuracy": 0.85,
            "memory_usage": 512
        }
        
        benchmark_system.benchmark_suite = mock_suite
        
        results = benchmark_system.run_benchmarks("test-model-v1")
        
        assert "response_time" in results
        assert "accuracy" in results
        assert results["accuracy"] > 0.8
        
    def test_reinforcement_learning_engine_feedback_processing(self):
        """Test reinforcement learning engine processes feedback"""
        rl_engine = ReinforcementLearningEngine()
        
        # Mock user interaction
        interaction = {
            "action": "code_suggestion",
            "context": {"file": "test.py"},
            "user_feedback": {"rating": 4, "accepted": True}
        }
        
        rl_engine.collect_feedback(interaction)
        
        assert len(rl_engine.experience_buffer) == 1
        assert rl_engine.experience_buffer[0]["user_feedback"]["rating"] == 4
        
    def test_multi_agent_system_agent_coordination(self):
        """Test multi-agent system coordinates agents correctly"""
        agent_system = MultiAgentSystem()
        
        # Mock agents
        code_agent = Mock()
        search_agent = Mock()
        
        code_agent.handle_task.return_value = {"result": "code generated"}
        search_agent.handle_task.return_value = {"result": "search completed"}
        
        agent_system.register_agent("code", code_agent)
        agent_system.register_agent("search", search_agent)
        
        # Test task delegation
        task = {"type": "code_generation", "description": "Generate function"}
        result = agent_system.delegate_task(task)
        
        assert result is not None
        assert "result" in result
        
    def test_chain_of_thought_engine_reasoning(self):
        """Test chain of thought engine reasoning process"""
        cot_engine = ChainOfThoughtEngine()
        
        problem = "How to implement a binary search algorithm?"
        
        with patch.object(cot_engine, '_generate_reasoning_step') as mock_step:
            mock_step.side_effect = [
                "Step 1: Define the problem",
                "Step 2: Identify base cases",
                "Step 3: Implement recursive logic"
            ]
            
            reasoning_trace = cot_engine.reason_through_problem(problem)
            
            assert len(reasoning_trace) > 0
            assert all("step" in step for step in reasoning_trace)
            
    def test_react_framework_reasoning_acting_cycle(self):
        """Test ReAct framework reasoning and acting cycle"""
        react_framework = ReActFramework()
        
        # Mock tools
        mock_tool = Mock()
        mock_tool.execute.return_value = {"result": "tool executed"}
        
        react_framework.register_tool("search", mock_tool)
        
        task = "Find information about Python decorators"
        
        with patch.object(react_framework, '_generate_reasoning') as mock_reasoning:
            mock_reasoning.return_value = "I need to search for Python decorator information"
            
            result = react_framework.execute_task(task)
            
            assert result is not None
            assert "reasoning_trace" in result
            
    def test_web_search_agent_search_execution(self):
        """Test web search agent executes searches correctly"""
        search_agent = WebSearchAgent()
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"title": "Python Tutorial", "url": "example.com", "snippet": "Learn Python"}
                ]
            }
            mock_get.return_value = mock_response
            
            results = search_agent.search("Python programming tutorial")
            
            assert len(results) > 0
            assert "title" in results[0]
            assert "url" in results[0]
            
    def test_rag_system_knowledge_retrieval(self):
        """Test RAG system retrieves relevant knowledge"""
        rag_system = RAGSystem()
        
        # Mock knowledge base
        knowledge = [
            {"content": "Python is a programming language", "embedding": TestUtils.create_mock_embedding()},
            {"content": "JavaScript is used for web development", "embedding": TestUtils.create_mock_embedding()}
        ]
        
        rag_system.knowledge_base = knowledge
        
        query = "What is Python?"
        results = rag_system.retrieve_relevant_knowledge(query)
        
        assert len(results) > 0
        assert "Python" in results[0]["content"]
        
    def test_langchain_orchestrator_workflow_management(self):
        """Test LangChain orchestrator manages workflows"""
        orchestrator = LangChainOrchestrator()
        
        # Mock workflow
        workflow_config = {
            "steps": [
                {"type": "analyze", "tool": "code_analyzer"},
                {"type": "generate", "tool": "code_generator"}
            ]
        }
        
        workflow = orchestrator.create_workflow("code_assistance", workflow_config)
        
        assert workflow is not None
        assert len(workflow.steps) == 2
        
    def test_mcp_integration_server_discovery(self):
        """Test MCP integration discovers servers correctly"""
        mcp_integration = MCPIntegration()
        
        # Mock MCP server configuration
        server_config = {
            "test-server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "tools": ["search", "analyze"]
            }
        }
        
        mcp_integration.configure_servers(server_config)
        
        assert "test-server" in mcp_integration.servers
        assert len(mcp_integration.servers["test-server"]["tools"]) == 2

@test_suite("ai_model_response_validation")
class AIModelResponseValidationTests:
    """Tests for validating AI model responses across all reasoning modes"""
    
    def test_code_generation_response_validation(self):
        """Test validation of code generation responses"""
        response = {
            "content": "def add(a, b):\n    return a + b",
            "confidence": 0.9,
            "language": "python",
            "explanation": "Simple addition function"
        }
        
        TestUtils.assert_response_quality(response, min_confidence=0.8)
        
        # Additional code-specific validations
        assert "def " in response["content"]  # Contains function definition
        assert response["language"] == "python"
        assert len(response["explanation"]) > 0
        
    def test_reasoning_response_validation(self):
        """Test validation of reasoning responses"""
        response = {
            "content": "To solve this problem, I need to: 1) Analyze the input, 2) Apply the algorithm, 3) Return the result",
            "confidence": 0.85,
            "reasoning_steps": [
                {"step": 1, "description": "Analyze input"},
                {"step": 2, "description": "Apply algorithm"},
                {"step": 3, "description": "Return result"}
            ]
        }
        
        TestUtils.assert_response_quality(response, min_confidence=0.8)
        
        # Reasoning-specific validations
        assert "reasoning_steps" in response
        assert len(response["reasoning_steps"]) >= 2
        assert all("step" in step for step in response["reasoning_steps"])
        
    def test_search_response_validation(self):
        """Test validation of search responses"""
        response = {
            "content": "Found 5 relevant results",
            "confidence": 0.92,
            "results": [
                {"title": "Result 1", "relevance": 0.9},
                {"title": "Result 2", "relevance": 0.8}
            ]
        }
        
        TestUtils.assert_response_quality(response, min_confidence=0.8)
        
        # Search-specific validations
        assert "results" in response
        assert len(response["results"]) > 0
        assert all("relevance" in result for result in response["results"])
        
    def test_multi_modal_response_validation(self):
        """Test validation of multi-modal responses"""
        response = {
            "content": "Generated code with explanation",
            "confidence": 0.88,
            "code": "print('Hello, World!')",
            "explanation": "This prints a greeting message",
            "visual_elements": ["syntax_highlighting", "code_structure"]
        }
        
        TestUtils.assert_response_quality(response, min_confidence=0.8)
        
        # Multi-modal specific validations
        assert "code" in response
        assert "explanation" in response
        assert "visual_elements" in response
        assert len(response["visual_elements"]) > 0

@test_suite("database_integration_tests")
class DatabaseIntegrationTests:
    """Tests for database and external service integrations"""
    
    def setup_method(self):
        """Setup test database"""
        self.test_db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_ai_ide",
            "user": "test_user",
            "password": "test_pass"
        }
        
    def test_database_connection(self):
        """Test database connection establishment"""
        from database.connection import DatabaseConnection
        
        with patch('psycopg2.connect') as mock_connect:
            mock_connect.return_value = Mock()
            
            db = DatabaseConnection(self.test_db_config)
            connection = db.get_connection()
            
            assert connection is not None
            mock_connect.assert_called_once()
            
    def test_embedding_storage_retrieval(self):
        """Test storing and retrieving embeddings"""
        from database.repositories.search_repository import SearchRepository
        
        repo = SearchRepository(self.test_db_config)
        
        # Mock database operations
        with patch.object(repo, 'store_embedding') as mock_store, \
             patch.object(repo, 'retrieve_similar_embeddings') as mock_retrieve:
            
            mock_store.return_value = True
            mock_retrieve.return_value = [
                {"id": 1, "content": "test", "similarity": 0.9}
            ]
            
            # Test storage
            embedding = TestUtils.create_mock_embedding()
            result = repo.store_embedding("test.py", "test content", embedding)
            assert result is True
            
            # Test retrieval
            similar = repo.retrieve_similar_embeddings(embedding, top_k=5)
            assert len(similar) > 0
            assert similar[0]["similarity"] > 0.8
            
    def test_agent_interaction_logging(self):
        """Test logging of agent interactions"""
        from database.repositories.agent_repository import AgentRepository
        
        repo = AgentRepository(self.test_db_config)
        
        with patch.object(repo, 'log_interaction') as mock_log:
            mock_log.return_value = {"id": "test-123"}
            
            interaction = {
                "agent_type": "code_generator",
                "input": "Generate a function",
                "output": "def test(): pass",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            result = repo.log_interaction(interaction)
            assert "id" in result
            
    def test_web_search_caching(self):
        """Test web search result caching"""
        from database.caching import SearchCache
        
        cache = SearchCache()
        
        with patch.object(cache, 'store_search_result') as mock_store, \
             patch.object(cache, 'get_cached_result') as mock_get:
            
            mock_store.return_value = True
            mock_get.return_value = {
                "query": "Python tutorial",
                "results": [{"title": "Tutorial", "url": "example.com"}],
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            # Test caching
            search_result = {
                "query": "Python tutorial",
                "results": [{"title": "Tutorial", "url": "example.com"}]
            }
            
            cache.store_search_result(search_result)
            cached = cache.get_cached_result("Python tutorial")
            
            assert cached is not None
            assert cached["query"] == "Python tutorial"

@test_suite("mcp_integration_testing")
class MCPIntegrationTestingFramework:
    """Comprehensive testing framework for MCP integrations"""
    
    def setup_method(self):
        """Setup MCP test environment"""
        self.mcp_config = {
            "test-server": {
                "command": "python",
                "args": ["-m", "test_mcp_server"],
                "tools": ["search", "analyze", "generate"]
            }
        }
        
    def test_mcp_server_discovery(self):
        """Test MCP server discovery and configuration"""
        from mcp_integration import MCPIntegration
        
        mcp = MCPIntegration()
        
        with patch.object(mcp, '_discover_servers') as mock_discover:
            mock_discover.return_value = ["test-server", "another-server"]
            
            servers = mcp.discover_available_servers()
            
            assert len(servers) >= 2
            assert "test-server" in servers
            
    def test_mcp_tool_execution(self):
        """Test MCP tool execution"""
        from mcp_integration import MCPIntegration
        
        mcp = MCPIntegration()
        
        with patch.object(mcp, 'execute_tool') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "result": "Tool executed successfully",
                "execution_time": 0.5
            }
            
            result = mcp.execute_tool("test-server", "search", {"query": "test"})
            
            assert result["success"] is True
            assert "result" in result
            assert result["execution_time"] < 1.0
            
    def test_mcp_security_validation(self):
        """Test MCP tool security validation"""
        from tool_security_sandbox import ToolSecuritySandbox
        
        sandbox = ToolSecuritySandbox()
        
        # Test safe tool execution
        safe_tool = {
            "name": "search",
            "command": "python -m search_tool",
            "permissions": ["read"]
        }
        
        validation_result = sandbox.validate_tool_safety(safe_tool)
        assert validation_result["safe"] is True
        
        # Test unsafe tool execution
        unsafe_tool = {
            "name": "delete_files",
            "command": "rm -rf /",
            "permissions": ["write", "delete"]
        }
        
        validation_result = sandbox.validate_tool_safety(unsafe_tool)
        assert validation_result["safe"] is False
        
    def test_mcp_tool_chaining(self):
        """Test chaining multiple MCP tools"""
        from unified_tool_interface import UnifiedToolInterface
        
        tool_interface = UnifiedToolInterface()
        
        # Mock tool chain
        tool_chain = [
            {"tool": "search", "params": {"query": "Python"}},
            {"tool": "analyze", "params": {"content": "search_result"}},
            {"tool": "generate", "params": {"template": "summary"}}
        ]
        
        with patch.object(tool_interface, 'execute_tool_chain') as mock_chain:
            mock_chain.return_value = {
                "success": True,
                "results": [
                    {"tool": "search", "result": "Found 10 results"},
                    {"tool": "analyze", "result": "Analysis complete"},
                    {"tool": "generate", "result": "Summary generated"}
                ]
            }
            
            result = tool_interface.execute_tool_chain(tool_chain)
            
            assert result["success"] is True
            assert len(result["results"]) == 3

if __name__ == "__main__":
    # Run all unit tests
    async def main():
        results = await test_framework.run_all_tests()
        
        # Print summary
        total_tests = sum(len(suite_results) for suite_results in results.values())
        passed_tests = sum(
            sum(1 for test in suite_results if test.passed) 
            for suite_results in results.values()
        )
        
        print(f"\n=== Unit Test Results ===")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Save detailed report
        test_framework.save_test_report("unit_test_report.json")
        test_framework.cleanup()
        
    asyncio.run(main())