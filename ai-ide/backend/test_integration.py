#!/usr/bin/env python3
"""
Comprehensive test suite for PocketFlow integration
Tests the ported nodes and cross-language communication
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pocketflow_integration import (
    create_ai_ide_flow, AIIDEFlow, AIIDESharedMemory,
    EnhancedMainDecisionAgent, SemanticRouter, CrossLanguageErrorHandler,
    ReadFileActionNode, GrepSearchActionNode, ListDirActionNode,
    DeleteFileActionNode, EditFileActionNode, SemanticSearchNode,
    ReasoningNode, error_handler
)

class TestPocketFlowIntegration(unittest.TestCase):
    """Test suite for PocketFlow integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.flow = create_ai_ide_flow(self.test_dir)
        
        # Create test files
        self.test_file = os.path.join(self.test_dir, "test.py")
        with open(self.test_file, 'w') as f:
            f.write("""def hello_world():
    print("Hello, World!")
    return "Hello"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_shared_memory_initialization(self):
        """Test AIIDESharedMemory initialization"""
        shared = AIIDESharedMemory(self.test_dir)
        
        self.assertEqual(shared.get("working_dir"), self.test_dir)
        self.assertEqual(shared.get("history"), [])
        self.assertIsInstance(shared.get("semantic_context"), dict)
        self.assertIsInstance(shared.get("reasoning_trace"), list)
    
    def test_semantic_router(self):
        """Test semantic routing functionality"""
        router = SemanticRouter()
        
        # Test code generation routing
        self.assertEqual(router.route_task("write a function"), "code_generation")
        self.assertEqual(router.route_task("create a class"), "code_generation")
        self.assertEqual(router.route_task("implement sorting"), "code_generation")
        
        # Test search routing
        self.assertEqual(router.route_task("find all functions"), "semantic_search")
        self.assertEqual(router.route_task("search for imports"), "semantic_search")
        
        # Test reasoning routing
        self.assertEqual(router.route_task("explain this algorithm"), "reasoning")
        self.assertEqual(router.route_task("think about this problem"), "reasoning")
    
    def test_error_handler(self):
        """Test cross-language error handling"""
        handler = CrossLanguageErrorHandler()
        
        # Test error logging
        test_error = ValueError("Test error")
        handler.log_error(test_error, "test context")
        
        summary = handler.get_error_summary()
        self.assertIn("ValueError", summary["error_counts"])
        self.assertEqual(summary["error_counts"]["ValueError"], 1)
        self.assertEqual(len(summary["recent_errors"]), 1)
        
        # Test retry logic
        self.assertFalse(handler.should_retry(ValueError("test")))
        self.assertTrue(handler.should_retry(ConnectionError("test")))
    
    @patch('pocketflow_integration.read_file')
    def test_read_file_node(self, mock_read_file):
        """Test ReadFileActionNode"""
        mock_read_file.return_value = ("test content", True)
        
        shared = AIIDESharedMemory(self.test_dir)
        shared.set("history", [{
            "tool": "read_file",
            "reason": "Test read",
            "params": {"target_file": "test.py"},
            "result": None
        }])
        
        node = ReadFileActionNode()
        prep_result = node.prep(shared)
        exec_result = node.exec(prep_result)
        next_node = node.post(shared, prep_result, exec_result)
        
        self.assertEqual(next_node, "decide_next")
        history = shared.get("history")
        self.assertTrue(history[0]["result"]["success"])
        self.assertEqual(history[0]["result"]["content"], "test content")
    
    @patch('pocketflow_integration.grep_search')
    def test_grep_search_node(self, mock_grep_search):
        """Test GrepSearchActionNode"""
        mock_grep_search.return_value = (True, [
            {"file": "test.py", "line": 1, "content": "def hello_world():"}
        ])
        
        shared = AIIDESharedMemory(self.test_dir)
        shared.set("history", [{
            "tool": "grep_search",
            "reason": "Test search",
            "params": {"query": "hello", "case_sensitive": False},
            "result": None
        }])
        
        node = GrepSearchActionNode()
        prep_result = node.prep(shared)
        exec_result = node.exec(prep_result)
        next_node = node.post(shared, prep_result, exec_result)
        
        self.assertEqual(next_node, "decide_next")
        history = shared.get("history")
        self.assertTrue(history[0]["result"]["success"])
        self.assertEqual(len(history[0]["result"]["matches"]), 1)
    
    @patch('pocketflow_integration.list_dir')
    def test_list_dir_node(self, mock_list_dir):
        """Test ListDirActionNode"""
        mock_list_dir.return_value = (True, "test/\n  test.py")
        
        shared = AIIDESharedMemory(self.test_dir)
        shared.set("history", [{
            "tool": "list_dir",
            "reason": "Test listing",
            "params": {"relative_workspace_path": "."},
            "result": None
        }])
        
        node = ListDirActionNode()
        prep_result = node.prep(shared)
        exec_result = node.exec(prep_result)
        next_node = node.post(shared, prep_result, exec_result)
        
        self.assertEqual(next_node, "decide_next")
        history = shared.get("history")
        self.assertTrue(history[0]["result"]["success"])
        self.assertIn("test.py", history[0]["result"]["tree_visualization"])
    
    @patch('pocketflow_integration.call_llm')
    @patch('pocketflow_integration.replace_file')
    @patch('pocketflow_integration.read_file')
    def test_edit_file_node(self, mock_read_file, mock_replace_file, mock_call_llm):
        """Test EditFileActionNode"""
        mock_read_file.return_value = ("original content", True)
        mock_replace_file.return_value = (True, "File updated")
        mock_call_llm.return_value = "updated content"
        
        shared = AIIDESharedMemory(self.test_dir)
        shared.set("history", [{
            "tool": "edit_file",
            "reason": "Test edit",
            "params": {
                "target_file": "test.py",
                "instructions": "Add a comment",
                "code_edit": "# Comment\noriginal content"
            },
            "result": None
        }])
        
        node = EditFileActionNode()
        prep_result = node.prep(shared)
        exec_result = node.exec(prep_result)
        next_node = node.post(shared, prep_result, exec_result)
        
        self.assertEqual(next_node, "decide_next")
        history = shared.get("history")
        self.assertTrue(history[0]["result"]["success"])
        self.assertEqual(history[0]["result"]["operations"], 1)
    
    @patch('pocketflow_integration.call_llm')
    def test_enhanced_main_decision_agent(self, mock_call_llm):
        """Test EnhancedMainDecisionAgent"""
        mock_call_llm.return_value = """```yaml
tool: read_file
reason: |
  User wants to read a file
params:
  target_file: test.py
```"""
        
        shared = AIIDESharedMemory(self.test_dir)
        shared.set("user_query", "read test.py file")
        
        agent = EnhancedMainDecisionAgent()
        prep_result = agent.prep(shared)
        exec_result = agent.exec(prep_result)
        next_node = agent.post(shared, prep_result, exec_result)
        
        self.assertEqual(exec_result["tool"], "read_file")
        self.assertEqual(exec_result["params"]["target_file"], "test.py")
        self.assertEqual(next_node, "read_file")
    
    def test_flow_initialization(self):
        """Test AIIDEFlow initialization"""
        flow = AIIDEFlow(self.test_dir)
        
        self.assertIsInstance(flow.shared, AIIDESharedMemory)
        self.assertEqual(flow.shared.get("working_dir"), self.test_dir)
        self.assertIn("main_decision", flow.nodes)
        self.assertIn("read_file", flow.nodes)
        self.assertIn("edit_file", flow.nodes)
        self.assertIn("semantic_search", flow.nodes)
    
    def test_dynamic_flow_generation(self):
        """Test dynamic flow generation"""
        flow = AIIDEFlow(self.test_dir)
        
        # Test different task types
        code_flow = flow.generate_dynamic_flow("code_generation", "medium")
        self.assertIn("main_decision", code_flow)
        self.assertIn("edit_file", code_flow)
        
        search_flow = flow.generate_dynamic_flow("semantic_search", "low")
        self.assertIn("main_decision", search_flow)
        self.assertIn("semantic_search", search_flow)
        
        refactor_flow = flow.generate_dynamic_flow("refactoring", "high")
        self.assertIn("read_file", refactor_flow)
        self.assertIn("reasoning_task", refactor_flow)
        self.assertIn("edit_file", refactor_flow)
    
    @patch('pocketflow_integration.call_llm')
    def test_task_execution_code_generation(self, mock_call_llm):
        """Test complete task execution for code generation"""
        # Mock LLM responses
        mock_call_llm.side_effect = [
            # Main decision response
            """```yaml
tool: edit_file
reason: |
  User wants to generate code
params:
  target_file: new_code.py
  instructions: Create a hello world function
  code_edit: |
    def hello():
        print("Hello, World!")
        return "Hello"
```""",
            # Edit file response
            """def hello():
    print("Hello, World!")
    return "Hello"
"""
        ]
        
        task = {
            "id": "test_001",
            "type": "code_generation",
            "input": {"prompt": "create a hello world function"},
            "context": {"working_dir": self.test_dir}
        }
        
        with patch('pocketflow_integration.replace_file') as mock_replace:
            mock_replace.return_value = (True, "File created")
            
            result = self.flow.execute_task(task)
            
            self.assertTrue(result["success"])
            self.assertIn("code", result["result"])
            self.assertGreater(len(result["history"]), 0)
            self.assertEqual(result["task_id"], "test_001")
    
    @patch('pocketflow_integration.get_semantic_index')
    def test_task_execution_semantic_search(self, mock_get_semantic_index):
        """Test complete task execution for semantic search"""
        # Mock semantic index
        mock_index = MagicMock()
        mock_index.search_semantic.return_value = [
            {
                "file": "test.py",
                "score": 8.5,
                "context": MagicMock(language="python"),
                "matches": [
                    {"line": 1, "content": "def hello_world():", "type": "function"}
                ]
            }
        ]
        mock_get_semantic_index.return_value = mock_index
        
        task = {
            "id": "test_002",
            "type": "semantic_search",
            "input": {"query": "hello function"},
            "context": {"working_dir": self.test_dir}
        }
        
        with patch('pocketflow_integration.call_llm') as mock_call_llm:
            mock_call_llm.return_value = """```yaml
tool: semantic_search
reason: |
  User wants to search for code
params:
  query: hello function
  max_results: 10
```"""
            
            result = self.flow.execute_task(task)
            
            self.assertTrue(result["success"])
            self.assertIn("matches", result["result"])
            self.assertGreater(len(result["result"]["matches"]), 0)
    
    def test_task_execution_error_handling(self):
        """Test error handling in task execution"""
        task = {
            "id": "test_error",
            "type": "invalid_type",
            "input": {"prompt": "test"},
            "context": {}
        }
        
        # This should not crash but handle the error gracefully
        result = self.flow.execute_task(task)
        
        # Should still return a result structure
        self.assertIn("success", result)
        self.assertIn("task_id", result)
        self.assertEqual(result["task_id"], "test_error")
    
    def test_complexity_assessment(self):
        """Test task complexity assessment"""
        flow = AIIDEFlow(self.test_dir)
        
        # High complexity
        high_query = "design a complex architecture with multiple patterns and optimization"
        self.assertEqual(flow._assess_task_complexity(high_query), "high")
        
        # Low complexity
        low_query = "fix a simple typo in the comment"
        self.assertEqual(flow._assess_task_complexity(low_query), "low")
        
        # Medium complexity
        medium_query = "create a function to process data"
        self.assertEqual(flow._assess_task_complexity(medium_query), "medium")

class TestCrossLanguageCommunication(unittest.TestCase):
    """Test cross-language communication features"""
    
    def test_json_serialization(self):
        """Test JSON serialization of results"""
        result = {
            "success": True,
            "result": {"code": "print('hello')", "confidence": 0.95},
            "history": [{"tool": "edit_file", "success": True}],
            "metrics": {"execution_time": 1.5}
        }
        
        # Should be serializable to JSON
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["success"], True)
        self.assertEqual(parsed["result"]["code"], "print('hello')")
    
    def test_error_response_format(self):
        """Test error response format for TypeScript bridge"""
        flow = create_ai_ide_flow()
        
        # Create a task that will fail by causing an exception in execute_task itself
        task = {
            "id": "error_test",
            "type": "code_generation",
            "input": {"prompt": "test"},
            "context": {}
        }
        
        # Mock shared memory to raise an exception during setup
        with patch.object(flow.shared, 'set', side_effect=Exception("Critical error")):
            result = flow.execute_task(task)
            
            # Should have proper error structure
            self.assertIn("success", result)
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("task_id", result)
            self.assertEqual(result["task_id"], "error_test")

def run_integration_tests():
    """Run all integration tests"""
    print("Running PocketFlow Integration Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPocketFlowIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossLanguageCommunication))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)