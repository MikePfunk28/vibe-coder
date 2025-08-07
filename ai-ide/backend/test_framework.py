"""
Comprehensive Testing Framework for AI IDE
Provides advanced unit and integration testing capabilities for all AI components.
"""

import pytest
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Represents the result of a test execution"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
@dataclass
class TestSuite:
    """Represents a collection of related tests"""
    name: str
    tests: List[Callable]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None

class AITestFramework:
    """
    Main testing framework for AI IDE components
    Provides comprehensive testing capabilities for all AI systems
    """
    
    def __init__(self):
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        self.mock_services = {}
        self.temp_dirs = []
        
    def register_test_suite(self, suite: TestSuite):
        """Register a test suite with the framework"""
        self.test_suites[suite.name] = suite
        logger.info(f"Registered test suite: {suite.name}")
        
    def create_mock_service(self, service_name: str, mock_config: Dict[str, Any]) -> Mock:
        """Create a mock service for testing"""
        mock_service = Mock()
        
        # Configure mock based on config
        for method_name, return_value in mock_config.items():
            if callable(return_value):
                setattr(mock_service, method_name, return_value)
            else:
                getattr(mock_service, method_name).return_value = return_value
                
        self.mock_services[service_name] = mock_service
        return mock_service
        
    def create_temp_directory(self) -> Path:
        """Create a temporary directory for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
        
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
            
        suite = self.test_suites[suite_name]
        results = []
        
        logger.info(f"Running test suite: {suite_name}")
        
        # Run setup if provided
        if suite.setup_func:
            try:
                await self._run_async_or_sync(suite.setup_func)
            except Exception as e:
                logger.error(f"Setup failed for suite {suite_name}: {e}")
                return []
                
        # Run each test
        for test_func in suite.tests:
            result = await self._run_single_test(test_func)
            results.append(result)
            self.test_results.append(result)
            
        # Run teardown if provided
        if suite.teardown_func:
            try:
                await self._run_async_or_sync(suite.teardown_func)
            except Exception as e:
                logger.error(f"Teardown failed for suite {suite_name}: {e}")
                
        return results
        
    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all registered test suites"""
        all_results = {}
        
        for suite_name in self.test_suites:
            results = await self.run_test_suite(suite_name)
            all_results[suite_name] = results
            
        return all_results
        
    async def _run_single_test(self, test_func: Callable) -> TestResult:
        """Run a single test function"""
        test_name = test_func.__name__
        start_time = time.time()
        
        try:
            await self._run_async_or_sync(test_func)
            execution_time = time.time() - start_time
            
            result = TestResult(
                test_name=test_name,
                passed=True,
                execution_time=execution_time
            )
            logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            logger.error(f"✗ {test_name} failed ({execution_time:.3f}s): {e}")
            
        return result
        
    async def _run_async_or_sync(self, func: Callable):
        """Run a function whether it's async or sync"""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return func()
            
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        avg_execution_time = np.mean([r.execution_time for r in self.test_results]) if self.test_results else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_execution_time": avg_execution_time
            },
            "test_results": [asdict(result) for result in self.test_results],
            "failed_tests": [
                asdict(result) for result in self.test_results 
                if not result.passed
            ]
        }
        
        return report
        
    def save_test_report(self, filepath: str):
        """Save test report to file"""
        report = self.generate_test_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
    def cleanup(self):
        """Clean up temporary resources"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
        self.mock_services.clear()

# Global test framework instance
test_framework = AITestFramework()

# Decorators for test registration
def test_suite(name: str, setup_func: Optional[Callable] = None, teardown_func: Optional[Callable] = None):
    """Decorator to register a test suite"""
    def decorator(cls):
        # Extract test methods from class
        test_methods = [getattr(cls, method) for method in dir(cls) if method.startswith('test_')]
        
        suite = TestSuite(
            name=name,
            tests=test_methods,
            setup_func=setup_func,
            teardown_func=teardown_func
        )
        
        test_framework.register_test_suite(suite)
        return cls
    return decorator

def async_test(func):
    """Decorator to mark a test as async"""
    func._is_async_test = True
    return func

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_mock_ai_response(content: str, confidence: float = 0.9) -> Dict[str, Any]:
        """Create a mock AI response"""
        return {
            "content": content,
            "confidence": confidence,
            "timestamp": time.time(),
            "model": "mock-model",
            "tokens_used": len(content.split())
        }
        
    @staticmethod
    def create_mock_code_context(file_path: str, content: str, cursor_pos: int = 0) -> Dict[str, Any]:
        """Create a mock code context"""
        return {
            "file_path": file_path,
            "content": content,
            "cursor_position": cursor_pos,
            "selected_text": None,
            "project_context": {
                "language": "python",
                "framework": "test"
            }
        }
        
    @staticmethod
    def create_mock_embedding(dimension: int = 1536) -> np.ndarray:
        """Create a mock embedding vector"""
        return np.random.rand(dimension).astype(np.float32)
        
    @staticmethod
    def assert_response_quality(response: Dict[str, Any], min_confidence: float = 0.7):
        """Assert that an AI response meets quality standards"""
        assert "content" in response, "Response must have content"
        assert "confidence" in response, "Response must have confidence score"
        assert response["confidence"] >= min_confidence, f"Confidence {response['confidence']} below threshold {min_confidence}"
        assert len(response["content"]) > 0, "Response content cannot be empty"
        
    @staticmethod
    def assert_semantic_similarity(embedding1: np.ndarray, embedding2: np.ndarray, min_similarity: float = 0.5):
        """Assert that two embeddings have minimum semantic similarity"""
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        assert similarity >= min_similarity, f"Semantic similarity {similarity} below threshold {min_similarity}"

if __name__ == "__main__":
    # Example usage
    async def main():
        # Run all tests
        results = await test_framework.run_all_tests()
        
        # Generate report
        report = test_framework.generate_test_report()
        print(f"Test Results: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
        
        # Cleanup
        test_framework.cleanup()
        
    asyncio.run(main())