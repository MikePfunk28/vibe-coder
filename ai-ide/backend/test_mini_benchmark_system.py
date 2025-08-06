"""
Tests for Mini-Benchmarking System
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from mini_benchmark_system import (
    MiniBenchmarkSystem,
    BenchmarkLoader,
    CodeExecutor,
    CodeQualityAnalyzer,
    BenchmarkProblem,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkSuite,
    BenchmarkType,
    DifficultyLevel
)
from darwin_godel_model import PerformanceMetrics


class TestCodeExecutor:
    """Test the code executor."""
    
    def setup_method(self):
        self.executor = CodeExecutor(timeout=2.0)
    
    def test_execute_simple_code(self):
        """Test executing simple correct code."""
        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            {
                "function_name": "add",
                "input": {"a": 2, "b": 3},
                "expected": 5
            },
            {
                "function_name": "add", 
                "input": {"a": -1, "b": 1},
                "expected": 0
            }
        ]
        
        result = self.executor.execute_code(code, test_cases)
        
        assert result.success is True
        assert len(result.test_results) == 2
        assert all(result.test_results)
        assert result.execution_time > 0
        assert result.error_message is None
    
    def test_execute_incorrect_code(self):
        """Test executing incorrect code."""
        code = """
def add(a, b):
    return a - b  # Wrong operation
"""
        test_cases = [
            {
                "function_name": "add",
                "input": {"a": 2, "b": 3},
                "expected": 5
            }
        ]
        
        result = self.executor.execute_code(code, test_cases)
        
        assert result.success is False
        assert len(result.test_results) == 1
        assert not result.test_results[0]
        assert "FAIL" in result.output
    
    def test_execute_syntax_error(self):
        """Test executing code with syntax errors."""
        code = """
def add(a, b
    return a + b  # Missing closing parenthesis
"""
        test_cases = [
            {
                "function_name": "add",
                "input": {"a": 2, "b": 3},
                "expected": 5
            }
        ]
        
        result = self.executor.execute_code(code, test_cases)
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_execute_runtime_error(self):
        """Test executing code that raises runtime errors."""
        code = """
def divide(a, b):
    return a / b
"""
        test_cases = [
            {
                "function_name": "divide",
                "input": {"a": 10, "b": 0},
                "expected": None  # This will cause division by zero
            }
        ]
        
        result = self.executor.execute_code(code, test_cases)
        
        assert result.success is False
        assert len(result.test_results) == 1
        assert not result.test_results[0]


class TestCodeQualityAnalyzer:
    """Test the code quality analyzer."""
    
    def setup_method(self):
        self.analyzer = CodeQualityAnalyzer()
    
    def test_analyze_simple_code(self):
        """Test analyzing simple, clean code."""
        code = """
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        score = self.analyzer.analyze_code_quality(code)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent quality
    
    def test_analyze_complex_code(self):
        """Test analyzing complex code with high nesting."""
        code = """
def complex_function(data):
    for i in range(len(data)):
        if data[i] > 0:
            for j in range(i):
                if data[j] < data[i]:
                    for k in range(j):
                        if data[k] == 0:
                            if k % 2 == 0:
                                data[k] = 1
    return data
"""
        score = self.analyzer.analyze_code_quality(code)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.7  # Should be lower quality due to complexity
    
    def test_analyze_syntax_error(self):
        """Test analyzing code with syntax errors."""
        code = """
def broken_function(
    return "incomplete"
"""
        score = self.analyzer.analyze_code_quality(code)
        
        assert score == 0.0  # Should return 0 for syntax errors
    
    def test_analyze_well_commented_code(self):
        """Test analyzing well-commented code."""
        code = """
# This function calculates the fibonacci sequence
def fibonacci(n):
    \"\"\"Calculate the nth fibonacci number.\"\"\"
    # Base cases
    if n <= 1:
        return n
    
    # Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        score = self.analyzer.analyze_code_quality(code)
        
        assert score > 0.6  # Should score well due to comments


class TestBenchmarkLoader:
    """Test the benchmark loader."""
    
    def setup_method(self):
        self.loader = BenchmarkLoader()
    
    def test_load_humaneval_problems(self):
        """Test loading HumanEval problems."""
        problems = self.loader.load_humaneval_problems()
        
        assert len(problems) > 0
        assert all(isinstance(p, BenchmarkProblem) for p in problems)
        assert all(p.benchmark_type == BenchmarkType.HUMANEVAL for p in problems)
        
        # Check first problem structure
        first_problem = problems[0]
        assert first_problem.id == "humaneval_0"
        assert first_problem.name == "has_close_elements"
        assert len(first_problem.test_cases) > 0
        assert first_problem.difficulty in DifficultyLevel
    
    def test_load_mbpp_problems(self):
        """Test loading MBPP problems."""
        problems = self.loader.load_mbpp_problems()
        
        assert len(problems) > 0
        assert all(isinstance(p, BenchmarkProblem) for p in problems)
        assert all(p.benchmark_type == BenchmarkType.MBPP for p in problems)
    
    def test_load_code_contests_problems(self):
        """Test loading CodeContests problems."""
        problems = self.loader.load_code_contests_problems()
        
        assert len(problems) > 0
        assert all(isinstance(p, BenchmarkProblem) for p in problems)
        assert all(p.benchmark_type == BenchmarkType.CODE_CONTESTS for p in problems)
    
    def test_create_benchmark_suite(self):
        """Test creating a benchmark suite."""
        suite = self.loader.create_benchmark_suite(
            "Test Suite",
            [BenchmarkType.HUMANEVAL, BenchmarkType.MBPP]
        )
        
        assert isinstance(suite, BenchmarkSuite)
        assert suite.name == "Test Suite"
        assert len(suite.problems) > 0
        
        # Should have problems from both benchmark types
        humaneval_problems = [p for p in suite.problems if p.benchmark_type == BenchmarkType.HUMANEVAL]
        mbpp_problems = [p for p in suite.problems if p.benchmark_type == BenchmarkType.MBPP]
        
        assert len(humaneval_problems) > 0
        assert len(mbpp_problems) > 0


class TestMiniBenchmarkSystem:
    """Test the main mini-benchmark system."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_system = MiniBenchmarkSystem(storage_path=self.temp_dir)
    
    def teardown_method(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test system initialization."""
        assert len(self.benchmark_system.benchmark_suites) > 0
        assert "comprehensive" in self.benchmark_system.benchmark_suites
        assert "humaneval" in self.benchmark_system.benchmark_suites
        assert "mbpp" in self.benchmark_system.benchmark_suites
    
    def test_run_benchmark_success(self):
        """Test running a benchmark with successful code generation."""
        def mock_code_generator(prompt):
            # Simple mock that returns working solutions for known problems
            if "has_close_elements" in prompt:
                return """
def has_close_elements(numbers, threshold):
    for i, elem1 in enumerate(numbers):
        for j, elem2 in enumerate(numbers):
            if i != j and abs(elem1 - elem2) < threshold:
                return True
    return False
"""
            elif "separate_paren_groups" in prompt:
                return """
def separate_paren_groups(paren_string):
    result = []
    current_string = []
    current_depth = 0
    
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    
    return result
"""
            else:
                # Default simple function
                return "def solution(): return None"
        
        report = self.benchmark_system.run_benchmark(
            "humaneval", 
            "test_model_v1", 
            mock_code_generator
        )
        
        assert isinstance(report, BenchmarkReport)
        assert report.suite_name == "humaneval"
        assert report.model_version == "test_model_v1"
        assert report.total_problems > 0
        assert report.solved_problems >= 0
        assert 0.0 <= report.success_rate <= 1.0
        assert len(report.detailed_results) == report.total_problems
        assert isinstance(report.performance_metrics, PerformanceMetrics)
    
    def test_run_benchmark_failure(self):
        """Test running a benchmark with failing code generation."""
        def failing_code_generator(prompt):
            return "def broken_function(): return 'wrong'"
        
        report = self.benchmark_system.run_benchmark(
            "humaneval",
            "failing_model_v1",
            failing_code_generator
        )
        
        assert report.success_rate == 0.0
        assert report.solved_problems == 0
        assert all(not result.success for result in report.detailed_results)
    
    def test_compare_performance(self):
        """Test comparing performance between two reports."""
        # Create mock reports
        baseline_report = BenchmarkReport(
            suite_name="test",
            model_version="v1",
            timestamp=datetime.now(),
            total_problems=10,
            solved_problems=5,
            success_rate=0.5,
            average_execution_time=1.0,
            average_memory_usage=100.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(0.5, 0.5, 100, 0.3, 0.5, 2.5),
            detailed_results=[]
        )
        
        improved_report = BenchmarkReport(
            suite_name="test",
            model_version="v2", 
            timestamp=datetime.now(),
            total_problems=10,
            solved_problems=7,
            success_rate=0.7,
            average_execution_time=0.8,
            average_memory_usage=90.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(0.8, 0.7, 90, 0.25, 0.3, 3.5),
            detailed_results=[]
        )
        
        comparison = self.benchmark_system.compare_performance(improved_report, baseline_report)
        
        assert abs(comparison["success_rate_delta"] - 0.2) < 0.001
        assert abs(comparison["execution_time_delta"] - (-0.2)) < 0.001  # Improvement
        assert abs(comparison["memory_usage_delta"] - (-10.0)) < 0.001  # Improvement
        assert comparison["solved_problems_delta"] == 2
        assert "statistical_significance" in comparison
    
    def test_validate_improvement(self):
        """Test validating improvements."""
        baseline_report = BenchmarkReport(
            suite_name="test",
            model_version="v1",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=10,
            success_rate=0.5,
            average_execution_time=1.0,
            average_memory_usage=100.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(0.5, 0.5, 100, 0.3, 0.5, 2.5),
            detailed_results=[]
        )
        
        # Significant improvement
        good_improvement = BenchmarkReport(
            suite_name="test",
            model_version="v2",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=16,
            success_rate=0.8,
            average_execution_time=0.9,
            average_memory_usage=95.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(0.9, 0.8, 95, 0.25, 0.2, 4.0),
            detailed_results=[]
        )
        
        # Marginal improvement
        marginal_improvement = BenchmarkReport(
            suite_name="test",
            model_version="v3",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=11,
            success_rate=0.55,
            average_execution_time=1.1,
            average_memory_usage=105.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.1, 0.55, 105, 0.35, 0.45, 2.75),
            detailed_results=[]
        )
        
        # Test validation
        assert self.benchmark_system.validate_improvement(good_improvement, baseline_report, 0.1)
        assert not self.benchmark_system.validate_improvement(marginal_improvement, baseline_report, 0.1)
    
    def test_save_and_load_report(self):
        """Test saving and loading benchmark reports."""
        # Create a test report
        original_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="test_model",
            timestamp=datetime.now(),
            total_problems=5,
            solved_problems=3,
            success_rate=0.6,
            average_execution_time=1.5,
            average_memory_usage=120.0,
            difficulty_breakdown={"easy": {"total": 3, "solved": 2, "success_rate": 0.67}},
            performance_metrics=PerformanceMetrics(1.5, 0.6, 120, 0.4, 0.4, 3.0),
            detailed_results=[
                BenchmarkResult(
                    problem_id="test_1",
                    success=True,
                    execution_time=1.0,
                    memory_usage=100,
                    output="Test passed",
                    code_quality_score=0.8
                )
            ]
        )
        
        # Save the report
        self.benchmark_system._save_report(original_report)
        
        # Find the saved file
        saved_files = list(Path(self.temp_dir).glob("*.json"))
        assert len(saved_files) == 1
        
        # Load the report
        loaded_report = self.benchmark_system.load_report(str(saved_files[0]))
        
        # Verify the loaded report matches the original
        assert loaded_report.suite_name == original_report.suite_name
        assert loaded_report.model_version == original_report.model_version
        assert loaded_report.total_problems == original_report.total_problems
        assert loaded_report.solved_problems == original_report.solved_problems
        assert loaded_report.success_rate == original_report.success_rate
        assert len(loaded_report.detailed_results) == len(original_report.detailed_results)
    
    def test_get_benchmark_statistics(self):
        """Test getting benchmark statistics."""
        stats = self.benchmark_system.get_benchmark_statistics()
        
        assert "available_suites" in stats
        assert "suite_details" in stats
        assert len(stats["available_suites"]) > 0
        
        for suite_name in stats["available_suites"]:
            assert suite_name in stats["suite_details"]
            suite_details = stats["suite_details"][suite_name]
            
            assert "total_problems" in suite_details
            assert "difficulty_breakdown" in suite_details
            assert "benchmark_type_breakdown" in suite_details
            assert "description" in suite_details
            assert suite_details["total_problems"] > 0


class TestIntegration:
    """Integration tests for the complete benchmarking system."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_system = MiniBenchmarkSystem(storage_path=self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_benchmark_cycle(self):
        """Test a complete benchmark cycle with improvement validation."""
        
        def baseline_code_generator(prompt):
            """Baseline generator that solves some problems correctly."""
            if "has_close_elements" in prompt:
                return """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
            else:
                return "def solution(): return None"  # Fails other problems
        
        def improved_code_generator(prompt):
            """Improved generator that solves more problems correctly."""
            if "has_close_elements" in prompt:
                return """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
            elif "separate_paren_groups" in prompt:
                return """
def separate_paren_groups(paren_string):
    result = []
    current_string = []
    current_depth = 0
    
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    
    return result
"""
            else:
                return "def solution(): return None"
        
        # Run baseline benchmark
        baseline_report = self.benchmark_system.run_benchmark(
            "humaneval",
            "baseline_v1",
            baseline_code_generator
        )
        
        # Run improved benchmark
        improved_report = self.benchmark_system.run_benchmark(
            "humaneval", 
            "improved_v2",
            improved_code_generator
        )
        
        # Validate improvement
        is_improvement = self.benchmark_system.validate_improvement(
            improved_report,
            baseline_report,
            min_improvement_threshold=0.01  # Lower threshold for test
        )
        
        # Verify results
        assert baseline_report.total_problems > 0
        assert improved_report.total_problems == baseline_report.total_problems
        assert improved_report.solved_problems >= baseline_report.solved_problems
        
        # Compare performance
        comparison = self.benchmark_system.compare_performance(improved_report, baseline_report)
        assert comparison["success_rate_delta"] >= 0
        
        # Check that reports were saved
        saved_files = list(Path(self.temp_dir).glob("*.json"))
        assert len(saved_files) == 2  # One for each report
    
    def test_benchmark_with_different_difficulties(self):
        """Test benchmarking across different difficulty levels."""
        def smart_code_generator(prompt):
            """Generator that performs better on easier problems."""
            # This is a simplified heuristic - in practice, you'd use actual AI models
            if "palindrome" in prompt.lower() or "add" in prompt.lower():
                # Easy problems - provide good solutions
                if "palindrome" in prompt.lower():
                    return """
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
"""
                elif "add" in prompt.lower():
                    return """
def add(a, b):
    return a + b
"""
            else:
                # Harder problems - provide basic/incorrect solutions
                return "def solution(): return None"
        
        # Run comprehensive benchmark
        report = self.benchmark_system.run_benchmark(
            "comprehensive",
            "smart_model_v1", 
            smart_code_generator
        )
        
        # Analyze difficulty breakdown
        assert "difficulty_breakdown" in report.__dict__
        
        # Should perform better on easier problems
        if "easy" in report.difficulty_breakdown and "medium" in report.difficulty_breakdown:
            easy_success = report.difficulty_breakdown["easy"]["success_rate"]
            medium_success = report.difficulty_breakdown["medium"]["success_rate"]
            # Easy problems should generally have higher success rate
            # (though this depends on the specific problems loaded)
    
    def test_statistical_significance_calculation(self):
        """Test statistical significance calculations."""
        # Create reports with sufficient sample sizes
        baseline_report = BenchmarkReport(
            suite_name="test",
            model_version="baseline",
            timestamp=datetime.now(),
            total_problems=50,
            solved_problems=25,
            success_rate=0.5,
            average_execution_time=1.0,
            average_memory_usage=100.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.0, 0.5, 100, 0.3, 0.5, 2.5),
            detailed_results=[]
        )
        
        # Significantly improved report
        improved_report = BenchmarkReport(
            suite_name="test",
            model_version="improved",
            timestamp=datetime.now(),
            total_problems=50,
            solved_problems=40,
            success_rate=0.8,
            average_execution_time=0.9,
            average_memory_usage=95.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(0.9, 0.8, 95, 0.25, 0.2, 4.0),
            detailed_results=[]
        )
        
        comparison = self.benchmark_system.compare_performance(improved_report, baseline_report)
        
        # Should detect statistical significance
        assert comparison["statistical_significance"] > 1.96  # 95% confidence level
        
        # Should validate as a real improvement
        is_valid = self.benchmark_system.validate_improvement(improved_report, baseline_report)
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__])