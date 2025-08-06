"""
Mini-Benchmarking System for Darwin-Gödel Model

This module implements a comprehensive benchmarking system using established, trusted
coding benchmarks to validate that the Darwin-Gödel model improvements are actually
working and the model is getting better over time.

Based on industry-standard benchmarks:
- HumanEval: Hand-written programming problems
- MBPP (Mostly Basic Python Problems): Crowd-sourced Python programming problems
- CodeContests: Programming contest problems
- APPS: All Programming Problems Subset
- CodeT5 benchmarks: Code understanding and generation tasks
"""

import ast
import json
import logging
import subprocess
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
from darwin_godel_model import PerformanceMetrics


class BenchmarkType(Enum):
    """Types of coding benchmarks."""
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    CODE_CONTESTS = "code_contests"
    APPS = "apps"
    CODE_T5 = "code_t5"
    CUSTOM = "custom"


class DifficultyLevel(Enum):
    """Difficulty levels for benchmark problems."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class BenchmarkProblem:
    """Represents a single benchmark problem."""
    id: str
    name: str
    description: str
    prompt: str
    solution: str
    test_cases: List[Dict[str, Any]]
    difficulty: DifficultyLevel
    benchmark_type: BenchmarkType
    tags: List[str] = field(default_factory=list)
    time_limit: float = 5.0  # seconds
    memory_limit: int = 256  # MB
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark problem."""
    problem_id: str
    success: bool
    execution_time: float
    memory_usage: int
    output: str
    error_message: Optional[str] = None
    test_results: List[bool] = field(default_factory=list)
    code_quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkSuite:
    """A collection of benchmark problems."""
    name: str
    description: str
    problems: List[BenchmarkProblem]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    suite_name: str
    model_version: str
    timestamp: datetime
    total_problems: int
    solved_problems: int
    success_rate: float
    average_execution_time: float
    average_memory_usage: float
    difficulty_breakdown: Dict[str, Dict[str, Any]]
    performance_metrics: PerformanceMetrics
    detailed_results: List[BenchmarkResult]
    statistical_significance: Optional[Dict[str, float]] = None


class CodeExecutor:
    """Safely executes code for benchmarking."""
    
    def __init__(self, timeout: float = 5.0, memory_limit: int = 256):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.logger = logging.getLogger(__name__)
    
    def execute_code(self, code: str, test_cases: List[Dict[str, Any]]) -> BenchmarkResult:
        """Execute code against test cases and return results."""
        start_time = time.time()
        test_results = []
        error_message = None
        output = ""
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # First check if code has syntax errors
            try:
                ast.parse(code)
            except SyntaxError as e:
                error_message = f"Syntax error: {str(e)}"
                return BenchmarkResult(
                    problem_id="",
                    success=False,
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    output="",
                    error_message=error_message,
                    test_results=[]
                )
            
            # Run each test case
            for i, test_case in enumerate(test_cases):
                try:
                    result = self._run_single_test(temp_file, test_case)
                    test_results.append(result)
                    if result:
                        output += f"Test {i+1}: PASS\n"
                    else:
                        output += f"Test {i+1}: FAIL\n"
                except Exception as e:
                    test_results.append(False)
                    output += f"Test {i+1}: ERROR - {str(e)}\n"
                    if error_message is None:
                        error_message = str(e)
            
            execution_time = time.time() - start_time
            success = all(test_results)
            
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
            
            return BenchmarkResult(
                problem_id="",  # Will be set by caller
                success=success,
                execution_time=execution_time,
                memory_usage=0,  # Simplified for now
                output=output,
                error_message=error_message,
                test_results=test_results
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                problem_id="",
                success=False,
                execution_time=execution_time,
                memory_usage=0,
                output="",
                error_message=str(e),
                test_results=[]
            )
    
    def _run_single_test(self, code_file: str, test_case: Dict[str, Any]) -> bool:
        """Run a single test case."""
        try:
            # Read the code file
            with open(code_file, 'r') as f:
                code_content = f.read()
            
            # Create test script with proper escaping
            inputs = test_case.get('input', {})
            expected = test_case.get('expected', None)
            function_name = test_case.get('function_name', '')
            
            test_script = f"""
import sys
import json

# Execute the code
code = {repr(code_content)}
exec(code)

# Run the test
inputs = {repr(inputs)}
expected = {repr(expected)}
function_name = {repr(function_name)}

try:
    if function_name and function_name in globals():
        if isinstance(inputs, dict):
            result = globals()[function_name](**inputs)
        elif isinstance(inputs, list):
            result = globals()[function_name](*inputs)
        else:
            result = globals()[function_name](inputs)
    else:
        result = None
    
    # Compare result
    if result == expected:
        print("PASS")
    else:
        print(f"FAIL: Expected {{expected}}, got {{result}}")
        
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            
            # Execute test script
            result = subprocess.run(
                ['python', '-c', test_script],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return "PASS" in result.stdout
            
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            # Set error message for debugging
            return False


class BenchmarkLoader:
    """Loads benchmark problems from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_humaneval_problems(self) -> List[BenchmarkProblem]:
        """Load HumanEval benchmark problems."""
        # Sample HumanEval problems (in practice, you'd load from the actual dataset)
        problems = [
            BenchmarkProblem(
                id="humaneval_0",
                name="has_close_elements",
                description="Check if in given list of numbers, are any two numbers closer to each other than given threshold.",
                prompt="""def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
""",
                solution="""def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
""",
                test_cases=[
                    {
                        "function_name": "has_close_elements",
                        "input": {"numbers": [1.0, 2.0, 3.0], "threshold": 0.5},
                        "expected": False
                    },
                    {
                        "function_name": "has_close_elements", 
                        "input": {"numbers": [1.0, 2.8, 3.0, 4.0, 5.0, 2.0], "threshold": 0.3},
                        "expected": True
                    }
                ],
                difficulty=DifficultyLevel.EASY,
                benchmark_type=BenchmarkType.HUMANEVAL,
                tags=["list", "comparison", "threshold"]
            ),
            
            BenchmarkProblem(
                id="humaneval_1",
                name="separate_paren_groups",
                description="Input to this function is a string containing multiple groups of nested parentheses.",
                prompt="""def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
""",
                solution="""def separate_paren_groups(paren_string: str) -> List[str]:
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
""",
                test_cases=[
                    {
                        "function_name": "separate_paren_groups",
                        "input": {"paren_string": "( ) (( )) (( )( ))"},
                        "expected": ["()", "(())", "(()())"]
                    }
                ],
                difficulty=DifficultyLevel.MEDIUM,
                benchmark_type=BenchmarkType.HUMANEVAL,
                tags=["string", "parsing", "parentheses"]
            )
        ]
        
        self.logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
    
    def load_mbpp_problems(self) -> List[BenchmarkProblem]:
        """Load MBPP (Mostly Basic Python Problems) benchmark problems."""
        problems = [
            BenchmarkProblem(
                id="mbpp_1",
                name="min_cost",
                description="Find minimum cost to reach the last cell of the matrix from the first cell.",
                prompt="""def min_cost(cost, m, n):
    \"\"\"
    Find minimum cost to reach the last cell of the matrix from the first cell.
    \"\"\"
""",
                solution="""def min_cost(cost, m, n):
    tc = [[0 for x in range(n)] for x in range(m)]
    tc[0][0] = cost[0][0]
    
    for i in range(1, m):
        tc[i][0] = tc[i-1][0] + cost[i][0]
    
    for j in range(1, n):
        tc[0][j] = tc[0][j-1] + cost[0][j]
    
    for i in range(1, m):
        for j in range(1, n):
            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j]
    
    return tc[m-1][n-1]
""",
                test_cases=[
                    {
                        "function_name": "min_cost",
                        "input": {"cost": [[1, 2, 3], [4, 8, 2], [1, 5, 3]], "m": 3, "n": 3},
                        "expected": 8
                    }
                ],
                difficulty=DifficultyLevel.MEDIUM,
                benchmark_type=BenchmarkType.MBPP,
                tags=["dynamic_programming", "matrix", "optimization"]
            ),
            
            BenchmarkProblem(
                id="mbpp_2", 
                name="is_palindrome",
                description="Check if a string is a palindrome.",
                prompt="""def is_palindrome(s):
    \"\"\"
    Check if a string is a palindrome.
    \"\"\"
""",
                solution="""def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""",
                test_cases=[
                    {
                        "function_name": "is_palindrome",
                        "input": {"s": "racecar"},
                        "expected": True
                    },
                    {
                        "function_name": "is_palindrome",
                        "input": {"s": "hello"},
                        "expected": False
                    }
                ],
                difficulty=DifficultyLevel.EASY,
                benchmark_type=BenchmarkType.MBPP,
                tags=["string", "palindrome", "basic"]
            )
        ]
        
        self.logger.info(f"Loaded {len(problems)} MBPP problems")
        return problems
    
    def load_code_contests_problems(self) -> List[BenchmarkProblem]:
        """Load CodeContests benchmark problems."""
        problems = [
            BenchmarkProblem(
                id="codecontests_1",
                name="two_sum",
                description="Find two numbers in array that sum to target.",
                prompt="""def two_sum(nums, target):
    \"\"\"
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    \"\"\"
""",
                solution="""def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
""",
                test_cases=[
                    {
                        "function_name": "two_sum",
                        "input": {"nums": [2, 7, 11, 15], "target": 9},
                        "expected": [0, 1]
                    },
                    {
                        "function_name": "two_sum",
                        "input": {"nums": [3, 2, 4], "target": 6},
                        "expected": [1, 2]
                    }
                ],
                difficulty=DifficultyLevel.EASY,
                benchmark_type=BenchmarkType.CODE_CONTESTS,
                tags=["array", "hash_table", "two_pointers"]
            )
        ]
        
        self.logger.info(f"Loaded {len(problems)} CodeContests problems")
        return problems
    
    def create_benchmark_suite(self, name: str, benchmark_types: List[BenchmarkType]) -> BenchmarkSuite:
        """Create a benchmark suite from specified benchmark types."""
        all_problems = []
        
        for benchmark_type in benchmark_types:
            if benchmark_type == BenchmarkType.HUMANEVAL:
                all_problems.extend(self.load_humaneval_problems())
            elif benchmark_type == BenchmarkType.MBPP:
                all_problems.extend(self.load_mbpp_problems())
            elif benchmark_type == BenchmarkType.CODE_CONTESTS:
                all_problems.extend(self.load_code_contests_problems())
        
        return BenchmarkSuite(
            name=name,
            description=f"Benchmark suite with {len(all_problems)} problems from {len(benchmark_types)} benchmark types",
            problems=all_problems
        )


class CodeQualityAnalyzer:
    """Analyzes code quality metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_code_quality(self, code: str) -> float:
        """Analyze code quality and return a score (0-1)."""
        try:
            tree = ast.parse(code)
            
            # Calculate various metrics
            complexity_score = self._calculate_complexity(tree)
            readability_score = self._calculate_readability(code)
            structure_score = self._calculate_structure(tree)
            
            # Weighted average
            quality_score = (
                complexity_score * 0.4 +
                readability_score * 0.3 +
                structure_score * 0.3
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except SyntaxError:
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error analyzing code quality: {e}")
            return 0.5  # Default score
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity score."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        # Normalize to 0-1 scale (lower complexity is better)
        normalized = max(0.0, 1.0 - (complexity - 1) / 20.0)
        return normalized
    
    def _calculate_readability(self, code: str) -> float:
        """Calculate readability score based on various factors."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # Average line length (shorter is generally better)
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        line_length_score = max(0.0, 1.0 - (avg_line_length - 40) / 80.0)
        
        # Comment ratio
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comment_lines / len(non_empty_lines) if non_empty_lines else 0
        comment_score = min(1.0, comment_ratio * 2)  # Up to 50% comments is good
        
        # Combine scores
        readability = (line_length_score * 0.7 + comment_score * 0.3)
        return max(0.0, min(1.0, readability))
    
    def _calculate_structure(self, tree: ast.AST) -> float:
        """Calculate structural quality score."""
        function_count = 0
        class_count = 0
        max_nesting = 0
        
        def get_nesting_level(node, level=0):
            nonlocal max_nesting
            max_nesting = max(max_nesting, level)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    get_nesting_level(child, level + 1)
                else:
                    get_nesting_level(child, level)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
        
        get_nesting_level(tree)
        
        # Prefer moderate function count and low nesting
        function_score = 1.0 if function_count > 0 else 0.5
        nesting_score = max(0.0, 1.0 - max_nesting / 5.0)
        
        return (function_score * 0.5 + nesting_score * 0.5)


class MiniBenchmarkSystem:
    """Main mini-benchmarking system for validating Darwin-Gödel model improvements."""
    
    def __init__(self, storage_path: str = "benchmark_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.benchmark_loader = BenchmarkLoader()
        self.code_executor = CodeExecutor()
        self.quality_analyzer = CodeQualityAnalyzer()
        
        # Load default benchmark suites
        self.benchmark_suites = {
            "comprehensive": self.benchmark_loader.create_benchmark_suite(
                "Comprehensive Coding Benchmark",
                [BenchmarkType.HUMANEVAL, BenchmarkType.MBPP, BenchmarkType.CODE_CONTESTS]
            ),
            "humaneval": self.benchmark_loader.create_benchmark_suite(
                "HumanEval Benchmark",
                [BenchmarkType.HUMANEVAL]
            ),
            "mbpp": self.benchmark_loader.create_benchmark_suite(
                "MBPP Benchmark", 
                [BenchmarkType.MBPP]
            )
        }
        
        self.logger.info(f"MiniBenchmarkSystem initialized with {len(self.benchmark_suites)} benchmark suites")
    
    def run_benchmark(self, suite_name: str, model_version: str, 
                     code_generator_func: callable) -> BenchmarkReport:
        """Run a benchmark suite and return comprehensive results."""
        if suite_name not in self.benchmark_suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")
        
        suite = self.benchmark_suites[suite_name]
        self.logger.info(f"Running benchmark suite '{suite_name}' with {len(suite.problems)} problems")
        
        start_time = time.time()
        results = []
        
        for problem in suite.problems:
            self.logger.debug(f"Running problem: {problem.name}")
            
            try:
                # Generate code using the provided function
                generated_code = code_generator_func(problem.prompt)
                
                # Execute and test the code
                result = self.code_executor.execute_code(generated_code, problem.test_cases)
                result.problem_id = problem.id
                
                # Analyze code quality
                result.code_quality_score = self.quality_analyzer.analyze_code_quality(generated_code)
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error running problem {problem.id}: {e}")
                results.append(BenchmarkResult(
                    problem_id=problem.id,
                    success=False,
                    execution_time=0.0,
                    memory_usage=0,
                    output="",
                    error_message=str(e),
                    test_results=[]
                ))
        
        # Calculate overall metrics
        total_problems = len(results)
        solved_problems = sum(1 for r in results if r.success)
        success_rate = solved_problems / total_problems if total_problems > 0 else 0.0
        
        avg_execution_time = np.mean([r.execution_time for r in results])
        avg_memory_usage = np.mean([r.memory_usage for r in results])
        
        # Calculate difficulty breakdown
        difficulty_breakdown = {}
        for difficulty in DifficultyLevel:
            difficulty_problems = [p for p in suite.problems if p.difficulty == difficulty]
            difficulty_results = [r for r in results 
                                if any(p.id == r.problem_id and p.difficulty == difficulty 
                                      for p in suite.problems)]
            
            if difficulty_problems:
                solved = sum(1 for r in difficulty_results if r.success)
                difficulty_breakdown[difficulty.value] = {
                    "total": len(difficulty_problems),
                    "solved": solved,
                    "success_rate": solved / len(difficulty_problems)
                }
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            response_time=avg_execution_time,
            accuracy_score=success_rate,
            memory_usage=int(avg_memory_usage),
            cpu_usage=0.0,  # Not measured in this context
            error_rate=1.0 - success_rate,
            user_satisfaction=success_rate * 5.0  # Scale to 1-5
        )
        
        # Create report
        report = BenchmarkReport(
            suite_name=suite_name,
            model_version=model_version,
            timestamp=datetime.now(),
            total_problems=total_problems,
            solved_problems=solved_problems,
            success_rate=success_rate,
            average_execution_time=avg_execution_time,
            average_memory_usage=avg_memory_usage,
            difficulty_breakdown=difficulty_breakdown,
            performance_metrics=performance_metrics,
            detailed_results=results
        )
        
        # Save report
        self._save_report(report)
        
        total_time = time.time() - start_time
        self.logger.info(f"Benchmark completed in {total_time:.2f}s: {solved_problems}/{total_problems} problems solved ({success_rate:.1%})")
        
        return report
    
    def compare_performance(self, current_report: BenchmarkReport, 
                          baseline_report: BenchmarkReport) -> Dict[str, float]:
        """Compare performance between two benchmark reports."""
        comparison = {
            "success_rate_delta": current_report.success_rate - baseline_report.success_rate,
            "execution_time_delta": current_report.average_execution_time - baseline_report.average_execution_time,
            "memory_usage_delta": current_report.average_memory_usage - baseline_report.average_memory_usage,
            "solved_problems_delta": current_report.solved_problems - baseline_report.solved_problems
        }
        
        # Calculate statistical significance (simplified)
        if current_report.total_problems > 10 and baseline_report.total_problems > 10:
            # Simple z-test for proportions
            p1 = current_report.success_rate
            p2 = baseline_report.success_rate
            n1 = current_report.total_problems
            n2 = baseline_report.total_problems
            
            pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
            
            if se > 0:
                z_score = (p1 - p2) / se
                comparison["statistical_significance"] = abs(z_score)
            else:
                comparison["statistical_significance"] = 0.0
        else:
            comparison["statistical_significance"] = 0.0
        
        return comparison
    
    def validate_improvement(self, improvement_report: BenchmarkReport, 
                           baseline_report: BenchmarkReport,
                           min_improvement_threshold: float = 0.05) -> bool:
        """Validate if an improvement is statistically significant and meaningful."""
        comparison = self.compare_performance(improvement_report, baseline_report)
        
        # Check if improvement meets threshold
        success_rate_improvement = comparison["success_rate_delta"] >= min_improvement_threshold
        
        # Check statistical significance
        statistically_significant = comparison.get("statistical_significance", 0) >= 1.96  # 95% confidence
        
        # Check that execution time didn't degrade significantly
        execution_time_acceptable = comparison["execution_time_delta"] <= 1.0  # Max 1 second increase
        
        is_valid = success_rate_improvement and statistically_significant and execution_time_acceptable
        
        self.logger.info(f"Improvement validation: {is_valid}")
        self.logger.info(f"  Success rate improvement: {comparison['success_rate_delta']:.3f} (threshold: {min_improvement_threshold})")
        self.logger.info(f"  Statistical significance: {comparison.get('statistical_significance', 0):.2f}")
        self.logger.info(f"  Execution time delta: {comparison['execution_time_delta']:.3f}s")
        
        return is_valid
    
    def _save_report(self, report: BenchmarkReport):
        """Save benchmark report to storage."""
        filename = f"{report.suite_name}_{report.model_version}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename
        
        # Convert report to JSON-serializable format
        report_dict = {
            "suite_name": report.suite_name,
            "model_version": report.model_version,
            "timestamp": report.timestamp.isoformat(),
            "total_problems": report.total_problems,
            "solved_problems": report.solved_problems,
            "success_rate": report.success_rate,
            "average_execution_time": report.average_execution_time,
            "average_memory_usage": report.average_memory_usage,
            "difficulty_breakdown": report.difficulty_breakdown,
            "performance_metrics": {
                "response_time": report.performance_metrics.response_time,
                "accuracy_score": report.performance_metrics.accuracy_score,
                "memory_usage": report.performance_metrics.memory_usage,
                "cpu_usage": report.performance_metrics.cpu_usage,
                "error_rate": report.performance_metrics.error_rate,
                "user_satisfaction": report.performance_metrics.user_satisfaction,
                "timestamp": report.performance_metrics.timestamp.isoformat()
            },
            "detailed_results": [
                {
                    "problem_id": r.problem_id,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "memory_usage": r.memory_usage,
                    "output": r.output,
                    "error_message": r.error_message,
                    "test_results": r.test_results,
                    "code_quality_score": r.code_quality_score,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in report.detailed_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Benchmark report saved to {filepath}")
    
    def load_report(self, filepath: str) -> BenchmarkReport:
        """Load a benchmark report from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to objects
        performance_metrics = PerformanceMetrics(
            response_time=data["performance_metrics"]["response_time"],
            accuracy_score=data["performance_metrics"]["accuracy_score"],
            memory_usage=data["performance_metrics"]["memory_usage"],
            cpu_usage=data["performance_metrics"]["cpu_usage"],
            error_rate=data["performance_metrics"]["error_rate"],
            user_satisfaction=data["performance_metrics"]["user_satisfaction"],
            timestamp=datetime.fromisoformat(data["performance_metrics"]["timestamp"])
        )
        
        detailed_results = [
            BenchmarkResult(
                problem_id=r["problem_id"],
                success=r["success"],
                execution_time=r["execution_time"],
                memory_usage=r["memory_usage"],
                output=r["output"],
                error_message=r["error_message"],
                test_results=r["test_results"],
                code_quality_score=r["code_quality_score"],
                timestamp=datetime.fromisoformat(r["timestamp"])
            )
            for r in data["detailed_results"]
        ]
        
        return BenchmarkReport(
            suite_name=data["suite_name"],
            model_version=data["model_version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            total_problems=data["total_problems"],
            solved_problems=data["solved_problems"],
            success_rate=data["success_rate"],
            average_execution_time=data["average_execution_time"],
            average_memory_usage=data["average_memory_usage"],
            difficulty_breakdown=data["difficulty_breakdown"],
            performance_metrics=performance_metrics,
            detailed_results=detailed_results
        )
    
    def get_benchmark_statistics(self) -> Dict[str, Any]:
        """Get statistics about available benchmarks."""
        stats = {
            "available_suites": list(self.benchmark_suites.keys()),
            "suite_details": {}
        }
        
        for name, suite in self.benchmark_suites.items():
            difficulty_counts = {}
            for difficulty in DifficultyLevel:
                count = sum(1 for p in suite.problems if p.difficulty == difficulty)
                if count > 0:
                    difficulty_counts[difficulty.value] = count
            
            benchmark_type_counts = {}
            for benchmark_type in BenchmarkType:
                count = sum(1 for p in suite.problems if p.benchmark_type == benchmark_type)
                if count > 0:
                    benchmark_type_counts[benchmark_type.value] = count
            
            stats["suite_details"][name] = {
                "total_problems": len(suite.problems),
                "difficulty_breakdown": difficulty_counts,
                "benchmark_type_breakdown": benchmark_type_counts,
                "description": suite.description
            }
        
        return stats