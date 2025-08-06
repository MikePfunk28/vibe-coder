"""
Tests for Benchmark Validation and Comparison System
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from benchmark_validation_system import (
    BenchmarkValidationSystem,
    StatisticalAnalyzer,
    BaselineManager,
    PerformanceVisualizer,
    ValidationResult,
    PerformanceComparison,
    BaselineMetrics
)
from mini_benchmark_system import (
    MiniBenchmarkSystem,
    BenchmarkReport,
    BenchmarkResult,
    DifficultyLevel
)
from darwin_godel_model import PerformanceMetrics


class TestStatisticalAnalyzer:
    """Test the statistical analyzer."""
    
    def setup_method(self):
        self.analyzer = StatisticalAnalyzer()
    
    def test_calculate_statistical_significance(self):
        """Test statistical significance calculation."""
        # Create mock results
        baseline_results = [
            BenchmarkResult("1", True, 1.0, 100, "pass"),
            BenchmarkResult("2", False, 1.0, 100, "fail"),
            BenchmarkResult("3", True, 1.0, 100, "pass"),
            BenchmarkResult("4", False, 1.0, 100, "fail"),
            BenchmarkResult("5", True, 1.0, 100, "pass")
        ]
        
        comparison_results = [
            BenchmarkResult("1", True, 0.8, 90, "pass"),
            BenchmarkResult("2", True, 0.8, 90, "pass"),
            BenchmarkResult("3", True, 0.8, 90, "pass"),
            BenchmarkResult("4", True, 0.8, 90, "pass"),
            BenchmarkResult("5", False, 0.8, 90, "fail")
        ]
        
        stats = self.analyzer.calculate_statistical_significance(baseline_results, comparison_results)
        
        assert "z_score" in stats
        assert "p_value_z_test" in stats
        assert "effect_size_cohens_h" in stats
        assert "baseline_success_rate" in stats
        assert "comparison_success_rate" in stats
        
        assert stats["baseline_success_rate"] == 0.6  # 3/5
        assert stats["comparison_success_rate"] == 0.8  # 4/5
        assert stats["sample_size_baseline"] == 5
        assert stats["sample_size_comparison"] == 5
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        # Test with typical values
        lower, upper = self.analyzer.calculate_confidence_interval(0.8, 100, 0.95)
        
        assert 0.0 <= lower <= upper <= 1.0
        assert lower < 0.8 < upper  # Should contain the point estimate
        
        # Test edge cases
        lower_zero, upper_zero = self.analyzer.calculate_confidence_interval(0.0, 100, 0.95)
        assert lower_zero == 0.0
        
        lower_one, upper_one = self.analyzer.calculate_confidence_interval(1.0, 100, 0.95)
        assert upper_one == 1.0
    
    def test_perform_regression_analysis(self):
        """Test regression analysis."""
        # Create mock reports with trend
        base_time = datetime.now()
        reports = []
        
        for i in range(5):
            report = BenchmarkReport(
                suite_name="test",
                model_version=f"v{i}",
                timestamp=base_time + timedelta(hours=i),
                total_problems=10,
                solved_problems=5 + i,  # Improving trend
                success_rate=(5 + i) / 10,
                average_execution_time=2.0 - i * 0.1,  # Improving trend
                average_memory_usage=100.0,
                difficulty_breakdown={},
                performance_metrics=PerformanceMetrics(2.0 - i * 0.1, (5 + i) / 10, 100, 0.3, 0.5, 3.0),
                detailed_results=[]
            )
            reports.append(report)
        
        regression = self.analyzer.perform_regression_analysis(reports)
        
        assert "success_rate_trend" in regression
        assert "execution_time_trend" in regression
        assert "data_points" in regression
        
        # Should detect positive trend in success rate
        assert regression["success_rate_trend"]["slope"] > 0
        # Should detect negative trend in execution time (improvement)
        assert regression["execution_time_trend"]["slope"] < 0
    
    def test_regression_analysis_insufficient_data(self):
        """Test regression analysis with insufficient data."""
        reports = [Mock(), Mock()]  # Only 2 reports
        
        regression = self.analyzer.perform_regression_analysis(reports)
        
        assert "error" in regression
        assert "Insufficient data" in regression["error"]


class TestBaselineManager:
    """Test the baseline manager."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_manager = BaselineManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_baseline(self):
        """Test storing and retrieving baselines."""
        # Create a test report
        report = BenchmarkReport(
            suite_name="test_suite",
            model_version="test_model_v1",
            timestamp=datetime.now(),
            total_problems=10,
            solved_problems=7,
            success_rate=0.7,
            average_execution_time=1.5,
            average_memory_usage=120.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.5, 0.7, 120, 0.4, 0.3, 3.5),
            detailed_results=[
                BenchmarkResult("1", True, 1.0, 100, "pass", code_quality_score=0.8),
                BenchmarkResult("2", False, 2.0, 150, "fail", code_quality_score=0.6)
            ]
        )
        
        # Store baseline
        baseline_id = self.baseline_manager.store_baseline(report)
        assert baseline_id is not None
        
        # Retrieve baseline
        retrieved = self.baseline_manager.get_baseline("test_suite", "test_model_v1")
        
        assert retrieved is not None
        assert retrieved.model_version == "test_model_v1"
        assert retrieved.benchmark_suite == "test_suite"
        assert retrieved.success_rate == 0.7
        assert retrieved.execution_time == 1.5
        assert retrieved.memory_usage == 120.0
        assert retrieved.sample_size == 10
    
    def test_get_baseline_history(self):
        """Test getting baseline history."""
        # Store multiple baselines
        base_time = datetime.now()
        
        for i in range(3):
            report = BenchmarkReport(
                suite_name="test_suite",
                model_version=f"model_v{i}",
                timestamp=base_time + timedelta(hours=i),
                total_problems=10,
                solved_problems=5 + i,
                success_rate=(5 + i) / 10,
                average_execution_time=1.0,
                average_memory_usage=100.0,
                difficulty_breakdown={},
                performance_metrics=PerformanceMetrics(1.0, (5 + i) / 10, 100, 0.3, 0.5, 3.0),
                detailed_results=[]
            )
            self.baseline_manager.store_baseline(report)
        
        # Get history
        history = self.baseline_manager.get_baseline_history("test_suite", limit=5)
        
        assert len(history) == 3
        # Should be ordered by timestamp (most recent first)
        assert history[0].model_version == "model_v2"
        assert history[1].model_version == "model_v1"
        assert history[2].model_version == "model_v0"
    
    def test_get_nonexistent_baseline(self):
        """Test retrieving non-existent baseline."""
        baseline = self.baseline_manager.get_baseline("nonexistent_suite")
        assert baseline is None


class TestBenchmarkValidationSystem:
    """Test the main validation system."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the benchmark system
        self.mock_benchmark_system = Mock(spec=MiniBenchmarkSystem)
        
        self.validation_system = BenchmarkValidationSystem(
            self.mock_benchmark_system,
            storage_path=self.temp_dir
        )
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_establish_baseline(self):
        """Test establishing a baseline."""
        report = BenchmarkReport(
            suite_name="test_suite",
            model_version="baseline_v1",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=15,
            success_rate=0.75,
            average_execution_time=1.2,
            average_memory_usage=110.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.2, 0.75, 110, 0.35, 0.25, 3.75),
            detailed_results=[]
        )
        
        baseline_id = self.validation_system.establish_baseline(report)
        
        assert baseline_id is not None
        
        # Verify baseline was stored
        retrieved = self.validation_system.baseline_manager.get_baseline("test_suite")
        assert retrieved is not None
        assert retrieved.success_rate == 0.75
    
    def test_validate_improvement_with_baseline(self):
        """Test validating improvement against stored baseline."""
        # First establish a baseline
        baseline_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="baseline_v1",
            timestamp=datetime.now() - timedelta(days=1),
            total_problems=20,
            solved_problems=10,
            success_rate=0.5,
            average_execution_time=2.0,
            average_memory_usage=150.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(2.0, 0.5, 150, 0.4, 0.5, 2.5),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 10, 2.0, 150, "result") for i in range(20)]
        )
        
        self.validation_system.establish_baseline(baseline_report)
        
        # Create an improved report
        improved_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="improved_v2",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=16,
            success_rate=0.8,
            average_execution_time=1.5,
            average_memory_usage=130.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.5, 0.8, 130, 0.3, 0.2, 4.0),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 16, 1.5, 130, "result") for i in range(20)]
        )
        
        # Validate improvement
        validation = self.validation_system.validate_improvement(
            improved_report,
            min_improvement_threshold=0.1
        )
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid_improvement  # Should be valid with 30% improvement
        assert abs(validation.validation_details["success_rate_improvement"] - 0.3) < 0.001
        assert abs(validation.validation_details["execution_time_improvement"] - 0.5) < 0.001
    
    def test_validate_improvement_insufficient_improvement(self):
        """Test validation with insufficient improvement."""
        baseline_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="baseline_v1",
            timestamp=datetime.now() - timedelta(days=1),
            total_problems=20,
            solved_problems=15,
            success_rate=0.75,
            average_execution_time=1.0,
            average_memory_usage=100.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.0, 0.75, 100, 0.3, 0.25, 3.75),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 15, 1.0, 100, "result") for i in range(20)]
        )
        
        # Marginal improvement
        marginal_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="marginal_v2",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=16,
            success_rate=0.8,
            average_execution_time=1.0,
            average_memory_usage=100.0,
            difficulty_breakdown={},
            performance_metrics=PerformanceMetrics(1.0, 0.8, 100, 0.3, 0.2, 4.0),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 16, 1.0, 100, "result") for i in range(20)]
        )
        
        validation = self.validation_system.validate_improvement(
            marginal_report,
            baseline_report,
            min_improvement_threshold=0.1  # 10% threshold, but only 5% improvement
        )
        
        assert not validation.is_valid_improvement
        assert any("below threshold" in rec for rec in validation.recommendations)
    
    def test_create_detailed_comparison(self):
        """Test creating detailed performance comparison."""
        baseline_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="baseline_v1",
            timestamp=datetime.now() - timedelta(days=1),
            total_problems=20,
            solved_problems=10,
            success_rate=0.5,
            average_execution_time=2.0,
            average_memory_usage=150.0,
            difficulty_breakdown={
                "easy": {"total": 10, "solved": 8, "success_rate": 0.8},
                "medium": {"total": 10, "solved": 2, "success_rate": 0.2}
            },
            performance_metrics=PerformanceMetrics(2.0, 0.5, 150, 0.4, 0.5, 2.5),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 10, 2.0, 150, "result") for i in range(20)]
        )
        
        improved_report = BenchmarkReport(
            suite_name="test_suite",
            model_version="improved_v2",
            timestamp=datetime.now(),
            total_problems=20,
            solved_problems=16,
            success_rate=0.8,
            average_execution_time=1.5,
            average_memory_usage=130.0,
            difficulty_breakdown={
                "easy": {"total": 10, "solved": 9, "success_rate": 0.9},
                "medium": {"total": 10, "solved": 7, "success_rate": 0.7}
            },
            performance_metrics=PerformanceMetrics(1.5, 0.8, 130, 0.3, 0.2, 4.0),
            detailed_results=[BenchmarkResult(f"test_{i}", i < 16, 1.5, 130, "result") for i in range(20)]
        )
        
        comparison = self.validation_system.create_detailed_comparison(baseline_report, improved_report)
        
        assert isinstance(comparison, PerformanceComparison)
        assert abs(comparison.overall_improvement - 0.3) < 0.001
        assert comparison.baseline_report == baseline_report
        assert comparison.comparison_report == improved_report
        assert "easy" in comparison.difficulty_improvements
        assert "medium" in comparison.difficulty_improvements
        assert abs(comparison.difficulty_improvements["easy"] - 0.1) < 0.001  # 0.9 - 0.8
        assert abs(comparison.difficulty_improvements["medium"] - 0.5) < 0.001  # 0.7 - 0.2
    
    def test_create_performance_report(self):
        """Test creating a comprehensive performance report."""
        # Create mock comparison
        baseline_report = Mock(spec=BenchmarkReport)
        baseline_report.model_version = "baseline_v1"
        baseline_report.success_rate = 0.5
        baseline_report.average_execution_time = 2.0
        baseline_report.average_memory_usage = 150.0
        
        comparison_report = Mock(spec=BenchmarkReport)
        comparison_report.model_version = "improved_v2"
        comparison_report.success_rate = 0.8
        comparison_report.average_execution_time = 1.5
        comparison_report.average_memory_usage = 130.0
        
        comparison = PerformanceComparison(
            baseline_report=baseline_report,
            comparison_report=comparison_report,
            overall_improvement=0.3,
            statistical_significance=0.01,
            effect_size=0.6,
            difficulty_improvements={"easy": 0.1, "medium": 0.5},
            benchmark_type_improvements={},
            regression_analysis={
                "success_rate_change": 0.3,
                "execution_time_change": 0.5,
                "memory_usage_change": 20.0
            },
            confidence_interval=(0.7, 0.9)
        )
        
        # Mock visualization to avoid matplotlib issues in tests
        with patch.object(self.validation_system.visualizer, 'create_performance_comparison_chart', 
                         return_value="/mock/path/chart.png"):
            report = self.validation_system.create_performance_report(comparison, include_visualizations=True)
        
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "difficulty_breakdown" in report
        assert "recommendations" in report
        assert "timestamp" in report
        
        assert report["summary"]["baseline_model"] == "baseline_v1"
        assert report["summary"]["comparison_model"] == "improved_v2"
        assert report["summary"]["overall_improvement"] == 0.3
        assert report["summary"]["is_significant"] is True  # p < 0.05
    
    def test_get_validation_statistics(self):
        """Test getting validation statistics."""
        stats = self.validation_system.get_validation_statistics()
        
        assert "available_baselines" in stats
        assert "visualization_count" in stats
        assert "storage_path" in stats
        assert "supported_benchmark_types" in stats
        assert "supported_difficulty_levels" in stats
        
        assert isinstance(stats["available_baselines"], int)
        assert isinstance(stats["supported_benchmark_types"], list)
        assert isinstance(stats["supported_difficulty_levels"], list)


class TestIntegration:
    """Integration tests for the validation system."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a real benchmark system for integration testing
        self.benchmark_system = MiniBenchmarkSystem(storage_path=str(Path(self.temp_dir) / "benchmarks"))
        self.validation_system = BenchmarkValidationSystem(
            self.benchmark_system,
            storage_path=str(Path(self.temp_dir) / "validation")
        )
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_validation_workflow(self):
        """Test a complete validation workflow."""
        
        def baseline_code_generator(prompt):
            """Baseline generator with moderate performance."""
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
                return "def solution(): return None"
        
        def improved_code_generator(prompt):
            """Improved generator with better performance."""
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
        
        # 1. Run baseline benchmark
        baseline_report = self.benchmark_system.run_benchmark(
            "humaneval",
            "baseline_model_v1",
            baseline_code_generator
        )
        
        # 2. Establish baseline
        baseline_id = self.validation_system.establish_baseline(baseline_report)
        assert baseline_id is not None
        
        # 3. Run improved benchmark
        improved_report = self.benchmark_system.run_benchmark(
            "humaneval",
            "improved_model_v2",
            improved_code_generator
        )
        
        # 4. Validate improvement
        validation = self.validation_system.validate_improvement(
            improved_report,
            min_improvement_threshold=0.01  # Low threshold for test
        )
        
        assert isinstance(validation, ValidationResult)
        # Should show improvement since improved generator solves more problems
        assert validation.validation_details["success_rate_improvement"] >= 0
        
        # 5. Create detailed comparison
        comparison = self.validation_system.create_detailed_comparison(
            baseline_report,
            improved_report
        )
        
        assert isinstance(comparison, PerformanceComparison)
        assert comparison.overall_improvement >= 0
        
        # 6. Create performance report
        with patch.object(self.validation_system.visualizer, 'create_performance_comparison_chart',
                         return_value="/mock/path/chart.png"):
            report = self.validation_system.create_performance_report(comparison)
        
        assert "summary" in report
        assert report["summary"]["baseline_model"] == "baseline_model_v1"
        assert report["summary"]["comparison_model"] == "improved_model_v2"
        
        # 7. Check validation statistics
        stats = self.validation_system.get_validation_statistics()
        # Check that we have at least one baseline stored
        baseline_history = self.validation_system.baseline_manager.get_baseline_history("humaneval", 10)
        assert len(baseline_history) >= 1


if __name__ == "__main__":
    pytest.main([__file__])