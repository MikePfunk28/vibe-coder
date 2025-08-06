"""
Benchmark Validation and Comparison System

This module provides advanced validation and comparison capabilities for the
mini-benchmarking system, including statistical analysis, performance visualization,
and integration with the Darwin-Gödel model for improvement validation.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

from mini_benchmark_system import (
    MiniBenchmarkSystem, 
    BenchmarkReport, 
    BenchmarkResult,
    BenchmarkType,
    DifficultyLevel
)
from darwin_godel_model import DarwinGodelModel, PerformanceMetrics


@dataclass
class ValidationResult:
    """Result of benchmark validation."""
    is_valid_improvement: bool
    confidence_level: float
    statistical_significance: float
    effect_size: float
    validation_details: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceComparison:
    """Detailed performance comparison between two benchmark reports."""
    baseline_report: BenchmarkReport
    comparison_report: BenchmarkReport
    overall_improvement: float
    statistical_significance: float
    effect_size: float
    difficulty_improvements: Dict[str, float]
    benchmark_type_improvements: Dict[str, float]
    regression_analysis: Dict[str, Any]
    confidence_interval: Tuple[float, float]


@dataclass
class BaselineMetrics:
    """Baseline performance metrics for comparison."""
    id: str
    model_version: str
    benchmark_suite: str
    success_rate: float
    execution_time: float
    memory_usage: float
    code_quality_score: float
    timestamp: datetime
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatisticalAnalyzer:
    """Performs statistical analysis on benchmark results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_statistical_significance(self, 
                                         baseline_results: List[BenchmarkResult],
                                         comparison_results: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate statistical significance using multiple tests."""
        
        # Extract success rates
        baseline_successes = [1 if r.success else 0 for r in baseline_results]
        comparison_successes = [1 if r.success else 0 for r in comparison_results]
        
        # Proportion test (z-test)
        n1, n2 = len(baseline_successes), len(comparison_successes)
        p1, p2 = np.mean(baseline_successes), np.mean(comparison_successes)
        
        # Pooled proportion
        if n1 == 0 or n2 == 0:
            pooled_p = 0.5  # Default when no data
            se = 0
        else:
            pooled_p = (sum(baseline_successes) + sum(comparison_successes)) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        if se > 0:
            z_score = (p2 - p1) / se
            if SCIPY_AVAILABLE:
                p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                # Approximate p-value calculation without scipy
                if abs(z_score) > 1.96:
                    p_value_z = 0.05  # Significant at 95% level
                elif abs(z_score) > 1.645:
                    p_value_z = 0.1   # Significant at 90% level
                else:
                    p_value_z = 0.5   # Not significant
        else:
            z_score = 0
            p_value_z = 1.0
        
        # Chi-square test (if scipy available)
        if SCIPY_AVAILABLE:
            contingency_table = [
                [sum(baseline_successes), n1 - sum(baseline_successes)],
                [sum(comparison_successes), n2 - sum(comparison_successes)]
            ]
            
            try:
                chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            except ValueError:
                chi2, p_value_chi2 = 0, 1.0
            
            # Fisher's exact test (for small samples)
            try:
                odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)
            except ValueError:
                odds_ratio, p_value_fisher = 1.0, 1.0
        else:
            chi2, p_value_chi2 = 0, 1.0
            odds_ratio, p_value_fisher = 1.0, 1.0
        
        # Effect size (Cohen's h for proportions)
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        return {
            "z_score": z_score,
            "p_value_z_test": p_value_z,
            "p_value_chi2": p_value_chi2,
            "p_value_fisher": p_value_fisher,
            "effect_size_cohens_h": h,
            "baseline_success_rate": p1,
            "comparison_success_rate": p2,
            "sample_size_baseline": n1,
            "sample_size_comparison": n2
        }
    
    def calculate_confidence_interval(self, 
                                    success_rate: float, 
                                    sample_size: int, 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for success rate."""
        if sample_size == 0:
            return (0.0, 0.0)
        
        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        else:
            # Approximate z-score for 95% confidence
            z_score = 1.96 if confidence_level == 0.95 else 2.58
        se = np.sqrt(success_rate * (1 - success_rate) / sample_size)
        
        lower = max(0.0, success_rate - z_score * se)
        upper = min(1.0, success_rate + z_score * se)
        
        return (lower, upper)
    
    def perform_regression_analysis(self, 
                                  reports: List[BenchmarkReport]) -> Dict[str, Any]:
        """Perform regression analysis on benchmark performance over time."""
        if len(reports) < 3:
            return {"error": "Insufficient data for regression analysis"}
        
        # Sort reports by timestamp
        sorted_reports = sorted(reports, key=lambda r: r.timestamp)
        
        # Extract data
        timestamps = [(r.timestamp - sorted_reports[0].timestamp).total_seconds() / 3600 
                     for r in sorted_reports]  # Hours since first report
        success_rates = [r.success_rate for r in sorted_reports]
        execution_times = [r.average_execution_time for r in sorted_reports]
        
        if SCIPY_AVAILABLE:
            # Linear regression for success rate
            slope_success, intercept_success, r_value_success, p_value_success, se_success = \
                stats.linregress(timestamps, success_rates)
            
            # Linear regression for execution time
            slope_time, intercept_time, r_value_time, p_value_time, se_time = \
                stats.linregress(timestamps, execution_times)
        else:
            # Simple linear regression approximation
            slope_success = (success_rates[-1] - success_rates[0]) / (timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 0
            intercept_success = success_rates[0]
            r_value_success = 0.5  # Placeholder
            p_value_success = 0.1  # Placeholder
            se_success = 0.1  # Placeholder
            
            slope_time = (execution_times[-1] - execution_times[0]) / (timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 0
            intercept_time = execution_times[0]
            r_value_time = 0.5  # Placeholder
            p_value_time = 0.1  # Placeholder
            se_time = 0.1  # Placeholder
        
        return {
            "success_rate_trend": {
                "slope": slope_success,
                "intercept": intercept_success,
                "r_squared": r_value_success ** 2,
                "p_value": p_value_success,
                "standard_error": se_success
            },
            "execution_time_trend": {
                "slope": slope_time,
                "intercept": intercept_time,
                "r_squared": r_value_time ** 2,
                "p_value": p_value_time,
                "standard_error": se_time
            },
            "data_points": len(reports),
            "time_span_hours": max(timestamps)
        }


class PerformanceVisualizer:
    """Creates visualizations for benchmark performance."""
    
    def __init__(self, output_dir: str = "benchmark_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set style if available
        if VISUALIZATION_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            except:
                # Fallback to default style
                pass
    
    def create_performance_comparison_chart(self, 
                                          baseline: BenchmarkReport,
                                          comparison: BenchmarkReport,
                                          save_path: Optional[str] = None) -> str:
        """Create a comprehensive performance comparison chart."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Matplotlib not available for visualization")
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate comparison
        models = [baseline.model_version, comparison.model_version]
        success_rates = [baseline.success_rate, comparison.success_rate]
        
        bars1 = ax1.bar(models, success_rates, color=['lightcoral', 'lightblue'])
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Execution time comparison
        exec_times = [baseline.average_execution_time, comparison.average_execution_time]
        bars2 = ax2.bar(models, exec_times, color=['lightcoral', 'lightblue'])
        ax2.set_title('Average Execution Time')
        ax2.set_ylabel('Time (seconds)')
        
        for bar, time in zip(bars2, exec_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # Difficulty breakdown
        if baseline.difficulty_breakdown and comparison.difficulty_breakdown:
            difficulties = list(set(baseline.difficulty_breakdown.keys()) | 
                              set(comparison.difficulty_breakdown.keys()))
            
            baseline_rates = [baseline.difficulty_breakdown.get(d, {}).get('success_rate', 0) 
                            for d in difficulties]
            comparison_rates = [comparison.difficulty_breakdown.get(d, {}).get('success_rate', 0) 
                              for d in difficulties]
            
            x = np.arange(len(difficulties))
            width = 0.35
            
            ax3.bar(x - width/2, baseline_rates, width, label=baseline.model_version, 
                   color='lightcoral')
            ax3.bar(x + width/2, comparison_rates, width, label=comparison.model_version,
                   color='lightblue')
            
            ax3.set_title('Success Rate by Difficulty')
            ax3.set_ylabel('Success Rate')
            ax3.set_xlabel('Difficulty Level')
            ax3.set_xticks(x)
            ax3.set_xticklabels(difficulties)
            ax3.legend()
            ax3.set_ylim(0, 1)
        
        # Problems solved comparison
        problems_solved = [baseline.solved_problems, comparison.solved_problems]
        total_problems = [baseline.total_problems, comparison.total_problems]
        
        bars4 = ax4.bar(models, problems_solved, color=['lightcoral', 'lightblue'])
        ax4.set_title('Problems Solved')
        ax4.set_ylabel('Number of Problems')
        
        for bar, solved, total in zip(bars4, problems_solved, total_problems):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{solved}/{total}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"comparison_{baseline.model_version}_vs_{comparison.model_version}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance comparison chart saved to {save_path}")
        return str(save_path)
    
    def create_performance_trend_chart(self, 
                                     reports: List[BenchmarkReport],
                                     save_path: Optional[str] = None) -> str:
        """Create a performance trend chart over time."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Matplotlib not available for visualization")
            
        if len(reports) < 2:
            raise ValueError("Need at least 2 reports for trend analysis")
        
        # Sort by timestamp
        sorted_reports = sorted(reports, key=lambda r: r.timestamp)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        timestamps = [r.timestamp for r in sorted_reports]
        success_rates = [r.success_rate for r in sorted_reports]
        execution_times = [r.average_execution_time for r in sorted_reports]
        memory_usage = [r.average_memory_usage for r in sorted_reports]
        
        # Success rate trend
        ax1.plot(timestamps, success_rates, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Success Rate Trend Over Time')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Execution time trend
        ax2.plot(timestamps, execution_times, marker='s', color='orange', linewidth=2, markersize=6)
        ax2.set_title('Execution Time Trend Over Time')
        ax2.set_ylabel('Average Execution Time (s)')
        ax2.grid(True, alpha=0.3)
        
        # Memory usage trend
        ax3.plot(timestamps, memory_usage, marker='^', color='green', linewidth=2, markersize=6)
        ax3.set_title('Memory Usage Trend Over Time')
        ax3.set_ylabel('Average Memory Usage (MB)')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"performance_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance trend chart saved to {save_path}")
        return str(save_path)


class BaselineManager:
    """Manages baseline performance metrics for comparison."""
    
    def __init__(self, storage_path: str = "baseline_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.db_path = self.storage_path / "baselines.db"
        self.logger = logging.getLogger(__name__)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the baseline storage database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    benchmark_suite TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    code_quality_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_baselines_suite_version 
                ON baselines (benchmark_suite, model_version)
            """)
    
    def store_baseline(self, report: BenchmarkReport) -> str:
        """Store a benchmark report as a baseline."""
        baseline_id = str(uuid.uuid4())
        
        # Calculate average code quality score
        avg_quality = np.mean([r.code_quality_score for r in report.detailed_results]) \
                     if report.detailed_results else 0.0
        
        baseline = BaselineMetrics(
            id=baseline_id,
            model_version=report.model_version,
            benchmark_suite=report.suite_name,
            success_rate=report.success_rate,
            execution_time=report.average_execution_time,
            memory_usage=report.average_memory_usage,
            code_quality_score=avg_quality,
            timestamp=report.timestamp,
            sample_size=report.total_problems
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO baselines 
                (id, model_version, benchmark_suite, success_rate, execution_time, 
                 memory_usage, code_quality_score, timestamp, sample_size, metadata, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.id,
                baseline.model_version,
                baseline.benchmark_suite,
                baseline.success_rate,
                baseline.execution_time,
                baseline.memory_usage,
                baseline.code_quality_score,
                baseline.timestamp.isoformat(),
                baseline.sample_size,
                json.dumps(baseline.metadata),
                True
            ))
        
        self.logger.info(f"Stored baseline {baseline_id} for {baseline.model_version} on {baseline.benchmark_suite}")
        return baseline_id
    
    def get_baseline(self, benchmark_suite: str, model_version: Optional[str] = None) -> Optional[BaselineMetrics]:
        """Get the most recent baseline for a benchmark suite."""
        with sqlite3.connect(self.db_path) as conn:
            if model_version:
                cursor = conn.execute("""
                    SELECT * FROM baselines 
                    WHERE benchmark_suite = ? AND model_version = ? AND is_active = 1
                    ORDER BY timestamp DESC LIMIT 1
                """, (benchmark_suite, model_version))
            else:
                cursor = conn.execute("""
                    SELECT * FROM baselines 
                    WHERE benchmark_suite = ? AND is_active = 1
                    ORDER BY timestamp DESC LIMIT 1
                """, (benchmark_suite,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return BaselineMetrics(
                id=row[0],
                model_version=row[1],
                benchmark_suite=row[2],
                success_rate=row[3],
                execution_time=row[4],
                memory_usage=row[5],
                code_quality_score=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                sample_size=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
    
    def get_baseline_history(self, benchmark_suite: str, limit: int = 10) -> List[BaselineMetrics]:
        """Get baseline history for a benchmark suite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM baselines 
                WHERE benchmark_suite = ? AND is_active = 1
                ORDER BY timestamp DESC LIMIT ?
            """, (benchmark_suite, limit))
            
            baselines = []
            for row in cursor.fetchall():
                baselines.append(BaselineMetrics(
                    id=row[0],
                    model_version=row[1],
                    benchmark_suite=row[2],
                    success_rate=row[3],
                    execution_time=row[4],
                    memory_usage=row[5],
                    code_quality_score=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    sample_size=row[8],
                    metadata=json.loads(row[9]) if row[9] else {}
                ))
            
            return baselines


class BenchmarkValidationSystem:
    """Main validation and comparison system for benchmarks."""
    
    def __init__(self, 
                 benchmark_system: MiniBenchmarkSystem,
                 storage_path: str = "validation_storage"):
        self.benchmark_system = benchmark_system
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = PerformanceVisualizer(str(self.storage_path / "visualizations"))
        self.baseline_manager = BaselineManager(str(self.storage_path / "baselines"))
        
        self.logger.info("BenchmarkValidationSystem initialized")
    
    def validate_improvement(self, 
                           current_report: BenchmarkReport,
                           baseline_report: Optional[BenchmarkReport] = None,
                           min_improvement_threshold: float = 0.05,
                           confidence_level: float = 0.95) -> ValidationResult:
        """Validate if a benchmark report represents a significant improvement."""
        
        # Get baseline if not provided
        if baseline_report is None:
            baseline_metrics = self.baseline_manager.get_baseline(current_report.suite_name)
            if baseline_metrics is None:
                return ValidationResult(
                    is_valid_improvement=False,
                    confidence_level=0.0,
                    statistical_significance=0.0,
                    effect_size=0.0,
                    validation_details={"error": "No baseline available for comparison"},
                    recommendations=["Establish a baseline by running initial benchmarks"]
                )
            
            # Create a mock baseline report for comparison
            solved_count = int(baseline_metrics.success_rate * baseline_metrics.sample_size)
            detailed_results = []
            for i in range(baseline_metrics.sample_size):
                detailed_results.append(BenchmarkResult(
                    problem_id=f"baseline_{i}",
                    success=i < solved_count,
                    execution_time=baseline_metrics.execution_time,
                    memory_usage=int(baseline_metrics.memory_usage),
                    output="baseline_result"
                ))
            
            baseline_report = BenchmarkReport(
                suite_name=baseline_metrics.benchmark_suite,
                model_version=baseline_metrics.model_version,
                timestamp=baseline_metrics.timestamp,
                total_problems=baseline_metrics.sample_size,
                solved_problems=solved_count,
                success_rate=baseline_metrics.success_rate,
                average_execution_time=baseline_metrics.execution_time,
                average_memory_usage=baseline_metrics.memory_usage,
                difficulty_breakdown={},
                performance_metrics=PerformanceMetrics(
                    baseline_metrics.execution_time,
                    baseline_metrics.success_rate,
                    int(baseline_metrics.memory_usage),
                    0.0, 1.0 - baseline_metrics.success_rate,
                    baseline_metrics.success_rate * 5.0
                ),
                detailed_results=detailed_results
            )
        
        # Perform statistical analysis
        stats_result = self.statistical_analyzer.calculate_statistical_significance(
            baseline_report.detailed_results,
            current_report.detailed_results
        )
        
        # Calculate improvement metrics
        success_rate_improvement = current_report.success_rate - baseline_report.success_rate
        execution_time_improvement = baseline_report.average_execution_time - current_report.average_execution_time
        
        # Determine if improvement is significant
        is_statistically_significant = stats_result["p_value_z_test"] < (1 - confidence_level)
        meets_improvement_threshold = success_rate_improvement >= min_improvement_threshold
        no_significant_regression = execution_time_improvement >= -1.0  # Max 1 second regression
        
        is_valid = is_statistically_significant and meets_improvement_threshold and no_significant_regression
        
        # Generate recommendations
        recommendations = []
        if not meets_improvement_threshold:
            recommendations.append(f"Success rate improvement ({success_rate_improvement:.1%}) below threshold ({min_improvement_threshold:.1%})")
        if not is_statistically_significant:
            recommendations.append(f"Improvement not statistically significant (p-value: {stats_result['p_value_z_test']:.3f})")
        if execution_time_improvement < -1.0:
            recommendations.append(f"Significant execution time regression ({-execution_time_improvement:.2f}s)")
        
        if is_valid:
            recommendations.append("Improvement validated - safe to deploy")
        else:
            recommendations.append("Improvement not validated - consider additional optimization")
        
        return ValidationResult(
            is_valid_improvement=is_valid,
            confidence_level=confidence_level,
            statistical_significance=stats_result["p_value_z_test"],
            effect_size=stats_result["effect_size_cohens_h"],
            validation_details={
                "success_rate_improvement": success_rate_improvement,
                "execution_time_improvement": execution_time_improvement,
                "statistical_analysis": stats_result,
                "baseline_model": baseline_report.model_version,
                "current_model": current_report.model_version
            },
            recommendations=recommendations
        )
    
    def create_detailed_comparison(self, 
                                 baseline_report: BenchmarkReport,
                                 comparison_report: BenchmarkReport) -> PerformanceComparison:
        """Create a detailed performance comparison between two reports."""
        
        # Statistical analysis
        stats_result = self.statistical_analyzer.calculate_statistical_significance(
            baseline_report.detailed_results,
            comparison_report.detailed_results
        )
        
        # Overall improvement
        overall_improvement = comparison_report.success_rate - baseline_report.success_rate
        
        # Difficulty-based improvements
        difficulty_improvements = {}
        for difficulty in DifficultyLevel:
            baseline_diff = baseline_report.difficulty_breakdown.get(difficulty.value, {})
            comparison_diff = comparison_report.difficulty_breakdown.get(difficulty.value, {})
            
            if baseline_diff and comparison_diff:
                improvement = comparison_diff.get('success_rate', 0) - baseline_diff.get('success_rate', 0)
                difficulty_improvements[difficulty.value] = improvement
        
        # Benchmark type improvements (if available)
        benchmark_type_improvements = {}
        # This would require additional data structure in reports
        
        # Regression analysis (simplified)
        regression_analysis = {
            "success_rate_change": overall_improvement,
            "execution_time_change": baseline_report.average_execution_time - comparison_report.average_execution_time,
            "memory_usage_change": baseline_report.average_memory_usage - comparison_report.average_memory_usage
        }
        
        # Confidence interval for improvement
        confidence_interval = self.statistical_analyzer.calculate_confidence_interval(
            comparison_report.success_rate,
            comparison_report.total_problems
        )
        
        return PerformanceComparison(
            baseline_report=baseline_report,
            comparison_report=comparison_report,
            overall_improvement=overall_improvement,
            statistical_significance=stats_result["p_value_z_test"],
            effect_size=stats_result["effect_size_cohens_h"],
            difficulty_improvements=difficulty_improvements,
            benchmark_type_improvements=benchmark_type_improvements,
            regression_analysis=regression_analysis,
            confidence_interval=confidence_interval
        )
    
    def create_performance_report(self, 
                                comparison: PerformanceComparison,
                                include_visualizations: bool = True) -> Dict[str, Any]:
        """Create a comprehensive performance report."""
        
        report = {
            "summary": {
                "baseline_model": comparison.baseline_report.model_version,
                "comparison_model": comparison.comparison_report.model_version,
                "overall_improvement": comparison.overall_improvement,
                "statistical_significance": comparison.statistical_significance,
                "effect_size": comparison.effect_size,
                "is_significant": comparison.statistical_significance < 0.05
            },
            "detailed_metrics": {
                "success_rate": {
                    "baseline": comparison.baseline_report.success_rate,
                    "comparison": comparison.comparison_report.success_rate,
                    "improvement": comparison.overall_improvement,
                    "confidence_interval": comparison.confidence_interval
                },
                "execution_time": {
                    "baseline": comparison.baseline_report.average_execution_time,
                    "comparison": comparison.comparison_report.average_execution_time,
                    "improvement": comparison.regression_analysis["execution_time_change"]
                },
                "memory_usage": {
                    "baseline": comparison.baseline_report.average_memory_usage,
                    "comparison": comparison.comparison_report.average_memory_usage,
                    "improvement": comparison.regression_analysis["memory_usage_change"]
                }
            },
            "difficulty_breakdown": comparison.difficulty_improvements,
            "recommendations": self._generate_recommendations(comparison),
            "timestamp": datetime.now().isoformat()
        }
        
        if include_visualizations:
            try:
                chart_path = self.visualizer.create_performance_comparison_chart(
                    comparison.baseline_report,
                    comparison.comparison_report
                )
                report["visualization_path"] = chart_path
            except Exception as e:
                self.logger.warning(f"Failed to create visualization: {e}")
        
        return report
    
    def _generate_recommendations(self, comparison: PerformanceComparison) -> List[str]:
        """Generate recommendations based on performance comparison."""
        recommendations = []
        
        if comparison.overall_improvement > 0.1:
            recommendations.append("Significant improvement detected - recommend deployment")
        elif comparison.overall_improvement > 0.05:
            recommendations.append("Moderate improvement - consider additional validation")
        elif comparison.overall_improvement > 0:
            recommendations.append("Minor improvement - monitor for consistency")
        else:
            recommendations.append("No improvement or regression - investigate issues")
        
        if comparison.statistical_significance < 0.01:
            recommendations.append("Highly statistically significant results")
        elif comparison.statistical_significance < 0.05:
            recommendations.append("Statistically significant results")
        else:
            recommendations.append("Results not statistically significant - need more data")
        
        # Execution time recommendations
        exec_time_change = comparison.regression_analysis["execution_time_change"]
        if exec_time_change > 0.5:
            recommendations.append("Significant execution time improvement")
        elif exec_time_change < -0.5:
            recommendations.append("Execution time regression - investigate performance issues")
        
        return recommendations
    
    def establish_baseline(self, report: BenchmarkReport) -> str:
        """Establish a new baseline from a benchmark report."""
        baseline_id = self.baseline_manager.store_baseline(report)
        self.logger.info(f"Established baseline {baseline_id} for {report.suite_name}")
        return baseline_id
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the validation system."""
        return {
            "available_baselines": len(self.baseline_manager.get_baseline_history("comprehensive", 100)),
            "visualization_count": len(list(self.visualizer.output_dir.glob("*.png"))),
            "storage_path": str(self.storage_path),
            "supported_benchmark_types": [bt.value for bt in BenchmarkType],
            "supported_difficulty_levels": [dl.value for dl in DifficultyLevel]
        }