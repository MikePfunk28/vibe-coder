"""
Tests for Darwin-Gödel Self-Improving Model System
"""

import ast
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from darwin_godel_model import (
    DarwinGodelModel,
    CodeAnalysisEngine,
    ImprovementGenerator,
    SafetyValidator,
    PerformanceMetrics,
    ImprovementOpportunity,
    CodeModification,
    SafetyResult,
    ImprovementType,
    SafetyLevel
)


class TestCodeAnalysisEngine:
    """Test the code analysis engine."""
    
    def setup_method(self):
        self.analyzer = CodeAnalysisEngine()
    
    def test_analyze_simple_code(self):
        """Test analyzing simple Python code."""
        code = """
def simple_function():
    return "hello world"
"""
        opportunities = self.analyzer.analyze_code("test.py", code)
        assert isinstance(opportunities, list)
    
    def test_analyze_nested_loops(self):
        """Test detection of nested loops."""
        code = """
def nested_loops():
    for i in range(10):
        for j in range(10):
            print(i, j)
"""
        opportunities = self.analyzer.analyze_code("test.py", code)
        
        # Should detect nested loop opportunity
        nested_loop_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.PERFORMANCE_OPTIMIZATION
            and "nested loop" in opp.description.lower()
        ]
        assert len(nested_loop_opportunities) > 0
    
    def test_analyze_long_function(self):
        """Test detection of long functions."""
        # Create a function with many statements
        statements = ["    print(f'statement {i}')" for i in range(25)]
        code = f"""
def long_function():
{chr(10).join(statements)}
"""
        opportunities = self.analyzer.analyze_code("test.py", code)
        
        # Should detect long function opportunity
        long_func_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.CODE_QUALITY
            and "too long" in opp.description.lower()
        ]
        assert len(long_func_opportunities) > 0
    
    def test_analyze_list_comprehension(self):
        """Test detection of list comprehensions that could be generators."""
        code = """
def list_comp_function():
    result = [x * 2 for x in range(1000)]
    return result
"""
        opportunities = self.analyzer.analyze_code("test.py", code)
        
        # Should detect memory optimization opportunity
        memory_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.MEMORY_OPTIMIZATION
        ]
        assert len(memory_opportunities) > 0
    
    def test_analyze_invalid_syntax(self):
        """Test handling of invalid Python syntax."""
        code = "def invalid_function( print('missing colon')"
        
        # Should not raise exception, just return empty list
        opportunities = self.analyzer.analyze_code("test.py", code)
        assert isinstance(opportunities, list)


class TestImprovementGenerator:
    """Test the improvement generator."""
    
    def setup_method(self):
        self.generator = ImprovementGenerator()
    
    def test_generate_memory_optimization(self):
        """Test generation of memory optimization improvements."""
        opportunity = ImprovementOpportunity(
            id="test-1",
            improvement_type=ImprovementType.MEMORY_OPTIMIZATION,
            description="List comprehension could be converted to generator for memory efficiency",
            target_file="test.py",
            target_function="test_func",
            current_code="[x * 2 for x in range(100)]",
            estimated_benefit=0.3,
            confidence_score=0.8,
            risk_level=SafetyLevel.SAFE
        )
        
        modifications = self.generator.generate_improvements([opportunity])
        assert len(modifications) == 1
        
        modification = modifications[0]
        assert modification.modification_type == ImprovementType.MEMORY_OPTIMIZATION
        assert modification.modified_code == "(x * 2 for x in range(100))"
    
    def test_generate_performance_optimization(self):
        """Test generation of performance optimization improvements."""
        opportunity = ImprovementOpportunity(
            id="test-2",
            improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
            description="Nested loop detected - consider optimization",
            target_file="test.py",
            target_function="test_func",
            current_code="for i in range(10):\n    for j in range(10):\n        pass",
            estimated_benefit=0.4,
            confidence_score=0.7,
            risk_level=SafetyLevel.MODERATE_RISK
        )
        
        modifications = self.generator.generate_improvements([opportunity])
        assert len(modifications) == 1
        
        modification = modifications[0]
        assert modification.modification_type == ImprovementType.PERFORMANCE_OPTIMIZATION
    
    def test_generate_quality_improvement(self):
        """Test generation of code quality improvements."""
        opportunity = ImprovementOpportunity(
            id="test-3",
            improvement_type=ImprovementType.CODE_QUALITY,
            description="Function is too long - consider refactoring",
            target_file="test.py",
            target_function="long_func",
            current_code="def long_func():\n    # many lines...",
            estimated_benefit=0.2,
            confidence_score=0.8,
            risk_level=SafetyLevel.SAFE
        )
        
        modifications = self.generator.generate_improvements([opportunity])
        assert len(modifications) == 1
        
        modification = modifications[0]
        assert modification.modification_type == ImprovementType.CODE_QUALITY


class TestSafetyValidator:
    """Test the safety validator."""
    
    def setup_method(self):
        self.validator = SafetyValidator()
    
    def test_validate_safe_modification(self):
        """Test validation of a safe code modification."""
        modification = CodeModification(
            id="test-1",
            opportunity_id="opp-1",
            original_code="x = [i for i in range(10)]",
            modified_code="x = (i for i in range(10))",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.MEMORY_OPTIMIZATION,
            rationale="Convert to generator",
            estimated_impact=0.2,
            safety_score=0.9
        )
        
        result = self.validator.validate_safety(modification)
        assert result.is_safe
        assert result.safety_level == SafetyLevel.SAFE
        assert len(result.risk_factors) == 0
    
    def test_validate_unsafe_modification(self):
        """Test validation of an unsafe code modification."""
        modification = CodeModification(
            id="test-2",
            opportunity_id="opp-2",
            original_code="print('hello')",
            modified_code="exec('print(\"hello\")')",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
            rationale="Dynamic execution",
            estimated_impact=0.5,
            safety_score=0.1
        )
        
        result = self.validator.validate_safety(modification)
        assert not result.is_safe
        assert result.safety_level == SafetyLevel.UNSAFE
        assert len(result.risk_factors) > 0
        assert any("exec" in factor for factor in result.risk_factors)
    
    def test_validate_high_impact_modification(self):
        """Test validation of high impact modifications."""
        modification = CodeModification(
            id="test-3",
            opportunity_id="opp-3",
            original_code="def func(): pass",
            modified_code="def func(): return 'modified'",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.ALGORITHM_ENHANCEMENT,
            rationale="Major algorithm change",
            estimated_impact=0.8,  # High impact
            safety_score=0.7
        )
        
        result = self.validator.validate_safety(modification)
        assert result.safety_level in [SafetyLevel.MODERATE_RISK, SafetyLevel.HIGH_RISK]
        assert any("High impact" in factor for factor in result.risk_factors)
    
    def test_validate_syntax_error(self):
        """Test validation of modifications with syntax errors."""
        modification = CodeModification(
            id="test-4",
            opportunity_id="opp-4",
            original_code="def func(): pass",
            modified_code="def func( print('invalid')",  # Missing colon and parenthesis
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.CODE_QUALITY,
            rationale="Refactoring",
            estimated_impact=0.3,
            safety_score=0.5
        )
        
        result = self.validator.validate_safety(modification)
        assert not result.is_safe
        assert result.safety_level == SafetyLevel.UNSAFE
        assert any("syntax" in factor.lower() for factor in result.risk_factors)


class TestDarwinGodelModel:
    """Test the main Darwin-Gödel model."""
    
    def setup_method(self):
        self.model = DarwinGodelModel()
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.base_model == "default"
        assert self.model.safety_threshold == 0.7
        assert isinstance(self.model.code_analyzer, CodeAnalysisEngine)
        assert isinstance(self.model.improvement_generator, ImprovementGenerator)
        assert isinstance(self.model.safety_validator, SafetyValidator)
        assert self.model.improvement_candidates == []
        assert self.model.version_history == []
        assert self.model.performance_baseline is None
    
    def test_analyze_performance_baseline(self):
        """Test performance analysis with baseline establishment."""
        metrics = PerformanceMetrics(
            response_time=0.5,
            accuracy_score=0.9,
            memory_usage=100,
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        
        opportunities = self.model.analyze_performance(metrics)
        
        # Should establish baseline
        assert self.model.performance_baseline is not None
        assert self.model.performance_baseline.response_time == 0.5
        assert isinstance(opportunities, list)
    
    def test_analyze_performance_degradation(self):
        """Test detection of performance degradation."""
        # Set baseline
        baseline = PerformanceMetrics(
            response_time=0.5,
            accuracy_score=0.9,
            memory_usage=100,
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        self.model.performance_baseline = baseline
        
        # Create degraded metrics
        degraded = PerformanceMetrics(
            response_time=0.7,  # 40% increase
            accuracy_score=0.9,
            memory_usage=140,   # 40% increase
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        
        opportunities = self.model.analyze_performance(degraded)
        
        # Should detect both response time and memory degradation
        assert len(opportunities) >= 2
        
        perf_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.PERFORMANCE_OPTIMIZATION
        ]
        memory_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.MEMORY_OPTIMIZATION
        ]
        
        assert len(perf_opportunities) > 0
        assert len(memory_opportunities) > 0
    
    def test_generate_improvements(self):
        """Test improvement generation with safety validation."""
        opportunities = [
            ImprovementOpportunity(
                id="test-1",
                improvement_type=ImprovementType.MEMORY_OPTIMIZATION,
                description="List comprehension could be converted to generator",
                target_file="test.py",
                target_function="test_func",
                current_code="[x * 2 for x in range(100)]",
                estimated_benefit=0.3,
                confidence_score=0.8,
                risk_level=SafetyLevel.SAFE
            ),
            ImprovementOpportunity(
                id="test-2",
                improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                description="Unsafe operation",
                target_file="test.py",
                target_function="unsafe_func",
                current_code="exec('dangerous code')",
                estimated_benefit=0.5,
                confidence_score=0.3,
                risk_level=SafetyLevel.UNSAFE
            )
        ]
        
        modifications = self.model.generate_improvements(opportunities)
        
        # Should only return safe modifications
        assert len(modifications) >= 1
        for mod in modifications:
            safety_result = self.model.validate_safety(mod)
            assert safety_result.is_safe
    
    def test_apply_improvement(self):
        """Test applying an improvement."""
        modification = CodeModification(
            id="test-1",
            opportunity_id="opp-1",
            original_code="x = [i for i in range(10)]",
            modified_code="x = (i for i in range(10))",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.MEMORY_OPTIMIZATION,
            rationale="Convert to generator",
            estimated_impact=0.2,
            safety_score=0.9
        )
        
        result = self.model.apply_improvement(modification)
        assert result is True
        
        # Should add to version history
        assert len(self.model.version_history) == 1
        version = self.model.version_history[0]
        assert version["modification_id"] == "test-1"
        assert version["file_path"] == "test.py"
    
    def test_rollback_to_version(self):
        """Test rolling back to a previous version."""
        # First apply an improvement
        modification = CodeModification(
            id="test-1",
            opportunity_id="opp-1",
            original_code="original",
            modified_code="modified",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.CODE_QUALITY,
            rationale="Test modification",
            estimated_impact=0.2,
            safety_score=0.9
        )
        
        self.model.apply_improvement(modification)
        version_id = self.model.version_history[0]["id"]
        
        # Test rollback
        result = self.model.rollback_to_version(version_id)
        assert result is True
    
    def test_rollback_nonexistent_version(self):
        """Test rolling back to a non-existent version."""
        result = self.model.rollback_to_version("nonexistent-id")
        assert result is False
    
    def test_get_improvement_history(self):
        """Test getting improvement history."""
        # Initially empty
        history = self.model.get_improvement_history()
        assert history == []
        
        # Apply an improvement
        modification = CodeModification(
            id="test-1",
            opportunity_id="opp-1",
            original_code="original",
            modified_code="modified",
            file_path="test.py",
            line_start=1,
            line_end=1,
            modification_type=ImprovementType.CODE_QUALITY,
            rationale="Test modification",
            estimated_impact=0.2,
            safety_score=0.9
        )
        
        self.model.apply_improvement(modification)
        
        # Should have history
        history = self.model.get_improvement_history()
        assert len(history) == 1
        assert history[0]["modification_id"] == "test-1"
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Initially None
        metrics = self.model.get_performance_metrics()
        assert metrics is None
        
        # Set baseline
        baseline = PerformanceMetrics(
            response_time=0.5,
            accuracy_score=0.9,
            memory_usage=100,
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        self.model.performance_baseline = baseline
        
        # Should return baseline
        metrics = self.model.get_performance_metrics()
        assert metrics is not None
        assert metrics.response_time == 0.5


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_improvement_cycle(self):
        """Test a complete improvement cycle from analysis to application."""
        model = DarwinGodelModel()
        
        # 1. Establish performance baseline
        baseline_metrics = PerformanceMetrics(
            response_time=0.5,
            accuracy_score=0.9,
            memory_usage=100,
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        
        opportunities = model.analyze_performance(baseline_metrics)
        assert model.performance_baseline is not None
        
        # 2. Simulate performance degradation
        degraded_metrics = PerformanceMetrics(
            response_time=0.7,  # Degraded
            accuracy_score=0.9,
            memory_usage=140,   # Degraded
            cpu_usage=0.3,
            error_rate=0.01,
            user_satisfaction=4.5
        )
        
        opportunities = model.analyze_performance(degraded_metrics)
        assert len(opportunities) > 0
        
        # 3. Generate improvements
        modifications = model.generate_improvements(opportunities)
        
        # 4. Apply safe improvements
        applied_count = 0
        for modification in modifications:
            if model.apply_improvement(modification):
                applied_count += 1
        
        # Should have applied at least some improvements
        assert applied_count > 0
        assert len(model.get_improvement_history()) == applied_count
    
    def test_code_analysis_integration(self):
        """Test integration with actual code analysis."""
        model = DarwinGodelModel()
        
        # Analyze some sample code
        sample_code = """
def inefficient_function():
    result = []
    for i in range(1000):
        for j in range(100):
            if i * j % 2 == 0:
                result.append(i * j)
    return result

def long_function_that_does_too_much():
    # This function is intentionally long
    print("Step 1")
    print("Step 2")
    print("Step 3")
    print("Step 4")
    print("Step 5")
    print("Step 6")
    print("Step 7")
    print("Step 8")
    print("Step 9")
    print("Step 10")
    print("Step 11")
    print("Step 12")
    print("Step 13")
    print("Step 14")
    print("Step 15")
    print("Step 16")
    print("Step 17")
    print("Step 18")
    print("Step 19")
    print("Step 20")
    print("Step 21")
    return "done"
"""
        
        opportunities = model.code_analyzer.analyze_code("sample.py", sample_code)
        assert len(opportunities) > 0
        
        # Should find nested loops and long function
        perf_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.PERFORMANCE_OPTIMIZATION
        ]
        quality_opportunities = [
            opp for opp in opportunities 
            if opp.improvement_type == ImprovementType.CODE_QUALITY
        ]
        
        assert len(perf_opportunities) > 0  # Nested loops
        assert len(quality_opportunities) > 0  # Long function
        
        # Generate and validate improvements
        modifications = model.generate_improvements(opportunities)
        assert len(modifications) > 0
        
        # All generated modifications should be safe
        for modification in modifications:
            safety_result = model.validate_safety(modification)
            assert safety_result.is_safe or safety_result.safety_level == SafetyLevel.MODERATE_RISK


if __name__ == "__main__":
    pytest.main([__file__])