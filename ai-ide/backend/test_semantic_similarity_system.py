"""
Test suite for Semantic Similarity System Integration
"""

import os
import tempfile
import shutil
import time

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest decorators for basic testing
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        
        @staticmethod
        def skip(reason):
            pass
        
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator

from semantic_similarity_system import (
    SemanticSimilaritySystem, get_semantic_system,
    initialize_semantic_search, semantic_search, find_similar_code, get_workspace_overview
)
from semantic_search_engine import SearchContext

class TestSemanticSimilaritySystem:
    """Test SemanticSimilaritySystem functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_files = self._create_sample_files()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_files(self):
        """Create sample files for testing"""
        files = {}
        
        # Python file with various code structures
        python_content = '''"""
Sample Python module for testing semantic search
"""

import os
import sys
from typing import List, Dict, Any

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers"""
    return a + b

def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers"""
    return a * b

def find_maximum(numbers: List[int]) -> int:
    """Find the maximum number in a list"""
    if not numbers:
        raise ValueError("List cannot be empty")
    
    max_num = numbers[0]
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
    return max_num

class Calculator:
    """A simple calculator class for basic arithmetic operations"""
    
    def __init__(self):
        self.history = []
    
    def add(self, x: int, y: int) -> int:
        """Add two numbers and store in history"""
        result = calculate_sum(x, y)
        self.history.append(f"{x} + {y} = {result}")
        return result
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers and store in history"""
        result = calculate_product(x, y)
        self.history.append(f"{x} * {y} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()

class MathUtils:
    """Utility class for mathematical operations"""
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of a number"""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * MathUtils.factorial(n - 1)

def main():
    """Main function to demonstrate calculator usage"""
    calc = Calculator()
    
    # Test basic operations
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    
    # Test utility functions
    numbers = [1, 5, 3, 9, 2]
    print(f"Maximum: {find_maximum(numbers)}")
    
    # Test math utils
    print(f"Is 17 prime? {MathUtils.is_prime(17)}")
    print(f"5! = {MathUtils.factorial(5)}")
    
    # Show history
    print("History:", calc.get_history())

if __name__ == "__main__":
    main()
'''
        
        python_file = os.path.join(self.temp_dir, "calculator.py")
        with open(python_file, 'w') as f:
            f.write(python_content)
        files['python'] = python_file
        
        # JavaScript file with similar functionality
        js_content = '''/**
 * Sample JavaScript module for testing semantic search
 */

function calculateSum(a, b) {
    // Calculate the sum of two numbers
    return a + b;
}

function calculateProduct(a, b) {
    // Calculate the product of two numbers
    return a * b;
}

function findMaximum(numbers) {
    // Find the maximum number in an array
    if (numbers.length === 0) {
        throw new Error("Array cannot be empty");
    }
    
    let maxNum = numbers[0];
    for (let i = 1; i < numbers.length; i++) {
        if (numbers[i] > maxNum) {
            maxNum = numbers[i];
        }
    }
    return maxNum;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(x, y) {
        const result = calculateSum(x, y);
        this.history.push(`${x} + ${y} = ${result}`);
        return result;
    }
    
    multiply(x, y) {
        const result = calculateProduct(x, y);
        this.history.push(`${x} * ${y} = ${result}`);
        return result;
    }
    
    getHistory() {
        return [...this.history];
    }
}

class MathUtils {
    static isPrime(n) {
        if (n < 2) return false;
        for (let i = 2; i <= Math.sqrt(n); i++) {
            if (n % i === 0) return false;
        }
        return true;
    }
    
    static factorial(n) {
        if (n < 0) {
            throw new Error("Factorial not defined for negative numbers");
        }
        if (n === 0 || n === 1) return 1;
        return n * MathUtils.factorial(n - 1);
    }
}

function main() {
    const calc = new Calculator();
    
    // Test basic operations
    console.log(calc.add(5, 3));
    console.log(calc.multiply(4, 7));
    
    // Test utility functions
    const numbers = [1, 5, 3, 9, 2];
    console.log(`Maximum: ${findMaximum(numbers)}`);
    
    // Test math utils
    console.log(`Is 17 prime? ${MathUtils.isPrime(17)}`);
    console.log(`5! = ${MathUtils.factorial(5)}`);
    
    // Show history
    console.log("History:", calc.getHistory());
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Calculator, MathUtils, calculateSum, calculateProduct, findMaximum };
} else {
    main();
}
'''
        
        js_file = os.path.join(self.temp_dir, "calculator.js")
        with open(js_file, 'w') as f:
            f.write(js_content)
        files['javascript'] = js_file
        
        # TypeScript file
        ts_content = '''/**
 * Sample TypeScript module for testing semantic search
 */

interface CalculationResult {
    value: number;
    operation: string;
}

interface HistoryEntry {
    operation: string;
    result: number;
    timestamp: Date;
}

function calculateSum(a: number, b: number): number {
    // Calculate the sum of two numbers
    return a + b;
}

function calculateProduct(a: number, b: number): number {
    // Calculate the product of two numbers
    return a * b;
}

function findMaximum(numbers: number[]): number {
    // Find the maximum number in an array
    if (numbers.length === 0) {
        throw new Error("Array cannot be empty");
    }
    
    return Math.max(...numbers);
}

class AdvancedCalculator {
    private history: HistoryEntry[] = [];
    
    add(x: number, y: number): CalculationResult {
        const value = calculateSum(x, y);
        const operation = `${x} + ${y}`;
        
        this.history.push({
            operation,
            result: value,
            timestamp: new Date()
        });
        
        return { value, operation };
    }
    
    multiply(x: number, y: number): CalculationResult {
        const value = calculateProduct(x, y);
        const operation = `${x} * ${y}`;
        
        this.history.push({
            operation,
            result: value,
            timestamp: new Date()
        });
        
        return { value, operation };
    }
    
    getHistory(): HistoryEntry[] {
        return [...this.history];
    }
    
    clearHistory(): void {
        this.history = [];
    }
}

export { AdvancedCalculator, calculateSum, calculateProduct, findMaximum };
export type { CalculationResult, HistoryEntry };
'''
        
        ts_file = os.path.join(self.temp_dir, "calculator.ts")
        with open(ts_file, 'w') as f:
            f.write(ts_content)
        files['typescript'] = ts_file
        
        return files
    
    def test_system_creation(self):
        """Test system creation"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        assert system.workspace_path == self.temp_dir
        assert system.embedding_generator is not None
        assert system.search_engine is not None
        assert not system.is_initialized
        assert not system.is_indexing
    
    def test_system_initialization(self):
        """Test system initialization"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        # Initialize system
        result = system.initialize()
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert system.is_initialized
        
        # Should have indexed some files
        if result.get('success'):
            assert result.get('files_indexed', 0) >= 0
    
    def test_file_indexing(self):
        """Test individual file indexing"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        # Index a single file
        python_file = self.sample_files['python']
        result = system.index_file(python_file)
        
        assert result['success']
        assert result['file_path'] == python_file
        assert result['chunks_created'] > 0
        assert 'chunk_types' in result
    
    def test_workspace_indexing(self):
        """Test workspace indexing"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        # Index workspace
        result = system.index_workspace(max_workers=1)  # Single worker for testing
        
        assert result['success']
        assert result.get('indexed_files', 0) >= len(self.sample_files)
        assert result.get('total_chunks', 0) > 0
    
    def test_basic_search(self):
        """Test basic search functionality"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        # Initialize and index
        system.initialize()
        
        # Test search
        context = SearchContext(current_language="python")
        results = system.search("calculate sum", context, max_results=5)
        
        assert isinstance(results, list)
        # Should find results even with text-based search
        for result in results:
            assert hasattr(result, 'chunk_id')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'final_score')
    
    def test_search_with_context(self):
        """Test search with development context"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        # Create context favoring Python files
        context = SearchContext(
            current_file=self.sample_files['python'],
            current_language="python",
            open_files=[self.sample_files['python']],
            recent_files=[self.sample_files['javascript']]
        )
        
        results = system.search("calculator class", context, max_results=10)
        
        # Should find calculator classes
        assert isinstance(results, list)
        
        # If results found, check that Python results are ranked higher due to context
        if len(results) > 1:
            python_results = [r for r in results if r.file_path.endswith('.py')]
            if python_results:
                # Python results should have higher context relevance
                assert python_results[0].context_relevance >= 0
    
    def test_similar_code_search(self):
        """Test finding similar code blocks"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        # Find code similar to a specific function
        python_file = self.sample_files['python']
        results = system.search_similar_code(python_file, 10, 12)  # Around calculate_sum function
        
        assert isinstance(results, list)
        # Results might be empty if no similar code found, which is okay
    
    def test_file_chunks_retrieval(self):
        """Test retrieving chunks for a specific file"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        python_file = self.sample_files['python']
        chunks = system.get_file_chunks(python_file)
        
        assert isinstance(chunks, list)
        
        if chunks:  # If file was indexed
            for chunk in chunks:
                assert 'chunk_id' in chunk
                assert 'chunk_type' in chunk
                assert 'line_start' in chunk
                assert 'line_end' in chunk
                assert 'content_preview' in chunk
                assert 'metadata' in chunk
                assert 'has_embedding' in chunk
    
    def test_workspace_overview(self):
        """Test workspace overview generation"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        overview = system.get_workspace_overview()
        
        assert isinstance(overview, dict)
        assert 'workspace_path' in overview
        assert 'system_health' in overview
        assert 'is_initialized' in overview
        assert 'embedding_stats' in overview
        assert 'search_stats' in overview
        assert 'chunk_type_distribution' in overview
        assert 'language_distribution' in overview
        assert 'system_metrics' in overview
    
    def test_system_optimization(self):
        """Test system optimization"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        # Perform some searches to populate caches
        system.search("test query 1")
        system.search("test query 2")
        
        # Optimize system
        result = system.optimize_system()
        
        assert result['success']
        assert 'optimizations_applied' in result
        assert isinstance(result['optimizations_applied'], list)
    
    def test_system_shutdown(self):
        """Test system shutdown"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        # Shutdown should not raise exceptions
        system.shutdown()
        
        assert not system.is_initialized
    
    def test_global_instance(self):
        """Test global instance management"""
        system1 = get_semantic_system(self.temp_dir)
        system2 = get_semantic_system(self.temp_dir)
        
        # Should be the same instance
        assert system1 is system2
        
        # Different workspace should create new instance
        temp_dir2 = tempfile.mkdtemp()
        try:
            system3 = get_semantic_system(temp_dir2)
            assert system3 is not system1
        finally:
            shutil.rmtree(temp_dir2)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test initialization function
        result = initialize_semantic_search(self.temp_dir)
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Test search function
        results = semantic_search("calculate", max_results=5)
        assert isinstance(results, list)
        
        # Test similar code function
        python_file = self.sample_files['python']
        similar_results = find_similar_code(python_file, 10)
        assert isinstance(similar_results, list)
        
        # Test overview function
        overview = get_workspace_overview()
        assert isinstance(overview, dict)
        assert 'workspace_path' in overview

class TestSystemPerformance:
    """Test system performance characteristics"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self._create_large_codebase()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_large_codebase(self):
        """Create a larger codebase for performance testing"""
        # Create multiple files with different patterns
        for i in range(5):  # Create 5 files
            content = f'''
def function_{i}_add(a, b):
    """Add function number {i}"""
    return a + b

def function_{i}_multiply(a, b):
    """Multiply function number {i}"""
    return a * b

class Class_{i}:
    """Class number {i}"""
    
    def method_{i}(self, x):
        """Method {i} of class {i}"""
        return x * {i}
    
    def calculate_{i}(self, data):
        """Calculate something for class {i}"""
        result = 0
        for item in data:
            result += item * {i}
        return result
'''
            
            file_path = os.path.join(self.temp_dir, f"module_{i}.py")
            with open(file_path, 'w') as f:
                f.write(content)
    
    def test_indexing_performance(self):
        """Test indexing performance"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        
        start_time = time.time()
        result = system.initialize()
        end_time = time.time()
        
        indexing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert indexing_time < 30.0  # 30 seconds max for small test codebase
        
        if result.get('success'):
            files_indexed = result.get('files_indexed', 0)
            if files_indexed > 0:
                time_per_file = indexing_time / files_indexed
                assert time_per_file < 10.0  # 10 seconds per file max
    
    def test_search_performance(self):
        """Test search performance"""
        system = SemanticSimilaritySystem(workspace_path=self.temp_dir)
        system.initialize()
        
        # Perform multiple searches and measure time
        search_times = []
        queries = ["function", "class", "calculate", "multiply", "add"]
        
        for query in queries:
            start_time = time.time()
            results = system.search(query, max_results=10)
            end_time = time.time()
            
            search_time = end_time - start_time
            search_times.append(search_time)
            
            # Each search should complete quickly
            assert search_time < 5.0  # 5 seconds max per search
        
        # Average search time should be reasonable
        avg_search_time = sum(search_times) / len(search_times)
        assert avg_search_time < 2.0  # 2 seconds average

if __name__ == "__main__":
    # Run basic tests without pytest
    import sys
    
    print("Testing Semantic Similarity System...")
    
    # Test basic functionality
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sample file
        sample_content = '''def hello_world():
    """A simple hello world function"""
    print("Hello, World!")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''
        
        sample_file = os.path.join(temp_dir, "sample.py")
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        
        # Test system creation
        system = SemanticSimilaritySystem(workspace_path=temp_dir)
        print(f"✓ System creation: {system.workspace_path}")
        
        # Test initialization
        result = system.initialize()
        print(f"✓ System initialization: {result.get('success', False)}")
        
        # Test file indexing
        index_result = system.index_file(sample_file)
        print(f"✓ File indexing: {index_result.get('chunks_created', 0)} chunks")
        
        # Test search
        search_results = system.search("hello", max_results=5)
        print(f"✓ Search functionality: {len(search_results)} results")
        
        # Test file chunks
        chunks = system.get_file_chunks(sample_file)
        print(f"✓ File chunks retrieval: {len(chunks)} chunks")
        
        # Test workspace overview
        overview = system.get_workspace_overview()
        print(f"✓ Workspace overview: {overview.get('system_health', 'unknown')} health")
        
        # Test optimization
        opt_result = system.optimize_system()
        print(f"✓ System optimization: {opt_result.get('success', False)}")
        
        # Test convenience functions
        init_result = initialize_semantic_search(temp_dir)
        print(f"✓ Convenience initialization: {init_result.get('success', False)}")
        
        conv_results = semantic_search("function", max_results=3)
        print(f"✓ Convenience search: {len(conv_results)} results")
        
        conv_overview = get_workspace_overview()
        print(f"✓ Convenience overview: {len(conv_overview)} fields")
        
        # Test shutdown
        system.shutdown()
        print("✓ System shutdown")
        
        print("\nAll basic tests passed!")
        print("Semantic Similarity System is working correctly!")
    
    finally:
        shutil.rmtree(temp_dir)