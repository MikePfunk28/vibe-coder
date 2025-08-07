"""
Comprehensive Test Configuration for AI IDE
Centralizes all test settings, thresholds, and configurations
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestThresholds:
    """Test quality thresholds"""
    semantic_accuracy: float = 0.7
    reasoning_quality: float = 0.75
    agent_coordination: float = 0.8
    web_search_accuracy: float = 0.7
    safety_validation: float = 0.6
    user_satisfaction: float = 0.75
    response_time_max: float = 5.0
    code_coverage: float = 0.8

@dataclass
class TestEnvironment:
    """Test environment configuration"""
    backend_url: str = "http://localhost:8000"
    test_database_url: str = "postgresql://test:test@localhost:5432/test_ai_ide"
    mock_ai_models: bool = True
    enable_web_search: bool = False
    enable_mcp_integration: bool = True
    log_level: str = "INFO"
    test_data_path: str = "test_data"
    temp_dir: str = "/tmp/ai_ide_tests"

class TestConfiguration:
    """Main test configuration class"""
    
    def __init__(self):
        self.thresholds = TestThresholds()
        self.environment = TestEnvironment()
        self.test_suites = self._configure_test_suites()
        self.ai_models = self._configure_ai_models()
        self.database = self._configure_database()
        self.mcp_servers = self._configure_mcp_servers()
        
    def _configure_test_suites(self) -> Dict[str, Dict[str, Any]]:
        """Configure test suite settings"""
        return {
            "unit_tests": {
                "enabled": True,
                "timeout": 30,
                "parallel": True,
                "coverage_required": True,
                "markers": ["unit"]
            },
            "integration_tests": {
                "enabled": True,
                "timeout": 120,
                "parallel": False,
                "requires_backend": True,
                "markers": ["integration"]
            },
            "ai_validation_tests": {
                "enabled": True,
                "timeout": 300,
                "parallel": False,
                "requires_ai_models": True,
                "markers": ["ai_model", "slow"]
            },
            "performance_tests": {
                "enabled": True,
                "timeout": 600,
                "parallel": False,
                "requires_benchmark_data": True,
                "markers": ["performance", "slow"]
            },
            "security_tests": {
                "enabled": True,
                "timeout": 180,
                "parallel": True,
                "requires_security_tools": True,
                "markers": ["security"]
            },
            "extension_tests": {
                "enabled": True,
                "timeout": 240,
                "parallel": False,
                "requires_vscode": True,
                "markers": ["extension"]
            }
        }
        
    def _configure_ai_models(self) -> Dict[str, Dict[str, Any]]:
        """Configure AI model settings for testing"""
        return {
            "mock_models": {
                "enabled": self.environment.mock_ai_models,
                "response_delay": 0.1,
                "error_rate": 0.05,
                "confidence_range": (0.7, 0.95)
            },
            "lm_studio": {
                "enabled": not self.environment.mock_ai_models,
                "host": "localhost",
                "port": 1234,
                "model": "qwen-coder-3",
                "timeout": 30
            },
            "embedding_model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "cache_embeddings": True
            }
        }
        
    def _configure_database(self) -> Dict[str, Any]:
        """Configure database settings for testing"""
        return {
            "url": self.environment.test_database_url,
            "pool_size": 5,
            "max_overflow": 10,
            "echo": False,
            "create_tables": True,
            "drop_tables_after": True,
            "fixtures": {
                "load_sample_data": True,
                "sample_embeddings": 100,
                "sample_interactions": 50
            }
        }
        
    def _configure_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Configure MCP servers for testing"""
        return {
            "test_server": {
                "enabled": self.environment.enable_mcp_integration,
                "command": "python",
                "args": ["-m", "test_mcp_server"],
                "tools": ["search", "analyze", "generate"],
                "timeout": 10
            },
            "mock_github": {
                "enabled": True,
                "mock_responses": True,
                "tools": ["search_repos", "get_file", "create_issue"]
            }
        }
        
    def get_pytest_args(self) -> List[str]:
        """Get pytest command line arguments"""
        args = [
            "-v",
            "--tb=short",
            "--strict-markers",
            "--color=yes",
            f"--cov-fail-under={int(self.thresholds.code_coverage * 100)}",
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ]
        
        # Add parallel execution for unit tests
        if self.test_suites["unit_tests"]["parallel"]:
            args.extend(["-n", "auto"])
            
        return args
        
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for testing"""
        return {
            "AI_IDE_TEST_MODE": "true",
            "AI_IDE_BACKEND_URL": self.environment.backend_url,
            "AI_IDE_TEST_DB_URL": self.environment.test_database_url,
            "AI_IDE_LOG_LEVEL": self.environment.log_level,
            "AI_IDE_MOCK_AI": str(self.environment.mock_ai_models),
            "AI_IDE_ENABLE_WEB_SEARCH": str(self.environment.enable_web_search),
            "AI_IDE_ENABLE_MCP": str(self.environment.enable_mcp_integration),
            "AI_IDE_TEST_DATA_PATH": self.environment.test_data_path,
            "AI_IDE_TEMP_DIR": self.environment.temp_dir
        }
        
    def validate_configuration(self) -> List[str]:
        """Validate test configuration and return any issues"""
        issues = []
        
        # Check required directories
        test_data_path = Path(self.environment.test_data_path)
        if not test_data_path.exists():
            issues.append(f"Test data directory does not exist: {test_data_path}")
            
        # Check database connection if not mocking
        if not self.environment.mock_ai_models:
            # Would check database connectivity here
            pass
            
        # Check AI model availability if not mocking
        if not self.environment.mock_ai_models:
            # Would check LM Studio connectivity here
            pass
            
        # Validate thresholds
        if self.thresholds.semantic_accuracy < 0.5:
            issues.append("Semantic accuracy threshold too low")
            
        if self.thresholds.code_coverage < 0.7:
            issues.append("Code coverage threshold too low")
            
        return issues
        
    def create_test_data_directory(self):
        """Create test data directory structure"""
        base_path = Path(self.environment.test_data_path)
        
        directories = [
            "embeddings",
            "code_samples",
            "search_results",
            "reasoning_traces",
            "agent_interactions",
            "benchmarks",
            "fixtures"
        ]
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
            
    def generate_sample_test_data(self):
        """Generate sample test data for testing"""
        base_path = Path(self.environment.test_data_path)
        
        # Sample code snippets
        code_samples = [
            {
                "file": "example1.py",
                "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python",
                "tags": ["recursion", "algorithm", "fibonacci"]
            },
            {
                "file": "example2.py", 
                "content": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "language": "python",
                "tags": ["search", "algorithm", "binary_search"]
            }
        ]
        
        import json
        with open(base_path / "code_samples" / "samples.json", "w") as f:
            json.dump(code_samples, f, indent=2)
            
        # Sample reasoning traces
        reasoning_traces = [
            {
                "problem": "Optimize bubble sort algorithm",
                "steps": [
                    "Analyze current O(n²) complexity",
                    "Identify nested loop inefficiency", 
                    "Consider quicksort alternative",
                    "Implement O(n log n) solution"
                ],
                "solution": "Replace with quicksort implementation"
            }
        ]
        
        with open(base_path / "reasoning_traces" / "traces.json", "w") as f:
            json.dump(reasoning_traces, f, indent=2)

# Global test configuration instance
test_config = TestConfiguration()

# Environment-specific configurations
class DevelopmentTestConfig(TestConfiguration):
    """Development environment test configuration"""
    
    def __init__(self):
        super().__init__()
        self.environment.mock_ai_models = True
        self.environment.enable_web_search = False
        self.thresholds.code_coverage = 0.7  # Lower for development

class CITestConfig(TestConfiguration):
    """CI/CD environment test configuration"""
    
    def __init__(self):
        super().__init__()
        self.environment.mock_ai_models = True
        self.environment.enable_web_search = False
        self.environment.enable_mcp_integration = False
        self.thresholds.code_coverage = 0.85  # Higher for CI

class ProductionTestConfig(TestConfiguration):
    """Production-like test configuration"""
    
    def __init__(self):
        super().__init__()
        self.environment.mock_ai_models = False
        self.environment.enable_web_search = True
        self.environment.enable_mcp_integration = True
        self.thresholds.code_coverage = 0.9  # Highest for production

def get_test_config() -> TestConfiguration:
    """Get test configuration based on environment"""
    env = os.getenv("AI_IDE_TEST_ENV", "development")
    
    if env == "ci":
        return CITestConfig()
    elif env == "production":
        return ProductionTestConfig()
    else:
        return DevelopmentTestConfig()

if __name__ == "__main__":
    # Initialize test configuration
    config = get_test_config()
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Test configuration is valid")
        
    # Create test data directory
    config.create_test_data_directory()
    config.generate_sample_test_data()
    
    print(f"Test configuration initialized for {os.getenv('AI_IDE_TEST_ENV', 'development')} environment")