# AI IDE Comprehensive Testing Framework

This document describes the comprehensive testing framework for the AI IDE project, covering all aspects of testing from unit tests to AI-specific validation.

## Overview

The testing framework provides:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component and system integration testing
- **AI-Specific Validation**: Semantic accuracy, reasoning quality, and agent coordination testing
- **Performance Tests**: Benchmarking and performance validation
- **Security Tests**: Safety validation and security scanning
- **Extension Tests**: VSCodium extension testing

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (for integration tests)
- Redis (for caching tests)

### Installation

```bash
# Install backend dependencies
cd ai-ide/backend
pip install -r requirements.txt

# Install extension dependencies
cd ../extensions/ai-assistant
npm install
```

### Running Tests

```bash
# Run all tests
python run_comprehensive_tests.py

# Run specific test types
python run_comprehensive_tests.py --types backend extension

# Run with dependency installation
python run_comprehensive_tests.py --install-deps

# Run in CI environment
python run_comprehensive_tests.py --env ci
```

## Test Structure

### Backend Tests

Located in `ai-ide/backend/`:

- `test_comprehensive_unit_tests.py` - Unit tests for all core components
- `test_ai_specific_validation.py` - AI-specific testing and validation
- `test_integration.py` - Integration tests
- `test_framework.py` - Core testing framework
- `test_config.py` - Test configuration management
- `test_runner.py` - Main test orchestrator

### Extension Tests

Located in `ai-ide/extensions/ai-assistant/src/test/`:

- `integration.test.ts` - Extension integration tests
- `unit/` - Unit tests for extension components

### Configuration

Test configuration is managed through `test_config.py`:

```python
# Development environment (default)
python run_comprehensive_tests.py --env development

# CI environment (mocked AI models)
python run_comprehensive_tests.py --env ci

# Production-like environment
python run_comprehensive_tests.py --env production
```

## Test Categories

### 1. Unit Tests

Test individual components in isolation:

```bash
# Run only unit tests
pytest -m "unit" -v
```

**Coverage includes:**
- PocketFlow integration
- LM Studio manager
- Qwen Coder agent
- Semantic search engine
- Code embedding generator
- Darwin-Gödel model
- Multi-agent system
- All other core components

### 2. Integration Tests

Test component interactions and system integration:

```bash
# Run integration tests
pytest -m "integration" -v
```

**Coverage includes:**
- VSCodium extension integration
- Backend service integration
- Database integration
- MCP server integration
- External service integration

### 3. AI-Specific Validation

Test AI model accuracy and quality:

```bash
# Run AI validation tests
pytest test_ai_specific_validation.py -v
```

**Coverage includes:**
- **Semantic Accuracy**: RAG and search functionality accuracy
- **Reasoning Quality**: Chain-of-thought and deep reasoning validation
- **Agent Coordination**: Multi-agent communication and coordination
- **Web Search Accuracy**: Search result relevance and accuracy
- **Safety Validation**: DGM modification safety with reasoning traces
- **User Satisfaction**: Multi-modal interaction testing

### 4. Performance Tests

Test system performance and benchmarks:

```bash
# Run performance tests
pytest -m "performance" -v
```

**Coverage includes:**
- Response time benchmarks
- Memory usage validation
- Throughput testing
- Load testing
- Performance regression detection

### 5. Security Tests

Test security and safety aspects:

```bash
# Run security tests
pytest -m "security" -v
```

**Coverage includes:**
- Code security scanning (Bandit)
- Dependency vulnerability checking (Safety)
- MCP tool security validation
- Improvement safety validation
- Input sanitization testing

## Test Quality Thresholds

The framework enforces quality thresholds:

- **Semantic Accuracy**: ≥70%
- **Reasoning Quality**: ≥75%
- **Agent Coordination**: ≥80%
- **Web Search Accuracy**: ≥70%
- **Safety Validation**: ≥60%
- **User Satisfaction**: ≥75%
- **Response Time**: ≤5.0 seconds
- **Code Coverage**: ≥80%

## CI/CD Integration

### GitHub Actions

The project includes comprehensive CI/CD workflows:

```yaml
# .github/workflows/comprehensive-tests.yml
- Backend unit tests
- Backend integration tests
- AI validation tests
- Extension tests
- Performance tests (scheduled)
- Security tests
- Comprehensive reporting
```

### Running in CI

```bash
# Set CI environment
export AI_IDE_TEST_ENV=ci

# Run tests
python run_comprehensive_tests.py --types all
```

## Test Data Management

### Test Data Structure

```
test_data/
├── embeddings/          # Sample embeddings
├── code_samples/        # Code snippets for testing
├── search_results/      # Mock search results
├── reasoning_traces/    # Sample reasoning traces
├── agent_interactions/  # Agent communication logs
├── benchmarks/         # Performance benchmarks
└── fixtures/           # Test fixtures
```

### Generating Test Data

```bash
# Generate sample test data
python test_config.py
```

## Mocking and Test Doubles

### AI Model Mocking

For CI and development environments, AI models are mocked:

```python
# Mock AI responses
mock_response = {
    "content": "Generated code or response",
    "confidence": 0.9,
    "reasoning": ["Step 1", "Step 2"],
    "agents_used": ["code_agent", "reasoning_agent"]
}
```

### External Service Mocking

External services (web search, MCP servers) are mocked in test environments:

```python
# Mock web search
mock_search_results = [
    {
        "title": "Python Tutorial",
        "url": "https://example.com",
        "snippet": "Learn Python programming"
    }
]
```

## Test Reporting

### Comprehensive Reports

The framework generates detailed test reports:

```json
{
  "summary": {
    "total_tests": 150,
    "passed": 142,
    "failed": 8,
    "success_rate": 94.7,
    "execution_time": 245.3
  },
  "detailed_results": { ... },
  "recommendations": [
    "Fix failing backend tests before deployment",
    "Address performance issues in search component"
  ]
}
```

### Coverage Reports

HTML coverage reports are generated:

```bash
# View coverage report
open htmlcov/index.html
```

## Debugging Tests

### Verbose Output

```bash
# Run with verbose output
python run_comprehensive_tests.py --verbose
```

### Individual Test Debugging

```bash
# Run specific test with debugging
pytest test_comprehensive_unit_tests.py::CoreComponentsUnitTests::test_semantic_search_engine_indexing -v -s
```

### Log Analysis

Test logs are available in:
- Console output (with `--verbose`)
- Test report JSON files
- Coverage reports

## Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_semantic_search_accuracy_with_code_context`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies**: Use `unittest.mock` for external services
4. **Test edge cases**: Include error conditions and boundary cases
5. **Maintain test data**: Keep test data current and relevant

### Test Organization

1. **Group related tests**: Use test classes for related functionality
2. **Use appropriate markers**: Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
3. **Keep tests independent**: Each test should be able to run in isolation
4. **Clean up resources**: Use fixtures and teardown methods

### Performance Considerations

1. **Use parallel execution**: Unit tests run in parallel by default
2. **Mock expensive operations**: Don't make real API calls in unit tests
3. **Optimize test data**: Use minimal test data sets
4. **Cache when appropriate**: Cache embeddings and other expensive computations

## Troubleshooting

### Common Issues

1. **Database connection errors**: Ensure PostgreSQL is running and accessible
2. **AI model timeouts**: Check LM Studio configuration or use mock mode
3. **Extension test failures**: Ensure Node.js dependencies are installed
4. **Permission errors**: Check file permissions and directory access

### Getting Help

1. Check test logs for detailed error messages
2. Run tests with `--verbose` for more information
3. Review test configuration in `test_config.py`
4. Check CI/CD workflow logs for environment-specific issues

## Contributing

When adding new features:

1. **Write tests first**: Follow TDD principles
2. **Maintain coverage**: Ensure new code has adequate test coverage
3. **Update documentation**: Update this README for new test types
4. **Run full test suite**: Ensure all tests pass before submitting PR

### Adding New Test Types

1. Create test file following naming convention: `test_new_feature.py`
2. Add appropriate markers: `@pytest.mark.unit` or `@pytest.mark.integration`
3. Update test configuration if needed
4. Add to CI/CD workflow if appropriate

## Conclusion

This comprehensive testing framework ensures the AI IDE maintains high quality, performance, and reliability across all components. Regular testing helps catch issues early and maintains confidence in the system's capabilities.

For questions or issues with the testing framework, please refer to the project documentation or create an issue in the repository.