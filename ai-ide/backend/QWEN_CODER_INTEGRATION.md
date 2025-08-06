# Qwen Coder 3 Integration Documentation

## Overview

This document describes the integration of Qwen Coder 3 into the AI IDE system. The integration provides advanced code generation, completion, refactoring, debugging, and optimization capabilities through a sophisticated agent-based architecture.

## Architecture

### Core Components

1. **QwenCoderAgent** (`qwen_coder_agent.py`)
   - Main agent class for code generation tasks
   - Extends existing `call_llm` utility
   - Supports streaming responses
   - Provides context-aware code generation

2. **QwenCoderAPI** (`qwen_coder_api.py`)
   - FastAPI-based REST endpoints
   - Supports both synchronous and streaming responses
   - Comprehensive error handling
   - Performance monitoring

3. **Integration Layer** (updated `main.py`)
   - Seamless integration with existing PocketFlow system
   - Fallback mechanisms for reliability
   - Enhanced task routing

4. **VSCode Extension Integration** (updated `PocketFlowBridge.ts`)
   - TypeScript interfaces for Qwen Coder functionality
   - Convenient wrapper methods
   - Streaming support

## Features

### Code Task Types

The system supports the following code generation tasks:

- **COMPLETION**: Complete partial code snippets
- **GENERATION**: Generate new code from descriptions
- **REFACTORING**: Improve existing code structure and quality
- **DEBUGGING**: Identify and fix code issues
- **DOCUMENTATION**: Generate comprehensive code documentation
- **EXPLANATION**: Provide detailed code explanations
- **OPTIMIZATION**: Optimize code for better performance

### Context Awareness

The agent uses rich context information:

```python
@dataclass
class CodeContext:
    language: str
    file_path: Optional[str] = None
    selected_text: Optional[str] = None
    cursor_position: Optional[int] = None
    surrounding_code: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None
    imports: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    classes: Optional[List[str]] = None
```

### Prompt Templates

Specialized prompt templates for different tasks:

- **Completion Prompts**: Focus on code continuation
- **Generation Prompts**: Comprehensive code creation
- **Refactoring Prompts**: Code improvement guidance
- **Debugging Prompts**: Issue identification and fixing
- **Documentation Prompts**: Comprehensive documentation generation
- **Explanation Prompts**: Detailed code analysis
- **Optimization Prompts**: Performance improvement focus

## API Endpoints

### REST API

The Qwen Coder API provides the following endpoints:

#### Code Completion
- `POST /complete` - Complete code (synchronous)
- `POST /complete/stream` - Complete code (streaming)

#### Code Generation
- `POST /generate` - Generate code (synchronous)
- `POST /generate/stream` - Generate code (streaming)

#### Code Operations
- `POST /refactor` - Refactor existing code
- `POST /debug` - Debug code issues
- `POST /document` - Generate documentation
- `POST /explain` - Explain code functionality
- `POST /optimize` - Optimize code performance

#### System Endpoints
- `GET /health` - Health check and status
- `GET /stats` - Performance statistics
- `GET /models` - Available model information

### Request/Response Format

#### Example Request
```json
{
  "prompt": "Create a function to calculate factorial",
  "context": {
    "language": "python",
    "file_path": "math_utils.py",
    "project_context": {
      "type": "utility_library"
    }
  },
  "max_tokens": 2048,
  "temperature": 0.3,
  "include_explanation": true
}
```

#### Example Response
```json
{
  "code": "def factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)",
  "language": "python",
  "confidence": 0.95,
  "explanation": "This function calculates the factorial of a number using recursion...",
  "execution_time": 1.234,
  "model_info": {
    "model_id": "microsoft/phi-4-reasoning-plus",
    "tokens_used": 156
  }
}
```

## Usage Examples

### Python Backend Usage

```python
from qwen_coder_agent import get_qwen_coder_agent, CodeRequest, CodeContext, CodeTaskType

# Initialize agent
agent = await get_qwen_coder_agent()

# Create context
context = CodeContext(
    language="python",
    file_path="example.py",
    selected_text="def incomplete_function():"
)

# Create request
request = CodeRequest(
    prompt="def incomplete_function():",
    task_type=CodeTaskType.COMPLETION,
    context=context
)

# Generate code
response = await agent.generate_code(request)
print(f"Generated code: {response.code}")
```

### Convenience Functions

```python
# Code completion
response = await complete_code(
    code="def calculate_sum(a, b):",
    language="python",
    context={'file_path': 'math.py'}
)

# Code generation
response = await generate_code(
    prompt="Create a REST API endpoint for user authentication",
    language="python",
    include_explanation=True
)

# Code refactoring
response = await refactor_code(
    code="# messy code here",
    language="python",
    refactoring_request="Improve readability and add error handling"
)
```

### VSCode Extension Usage

```typescript
import { PocketFlowBridge } from './services/PocketFlowBridge';

const bridge = new PocketFlowBridge(extensionPath);
await bridge.initialize();

// Complete code
const completion = await bridge.completeCode(
    "def calculate_",
    "python",
    { filePath: "math.py" }
);

// Generate code
const generation = await bridge.generateCode(
    "Create a function to validate email addresses",
    "python",
    { includeExplanation: true }
);

// Refactor code
const refactored = await bridge.refactorCode(
    selectedCode,
    "python",
    "Improve performance and add type hints"
);
```

## Configuration

### Environment Variables

- `LMSTUDIO_URL`: LM Studio server URL (default: http://localhost:1234)
- `LOG_DIR`: Directory for log files (default: logs)

### Model Configuration

The system automatically detects and uses the best available model from LM Studio:

- **Primary**: `microsoft/phi-4-reasoning-plus`
- **Fallback**: `phi-4-mini-instruct`
- **Code-specific**: Models optimized for code generation

## Performance Features

### Connection Pooling
- Efficient HTTP connection management
- Automatic retry mechanisms
- Circuit breaker patterns

### Caching
- Response caching for repeated requests
- Context caching for improved performance
- Model selection optimization

### Monitoring
- Request/response time tracking
- Success rate monitoring
- Model performance analytics
- Resource usage tracking

## Error Handling

### Graceful Degradation
1. **Qwen Coder Agent** (Primary)
2. **LM Studio Manager** (Fallback)
3. **PocketFlow Engine** (Secondary fallback)
4. **Static Templates** (Final fallback)

### Error Recovery
- Automatic retry with exponential backoff
- Circuit breaker for failing services
- Comprehensive error logging
- User-friendly error messages

## Testing

### Test Files
- `test_qwen_coder_integration.py` - Comprehensive integration tests
- `test_qwen_simple.py` - Basic functionality tests

### Test Coverage
- Code completion accuracy
- Generation quality
- Refactoring effectiveness
- Debugging capabilities
- Performance benchmarks
- Error handling scenarios

### Running Tests

```bash
# Basic functionality test
python ai-ide/backend/test_qwen_simple.py

# Comprehensive integration test
python ai-ide/backend/test_qwen_coder_integration.py

# Start API server for testing
python ai-ide/backend/start_qwen_coder_api.py --host localhost --port 8001
```

## Deployment

### Starting the API Server

```bash
# Development mode with auto-reload
python ai-ide/backend/start_qwen_coder_api.py --reload

# Production mode
python ai-ide/backend/start_qwen_coder_api.py --host 0.0.0.0 --port 8001
```

### Integration with VSCode

The Qwen Coder integration is automatically available through the existing PocketFlow bridge when the backend is running.

## Performance Benchmarks

### Expected Performance
- **Code Completion**: < 500ms response time
- **Code Generation**: < 2s for typical requests
- **Accuracy**: > 85% for code completions
- **Confidence**: > 0.8 for successful generations

### Optimization Features
- Streaming responses for real-time feedback
- Context-aware prompt optimization
- Intelligent model selection
- Response caching

## Future Enhancements

### Planned Features
1. **Multi-agent Collaboration**: Integration with other AI agents
2. **Advanced Context Windows**: Apple's interleaved context sliding windows
3. **Self-improvement**: Darwin-Gödel model integration
4. **Reinforcement Learning**: User feedback-based improvement
5. **Web Search Integration**: Real-time information retrieval

### Extensibility
The architecture is designed to support:
- Additional code task types
- New programming languages
- Custom prompt templates
- External tool integrations
- Advanced reasoning modes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Python path includes the backend directory
2. **LM Studio Connection**: Verify LM Studio is running on the correct port
3. **Model Loading**: Check that appropriate models are loaded in LM Studio
4. **Performance Issues**: Monitor system resources and adjust concurrency

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Monitor system health:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/stats
```

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use TypeScript best practices for extension code
- Include comprehensive docstrings
- Add type hints where appropriate

### Testing Requirements
- All new features must include tests
- Maintain > 80% code coverage
- Include performance benchmarks
- Test error handling scenarios

### Documentation
- Update this document for new features
- Include usage examples
- Document API changes
- Provide migration guides for breaking changes