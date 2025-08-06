# Semantic Similarity Search System

## Overview

The Semantic Similarity Search System provides advanced code search capabilities using semantic embeddings and context-aware ranking. It enables developers to find relevant code patterns, functions, and classes based on meaning rather than just text matching.

## Features

### Core Capabilities

1. **Code Embedding Generation**
   - Generates semantic embeddings for code chunks using sentence-transformers
   - Supports multiple programming languages (Python, JavaScript, TypeScript, Java, C++, etc.)
   - Incremental indexing for changed files
   - Vector storage using FAISS or ChromaDB

2. **Semantic Search Engine**
   - Context-aware search with intelligent ranking
   - Query preprocessing with search modifiers
   - Result caching for performance optimization
   - Fallback to text-based search when embeddings unavailable

3. **Context-Aware Ranking**
   - Considers current file, language, and development context
   - Weights results based on file proximity and recency
   - Adapts to user preferences and search patterns

## Architecture

```
SemanticSimilaritySystem
├── CodeEmbeddingGenerator
│   ├── CodeChunk (represents code segments)
│   ├── Embedding Model (sentence-transformers)
│   └── Vector Store (FAISS/ChromaDB)
└── SemanticSearchEngine
    ├── ContextAwareRanker
    ├── SearchCache
    └── Query Processor
```

## Installation

### Required Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install numpy scikit-learn

# Optional: For advanced semantic search
pip install sentence-transformers faiss-cpu chromadb
```

### Supported Models

- **Default**: `microsoft/codebert-base` (optimized for code)
- **Fallback**: `all-MiniLM-L6-v2` (smaller, faster)
- **Custom**: Any sentence-transformers compatible model

## Usage

### Basic Usage

```python
from semantic_similarity_system import (
    initialize_semantic_search, 
    semantic_search, 
    find_similar_code,
    get_workspace_overview
)

# Initialize the system
result = initialize_semantic_search("/path/to/workspace")
print(f"Initialization: {result['success']}")

# Perform semantic search
results = semantic_search("calculate sum function", max_results=10)
for result in results:
    print(f"Found: {result.file_path}:{result.line_start} (score: {result.final_score:.2f})")

# Find similar code blocks
similar = find_similar_code("calculator.py", 15, 25)
for result in similar:
    print(f"Similar: {result.file_path} - {result.snippet}")

# Get workspace overview
overview = get_workspace_overview()
print(f"Health: {overview['system_health']}")
print(f"Files indexed: {overview['embedding_stats']['total_files']}")
```

### Advanced Usage

```python
from semantic_similarity_system import SemanticSimilaritySystem
from semantic_search_engine import SearchContext

# Create system with custom model
system = SemanticSimilaritySystem(
    workspace_path="/path/to/workspace",
    model_name="microsoft/codebert-base"
)

# Initialize with force reindex
result = system.initialize(force_reindex=True)

# Create search context for better ranking
context = SearchContext(
    current_file="main.py",
    current_language="python",
    open_files=["main.py", "utils.py"],
    recent_files=["test.py", "config.py"],
    selected_text="def calculate",
    recent_searches=["function", "class"]
)

# Search with context and options
results = system.search(
    query="lang:python type:function calculate",
    context=context,
    max_results=20,
    chunk_types=["function", "class"],
    min_score=0.5
)

# Index specific file
index_result = system.index_file("new_file.py", force_reindex=True)

# Get file chunks
chunks = system.get_file_chunks("calculator.py")
for chunk in chunks:
    print(f"Chunk: {chunk['chunk_type']} at line {chunk['line_start']}")

# Optimize system performance
system.optimize_system()
```

### Search Query Syntax

The system supports advanced query syntax with modifiers:

```python
# Language filter
results = semantic_search("lang:python function definition")

# Chunk type filter
results = semantic_search("type:class calculator")

# File pattern filter
results = semantic_search("file:*.py import statement")

# Combined filters
results = semantic_search("lang:javascript type:function file:utils.* helper")
```

## API Reference

### SemanticSimilaritySystem

#### Methods

- `initialize(force_reindex=False)` - Initialize the system
- `index_workspace(extensions=None, max_workers=4)` - Index entire workspace
- `index_file(file_path, force_reindex=False)` - Index single file
- `search(query, context=None, max_results=10, **options)` - Perform search
- `search_similar_code(file_path, line_start, line_end=None, max_results=10)` - Find similar code
- `get_file_chunks(file_path)` - Get chunks for file
- `get_workspace_overview()` - Get system overview
- `optimize_system()` - Optimize performance
- `shutdown()` - Shutdown gracefully

### SearchContext

#### Properties

- `current_file` - Currently active file
- `current_language` - Current programming language
- `open_files` - List of open files
- `recent_files` - Recently accessed files
- `cursor_position` - Current cursor position (line, column)
- `selected_text` - Currently selected text
- `recent_searches` - Recent search queries
- `user_preferences` - User preference settings

### SearchResult

#### Properties

- `chunk_id` - Unique chunk identifier
- `file_path` - Path to the file
- `content` - Full chunk content
- `chunk_type` - Type (function, class, file, comment)
- `line_start/line_end` - Line range
- `similarity_score` - Semantic similarity score
- `context_relevance` - Context relevance score
- `final_score` - Combined final score
- `metadata` - Additional metadata
- `snippet` - Content snippet with highlights
- `highlights` - Highlight positions

## Performance Considerations

### Optimization Tips

1. **Model Selection**
   - Use `microsoft/codebert-base` for best code understanding
   - Use `all-MiniLM-L6-v2` for faster performance
   - Consider local models for privacy

2. **Indexing Strategy**
   - Index incrementally for large codebases
   - Use appropriate file extensions filter
   - Adjust max_workers based on system resources

3. **Search Performance**
   - Results are cached automatically
   - Use specific search modifiers to narrow results
   - Context-aware ranking improves relevance

4. **Memory Management**
   - System automatically manages embedding storage
   - Use `optimize_system()` periodically
   - Monitor system health via `get_workspace_overview()`

### System Requirements

- **Minimum**: 4GB RAM, text-based search only
- **Recommended**: 8GB RAM with sentence-transformers
- **Optimal**: 16GB RAM with FAISS/ChromaDB for large codebases

## Troubleshooting

### Common Issues

1. **"sentence-transformers not available"**
   ```bash
   pip install sentence-transformers
   ```

2. **"FAISS not available"**
   ```bash
   pip install faiss-cpu  # or faiss-gpu for GPU support
   ```

3. **"ChromaDB not available"**
   ```bash
   pip install chromadb
   ```

4. **Slow indexing performance**
   - Reduce `max_workers` parameter
   - Filter file extensions to relevant types only
   - Use incremental indexing instead of full reindex

5. **Poor search results**
   - Ensure files are properly indexed
   - Use more specific search queries
   - Provide better search context
   - Check system health via `get_workspace_overview()`

### Fallback Behavior

The system gracefully degrades when advanced features are unavailable:

- **No sentence-transformers**: Falls back to text-based search
- **No vector store**: Uses in-memory storage
- **No embeddings**: Uses keyword matching and metadata search

## Integration Examples

### VSCode Extension Integration

```typescript
// TypeScript extension code
import { spawn } from 'child_process';

class SemanticSearchProvider {
    private pythonProcess: any;
    
    async search(query: string, context: any): Promise<SearchResult[]> {
        return new Promise((resolve, reject) => {
            const python = spawn('python', ['-c', `
from semantic_similarity_system import semantic_search
from semantic_search_engine import SearchContext
import json

context = SearchContext(
    current_file="${context.currentFile}",
    current_language="${context.language}",
    open_files=${JSON.stringify(context.openFiles)}
)

results = semantic_search("${query}", context, max_results=10)
print(json.dumps([r.to_dict() for r in results]))
            `]);
            
            let output = '';
            python.stdout.on('data', (data) => output += data);
            python.on('close', (code) => {
                if (code === 0) {
                    resolve(JSON.parse(output));
                } else {
                    reject(new Error('Search failed'));
                }
            });
        });
    }
}
```

### Web API Integration

```python
from flask import Flask, request, jsonify
from semantic_similarity_system import get_semantic_system
from semantic_search_engine import SearchContext

app = Flask(__name__)
system = get_semantic_system()

@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    query = data.get('query', '')
    
    context = SearchContext(
        current_file=data.get('current_file'),
        current_language=data.get('current_language'),
        open_files=data.get('open_files', [])
    )
    
    results = system.search(query, context, max_results=data.get('max_results', 10))
    
    return jsonify({
        'results': [r.to_dict() for r in results],
        'total': len(results)
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_api():
    result = system.initialize()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

## Future Enhancements

### Planned Features

1. **Multi-modal Search**
   - Support for documentation and comments
   - Integration with external knowledge bases
   - Cross-repository search capabilities

2. **Learning and Adaptation**
   - User feedback integration
   - Personalized ranking algorithms
   - Automatic query expansion

3. **Advanced Analytics**
   - Code pattern analysis
   - Similarity clustering
   - Refactoring suggestions

4. **Performance Improvements**
   - Distributed indexing
   - GPU acceleration
   - Streaming search results

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-ide/backend

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_semantic_similarity_system.py
python test_code_embedding_generator.py
python test_semantic_search_engine.py
```

### Testing

The system includes comprehensive tests:

- Unit tests for individual components
- Integration tests for system workflows
- Performance tests for large codebases
- Fallback behavior tests

### Code Structure

```
ai-ide/backend/
├── code_embedding_generator.py      # Embedding generation
├── semantic_search_engine.py        # Search and ranking
├── semantic_similarity_system.py    # Main integration
├── test_*.py                        # Test suites
└── SEMANTIC_SEARCH_SYSTEM.md       # This documentation
```

## License

This semantic search system is part of the AI IDE project and follows the same licensing terms.