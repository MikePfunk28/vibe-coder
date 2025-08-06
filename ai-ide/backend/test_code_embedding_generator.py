"""
Test suite for Code Embedding Generation System
"""

import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

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

from code_embedding_generator import (
    CodeChunk, CodeEmbeddingGenerator, get_embedding_generator,
    SENTENCE_TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE, CHROMADB_AVAILABLE
)

class TestCodeChunk:
    """Test CodeChunk functionality"""
    
    def test_code_chunk_creation(self):
        """Test basic code chunk creation"""
        chunk = CodeChunk(
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            metadata={"name": "hello", "language": "python"}
        )
        
        assert chunk.file_path == "test.py"
        assert chunk.chunk_type == "function"
        assert chunk.line_start == 1
        assert chunk.line_end == 2
        assert chunk.metadata["name"] == "hello"
        assert chunk.hash is not None
    
    def test_chunk_serialization(self):
        """Test chunk to/from dict conversion"""
        chunk = CodeChunk(
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            metadata={"name": "hello"}
        )
        
        # Convert to dict and back
        chunk_dict = chunk.to_dict()
        restored_chunk = CodeChunk.from_dict(chunk_dict)
        
        assert restored_chunk.file_path == chunk.file_path
        assert restored_chunk.content == chunk.content
        assert restored_chunk.chunk_type == chunk.chunk_type
        assert restored_chunk.hash == chunk.hash

class TestCodeEmbeddingGenerator:
    """Test CodeEmbeddingGenerator functionality"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_python_file(self, temp_workspace):
        """Create sample Python file for testing"""
        content = '''"""
Sample Python module for testing
"""

import os
import sys

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def main():
    """Main function"""
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.multiply(4, 5))

if __name__ == "__main__":
    main()
'''
        
        file_path = os.path.join(temp_workspace, "calculator.py")
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path
    
    @pytest.fixture
    def sample_js_file(self, temp_workspace):
        """Create sample JavaScript file for testing"""
        content = '''/**
 * Sample JavaScript module for testing
 */

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
}

function createCalculator() {
    return new Calculator();
}

const defaultCalc = createCalculator();

export { Calculator, createCalculator, defaultCalc };
'''
        
        file_path = os.path.join(temp_workspace, "calculator.js")
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path
    
    def test_language_detection(self, temp_workspace):
        """Test programming language detection"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        
        assert generator._detect_language("test.py") == "python"
        assert generator._detect_language("test.js") == "javascript"
        assert generator._detect_language("test.ts") == "typescript"
        assert generator._detect_language("test.java") == "java"
        assert generator._detect_language("test.cpp") == "cpp"
        assert generator._detect_language("test.txt") == "text"
    
    def test_python_code_chunking(self, temp_workspace, sample_python_file):
        """Test Python code chunking"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        chunks = generator.chunk_code_file(sample_python_file)
        
        # Should have file, class, functions, and comment chunks
        assert len(chunks) > 0
        
        # Check chunk types
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert 'file' in chunk_types
        assert 'class' in chunk_types
        assert 'function' in chunk_types
        
        # Find Calculator class chunk
        calc_chunks = [c for c in chunks if c.chunk_type == 'class' and c.metadata.get('name') == 'Calculator']
        assert len(calc_chunks) == 1
        
        calc_chunk = calc_chunks[0]
        assert 'class Calculator:' in calc_chunk.content
        assert calc_chunk.metadata['language'] == 'python'
        
        # Find function chunks
        func_chunks = [c for c in chunks if c.chunk_type == 'function']
        func_names = [c.metadata.get('name') for c in func_chunks]
        
        expected_functions = ['__init__', 'add', 'multiply', 'main']
        for func_name in expected_functions:
            assert func_name in func_names
    
    def test_js_code_chunking(self, temp_workspace, sample_js_file):
        """Test JavaScript code chunking"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        chunks = generator.chunk_code_file(sample_js_file)
        
        # Should have file, class, and function chunks
        assert len(chunks) > 0
        
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert 'file' in chunk_types
        assert 'class' in chunk_types
        assert 'function' in chunk_types
        
        # Find function chunks
        func_chunks = [c for c in chunks if c.chunk_type in ['function', 'arrow_function']]
        func_names = [c.metadata.get('name') for c in func_chunks]
        
        # Should find createCalculator function
        assert 'createCalculator' in func_names
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_embedding_generation(self, temp_workspace, sample_python_file):
        """Test embedding generation"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace, model_name="all-MiniLM-L6-v2")
        
        # Wait for model to load
        if not generator.model:
            pytest.skip("Embedding model failed to load")
        
        chunks = generator.chunk_code_file(sample_python_file)
        chunks_with_embeddings = generator.generate_embeddings(chunks)
        
        assert len(chunks_with_embeddings) > 0
        
        # Check that embeddings were generated
        for chunk in chunks_with_embeddings:
            if chunk.content.strip():  # Only non-empty chunks should have embeddings
                assert chunk.embedding is not None
                assert isinstance(chunk.embedding, np.ndarray)
                assert chunk.embedding.shape[0] == generator.embedding_dim
                assert chunk.embedding_model == generator.model_name
    
    def test_embedding_text_preparation(self, temp_workspace):
        """Test embedding text preparation"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        
        chunk = CodeChunk(
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2,
            metadata={"name": "hello", "language": "python"}
        )
        
        embedding_text = generator._prepare_embedding_text(chunk)
        
        assert "Type: function" in embedding_text
        assert "Language: python" in embedding_text
        assert "Name: hello" in embedding_text
        assert "File: test.py" in embedding_text
        assert "def hello():" in embedding_text
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_file_indexing(self, temp_workspace, sample_python_file):
        """Test single file indexing"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace, model_name="all-MiniLM-L6-v2")
        
        if not generator.model:
            pytest.skip("Embedding model failed to load")
        
        chunks = generator.index_file(sample_python_file)
        
        assert len(chunks) > 0
        assert sample_python_file in generator.chunk_index
        
        # Check that chunks are stored
        chunk_ids = generator.chunk_index[sample_python_file]
        assert len(chunk_ids) > 0
        
        for chunk_id in chunk_ids:
            assert chunk_id in generator.chunks
            chunk = generator.chunks[chunk_id]
            assert chunk.embedding is not None
    
    def test_reindexing_detection(self, temp_workspace, sample_python_file):
        """Test reindexing detection"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        
        # First indexing
        generator.index_file(sample_python_file)
        
        # Should not need reindexing
        assert not generator._needs_reindexing(sample_python_file)
        
        # Modify file
        with open(sample_python_file, 'a') as f:
            f.write("\n# New comment\n")
        
        # Should need reindexing now
        assert generator._needs_reindexing(sample_python_file)
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_workspace_indexing(self, temp_workspace, sample_python_file, sample_js_file):
        """Test workspace indexing"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace, model_name="all-MiniLM-L6-v2")
        
        if not generator.model:
            pytest.skip("Embedding model failed to load")
        
        result = generator.index_workspace(max_workers=1)  # Use single worker for testing
        
        assert result['indexed_files'] >= 2  # At least the two sample files
        assert result['total_chunks'] > 0
        assert result['total_time'] > 0
        
        # Check that both files are indexed
        assert sample_python_file in generator.chunk_index
        assert sample_js_file in generator.chunk_index
    
    def test_stats_tracking(self, temp_workspace):
        """Test statistics tracking"""
        generator = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        
        stats = generator.get_stats()
        
        assert 'model_name' in stats
        assert 'embedding_dim' in stats
        assert 'vector_store_type' in stats
        assert 'total_files' in stats
        assert 'total_chunks' in stats
        assert 'total_embeddings' in stats
    
    def test_metadata_persistence(self, temp_workspace, sample_python_file):
        """Test metadata saving and loading"""
        # Create generator and index file
        generator1 = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        generator1.index_file(sample_python_file)
        
        # Save metadata
        generator1._save_metadata()
        
        # Create new generator and load metadata
        generator2 = CodeEmbeddingGenerator(workspace_path=temp_workspace)
        
        # Should have loaded the chunks
        assert sample_python_file in generator2.chunk_index
        assert len(generator2.chunks) > 0
    
    def test_global_instance(self, temp_workspace):
        """Test global instance management"""
        generator1 = get_embedding_generator(temp_workspace)
        generator2 = get_embedding_generator(temp_workspace)
        
        # Should be the same instance
        assert generator1 is generator2
        
        # Different workspace should create new instance
        temp_dir2 = tempfile.mkdtemp()
        try:
            generator3 = get_embedding_generator(temp_dir2)
            assert generator3 is not generator1
        finally:
            shutil.rmtree(temp_dir2)

class TestVectorStoreIntegration:
    """Test vector store integration"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_faiss_integration(self, temp_workspace):
        """Test FAISS vector store integration"""
        generator = CodeEmbeddingGenerator(
            workspace_path=temp_workspace,
            vector_store_type="faiss"
        )
        
        if generator.vector_store is not None:
            assert generator.vector_store_type == "faiss"
            # FAISS index should be initialized
            assert hasattr(generator.vector_store, 'ntotal')
    
    @pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
    def test_chroma_integration(self, temp_workspace):
        """Test ChromaDB vector store integration"""
        generator = CodeEmbeddingGenerator(
            workspace_path=temp_workspace,
            vector_store_type="chroma"
        )
        
        if generator.vector_store is not None:
            assert generator.vector_store_type == "chroma"
            # ChromaDB collection should be initialized
            assert hasattr(generator.vector_store, 'count')

if __name__ == "__main__":
    # Run basic tests without pytest
    import sys
    
    print("Testing Code Embedding Generator...")
    
    # Test basic functionality
    temp_dir = tempfile.mkdtemp()
    try:
        # Test chunk creation
        chunk = CodeChunk(
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            chunk_type="function",
            line_start=1,
            line_end=2
        )
        print(f"✓ CodeChunk creation: {chunk.chunk_type}")
        
        # Test generator creation
        generator = CodeEmbeddingGenerator(workspace_path=temp_dir)
        print(f"✓ Generator creation: {generator.model_name}")
        
        # Test language detection
        assert generator._detect_language("test.py") == "python"
        print("✓ Language detection")
        
        # Test code chunking
        sample_code = '''def add(a, b):
    """Add two numbers"""
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
'''
        
        sample_file = os.path.join(temp_dir, "sample.py")
        with open(sample_file, 'w') as f:
            f.write(sample_code)
        
        chunks = generator.chunk_code_file(sample_file)
        print(f"✓ Code chunking: {len(chunks)} chunks")
        
        # Test stats
        stats = generator.get_stats()
        print(f"✓ Stats tracking: {len(stats)} metrics")
        
        print("\nAll basic tests passed!")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("sentence-transformers available - advanced features enabled")
        else:
            print("sentence-transformers not available - install for full functionality")
            
        if FAISS_AVAILABLE:
            print("FAISS available - vector search enabled")
        else:
            print("FAISS not available - install faiss-cpu for vector search")
            
        if CHROMADB_AVAILABLE:
            print("ChromaDB available - vector database enabled")
        else:
            print("ChromaDB not available - install chromadb for vector database")
    
    finally:
        shutil.rmtree(temp_dir)