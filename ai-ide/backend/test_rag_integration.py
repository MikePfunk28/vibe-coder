"""
Integration tests for the complete RAG system
"""

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pytest

from rag_pipeline import ContextAwareRAGPipeline, QueryContext
from rag_system import (
    DocumentationSource,
    KnowledgeBaseRetrieval,
    SentenceTransformerEmbedding,
)


class TestRAGIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self):
        """Test the complete RAG workflow from ingestion to query processing."""
        # Create temporary storage
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize components
            embedding_model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
            knowledge_base = KnowledgeBaseRetrieval(
                embedding_model=embedding_model,
                storage_path=temp_dir
            )
            
            # Create mock documentation source
            mock_docs = [
                {
                    'url': 'https://python.org/docs/functions',
                    'title': 'Python Functions',
                    'content': '''# Python Functions

Functions in Python are defined using the def keyword. Here's the basic syntax:

```python
def function_name(parameters):
    """Docstring"""
    # Function body
    return value
```

## Examples

Here's a simple function:

```python
def greet(name):
    return f"Hello, {name}!"
```

Functions can have default parameters:

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```
''',
                    'fetched_at': datetime.now()
                },
                {
                    'url': 'https://python.org/docs/classes',
                    'title': 'Python Classes',
                    'content': '''# Python Classes

Classes in Python are defined using the class keyword:

```python
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
```

## Inheritance

Classes can inherit from other classes:

```python
class ChildClass(MyClass):
    def __init__(self, value, extra):
        super().__init__(value)
        self.extra = extra
```
''',
                    'fetched_at': datetime.now()
                },
                {
                    'url': 'https://stackoverflow.com/questions/python-debugging',
                    'title': 'Python Debugging Tips',
                    'content': '''# Debugging Python Code

Common debugging techniques:

1. Use print statements
2. Use the debugger (pdb)
3. Use logging
4. Check variable types

## Using pdb

```python
import pdb
pdb.set_trace()
```

## Common Errors

- NameError: Variable not defined
- TypeError: Wrong type used
- IndentationError: Incorrect indentation
''',
                    'fetched_at': datetime.now()
                }
            ]
            
            # Create mock source
            class MockDocumentationSource:
                async def fetch_documents(self, query=None, limit=100):
                    return mock_docs
                
                def extract_metadata(self, document):
                    from rag_system import DocumentMetadata, SourceType
                    return DocumentMetadata(
                        source_type=SourceType.DOCUMENTATION,
                        source_url=document['url'],
                        title=document['title'],
                        created_at=document['fetched_at'],
                        updated_at=document['fetched_at'],
                        language='en',
                        file_type='html',
                        checksum=str(hash(document['content']))
                    )
            
            # Add source and ingest documents
            mock_source = MockDocumentationSource()
            knowledge_base.add_source("python_docs", mock_source)
            
            await knowledge_base.ingest_from_source("python_docs", limit=10)
            
            # Verify ingestion
            assert len(knowledge_base.chunks) > 0
            assert knowledge_base.embeddings is not None
            
            # Initialize RAG pipeline
            rag_pipeline = ContextAwareRAGPipeline(knowledge_base, embedding_model)
            
            # Test different types of queries
            test_cases = [
                {
                    'query': 'how to define a function in python',
                    'context': QueryContext(
                        programming_language='python',
                        task_type='code_completion',
                        domain='programming'
                    ),
                    'expected_keywords': ['def', 'function', 'python']
                },
                {
                    'query': 'python class inheritance example',
                    'context': QueryContext(
                        programming_language='python',
                        task_type='documentation',
                        domain='programming'
                    ),
                    'expected_keywords': ['class', 'inheritance', 'super']
                },
                {
                    'query': 'debugging python code errors',
                    'context': QueryContext(
                        programming_language='python',
                        task_type='debugging',
                        domain='programming'
                    ),
                    'expected_keywords': ['debug', 'error', 'pdb']
                }
            ]
            
            # Process each test case
            for test_case in test_cases:
                response = await rag_pipeline.process_query(
                    test_case['query'],
                    test_case['context']
                )
                
                # Verify response structure
                assert response.query == test_case['query']
                assert response.synthesized_answer is not None
                assert len(response.synthesized_answer) > 50
                assert response.confidence_score > 0.0
                assert len(response.retrieved_chunks) > 0
                assert response.processing_time > 0.0
                assert 'overall' in response.quality_metrics
                
                # Verify content relevance
                answer_lower = response.synthesized_answer.lower()
                relevant_keywords = [
                    kw for kw in test_case['expected_keywords']
                    if kw.lower() in answer_lower
                ]
                assert len(relevant_keywords) > 0, f"No relevant keywords found in answer for query: {test_case['query']}"
                
                # Verify quality metrics
                assert 0.0 <= response.quality_metrics['overall'] <= 1.0
                assert response.quality_metrics['relevance'] > 0.0
                assert response.quality_metrics['completeness'] > 0.0
                
                print(f"✓ Query: {test_case['query']}")
                print(f"  Confidence: {response.confidence_score:.2f}")
                print(f"  Quality: {response.quality_metrics['overall']:.2f}")
                print(f"  Processing time: {response.processing_time:.2f}s")
                print(f"  Retrieved chunks: {len(response.retrieved_chunks)}")
                print(f"  Answer length: {len(response.synthesized_answer)} chars")
                print()
            
            # Test search functionality
            search_results = knowledge_base.search("python function definition", top_k=5)
            assert len(search_results) > 0
            assert all(hasattr(chunk, 'relevance_score') for chunk in search_results)
            
            # Test knowledge graph relationships
            if knowledge_base.chunks:
                sample_chunk = knowledge_base.chunks[0]
                related_chunks = knowledge_base.get_related_chunks(sample_chunk.id, max_related=3)
                # Should not crash, may or may not find related chunks
                assert isinstance(related_chunks, list)
            
            # Test save and load
            knowledge_base.save()
            
            # Create new instance and load
            new_kb = KnowledgeBaseRetrieval(
                embedding_model=embedding_model,
                storage_path=temp_dir
            )
            new_kb.load()
            
            assert len(new_kb.chunks) == len(knowledge_base.chunks)
            assert new_kb.embeddings is not None
            
            print("✓ All integration tests passed!")
            
        except Exception as e:
            print(f"Integration test failed: {e}")
            raise
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_performance(self):
        """Test RAG pipeline performance with multiple queries."""
        # Create minimal setup for performance testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            embedding_model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
            knowledge_base = KnowledgeBaseRetrieval(
                embedding_model=embedding_model,
                storage_path=temp_dir
            )
            
            # Add minimal test data
            from rag_system import DocumentChunk, DocumentMetadata, SourceType, ChunkType
            
            metadata = DocumentMetadata(
                source_type=SourceType.DOCUMENTATION,
                source_url="https://test.com",
                title="Test Doc"
            )
            
            test_chunks = []
            for i in range(10):
                chunk = DocumentChunk(
                    id=f"chunk_{i}",
                    content=f"This is test content {i} about Python programming and functions.",
                    chunk_type=ChunkType.PARAGRAPH,
                    metadata=metadata
                )
                test_chunks.append(chunk)
            
            knowledge_base.chunks = test_chunks
            
            # Generate embeddings
            contents = [chunk.content for chunk in test_chunks]
            embeddings = embedding_model.encode(contents)
            for chunk, embedding in zip(test_chunks, embeddings):
                chunk.embedding = embedding
            
            knowledge_base._update_embedding_matrix()
            
            # Initialize pipeline
            rag_pipeline = ContextAwareRAGPipeline(knowledge_base, embedding_model)
            
            # Test multiple queries for performance
            queries = [
                "python function definition",
                "programming concepts",
                "test content example",
                "python programming tutorial",
                "function implementation"
            ]
            
            context = QueryContext(
                programming_language='python',
                task_type='code_completion'
            )
            
            total_time = 0.0
            successful_queries = 0
            
            for query in queries:
                try:
                    response = await rag_pipeline.process_query(query, context)
                    total_time += response.processing_time
                    successful_queries += 1
                    
                    # Basic validation
                    assert response.synthesized_answer is not None
                    assert response.confidence_score >= 0.0
                    
                except Exception as e:
                    print(f"Query failed: {query} - {e}")
            
            # Performance assertions
            avg_time = total_time / successful_queries if successful_queries > 0 else 0
            assert successful_queries > 0, "No queries were successful"
            assert avg_time < 10.0, f"Average query time too high: {avg_time:.2f}s"
            
            # Get performance metrics
            metrics = rag_pipeline.get_performance_metrics()
            assert 'total_queries' in metrics
            assert metrics['total_queries'] == successful_queries
            
            print(f"✓ Performance test completed:")
            print(f"  Successful queries: {successful_queries}/{len(queries)}")
            print(f"  Average processing time: {avg_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(TestRAGIntegration().test_complete_rag_workflow())
    asyncio.run(TestRAGIntegration().test_rag_pipeline_performance())