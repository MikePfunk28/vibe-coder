"""
Tests for the Advanced RAG (Retrieval-Augmented Generation) System
"""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from rag_system import (
    ChunkType,
    DocumentChunk,
    DocumentMetadata,
    DocumentationSource,
    EmbeddingModel,
    GitHubSource,
    HierarchicalChunker,
    KnowledgeBaseRetrieval,
    KnowledgeGraph,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    SourceType,
    StackOverflowSource,
)


class TestDocumentMetadata:
    """Test DocumentMetadata class."""
    
    def test_create_metadata(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document",
            author="Test Author",
            language="en",
            tags=["python", "testing"]
        )
        
        assert metadata.source_type == SourceType.DOCUMENTATION
        assert metadata.source_url == "https://example.com/docs"
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.language == "en"
        assert metadata.tags == ["python", "testing"]


class TestDocumentChunk:
    """Test DocumentChunk class."""
    
    def test_create_chunk(self):
        """Test creating a document chunk."""
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is test content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata,
            keywords=["test", "content"],
            entities=["function:test"]
        )
        
        assert chunk.id == "test_chunk_1"
        assert chunk.content == "This is test content"
        assert chunk.chunk_type == ChunkType.PARAGRAPH
        assert chunk.keywords == ["test", "content"]
        assert chunk.entities == ["function:test"]
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document",
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )
        
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is test content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata,
            embedding=np.array([0.1, 0.2, 0.3])
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict['id'] == "test_chunk_1"
        assert chunk_dict['content'] == "This is test content"
        assert chunk_dict['chunk_type'] == "paragraph"
        assert chunk_dict['metadata']['source_type'] == "documentation"
        assert chunk_dict['metadata']['title'] == "Test Document"
        assert chunk_dict['embedding'] == [0.1, 0.2, 0.3]


class TestDocumentationSource:
    """Test DocumentationSource class."""
    
    @pytest.mark.asyncio
    async def test_fetch_documents_with_sitemap(self):
        """Test fetching documents using sitemap."""
        source = DocumentationSource("https://example.com", "https://example.com/sitemap.xml")
        
        # Mock sitemap response
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>"""
        
        # Mock page content
        page_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <h1>Test Page</h1>
                    <p>This is test content.</p>
                </main>
            </body>
        </html>
        """
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock()
            
            # Setup responses
            async def side_effect(*args, **kwargs):
                url = args[0] if args else kwargs.get('url', '')
                if 'sitemap' in url:
                    mock_response.text.return_value = sitemap_content
                else:
                    mock_response.text.return_value = page_content
                return mock_response
            
            mock_session.return_value.__aenter__.return_value.get = side_effect
            mock_session.return_value.__aexit__ = AsyncMock()
            
            documents = await source.fetch_documents(limit=2)
            
            assert len(documents) == 2
            assert all('title' in doc for doc in documents)
            assert all('content' in doc for doc in documents)
    
    def test_extract_metadata(self):
        """Test extracting metadata from documentation."""
        source = DocumentationSource("https://example.com")
        
        document = {
            'url': 'https://example.com/page1',
            'title': 'Test Page',
            'content': 'This is test content',
            'fetched_at': datetime(2023, 1, 1, 12, 0, 0)
        }
        
        metadata = source.extract_metadata(document)
        
        assert metadata.source_type == SourceType.DOCUMENTATION
        assert metadata.source_url == 'https://example.com/page1'
        assert metadata.title == 'Test Page'
        assert metadata.file_type == 'html'


class TestStackOverflowSource:
    """Test StackOverflowSource class."""
    
    @pytest.mark.asyncio
    async def test_fetch_documents(self):
        """Test fetching Stack Overflow documents."""
        source = StackOverflowSource()
        
        # Mock API response
        api_response = {
            'items': [
                {
                    'question_id': 12345,
                    'title': 'How to test Python code?',
                    'body': 'I need help with testing Python code.',
                    'score': 10,
                    'view_count': 1000,
                    'answer_count': 3,
                    'tags': ['python', 'testing'],
                    'creation_date': 1640995200,  # 2022-01-01
                    'last_activity_date': 1640995200,
                    'link': 'https://stackoverflow.com/questions/12345'
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=api_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aexit__ = AsyncMock()
            
            # Mock _fetch_answers to return empty list
            source._fetch_answers = AsyncMock(return_value=[])
            
            documents = await source.fetch_documents(query="python testing", limit=10)
            
            assert len(documents) == 1
            assert documents[0]['title'] == 'How to test Python code?'
            assert documents[0]['type'] == 'question'
            assert documents[0]['tags'] == ['python', 'testing']
    
    def test_extract_metadata(self):
        """Test extracting metadata from Stack Overflow document."""
        source = StackOverflowSource()
        
        document = {
            'id': 12345,
            'title': 'How to test Python code?',
            'body': 'I need help with testing Python code.',
            'tags': ['python', 'testing'],
            'creation_date': datetime(2022, 1, 1),
            'last_activity_date': datetime(2022, 1, 1),
            'link': 'https://stackoverflow.com/questions/12345',
            'type': 'question'
        }
        
        metadata = source.extract_metadata(document)
        
        assert metadata.source_type == SourceType.STACKOVERFLOW
        assert metadata.source_url == 'https://stackoverflow.com/questions/12345'
        assert metadata.title == 'How to test Python code?'
        assert metadata.tags == ['python', 'testing']


class TestGitHubSource:
    """Test GitHubSource class."""
    
    @pytest.mark.asyncio
    async def test_fetch_documents(self):
        """Test fetching GitHub documents."""
        source = GitHubSource(token="test_token")
        
        # Mock API response
        api_response = {
            'items': [
                {
                    'id': 12345,
                    'name': 'test-repo',
                    'full_name': 'user/test-repo',
                    'description': 'A test repository',
                    'language': 'Python',
                    'stargazers_count': 100,
                    'forks_count': 20,
                    'created_at': '2022-01-01T00:00:00Z',
                    'updated_at': '2022-01-01T00:00:00Z',
                    'html_url': 'https://github.com/user/test-repo',
                    'topics': ['python', 'testing']
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=api_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aexit__ = AsyncMock()
            
            # Mock _fetch_readme
            source._fetch_readme = AsyncMock(return_value="# Test Repo\nThis is a test repository.")
            
            documents = await source.fetch_documents(query="python", limit=10)
            
            assert len(documents) == 1
            assert documents[0]['name'] == 'test-repo'
            assert documents[0]['language'] == 'Python'
            assert documents[0]['readme_content'] == "# Test Repo\nThis is a test repository."
    
    def test_extract_metadata(self):
        """Test extracting metadata from GitHub document."""
        source = GitHubSource()
        
        document = {
            'id': 12345,
            'name': 'test-repo',
            'full_name': 'user/test-repo',
            'description': 'A test repository',
            'language': 'Python',
            'created_at': datetime(2022, 1, 1),
            'updated_at': datetime(2022, 1, 1),
            'html_url': 'https://github.com/user/test-repo',
            'topics': ['python', 'testing'],
            'readme_content': '# Test Repo'
        }
        
        metadata = source.extract_metadata(document)
        
        assert metadata.source_type == SourceType.GITHUB
        assert metadata.source_url == 'https://github.com/user/test-repo'
        assert metadata.title == 'user/test-repo'
        assert metadata.language == 'Python'
        assert metadata.tags == ['python', 'testing']


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return random embeddings for testing."""
        return np.random.rand(len(texts), self.dimension)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension


class TestSentenceTransformerEmbedding:
    """Test SentenceTransformerEmbedding class."""
    
    @patch('rag_system.SentenceTransformer')
    def test_encode(self, mock_transformer):
        """Test encoding texts."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_transformer.return_value = mock_model
        
        embedding_model = SentenceTransformerEmbedding("test-model")
        
        texts = ["Hello world", "Test text"]
        embeddings = embedding_model.encode(texts)
        
        assert embeddings.shape == (2, 2)
        assert np.array_equal(embeddings, np.array([[0.1, 0.2], [0.3, 0.4]]))
    
    @patch('rag_system.SentenceTransformer')
    def test_get_dimension(self, mock_transformer):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedding_model = SentenceTransformerEmbedding("test-model")
        
        assert embedding_model.get_dimension() == 384


class TestOpenAIEmbedding:
    """Test OpenAIEmbedding class."""
    
    @patch('rag_system.openai')
    def test_encode(self, mock_openai):
        """Test encoding texts with OpenAI."""
        mock_openai.Embedding.create.return_value = {
            'data': [
                {'embedding': [0.1, 0.2, 0.3]},
                {'embedding': [0.4, 0.5, 0.6]}
            ]
        }
        
        embedding_model = OpenAIEmbedding("test-key")
        
        texts = ["Hello world", "Test text"]
        embeddings = embedding_model.encode(texts)
        
        assert embeddings.shape == (2, 3)
        assert np.array_equal(embeddings, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    
    def test_get_dimension(self):
        """Test getting embedding dimension."""
        embedding_model = OpenAIEmbedding("test-key")
        assert embedding_model.get_dimension() == 1536


class TestHierarchicalChunker:
    """Test HierarchicalChunker class."""
    
    def test_chunk_document(self):
        """Test chunking a document hierarchically."""
        chunker = HierarchicalChunker(max_chunk_size=100, overlap_size=10)
        
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        content = """# Introduction
        
This is the introduction section.

## Getting Started

Here's how to get started:

1. Install the package
2. Import the module
3. Use the functions

```python
import example
result = example.function()
```

## Advanced Usage

For advanced usage, see the following example:

```python
import example
config = example.Config()
result = example.advanced_function(config)
```
"""
        
        chunks = chunker.chunk_document(content, metadata)
        
        # Should have document, sections, paragraphs, and code blocks
        assert len(chunks) > 1
        
        # Check document chunk
        doc_chunk = next(c for c in chunks if c.chunk_type == ChunkType.DOCUMENT)
        assert doc_chunk is not None
        assert doc_chunk.parent_id is None
        
        # Check section chunks
        section_chunks = [c for c in chunks if c.chunk_type == ChunkType.SECTION]
        assert len(section_chunks) >= 2  # Introduction and Getting Started sections
        
        # Check code block chunks
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE_BLOCK]
        assert len(code_chunks) >= 2  # Two code blocks in the content
    
    def test_split_into_sections(self):
        """Test splitting content into sections."""
        chunker = HierarchicalChunker()
        
        content = """# Section 1

Content for section 1.

## Section 2

Content for section 2.

### Section 3

Content for section 3."""
        
        sections = chunker._split_into_sections(content)
        
        assert len(sections) == 3
        assert sections[0]['title'] == 'Section 1'
        assert sections[1]['title'] == 'Section 2'
        assert sections[2]['title'] == 'Section 3'
    
    def test_extract_code_blocks(self):
        """Test extracting code blocks from content."""
        chunker = HierarchicalChunker()
        
        content = """Here's some code:

```python
def hello():
    print("Hello, world!")
```

And some inline code: `print("test")` and more code: `import os`.
"""
        
        code_blocks = chunker._extract_code_blocks(content)
        
        assert len(code_blocks) >= 2  # At least the function and one inline code
        assert 'def hello():' in code_blocks[0]


class TestKnowledgeGraph:
    """Test KnowledgeGraph class."""
    
    def test_add_entity(self):
        """Test adding an entity to the knowledge graph."""
        kg = KnowledgeGraph()
        
        entity_data = {'name': 'test_function', 'type': 'function'}
        kg.add_entity('function:test_function', entity_data)
        
        assert 'function:test_function' in kg.entities
        assert kg.entities['function:test_function'] == entity_data
    
    def test_add_relationship(self):
        """Test adding a relationship between entities."""
        kg = KnowledgeGraph()
        
        kg.add_relationship('entity1', 'entity2', 'related_to')
        
        assert ('entity1', 'entity2') in kg.relationships
        assert kg.relationships[('entity1', 'entity2')] == 'related_to'
        # Should also add reverse relationship for bidirectional types
        assert ('entity2', 'entity1') in kg.relationships
    
    def test_get_related_entities(self):
        """Test getting related entities."""
        kg = KnowledgeGraph()
        
        kg.add_relationship('entity1', 'entity2', 'related_to')
        kg.add_relationship('entity1', 'entity3', 'part_of')
        
        related = kg.get_related_entities('entity1')
        assert 'entity2' in related
        assert 'entity3' in related
        
        # Test filtering by relationship type
        related_to = kg.get_related_entities('entity1', 'related_to')
        assert 'entity2' in related_to
        assert 'entity3' not in related_to
    
    def test_extract_entities_from_chunk(self):
        """Test extracting entities from a document chunk."""
        kg = KnowledgeGraph()
        
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunk = DocumentChunk(
            id="test_chunk",
            content="""
            def test_function():
                import numpy
                from pandas import DataFrame
                result = calculate_value()
                return result
            
            class TestClass:
                pass
            """,
            chunk_type=ChunkType.CODE_BLOCK,
            metadata=metadata
        )
        
        entities = kg.extract_entities_from_chunk(chunk)
        
        # Should extract functions, classes, imports, etc.
        assert len(entities) > 0
        
        # Check for specific entity types
        function_entities = [e for e in entities if e.startswith('function:')]
        class_entities = [e for e in entities if e.startswith('class:')]
        import_entities = [e for e in entities if e.startswith('import:')]
        
        assert len(function_entities) > 0
        assert len(class_entities) > 0
        assert len(import_entities) > 0


class TestKnowledgeBaseRetrieval:
    """Test KnowledgeBaseRetrieval class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_model = MockEmbeddingModel()
        self.kb = KnowledgeBaseRetrieval(
            embedding_model=self.embedding_model,
            storage_path=self.temp_dir
        )
    
    def test_add_source(self):
        """Test adding a knowledge source."""
        self.setUp()
        
        source = DocumentationSource("https://example.com")
        self.kb.add_source("docs", source)
        
        assert "docs" in self.kb.sources
        assert self.kb.sources["docs"] == source
    
    @pytest.mark.asyncio
    async def test_ingest_from_source(self):
        """Test ingesting documents from a source."""
        self.setUp()
        
        # Create mock source
        mock_source = Mock()
        mock_source.fetch_documents = AsyncMock(return_value=[
            {
                'url': 'https://example.com/page1',
                'title': 'Test Page 1',
                'content': 'This is test content for page 1.',
                'fetched_at': datetime.now()
            },
            {
                'url': 'https://example.com/page2',
                'title': 'Test Page 2',
                'content': 'This is test content for page 2.',
                'fetched_at': datetime.now()
            }
        ])
        
        mock_source.extract_metadata = Mock(side_effect=lambda doc: DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url=doc['url'],
            title=doc['title'],
            created_at=doc['fetched_at']
        ))
        
        self.kb.add_source("test", mock_source)
        
        await self.kb.ingest_from_source("test", limit=2)
        
        # Should have chunks from both documents
        assert len(self.kb.chunks) > 0
        
        # Should have embeddings
        assert self.kb.embeddings is not None
        assert self.kb.embeddings.shape[0] == len(self.kb.chunks)
    
    def test_search(self):
        """Test searching for relevant chunks."""
        self.setUp()
        
        # Add some test chunks
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Python programming tutorial",
                chunk_type=ChunkType.PARAGRAPH,
                metadata=metadata,
                embedding=np.array([0.1, 0.2, 0.3])
            ),
            DocumentChunk(
                id="chunk2",
                content="JavaScript web development",
                chunk_type=ChunkType.PARAGRAPH,
                metadata=metadata,
                embedding=np.array([0.4, 0.5, 0.6])
            ),
            DocumentChunk(
                id="chunk3",
                content="Python data science",
                chunk_type=ChunkType.PARAGRAPH,
                metadata=metadata,
                embedding=np.array([0.1, 0.3, 0.2])
            )
        ]
        
        self.kb.chunks = chunks
        self.kb._update_embedding_matrix()
        
        # Mock the embedding model to return a query embedding similar to chunk1 and chunk3
        with patch.object(self.kb.embedding_model, 'encode') as mock_encode:
            mock_encode.return_value = np.array([[0.1, 0.25, 0.25]])  # Similar to chunk1 and chunk3
            
            results = self.kb.search("Python programming", top_k=2)
            
            assert len(results) <= 2
            assert all(isinstance(chunk, DocumentChunk) for chunk in results)
            assert all(hasattr(chunk, 'relevance_score') for chunk in results)
    
    def test_get_related_chunks(self):
        """Test getting related chunks through knowledge graph."""
        self.setUp()
        
        # Add test chunks with entities
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunk1 = DocumentChunk(
            id="chunk1",
            content="def test_function(): pass",
            chunk_type=ChunkType.CODE_BLOCK,
            metadata=metadata,
            entities=["function:test_function", "technology:python"]
        )
        
        chunk2 = DocumentChunk(
            id="chunk2",
            content="import test_module",
            chunk_type=ChunkType.CODE_BLOCK,
            metadata=metadata,
            entities=["import:test_module", "technology:python"]
        )
        
        self.kb.chunks = [chunk1, chunk2]
        
        # Add relationships to knowledge graph
        self.kb.knowledge_graph.add_relationship("function:test_function", "technology:python", "uses")
        self.kb.knowledge_graph.add_relationship("import:test_module", "technology:python", "uses")
        
        related_chunks = self.kb.get_related_chunks("chunk1", max_related=5)
        
        # Should find chunk2 through the shared "technology:python" entity
        assert len(related_chunks) > 0
        assert any(chunk.id == "chunk2" for chunk in related_chunks)
    
    def test_save_and_load(self):
        """Test saving and loading the knowledge base."""
        self.setUp()
        
        # Add test data
        metadata = DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url="https://example.com/docs",
            title="Test Document"
        )
        
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata,
            embedding=np.array([0.1, 0.2, 0.3]),
            entities=["function:test"]
        )
        
        self.kb.chunks = [chunk]
        self.kb._update_embedding_matrix()
        self.kb.knowledge_graph.add_entity("function:test", {"name": "test", "type": "function"})
        
        # Save
        self.kb.save()
        
        # Create new instance and load
        new_kb = KnowledgeBaseRetrieval(
            embedding_model=self.embedding_model,
            storage_path=self.temp_dir
        )
        new_kb.load()
        
        # Verify data was loaded correctly
        assert len(new_kb.chunks) == 1
        assert new_kb.chunks[0].id == "test_chunk"
        assert new_kb.chunks[0].content == "Test content"
        assert new_kb.embeddings is not None
        assert "function:test" in new_kb.knowledge_graph.entities


if __name__ == "__main__":
    # Run tests
    unittest.main()