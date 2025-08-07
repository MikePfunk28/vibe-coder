"""
Advanced RAG (Retrieval-Augmented Generation) System

This module implements a comprehensive RAG system with multi-source knowledge ingestion,
advanced embedding models, hierarchical document chunking, and knowledge graph construction.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of knowledge sources supported by the RAG system."""
    DOCUMENTATION = "documentation"
    STACKOVERFLOW = "stackoverflow"
    GITHUB = "github"
    LOCAL_FILE = "local_file"
    WEB_PAGE = "web_page"


class ChunkType(Enum):
    """Types of document chunks in the hierarchical structure."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"


@dataclass
class DocumentMetadata:
    """Metadata for a document in the knowledge base."""
    source_type: SourceType
    source_url: str
    title: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    file_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None


@dataclass
class DocumentChunk:
    """A hierarchical chunk of a document with metadata and embeddings."""
    id: str
    content: str
    chunk_type: ChunkType
    metadata: DocumentMetadata
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'id': self.id,
            'content': self.content,
            'chunk_type': self.chunk_type.value,
            'metadata': {
                'source_type': self.metadata.source_type.value,
                'source_url': self.metadata.source_url,
                'title': self.metadata.title,
                'author': self.metadata.author,
                'created_at': self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                'updated_at': self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                'language': self.metadata.language,
                'tags': self.metadata.tags,
                'file_type': self.metadata.file_type,
                'size_bytes': self.metadata.size_bytes,
                'checksum': self.metadata.checksum
            },
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'keywords': self.keywords,
            'entities': self.entities,
            'relationships': self.relationships,
            'relevance_score': self.relevance_score
        }


class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""
    
    @abstractmethod
    async def fetch_documents(self, query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch documents from the source."""
        pass
    
    @abstractmethod
    def extract_metadata(self, document: Dict[str, Any]) -> DocumentMetadata:
        """Extract metadata from a document."""
        pass


class DocumentationSource(KnowledgeSource):
    """Knowledge source for documentation websites."""
    
    def __init__(self, base_url: str, sitemap_url: Optional[str] = None):
        self.base_url = base_url
        self.sitemap_url = sitemap_url or urljoin(base_url, '/sitemap.xml')
        self.session = None
    
    async def fetch_documents(self, query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch documentation pages."""
        documents = []
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Try to get URLs from sitemap first
            urls = await self._get_urls_from_sitemap()
            if not urls:
                # Fallback to crawling
                urls = await self._crawl_documentation(limit)
            
            # Limit the number of URLs
            urls = urls[:limit]
            
            # Fetch content for each URL
            for url in urls:
                try:
                    document = await self._fetch_page_content(url)
                    if document:
                        documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    continue
        
        finally:
            if self.session:
                await self.session.close()
                self.session = None
        
        return documents
    
    async def _get_urls_from_sitemap(self) -> List[str]:
        """Extract URLs from sitemap."""
        try:
            async with self.session.get(self.sitemap_url) as response:
                if response.status == 200:
                    content = await response.text()
                    # Simple XML parsing for sitemap
                    urls = re.findall(r'<loc>(.*?)</loc>', content)
                    return [url for url in urls if url.startswith(self.base_url)]
        except Exception as e:
            logger.warning(f"Failed to fetch sitemap {self.sitemap_url}: {e}")
        
        return []
    
    async def _crawl_documentation(self, limit: int) -> List[str]:
        """Crawl documentation pages starting from base URL."""
        visited = set()
        to_visit = [self.base_url]
        urls = []
        
        while to_visit and len(urls) < limit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        urls.append(url)
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Find more documentation links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if href.startswith('/'):
                                href = urljoin(self.base_url, href)
                            
                            if (href.startswith(self.base_url) and 
                                href not in visited and 
                                href not in to_visit and
                                len(to_visit) < limit * 2):  # Limit queue size
                                to_visit.append(href)
            
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                continue
        
        return urls
    
    async def _fetch_page_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse content from a documentation page."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract title
                title_elem = soup.find('title') or soup.find('h1')
                title = title_elem.get_text().strip() if title_elem else url
                
                # Extract main content
                main_content = ""
                content_selectors = [
                    'main', 'article', '.content', '.documentation', 
                    '.docs-content', '#content', '.main-content'
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        main_content = content_elem.get_text(separator='\n').strip()
                        break
                
                if not main_content:
                    # Fallback to body content
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(separator='\n').strip()
                
                return {
                    'url': url,
                    'title': title,
                    'content': main_content,
                    'html': str(soup),
                    'fetched_at': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Error fetching page content from {url}: {e}")
            return None
    
    def extract_metadata(self, document: Dict[str, Any]) -> DocumentMetadata:
        """Extract metadata from a documentation document."""
        return DocumentMetadata(
            source_type=SourceType.DOCUMENTATION,
            source_url=document['url'],
            title=document['title'],
            created_at=document.get('fetched_at'),
            updated_at=document.get('fetched_at'),
            language='en',  # Default to English
            file_type='html',
            checksum=hashlib.md5(document['content'].encode()).hexdigest()
        )


class StackOverflowSource(KnowledgeSource):
    """Knowledge source for Stack Overflow questions and answers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.stackexchange.com/2.3"
    
    async def fetch_documents(self, query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch Stack Overflow questions and answers."""
        documents = []
        
        # Build API request
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': 'stackoverflow',
            'pagesize': min(limit, 100),  # API limit
            'filter': 'withbody'
        }
        
        if query:
            params['intitle'] = query
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/questions", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for question in data.get('items', []):
                            # Get question document
                            question_doc = {
                                'id': question['question_id'],
                                'title': question['title'],
                                'body': question.get('body', ''),
                                'score': question['score'],
                                'view_count': question['view_count'],
                                'answer_count': question['answer_count'],
                                'tags': question.get('tags', []),
                                'creation_date': datetime.fromtimestamp(question['creation_date']),
                                'last_activity_date': datetime.fromtimestamp(question['last_activity_date']),
                                'link': question['link'],
                                'type': 'question'
                            }
                            documents.append(question_doc)
                            
                            # Fetch answers for this question
                            if question['answer_count'] > 0:
                                answers = await self._fetch_answers(session, question['question_id'])
                                documents.extend(answers)
        
        except Exception as e:
            logger.error(f"Error fetching Stack Overflow data: {e}")
        
        return documents
    
    async def _fetch_answers(self, session: aiohttp.ClientSession, question_id: int) -> List[Dict[str, Any]]:
        """Fetch answers for a specific question."""
        answers = []
        
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': 'stackoverflow',
            'filter': 'withbody'
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            async with session.get(f"{self.base_url}/questions/{question_id}/answers", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for answer in data.get('items', []):
                        answer_doc = {
                            'id': answer['answer_id'],
                            'question_id': question_id,
                            'body': answer.get('body', ''),
                            'score': answer['score'],
                            'is_accepted': answer.get('is_accepted', False),
                            'creation_date': datetime.fromtimestamp(answer['creation_date']),
                            'last_activity_date': datetime.fromtimestamp(answer['last_activity_date']),
                            'type': 'answer'
                        }
                        answers.append(answer_doc)
        
        except Exception as e:
            logger.warning(f"Error fetching answers for question {question_id}: {e}")
        
        return answers
    
    def extract_metadata(self, document: Dict[str, Any]) -> DocumentMetadata:
        """Extract metadata from a Stack Overflow document."""
        if document['type'] == 'question':
            title = document['title']
            url = document['link']
        else:
            title = f"Answer to question {document['question_id']}"
            url = f"https://stackoverflow.com/a/{document['id']}"
        
        return DocumentMetadata(
            source_type=SourceType.STACKOVERFLOW,
            source_url=url,
            title=title,
            created_at=document.get('creation_date'),
            updated_at=document.get('last_activity_date'),
            language='en',
            tags=document.get('tags', []),
            checksum=hashlib.md5(document.get('body', '').encode()).hexdigest()
        )


class GitHubSource(KnowledgeSource):
    """Knowledge source for GitHub repositories."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {}
        if token:
            self.headers['Authorization'] = f'token {token}'
    
    async def fetch_documents(self, query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch GitHub repositories and their README files."""
        documents = []
        
        # Search for repositories
        search_query = query or "language:python stars:>100"
        params = {
            'q': search_query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': min(limit, 100)
        }
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f"{self.base_url}/search/repositories", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for repo in data.get('items', []):
                            # Fetch README content
                            readme_content = await self._fetch_readme(session, repo['full_name'])
                            
                            repo_doc = {
                                'id': repo['id'],
                                'name': repo['name'],
                                'full_name': repo['full_name'],
                                'description': repo.get('description', ''),
                                'readme_content': readme_content,
                                'language': repo.get('language'),
                                'stars': repo['stargazers_count'],
                                'forks': repo['forks_count'],
                                'created_at': datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00')),
                                'updated_at': datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00')),
                                'html_url': repo['html_url'],
                                'topics': repo.get('topics', [])
                            }
                            documents.append(repo_doc)
        
        except Exception as e:
            logger.error(f"Error fetching GitHub data: {e}")
        
        return documents
    
    async def _fetch_readme(self, session: aiohttp.ClientSession, repo_full_name: str) -> str:
        """Fetch README content for a repository."""
        try:
            async with session.get(f"{self.base_url}/repos/{repo_full_name}/readme") as response:
                if response.status == 200:
                    data = await response.json()
                    # Decode base64 content
                    import base64
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return content
        except Exception as e:
            logger.warning(f"Error fetching README for {repo_full_name}: {e}")
        
        return ""
    
    def extract_metadata(self, document: Dict[str, Any]) -> DocumentMetadata:
        """Extract metadata from a GitHub document."""
        return DocumentMetadata(
            source_type=SourceType.GITHUB,
            source_url=document['html_url'],
            title=document['full_name'],
            created_at=document.get('created_at'),
            updated_at=document.get('updated_at'),
            language=document.get('language'),
            tags=document.get('topics', []),
            checksum=hashlib.md5((document.get('description', '') + document.get('readme_content', '')).encode()).hexdigest()
        )


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformer embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model."""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = 1536  # Default for ada-002
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using OpenAI API."""
        import openai
        
        openai.api_key = self.api_key
        
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = []
            for item in response['data']:
                embeddings.append(item['embedding'])
            
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {e}")
            # Fallback to zeros
            return np.zeros((len(texts), self.dimension))
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimension


class HierarchicalChunker:
    """Hierarchical document chunker that creates structured chunks."""
    
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_document(self, content: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Chunk a document hierarchically."""
        chunks = []
        
        # Create document-level chunk
        doc_id = f"doc_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        doc_chunk = DocumentChunk(
            id=doc_id,
            content=content[:200] + "..." if len(content) > 200 else content,  # Summary
            chunk_type=ChunkType.DOCUMENT,
            metadata=metadata
        )
        chunks.append(doc_chunk)
        
        # Split into sections based on headers
        sections = self._split_into_sections(content)
        
        for i, section in enumerate(sections):
            section_id = f"{doc_id}_sec_{i}"
            section_chunk = DocumentChunk(
                id=section_id,
                content=section['content'],
                chunk_type=ChunkType.SECTION,
                metadata=metadata,
                parent_id=doc_id
            )
            doc_chunk.children_ids.append(section_id)
            chunks.append(section_chunk)
            
            # Further chunk sections into paragraphs
            paragraphs = self._split_into_paragraphs(section['content'])
            
            for j, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                    continue
                
                para_id = f"{section_id}_para_{j}"
                para_chunk = DocumentChunk(
                    id=para_id,
                    content=paragraph,
                    chunk_type=ChunkType.PARAGRAPH,
                    metadata=metadata,
                    parent_id=section_id
                )
                section_chunk.children_ids.append(para_id)
                chunks.append(para_chunk)
                
                # Extract code blocks
                code_blocks = self._extract_code_blocks(paragraph)
                for k, code_block in enumerate(code_blocks):
                    code_id = f"{para_id}_code_{k}"
                    code_chunk = DocumentChunk(
                        id=code_id,
                        content=code_block,
                        chunk_type=ChunkType.CODE_BLOCK,
                        metadata=metadata,
                        parent_id=para_id
                    )
                    para_chunk.children_ids.append(code_id)
                    chunks.append(code_chunk)
        
        return chunks
    
    def _split_into_sections(self, content: str) -> List[Dict[str, str]]:
        """Split content into sections based on headers."""
        sections = []
        
        # Split by markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = {'title': '', 'content': ''}
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': header_match.group(2),
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # If no headers found, treat entire content as one section
        if not sections:
            sections.append({'title': 'Main Content', 'content': content})
        
        return sections
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs."""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                result.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            result.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    result.append(current_chunk.strip())
        
        return result
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content."""
        code_blocks = []
        
        # Extract markdown code blocks
        code_pattern = r'```[\w]*\n(.*?)\n```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        code_blocks.extend(matches)
        
        # Extract inline code
        inline_code_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_code_pattern, content)
        code_blocks.extend([match for match in inline_matches if len(match) > 10])
        
        return code_blocks


class KnowledgeGraph:
    """Knowledge graph for storing relationships between entities."""
    
    def __init__(self):
        self.entities = {}  # entity_id -> entity_data
        self.relationships = {}  # (entity1, entity2) -> relationship_type
    
    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """Add an entity to the knowledge graph."""
        self.entities[entity_id] = entity_data
    
    def add_relationship(self, entity1: str, entity2: str, relationship_type: str):
        """Add a relationship between two entities."""
        self.relationships[(entity1, entity2)] = relationship_type
        # Add reverse relationship for undirected relationships
        if relationship_type in ['related_to', 'similar_to']:
            self.relationships[(entity2, entity1)] = relationship_type
    
    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """Get entities related to the given entity."""
        related = []
        
        for (e1, e2), rel_type in self.relationships.items():
            if e1 == entity_id:
                if relationship_type is None or rel_type == relationship_type:
                    related.append(e2)
            elif e2 == entity_id:
                if relationship_type is None or rel_type == relationship_type:
                    related.append(e1)
        
        return related
    
    def extract_entities_from_chunk(self, chunk: DocumentChunk) -> List[str]:
        """Extract entities from a document chunk."""
        entities = []
        content = chunk.content.lower()
        
        # Simple entity extraction patterns
        patterns = {
            'function': r'\b(\w+)\s*\(',
            'class': r'class\s+(\w+)',
            'variable': r'(\w+)\s*=',
            'import': r'import\s+(\w+)',
            'from_import': r'from\s+(\w+)\s+import',
            'api_endpoint': r'/(api/)?(\w+)',
            'technology': r'\b(python|javascript|react|vue|angular|django|flask|fastapi|numpy|pandas|tensorflow|pytorch)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[-1]  # Take the last group
                
                entity_id = f"{entity_type}:{match}"
                entities.append(entity_id)
                
                # Add to knowledge graph
                self.add_entity(entity_id, {
                    'name': match,
                    'type': entity_type,
                    'source_chunk': chunk.id
                })
        
        return entities
    
    def build_relationships(self, chunks: List[DocumentChunk]):
        """Build relationships between entities across chunks."""
        # Extract entities from all chunks
        for chunk in chunks:
            entities = self.extract_entities_from_chunk(chunk)
            chunk.entities = entities
            
            # Create co-occurrence relationships
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    self.add_relationship(entity1, entity2, 'co_occurs')
        
        # Create hierarchical relationships based on chunk hierarchy
        for chunk in chunks:
            if chunk.parent_id:
                parent_chunk = next((c for c in chunks if c.id == chunk.parent_id), None)
                if parent_chunk:
                    for entity in chunk.entities:
                        for parent_entity in parent_chunk.entities:
                            self.add_relationship(entity, parent_entity, 'part_of')


class KnowledgeBaseRetrieval:
    """Main knowledge base and retrieval system."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 storage_path: str = "knowledge_base"):
        self.embedding_model = embedding_model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.chunks = []  # In-memory storage for chunks
        self.embeddings = None  # Embedding matrix
        self.knowledge_graph = KnowledgeGraph()
        self.chunker = HierarchicalChunker()
        
        # Initialize knowledge sources
        self.sources = {}
    
    def add_source(self, name: str, source: KnowledgeSource):
        """Add a knowledge source."""
        self.sources[name] = source
    
    async def ingest_from_source(self, source_name: str, query: Optional[str] = None, limit: int = 100):
        """Ingest documents from a knowledge source."""
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        source = self.sources[source_name]
        logger.info(f"Ingesting documents from {source_name}...")
        
        # Fetch documents
        documents = await source.fetch_documents(query, limit)
        logger.info(f"Fetched {len(documents)} documents from {source_name}")
        
        # Process each document
        for doc in documents:
            try:
                metadata = source.extract_metadata(doc)
                content = self._extract_content(doc)
                
                if not content.strip():
                    continue
                
                # Chunk the document
                doc_chunks = self.chunker.chunk_document(content, metadata)
                
                # Generate embeddings
                contents = [chunk.content for chunk in doc_chunks]
                embeddings = self.embedding_model.encode(contents)
                
                # Assign embeddings to chunks
                for chunk, embedding in zip(doc_chunks, embeddings):
                    chunk.embedding = embedding
                
                # Add to knowledge base
                self.chunks.extend(doc_chunks)
                
                # Build knowledge graph relationships
                self.knowledge_graph.build_relationships(doc_chunks)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        # Update embedding matrix
        self._update_embedding_matrix()
        
        # Save to disk
        self.save()
        
        logger.info(f"Successfully ingested {len(documents)} documents. Total chunks: {len(self.chunks)}")
    
    def _extract_content(self, document: Dict[str, Any]) -> str:
        """Extract text content from a document."""
        # Try different content fields
        content_fields = ['content', 'body', 'readme_content', 'description', 'title']
        
        content_parts = []
        for field in content_fields:
            if field in document and document[field]:
                content_parts.append(str(document[field]))
        
        return '\n\n'.join(content_parts)
    
    def _update_embedding_matrix(self):
        """Update the embedding matrix with all chunk embeddings."""
        if not self.chunks:
            self.embeddings = None
            return
        
        embeddings_list = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings_list.append(chunk.embedding)
        
        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
        else:
            self.embeddings = None
    
    def search(self, query: str, top_k: int = 10, chunk_types: Optional[List[ChunkType]] = None) -> List[DocumentChunk]:
        """Search for relevant chunks using semantic similarity."""
        if not self.chunks or self.embeddings is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to filter
        
        # Filter by chunk types if specified
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                if chunk_types is None or chunk.chunk_type in chunk_types:
                    chunk.relevance_score = similarities[idx]
                    results.append(chunk)
                    
                    if len(results) >= top_k:
                        break
        
        return results
    
    def get_related_chunks(self, chunk_id: str, max_related: int = 5) -> List[DocumentChunk]:
        """Get chunks related to the given chunk through the knowledge graph."""
        chunk = next((c for c in self.chunks if c.id == chunk_id), None)
        if not chunk:
            return []
        
        related_chunks = []
        
        # Get related entities
        for entity in chunk.entities:
            related_entities = self.knowledge_graph.get_related_entities(entity)
            
            for related_entity in related_entities:
                # Find chunks containing this entity
                for other_chunk in self.chunks:
                    if (other_chunk.id != chunk_id and 
                        related_entity in other_chunk.entities and
                        other_chunk not in related_chunks):
                        related_chunks.append(other_chunk)
                        
                        if len(related_chunks) >= max_related:
                            break
                
                if len(related_chunks) >= max_related:
                    break
            
            if len(related_chunks) >= max_related:
                break
        
        return related_chunks
    
    def save(self):
        """Save the knowledge base to disk."""
        # Save chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(self.storage_path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(self.storage_path / "embeddings.npy", self.embeddings)
        
        # Save knowledge graph
        kg_data = {
            'entities': self.knowledge_graph.entities,
            'relationships': {f"{k[0]}|{k[1]}": v for k, v in self.knowledge_graph.relationships.items()}
        }
        with open(self.storage_path / "knowledge_graph.json", 'w') as f:
            json.dump(kg_data, f, indent=2, default=str)
        
        logger.info(f"Knowledge base saved to {self.storage_path}")
    
    def load(self):
        """Load the knowledge base from disk."""
        try:
            # Load chunks
            chunks_file = self.storage_path / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                
                self.chunks = []
                for chunk_data in chunks_data:
                    # Reconstruct metadata
                    metadata_data = chunk_data['metadata']
                    metadata = DocumentMetadata(
                        source_type=SourceType(metadata_data['source_type']),
                        source_url=metadata_data['source_url'],
                        title=metadata_data['title'],
                        author=metadata_data.get('author'),
                        created_at=datetime.fromisoformat(metadata_data['created_at']) if metadata_data.get('created_at') else None,
                        updated_at=datetime.fromisoformat(metadata_data['updated_at']) if metadata_data.get('updated_at') else None,
                        language=metadata_data.get('language'),
                        tags=metadata_data.get('tags', []),
                        file_type=metadata_data.get('file_type'),
                        size_bytes=metadata_data.get('size_bytes'),
                        checksum=metadata_data.get('checksum')
                    )
                    
                    # Reconstruct chunk
                    chunk = DocumentChunk(
                        id=chunk_data['id'],
                        content=chunk_data['content'],
                        chunk_type=ChunkType(chunk_data['chunk_type']),
                        metadata=metadata,
                        parent_id=chunk_data.get('parent_id'),
                        children_ids=chunk_data.get('children_ids', []),
                        embedding=np.array(chunk_data['embedding']) if chunk_data.get('embedding') else None,
                        keywords=chunk_data.get('keywords', []),
                        entities=chunk_data.get('entities', []),
                        relationships=chunk_data.get('relationships', {}),
                        relevance_score=chunk_data.get('relevance_score', 0.0)
                    )
                    self.chunks.append(chunk)
            
            # Load embeddings
            embeddings_file = self.storage_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
            
            # Load knowledge graph
            kg_file = self.storage_path / "knowledge_graph.json"
            if kg_file.exists():
                with open(kg_file, 'r') as f:
                    kg_data = json.load(f)
                
                self.knowledge_graph.entities = kg_data.get('entities', {})
                
                # Reconstruct relationships
                relationships = {}
                for key, value in kg_data.get('relationships', {}).items():
                    entity1, entity2 = key.split('|', 1)
                    relationships[(entity1, entity2)] = value
                self.knowledge_graph.relationships = relationships
            
            logger.info(f"Knowledge base loaded from {self.storage_path}. Chunks: {len(self.chunks)}")
        
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")