"""
Code Embedding Generation System
Advanced semantic code embeddings using sentence-transformers and vector storage
"""

import os
import json
import logging
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available. Install with: pip install faiss-cpu")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("chromadb not available. Install with: pip install chromadb")

logger = logging.getLogger('code_embedding_generator')

class CodeChunk:
    """Represents a chunk of code with metadata for embedding"""
    
    def __init__(self, file_path: str, content: str, chunk_type: str, 
                 line_start: int, line_end: int, metadata: Dict[str, Any] = None):
        self.file_path = file_path
        self.content = content
        self.chunk_type = chunk_type  # 'function', 'class', 'file', 'comment', etc.
        self.line_start = line_start
        self.line_end = line_end
        self.metadata = metadata or {}
        self.hash = self._compute_hash()
        self.embedding: Optional[np.ndarray] = None
        self.embedding_model: Optional[str] = None
        
    def _compute_hash(self) -> str:
        """Compute content hash for change detection"""
        content_str = f"{self.file_path}:{self.content}:{self.chunk_type}"
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'file_path': self.file_path,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'metadata': self.metadata,
            'hash': self.hash,
            'embedding_model': self.embedding_model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create from dictionary"""
        chunk = cls(
            file_path=data['file_path'],
            content=data['content'],
            chunk_type=data['chunk_type'],
            line_start=data['line_start'],
            line_end=data['line_end'],
            metadata=data.get('metadata', {})
        )
        chunk.hash = data.get('hash', chunk.hash)
        chunk.embedding_model = data.get('embedding_model')
        return chunk

class CodeEmbeddingGenerator:
    """Generates and manages code embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", 
                 workspace_path: str = None, vector_store_type: str = "faiss"):
        self.model_name = model_name
        self.workspace_path = workspace_path or os.getcwd()
        self.vector_store_type = vector_store_type.lower()
        
        # Initialize embedding model
        self.model = None
        self.embedding_dim = None
        self._init_embedding_model()
        
        # Storage paths
        self.embeddings_dir = os.path.join(self.workspace_path, '.ai-ide', 'embeddings')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Vector store
        self.vector_store = None
        self._init_vector_store()
        
        # Code chunks and metadata
        self.chunks: Dict[str, CodeChunk] = {}
        self.chunk_index: Dict[str, List[str]] = {}  # file_path -> chunk_ids
        self.metadata_file = os.path.join(self.embeddings_dir, 'chunks_metadata.json')
        
        # Performance tracking
        self.stats = {
            'total_chunks': 0,
            'total_embeddings': 0,
            'last_update': None,
            'embedding_times': [],
            'indexing_times': []
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_metadata()
    
    def _init_embedding_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Cannot initialize embedding model.")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Trying fallback model: all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                logger.info(f"Fallback model loaded. Dimension: {self.embedding_dim}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.model = None
    
    def _init_vector_store(self):
        """Initialize vector storage backend"""
        if self.vector_store_type == "faiss" and FAISS_AVAILABLE and self.embedding_dim:
            self._init_faiss_store()
        elif self.vector_store_type == "chroma" and CHROMADB_AVAILABLE:
            self._init_chroma_store()
        else:
            logger.warning(f"Vector store {self.vector_store_type} not available. Using in-memory storage.")
            self.vector_store = None
    
    def _init_faiss_store(self):
        """Initialize FAISS vector store"""
        try:
            index_file = os.path.join(self.embeddings_dir, 'faiss_index.bin')
            
            if os.path.exists(index_file):
                # Load existing index
                self.vector_store = faiss.read_index(index_file)
                logger.info(f"Loaded existing FAISS index with {self.vector_store.ntotal} vectors")
            else:
                # Create new index
                self.vector_store = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS store: {e}")
            self.vector_store = None
    
    def _init_chroma_store(self):
        """Initialize ChromaDB vector store"""
        try:
            chroma_dir = os.path.join(self.embeddings_dir, 'chroma_db')
            
            client = chromadb.PersistentClient(path=chroma_dir)
            self.vector_store = client.get_or_create_collection(
                name="code_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB store with {self.vector_store.count()} vectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB store: {e}")
            self.vector_store = None
    
    def _load_metadata(self):
        """Load existing chunks metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct chunks
                for chunk_id, chunk_data in data.get('chunks', {}).items():
                    chunk = CodeChunk.from_dict(chunk_data)
                    self.chunks[chunk_id] = chunk
                
                # Reconstruct file index
                self.chunk_index = data.get('chunk_index', {})
                self.stats = data.get('stats', self.stats)
                
                logger.info(f"Loaded {len(self.chunks)} existing code chunks")
                
            except Exception as e:
                logger.error(f"Failed to load chunks metadata: {e}")
    
    def _save_metadata(self):
        """Save chunks metadata"""
        try:
            data = {
                'version': '1.0',
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'vector_store_type': self.vector_store_type,
                'timestamp': datetime.now().isoformat(),
                'chunks': {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
                'chunk_index': self.chunk_index,
                'stats': self.stats
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save chunks metadata: {e}")
    
    def chunk_code_file(self, file_path: str, content: str = None) -> List[CodeChunk]:
        """Split code file into semantic chunks"""
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return []
        
        chunks = []
        lines = content.split('\n')
        language = self._detect_language(file_path)
        
        # Create file-level chunk
        file_chunk = CodeChunk(
            file_path=file_path,
            content=content[:1000],  # First 1000 chars for file overview
            chunk_type='file',
            line_start=1,
            line_end=len(lines),
            metadata={'language': language, 'total_lines': len(lines)}
        )
        chunks.append(file_chunk)
        
        # Extract function and class chunks
        if language == 'python':
            chunks.extend(self._chunk_python_code(file_path, content, lines))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._chunk_js_code(file_path, content, lines))
        elif language in ['java', 'cpp', 'c']:
            chunks.extend(self._chunk_c_like_code(file_path, content, lines))
        
        # Add comment chunks for significant comments
        comment_chunks = self._extract_comment_chunks(file_path, content, lines, language)
        chunks.extend(comment_chunks)
        
        return chunks
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'text')
    
    def _chunk_python_code(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract Python functions and classes as chunks"""
        chunks = []
        
        import re
        
        # Find function definitions
        func_pattern = r'^(\s*)def\s+(\w+)\s*\([^)]*\):'
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line)
            if match:
                indent_level = len(match.group(1))
                func_name = match.group(2)
                
                # Find function end
                func_end = self._find_python_block_end(lines, i, indent_level)
                func_content = '\n'.join(lines[i:func_end + 1])
                
                chunk = CodeChunk(
                    file_path=file_path,
                    content=func_content,
                    chunk_type='function',
                    line_start=i + 1,
                    line_end=func_end + 1,
                    metadata={'name': func_name, 'language': 'python'}
                )
                chunks.append(chunk)
        
        # Find class definitions
        class_pattern = r'^(\s*)class\s+(\w+)(?:\([^)]*\))?:'
        for i, line in enumerate(lines):
            match = re.match(class_pattern, line)
            if match:
                indent_level = len(match.group(1))
                class_name = match.group(2)
                
                # Find class end
                class_end = self._find_python_block_end(lines, i, indent_level)
                class_content = '\n'.join(lines[i:class_end + 1])
                
                chunk = CodeChunk(
                    file_path=file_path,
                    content=class_content,
                    chunk_type='class',
                    line_start=i + 1,
                    line_end=class_end + 1,
                    metadata={'name': class_name, 'language': 'python'}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _find_python_block_end(self, lines: List[str], start_line: int, base_indent: int) -> int:
        """Find the end of a Python code block"""
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i - 1
        
        return len(lines) - 1
    
    def _chunk_js_code(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript functions and classes as chunks"""
        chunks = []
        
        import re
        
        # Function patterns
        patterns = [
            (r'function\s+(\w+)\s*\([^)]*\)\s*{', 'function'),
            (r'(\w+)\s*:\s*function\s*\([^)]*\)\s*{', 'function'),
            (r'(\w+)\s*=\s*\([^)]*\)\s*=>\s*{', 'arrow_function'),
            (r'(\w+)\s*=\s*function\s*\([^)]*\)\s*{', 'function'),
            (r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*{', 'class')
        ]
        
        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    
                    # Find block end by counting braces
                    block_end = self._find_js_block_end(lines, i)
                    if block_end > i:
                        block_content = '\n'.join(lines[i:block_end + 1])
                        
                        chunk = CodeChunk(
                            file_path=file_path,
                            content=block_content,
                            chunk_type=chunk_type,
                            line_start=i + 1,
                            line_end=block_end + 1,
                            metadata={'name': name, 'language': 'javascript'}
                        )
                        chunks.append(chunk)
                    break
        
        return chunks
    
    def _find_js_block_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a JavaScript code block by counting braces"""
        brace_count = 0
        found_opening = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
                    if found_opening and brace_count == 0:
                        return i
        
        return len(lines) - 1
    
    def _chunk_c_like_code(self, file_path: str, content: str, lines: List[str]) -> List[CodeChunk]:
        """Extract C/C++/Java functions and classes as chunks"""
        chunks = []
        
        import re
        
        # Simple function pattern (this could be more sophisticated)
        func_pattern = r'^\s*(?:public|private|protected|static|\w+\s+)*(\w+)\s+(\w+)\s*\([^)]*\)\s*{'
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match and not line.strip().startswith('//'):
                func_name = match.group(2)
                
                # Find block end
                block_end = self._find_js_block_end(lines, i)  # Same logic as JS
                if block_end > i:
                    block_content = '\n'.join(lines[i:block_end + 1])
                    
                    chunk = CodeChunk(
                        file_path=file_path,
                        content=block_content,
                        chunk_type='function',
                        line_start=i + 1,
                        line_end=block_end + 1,
                        metadata={'name': func_name, 'language': self._detect_language(file_path)}
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_comment_chunks(self, file_path: str, content: str, lines: List[str], language: str) -> List[CodeChunk]:
        """Extract significant comment blocks as chunks"""
        chunks = []
        
        if language == 'python':
            comment_char = '#'
        elif language in ['javascript', 'typescript', 'java', 'cpp', 'c']:
            comment_char = '//'
        else:
            return chunks
        
        current_comment = []
        comment_start = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith(comment_char):
                if comment_start is None:
                    comment_start = i
                current_comment.append(stripped[len(comment_char):].strip())
            else:
                if current_comment and len(current_comment) >= 3:  # Only significant comment blocks
                    comment_content = '\n'.join(current_comment)
                    
                    chunk = CodeChunk(
                        file_path=file_path,
                        content=comment_content,
                        chunk_type='comment',
                        line_start=comment_start + 1,
                        line_end=i,
                        metadata={'language': language, 'comment_lines': len(current_comment)}
                    )
                    chunks.append(chunk)
                
                current_comment = []
                comment_start = None
        
        # Handle comment at end of file
        if current_comment and len(current_comment) >= 3:
            comment_content = '\n'.join(current_comment)
            
            chunk = CodeChunk(
                file_path=file_path,
                content=comment_content,
                chunk_type='comment',
                line_start=comment_start + 1,
                line_end=len(lines),
                metadata={'language': language, 'comment_lines': len(current_comment)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generate embeddings for code chunks"""
        if not self.model:
            logger.error("Embedding model not available")
            return chunks
        
        start_time = time.time()
        
        # Prepare texts for embedding
        texts = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.content.strip():
                # Create embedding text with context
                embedding_text = self._prepare_embedding_text(chunk)
                texts.append(embedding_text)
                valid_chunks.append(chunk)
        
        if not texts:
            return chunks
        
        try:
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(valid_chunks, embeddings):
                chunk.embedding = embedding
                chunk.embedding_model = self.model_name
            
            # Update stats
            embedding_time = time.time() - start_time
            self.stats['embedding_times'].append(embedding_time)
            self.stats['total_embeddings'] += len(valid_chunks)
            
            logger.info(f"Generated {len(valid_chunks)} embeddings in {embedding_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
        
        return valid_chunks
    
    def _prepare_embedding_text(self, chunk: CodeChunk) -> str:
        """Prepare text for embedding generation"""
        # Include chunk type and metadata in embedding text
        context_parts = []
        
        # Add chunk type
        context_parts.append(f"Type: {chunk.chunk_type}")
        
        # Add language
        if 'language' in chunk.metadata:
            context_parts.append(f"Language: {chunk.metadata['language']}")
        
        # Add name if available
        if 'name' in chunk.metadata:
            context_parts.append(f"Name: {chunk.metadata['name']}")
        
        # Add file context
        file_name = os.path.basename(chunk.file_path)
        context_parts.append(f"File: {file_name}")
        
        # Combine context with content
        context_str = " | ".join(context_parts)
        embedding_text = f"{context_str}\n\n{chunk.content}"
        
        # Limit text length for embedding model
        max_length = 512  # Most models have token limits
        if len(embedding_text) > max_length:
            embedding_text = embedding_text[:max_length]
        
        return embedding_text
    
    def index_file(self, file_path: str, force_reindex: bool = False) -> List[CodeChunk]:
        """Index a single file and generate embeddings"""
        with self._lock:
            # Check if file needs reindexing
            if not force_reindex and not self._needs_reindexing(file_path):
                logger.debug(f"File {file_path} already indexed and up to date")
                return self.chunk_index.get(file_path, [])
            
            start_time = time.time()
            
            try:
                # Remove existing chunks for this file
                self._remove_file_chunks(file_path)
                
                # Chunk the file
                chunks = self.chunk_code_file(file_path)
                if not chunks:
                    return []
                
                # Generate embeddings
                chunks_with_embeddings = self.generate_embeddings(chunks)
                
                # Store chunks
                chunk_ids = []
                for chunk in chunks_with_embeddings:
                    chunk_id = f"{file_path}:{chunk.chunk_type}:{chunk.line_start}"
                    self.chunks[chunk_id] = chunk
                    chunk_ids.append(chunk_id)
                
                # Update file index
                self.chunk_index[file_path] = chunk_ids
                
                # Add to vector store
                self._add_to_vector_store(chunks_with_embeddings)
                
                # Update stats
                indexing_time = time.time() - start_time
                self.stats['indexing_times'].append(indexing_time)
                self.stats['total_chunks'] += len(chunks_with_embeddings)
                self.stats['last_update'] = datetime.now().isoformat()
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Indexed {file_path}: {len(chunks_with_embeddings)} chunks in {indexing_time:.2f}s")
                return chunks_with_embeddings
                
            except Exception as e:
                logger.error(f"Failed to index file {file_path}: {e}")
                return []
    
    def _needs_reindexing(self, file_path: str) -> bool:
        """Check if file needs reindexing"""
        if file_path not in self.chunk_index:
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if any existing chunk has different hash
            for chunk_id in self.chunk_index[file_path]:
                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    if chunk.chunk_type == 'file':
                        # For file chunks, compare first 1000 chars
                        current_hash = hashlib.md5(f"{file_path}:{content[:1000]}:file".encode()).hexdigest()
                        if current_hash != chunk.hash:
                            return True
            
            return False
            
        except Exception:
            return True
    
    def _remove_file_chunks(self, file_path: str):
        """Remove all chunks for a file"""
        if file_path in self.chunk_index:
            for chunk_id in self.chunk_index[file_path]:
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]
            del self.chunk_index[file_path]
    
    def _add_to_vector_store(self, chunks: List[CodeChunk]):
        """Add chunks to vector store"""
        if not self.vector_store:
            return
        
        try:
            if self.vector_store_type == "faiss" and FAISS_AVAILABLE:
                self._add_to_faiss(chunks)
            elif self.vector_store_type == "chroma" and CHROMADB_AVAILABLE:
                self._add_to_chroma(chunks)
                
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
    
    def _add_to_faiss(self, chunks: List[CodeChunk]):
        """Add chunks to FAISS index"""
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.vector_store.add(embeddings_array)
            
            # Save index
            index_file = os.path.join(self.embeddings_dir, 'faiss_index.bin')
            faiss.write_index(self.vector_store, index_file)
    
    def _add_to_chroma(self, chunks: List[CodeChunk]):
        """Add chunks to ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk_id = f"{chunk.file_path}:{chunk.chunk_type}:{chunk.line_start}"
                documents.append(chunk.content)
                metadatas.append({
                    'file_path': chunk.file_path,
                    'chunk_type': chunk.chunk_type,
                    'line_start': chunk.line_start,
                    'line_end': chunk.line_end,
                    **chunk.metadata
                })
                ids.append(chunk_id)
                embeddings.append(chunk.embedding.tolist())
        
        if documents:
            self.vector_store.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
    
    def index_workspace(self, extensions: List[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """Index entire workspace with parallel processing"""
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h']
        
        start_time = time.time()
        indexed_files = []
        failed_files = []
        
        # Find all files to index
        files_to_index = []
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    if self._needs_reindexing(file_path):
                        files_to_index.append(file_path)
        
        logger.info(f"Found {len(files_to_index)} files to index")
        
        # Index files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.index_file, file_path): file_path 
                             for file_path in files_to_index}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    if chunks:
                        indexed_files.append(file_path)
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
                    failed_files.append(file_path)
        
        # Save final state
        if self.vector_store_type == "faiss" and FAISS_AVAILABLE and self.vector_store:
            index_file = os.path.join(self.embeddings_dir, 'faiss_index.bin')
            faiss.write_index(self.vector_store, index_file)
        
        total_time = time.time() - start_time
        
        result = {
            'indexed_files': len(indexed_files),
            'failed_files': len(failed_files),
            'total_chunks': self.stats['total_chunks'],
            'total_time': total_time,
            'files_per_second': len(indexed_files) / total_time if total_time > 0 else 0
        }
        
        logger.info(f"Workspace indexing complete: {result}")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        return {
            **self.stats,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'vector_store_type': self.vector_store_type,
            'total_files': len(self.chunk_index),
            'avg_embedding_time': np.mean(self.stats['embedding_times']) if self.stats['embedding_times'] else 0,
            'avg_indexing_time': np.mean(self.stats['indexing_times']) if self.stats['indexing_times'] else 0
        }

# Global instance
_embedding_generator = None

def get_embedding_generator(workspace_path: str = None, model_name: str = "microsoft/codebert-base") -> CodeEmbeddingGenerator:
    """Get or create embedding generator"""
    global _embedding_generator
    
    if _embedding_generator is None or (workspace_path and _embedding_generator.workspace_path != workspace_path):
        if workspace_path is None:
            workspace_path = os.getcwd()
        _embedding_generator = CodeEmbeddingGenerator(model_name=model_name, workspace_path=workspace_path)
    
    return _embedding_generator