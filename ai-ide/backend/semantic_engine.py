"""
Semantic Engine for AI IDE
Advanced semantic awareness and context management
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

logger = logging.getLogger('semantic_engine')

class CodeContext:
    """Represents semantic context of code"""
    
    def __init__(self, file_path: str, content: str, language: str = None):
        self.file_path = file_path
        self.content = content
        self.language = language or self._detect_language(file_path)
        self.hash = self._compute_hash(content)
        self.metadata = {}
        self.semantic_features = {}
        
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
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
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def extract_semantic_features(self) -> Dict[str, Any]:
        """Extract semantic features from code"""
        if self.semantic_features:
            return self.semantic_features
            
        features = {
            'functions': self._extract_functions(),
            'classes': self._extract_classes(),
            'imports': self._extract_imports(),
            'variables': self._extract_variables(),
            'comments': self._extract_comments(),
            'complexity': self._calculate_complexity(),
            'patterns': self._identify_patterns()
        }
        
        self.semantic_features = features
        return features
    
    def _extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function definitions"""
        functions = []
        
        if self.language == 'python':
            pattern = r'def\s+(\w+)\s*\([^)]*\):'
            matches = re.finditer(pattern, self.content, re.MULTILINE)
            
            for match in matches:
                func_name = match.group(1)
                line_num = self.content[:match.start()].count('\n') + 1
                
                # Extract docstring if present
                docstring = self._extract_docstring(match.end())
                
                functions.append({
                    'name': func_name,
                    'line': line_num,
                    'docstring': docstring,
                    'signature': match.group(0)
                })
        
        elif self.language in ['javascript', 'typescript']:
            # Function declarations and expressions
            patterns = [
                r'function\s+(\w+)\s*\([^)]*\)',
                r'(\w+)\s*:\s*function\s*\([^)]*\)',
                r'(\w+)\s*=\s*\([^)]*\)\s*=>',
                r'(\w+)\s*=\s*function\s*\([^)]*\)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, self.content, re.MULTILINE)
                for match in matches:
                    func_name = match.group(1)
                    line_num = self.content[:match.start()].count('\n') + 1
                    
                    functions.append({
                        'name': func_name,
                        'line': line_num,
                        'signature': match.group(0)
                    })
        
        return functions
    
    def _extract_classes(self) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []
        
        if self.language == 'python':
            pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
            matches = re.finditer(pattern, self.content, re.MULTILINE)
            
            for match in matches:
                class_name = match.group(1)
                line_num = self.content[:match.start()].count('\n') + 1
                
                classes.append({
                    'name': class_name,
                    'line': line_num,
                    'signature': match.group(0)
                })
        
        elif self.language in ['javascript', 'typescript']:
            pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
            matches = re.finditer(pattern, self.content, re.MULTILINE)
            
            for match in matches:
                class_name = match.group(1)
                line_num = self.content[:match.start()].count('\n') + 1
                
                classes.append({
                    'name': class_name,
                    'line': line_num,
                    'signature': match.group(0)
                })
        
        return classes
    
    def _extract_imports(self) -> List[Dict[str, Any]]:
        """Extract import statements"""
        imports = []
        
        if self.language == 'python':
            patterns = [
                r'import\s+([^\n]+)',
                r'from\s+([^\s]+)\s+import\s+([^\n]+)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, self.content, re.MULTILINE)
                for match in matches:
                    line_num = self.content[:match.start()].count('\n') + 1
                    imports.append({
                        'statement': match.group(0),
                        'line': line_num
                    })
        
        elif self.language in ['javascript', 'typescript']:
            patterns = [
                r'import\s+[^;]+;',
                r'const\s+[^=]+\s*=\s*require\([^)]+\)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, self.content, re.MULTILINE)
                for match in matches:
                    line_num = self.content[:match.start()].count('\n') + 1
                    imports.append({
                        'statement': match.group(0),
                        'line': line_num
                    })
        
        return imports
    
    def _extract_variables(self) -> List[Dict[str, Any]]:
        """Extract variable declarations"""
        variables = []
        
        if self.language == 'python':
            # Simple variable assignments
            pattern = r'^(\s*)(\w+)\s*=\s*([^\n]+)'
            matches = re.finditer(pattern, self.content, re.MULTILINE)
            
            for match in matches:
                var_name = match.group(2)
                line_num = self.content[:match.start()].count('\n') + 1
                
                variables.append({
                    'name': var_name,
                    'line': line_num,
                    'assignment': match.group(3).strip()
                })
        
        return variables
    
    def _extract_comments(self) -> List[Dict[str, Any]]:
        """Extract comments"""
        comments = []
        
        if self.language == 'python':
            pattern = r'#\s*([^\n]+)'
        elif self.language in ['javascript', 'typescript', 'java', 'cpp', 'c']:
            pattern = r'//\s*([^\n]+)'
        else:
            return comments
        
        matches = re.finditer(pattern, self.content, re.MULTILINE)
        for match in matches:
            line_num = self.content[:match.start()].count('\n') + 1
            comments.append({
                'text': match.group(1).strip(),
                'line': line_num
            })
        
        return comments
    
    def _extract_docstring(self, start_pos: int) -> Optional[str]:
        """Extract docstring after function definition"""
        remaining = self.content[start_pos:]
        
        # Look for triple quotes
        patterns = [r'"""([^"]+)"""', r"'''([^']+)'''"]
        
        for pattern in patterns:
            match = re.search(pattern, remaining, re.DOTALL)
            if match and match.start() < 100:  # Must be close to function def
                return match.group(1).strip()
        
        return None
    
    def _calculate_complexity(self) -> Dict[str, int]:
        """Calculate code complexity metrics"""
        lines = self.content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'cyclomatic_complexity': self._cyclomatic_complexity()
        }
    
    def _cyclomatic_complexity(self) -> int:
        """Calculate cyclomatic complexity"""
        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally']
        complexity = 1  # Base complexity
        
        for keyword in decision_keywords:
            pattern = rf'\b{keyword}\b'
            complexity += len(re.findall(pattern, self.content))
        
        return complexity
    
    def _identify_patterns(self) -> List[str]:
        """Identify common code patterns"""
        patterns = []
        
        # Design patterns
        if 'class' in self.content.lower():
            if 'singleton' in self.content.lower():
                patterns.append('singleton')
            if 'factory' in self.content.lower():
                patterns.append('factory')
            if 'observer' in self.content.lower():
                patterns.append('observer')
        
        # Common patterns
        if 'try:' in self.content and 'except' in self.content:
            patterns.append('error_handling')
        
        if 'async def' in self.content or 'await' in self.content:
            patterns.append('async_programming')
        
        if 'test_' in self.content or 'def test' in self.content:
            patterns.append('unit_testing')
        
        return patterns

class SemanticIndex:
    """Maintains semantic index of codebase"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.index_file = os.path.join(workspace_path, '.ai-ide', 'semantic_index.json')
        self.contexts: Dict[str, CodeContext] = {}
        self.load_index()
    
    def load_index(self):
        """Load existing semantic index"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct contexts from saved data
                for file_path, context_data in data.get('contexts', {}).items():
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as cf:
                            content = cf.read()
                            context = CodeContext(file_path, content)
                            context.semantic_features = context_data.get('semantic_features', {})
                            self.contexts[file_path] = context
                            
            except Exception as e:
                logger.warning(f"Failed to load semantic index: {e}")
    
    def save_index(self):
        """Save semantic index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            data = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'contexts': {}
            }
            
            for file_path, context in self.contexts.items():
                data['contexts'][file_path] = {
                    'hash': context.hash,
                    'language': context.language,
                    'semantic_features': context.semantic_features,
                    'metadata': context.metadata
                }
            
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save semantic index: {e}")
    
    def index_file(self, file_path: str) -> CodeContext:
        """Index a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            context = CodeContext(file_path, content)
            context.extract_semantic_features()
            
            self.contexts[file_path] = context
            return context
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return None
    
    def index_workspace(self, extensions: List[str] = None):
        """Index entire workspace"""
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']
        
        indexed_count = 0
        
        try:
            for root, dirs, files in os.walk(self.workspace_path):
                # Skip common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
                
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        
                        # Check if file needs reindexing
                        if self._needs_reindexing(file_path):
                            context = self.index_file(file_path)
                            if context:
                                indexed_count += 1
            
            logger.info(f"Indexed {indexed_count} files")
            self.save_index()
        except Exception as e:
            logger.error(f"Failed to index workspace: {e}")
            # Continue with empty index
    
    def _needs_reindexing(self, file_path: str) -> bool:
        """Check if file needs reindexing"""
        if file_path not in self.contexts:
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                current_hash = hashlib.md5(content.encode()).hexdigest()
                return current_hash != self.contexts[file_path].hash
        except:
            return True
    
    def search_semantic(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across indexed code"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for file_path, context in self.contexts.items():
            score = self._calculate_semantic_score(query_words, context)
            
            if score > 0:
                results.append({
                    'file': file_path,
                    'score': score,
                    'context': context,
                    'matches': self._find_matches(query_lower, context)
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    def _calculate_semantic_score(self, query_words: set, context: CodeContext) -> float:
        """Calculate semantic similarity score"""
        score = 0.0
        features = context.semantic_features
        
        # Check function names
        for func in features.get('functions', []):
            func_words = set(func['name'].lower().split('_'))
            intersection = query_words.intersection(func_words)
            if intersection:
                score += len(intersection) * 2.0  # Functions get higher weight
        
        # Check class names
        for cls in features.get('classes', []):
            cls_words = set(cls['name'].lower().split('_'))
            intersection = query_words.intersection(cls_words)
            if intersection:
                score += len(intersection) * 1.5
        
        # Check comments
        for comment in features.get('comments', []):
            comment_words = set(comment['text'].lower().split())
            intersection = query_words.intersection(comment_words)
            if intersection:
                score += len(intersection) * 0.5
        
        # Check content directly
        content_words = set(context.content.lower().split())
        intersection = query_words.intersection(content_words)
        score += len(intersection) * 0.1
        
        return score
    
    def _find_matches(self, query: str, context: CodeContext) -> List[Dict[str, Any]]:
        """Find specific matches in context"""
        matches = []
        lines = context.content.split('\n')
        
        for i, line in enumerate(lines):
            if query in line.lower():
                matches.append({
                    'line': i + 1,
                    'content': line.strip(),
                    'type': 'direct_match'
                })
        
        # Add semantic matches from features
        features = context.semantic_features
        
        for func in features.get('functions', []):
            if query in func['name'].lower():
                matches.append({
                    'line': func['line'],
                    'content': func['signature'],
                    'type': 'function_match'
                })
        
        return matches

class PerformanceTracker:
    """Tracks performance metrics for semantic operations"""
    
    def __init__(self):
        self.metrics = {
            'search_times': [],
            'index_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def record_search_time(self, duration: float):
        """Record search operation time"""
        self.metrics['search_times'].append(duration)
        
        # Keep only last 100 measurements
        if len(self.metrics['search_times']) > 100:
            self.metrics['search_times'] = self.metrics['search_times'][-100:]
    
    def record_index_time(self, duration: float):
        """Record indexing operation time"""
        self.metrics['index_times'].append(duration)
        
        if len(self.metrics['index_times']) > 100:
            self.metrics['index_times'] = self.metrics['index_times'][-100:]
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics['cache_misses'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        search_times = self.metrics['search_times']
        index_times = self.metrics['index_times']
        
        return {
            'avg_search_time': sum(search_times) / len(search_times) if search_times else 0,
            'avg_index_time': sum(index_times) / len(index_times) if index_times else 0,
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'total_searches': len(search_times),
            'total_indexes': len(index_times)
        }

# Global instances
_semantic_index = None
_performance_tracker = PerformanceTracker()

def get_semantic_index(workspace_path: str = None) -> SemanticIndex:
    """Get or create semantic index"""
    global _semantic_index
    
    if _semantic_index is None or (workspace_path and _semantic_index.workspace_path != workspace_path):
        if workspace_path is None:
            workspace_path = os.getcwd()
        _semantic_index = SemanticIndex(workspace_path)
    
    return _semantic_index

def get_performance_tracker() -> PerformanceTracker:
    """Get performance tracker"""
    return _performance_tracker