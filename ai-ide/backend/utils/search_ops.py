"""
Search operation utilities
"""
import os
import re
from typing import List, Dict

def grep_search(pattern: str, directory: str = ".", file_extensions: List[str] = None) -> List[Dict]:
    """Search for pattern in files"""
    results = []
    
    if file_extensions is None:
        file_extensions = ['.py', '.js', '.ts', '.txt', '.md']
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if re.search(pattern, line, re.IGNORECASE):
                                    results.append({
                                        'file': file_path,
                                        'line': line_num,
                                        'content': line.strip()
                                    })
                    except:
                        continue
    except Exception as e:
        print(f"Search error: {e}")
    
    return results