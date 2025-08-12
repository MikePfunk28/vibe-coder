"""
File reading utilities
"""
import os
from typing import Optional

def read_file(file_path: str) -> Optional[str]:
    """Read file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None