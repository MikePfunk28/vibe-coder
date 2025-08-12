"""
Directory operation utilities
"""
import os
from typing import List, Dict

def list_dir(directory: str = ".") -> List[Dict]:
    """List directory contents"""
    results = []
    
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            results.append({
                'name': item,
                'path': item_path,
                'type': 'directory' if os.path.isdir(item_path) else 'file',
                'size': os.path.getsize(item_path) if os.path.isfile(item_path) else 0
            })
    except Exception as e:
        print(f"Directory listing error: {e}")
    
    return results