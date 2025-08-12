"""
File deletion utilities
"""
import os

def delete_file(file_path: str) -> bool:
    """Delete a file"""
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False