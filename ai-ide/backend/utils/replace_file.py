"""
File replacement utilities
"""

def replace_file(file_path: str, content: str) -> bool:
    """Replace file content"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error replacing file {file_path}: {e}")
        return False