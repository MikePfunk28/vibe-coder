#!/usr/bin/env python3
"""
Setup script to create Ollama models from your LMStudio GGUF files
"""

import os
import subprocess
import tempfile

# Your LMStudio model paths
LMSTUDIO_MODELS = {
    "phi-4-reasoning-plus": r"C:\Users\mikep\.lmstudio\models\lmstudio-community\Phi-4-reasoning-plus-GGUF\Phi-4-reasoning-plus-Q4_K_M.gguf",
    # Add paths for other models when you find them
    # "phi-4-mini-reasoning": r"C:\Users\mikep\.lmstudio\models\...",
    # "phi-4-mini-instruct": r"C:\Users\mikep\.lmstudio\models\...",
}

def create_modelfile(model_name: str, gguf_path: str) -> str:
    """Create a Modelfile for Ollama"""
    modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"<|system|>
You are a helpful AI assistant.
<|user|>
{{{{ .Prompt }}}}
<|assistant|>
\"\"\"

PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
"""
    return modelfile_content

def setup_ollama_model(model_name: str, gguf_path: str):
    """Setup a model in Ollama"""
    print(f"Setting up {model_name} in Ollama...")

    # Check if GGUF file exists
    if not os.path.exists(gguf_path):
        print(f"Error: GGUF file not found at {gguf_path}")
        return False

    # Create temporary Modelfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
        f.write(create_modelfile(model_name, gguf_path))
        modelfile_path = f.name

    try:
        # Create model in Ollama
        result = subprocess.run([
            'ollama', 'create', model_name, '-f', modelfile_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully created {model_name} in Ollama")
            return True
        else:
            print(f"Error creating {model_name}: {result.stderr}")
            return False

    finally:
        # Clean up temporary file
        os.unlink(modelfile_path)

def main():
    """Main setup function"""
    print("Setting up Ollama models from LMStudio GGUF files...")

    # Check if Ollama is installed
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Ollama is not installed or not in PATH")
        print("Please install Ollama from https://ollama.ai/")
        return

    success_count = 0
    for model_name, gguf_path in LMSTUDIO_MODELS.items():
        if setup_ollama_model(model_name, gguf_path):
            success_count += 1

    print(f"\nSetup complete! Successfully created {success_count}/{len(LMSTUDIO_MODELS)} models")

    if success_count > 0:
        print("\nTo test a model, run:")
        print(f"ollama run {list(LMSTUDIO_MODELS.keys())[0]}")

if __name__ == "__main__":
    main()
