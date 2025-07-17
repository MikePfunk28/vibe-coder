#!/usr/bin/env python3
"""Test to verify all models are working with correct mappings"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from call_llm import call_llm, get_available_lmstudio_models

def test_exact_models():
    """Test each model with its exact mapping"""

    print("=== Testing Exact Model Mappings ===\n")

    # First, check what's actually available
    available_models = get_available_lmstudio_models()
    print("Available models in LMStudio:")
    for model in available_models:
        print(f"  - {model}")
    print()

    # Test each model type with the correct mapping
    test_cases = [
        ("phi-4-reasoning-plus", "microsoft/phi-4-reasoning-plus"),
        ("phi-4-mini-reasoning", "microsoft/phi-4-mini-reasoning"),
        ("phi-4-mini-instruct", "phi-4-mini-instruct")  # No microsoft/ prefix
    ]

    for model_name, expected_id in test_cases:
        print(f"Testing {model_name} (should map to {expected_id}):")

        if expected_id in available_models:
            try:
                response = call_llm(
                    f"Say 'Hello from {model_name}!' and confirm you are working correctly.",
                    model_name=model_name,
                    provider="lmstudio",
                    use_cache=False
                )
                print(f"  ✅ Success: {response.strip()}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"  ⚠️  Model {expected_id} not found in available models")
        print()

if __name__ == "__main__":
    test_exact_models()
