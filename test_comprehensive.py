#!/usr/bin/env python3
"""Comprehensive test of the enhanced vibe-coder system with local models"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from call_llm import call_llm, get_available_lmstudio_models

def test_comprehensive():
    """Test all the new features"""

    print("=== Enhanced Vibe-Coder System Test ===\n")

    # Test 1: Model Detection
    print("1. Testing model detection:")
    available_models = get_available_lmstudio_models()
    print(f"   Available models: {len(available_models)} models found")
    for model in available_models[:5]:  # Show first 5
        print(f"   - {model}")
    if len(available_models) > 5:
        print(f"   ... and {len(available_models) - 5} more")
    print()

    # Test 2: Provider Priority System
    print("2. Testing provider priority system:")
    try:
        response = call_llm("What is 2+2? Answer with just the number.", provider="lmstudio")
        print(f"   LMStudio response: {response.strip()}")
    except Exception as e:
        print(f"   LMStudio error: {e}")
    print()

    # Test 3: Model Flexibility
    print("3. Testing different model requests:")
    test_models = ["phi-4-reasoning-plus", "phi-4-mini-instruct", "nonexistent-model"]
    for model in test_models:
        try:
            response = call_llm(f"Say 'Hello from {model}!'", model=model, provider="lmstudio")
            print(f"   {model}: {response.strip()}")
        except Exception as e:
            print(f"   {model}: Error - {e}")
    print()

    # Test 4: Auto-fallback
    print("4. Testing auto-fallback system:")
    try:
        # This will try providers in order: lmstudio, ollama, anthropic
        response = call_llm("What is the capital of France? One word only.")
        print(f"   Auto-fallback response: {response.strip()}")
    except Exception as e:
        print(f"   Auto-fallback error: {e}")
    print()

    print("=== Test Complete ===")

if __name__ == "__main__":
    test_comprehensive()
