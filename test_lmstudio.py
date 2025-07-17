#!/usr/bin/env python3
"""Test script for LMStudio integration"""

import os
import sys
from utils.call_llm import call_llm

def test_lmstudio():
    """Test LMStudio integration"""
    print("Testing LMStudio integration...")

    # Set environment variables to use LMStudio
    os.environ["USE_LMSTUDIO"] = "true"
    os.environ["LMSTUDIO_URL"] = "http://localhost:1234"

    # Test prompt
    test_prompt = "Hello! Please respond with a brief greeting and confirm you're working."

    try:
        print(f"Sending prompt: {test_prompt}")
        response = call_llm(test_prompt, use_cache=False)
        print(f"Response: {response}")
        print("✅ LMStudio integration test passed!")
        return True
    except Exception as e:
        print(f"❌ LMStudio integration test failed: {e}")
        return False

def test_fallback():
    """Test fallback to Anthropic API"""
    print("\nTesting fallback to Anthropic API...")

    # Disable LMStudio to test fallback
    os.environ["USE_LMSTUDIO"] = "false"

    test_prompt = "Hello! Please respond with a brief greeting and confirm you're working."

    try:
        print(f"Sending prompt: {test_prompt}")
        response = call_llm(test_prompt, use_cache=False)
        print(f"Response: {response}")
        print("✅ Anthropic API fallback test passed!")
        return True
    except Exception as e:
        print(f"❌ Anthropic API fallback test failed: {e}")
        return False

def test_different_models():
    """Test different local models"""
    print("\nTesting different local models...")

    # Re-enable LMStudio
    os.environ["USE_LMSTUDIO"] = "true"

    models = ["phi-4-reasoning-plus", "phi-4-mini-reasoning", "phi-4-mini-instruct"]

    for model in models:
        try:
            print(f"\nTesting model: {model}")
            response = call_llm(
                "What is your name and model?",
                use_cache=False,
                model_name=model
            )
            print(f"Response from {model}: {response[:100]}...")
            print(f"✅ Model {model} test passed!")
        except Exception as e:
            print(f"❌ Model {model} test failed: {e}")

if __name__ == "__main__":
    print("LMStudio Integration Test Suite")
    print("=" * 50)

    # Test LMStudio integration
    lmstudio_success = test_lmstudio()

    # Test fallback (only if you have Anthropic API configured)
    # fallback_success = test_fallback()

    # Test different models
    test_different_models()

    print("\nTest suite completed!")
