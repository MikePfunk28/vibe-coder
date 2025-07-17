#!/usr/bin/env python3
"""Test the improved model selection system"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.call_llm import call_llm

def test_model_selection():
    """Test that the system selects appropriate models for different tasks"""

    print("=" * 60)
    print("Testing Model Selection System")
    print("=" * 60)

    # Test 1: Simple instruct task
    print("\n1. Testing simple instruct task:")
    instruct_prompt = "Write a simple hello world program in Python"
    try:
        response = call_llm(instruct_prompt, use_cache=False)
        print(f"✅ Instruct task completed successfully")
        print(f"Response preview: {response[:100]}...")
    except Exception as e:
        print(f"❌ Instruct task failed: {e}")

    # Test 2: Complex reasoning task
    print("\n2. Testing complex reasoning task:")
    reasoning_prompt = "Analyze this complex problem step by step: How would you design a system to solve reasoning chains with multiple logical steps?"
    try:
        response = call_llm(reasoning_prompt, use_cache=False)
        print(f"✅ Reasoning task completed successfully")
        print(f"Response preview: {response[:100]}...")
    except Exception as e:
        print(f"❌ Reasoning task failed: {e}")

    # Test 3: Explicit model selection
    print("\n3. Testing explicit model selection:")
    explicit_prompt = "Tell me about artificial intelligence"
    try:
        response = call_llm(explicit_prompt, model_name="phi-4-mini-instruct", use_cache=False)
        print(f"✅ Explicit model selection completed successfully")
        print(f"Response preview: {response[:100]}...")
    except Exception as e:
        print(f"❌ Explicit model selection failed: {e}")

    print("\n" + "=" * 60)
    print("Model Selection Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_model_selection()
