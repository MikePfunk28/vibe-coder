#!/usr/bin/env python3
"""Test smart model selection based on prompt type"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from call_llm import call_llm

def test_smart_selection():
    """Test that the system selects appropriate models for different prompt types"""

    print("=== Testing Smart Model Selection ===\n")

    # Test cases with different prompt types
    test_cases = [
        ("reasoning", "Analyze this step by step: If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?"),
        ("instruct", "Write a simple Python function that adds two numbers together."),
        ("reasoning", "Think through this problem: How would you approach debugging a program that crashes randomly?"),
        ("instruct", "Create a TODO list for learning Python programming."),
        ("reasoning", "What are the logical steps to solve this: A farmer has chickens and rabbits. Together they have 20 heads and 56 legs. How many of each animal does the farmer have?")
    ]

    for expected_type, prompt in test_cases:
        print(f"Testing {expected_type} prompt:")
        print(f"Prompt: {prompt}")

        try:
            response = call_llm(prompt, use_cache=False)
            print(f"✅ Response: {response[:200]}...")
            print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    test_smart_selection()
