#!/usr/bin/env python3
"""Test script to verify model detection in LMStudio"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from call_llm import (
    get_available_lmstudio_models,
    find_best_lmstudio_model,
    check_server_availability,
    LMSTUDIO_URL
)

def test_model_detection():
    """Test the model detection functionality"""
    print("Testing LMStudio model detection...")

    # Check if server is available
    server_available = check_server_availability(f"{LMSTUDIO_URL}/v1/models")
    print(f"LMStudio server available: {server_available}")

    if not server_available:
        print("LMStudio server not available - start it first")
        return

    # Get available models
    available_models = get_available_lmstudio_models()
    print(f"Available models: {available_models}")

    # Test model finding
    test_requests = [
        "phi-4-reasoning-plus",
        "phi-4-mini-reasoning",
        "phi-4-mini-instruct",
        "default",
        "nonexistent-model"
    ]

    for request in test_requests:
        best_model = find_best_lmstudio_model(request)
        print(f"Requested: {request} -> Best match: {best_model}")

if __name__ == "__main__":
    test_model_detection()
