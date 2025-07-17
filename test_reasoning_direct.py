#!/usr/bin/env python3
"""Direct test of the reasoning model"""

import requests
import json

def test_reasoning_model_direct():
    """Test the reasoning model directly"""

    print("Testing reasoning models directly with LMStudio...")

    # Test both reasoning models
    models_to_test = [
        "microsoft/phi-4-mini-reasoning",
        "microsoft/phi-4-reasoning-plus"
    ]

    for model_id in models_to_test:
        print(f"\n--- Testing {model_id} ---")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Analyze this step by step: What are the key factors in problem solving?"}
            ],
            "temperature": 0.7,
            "max_tokens": 500  # Reduced for faster response
        }

        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Increased timeout
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Success! Response: {content[:200]}...")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_reasoning_model_direct()
