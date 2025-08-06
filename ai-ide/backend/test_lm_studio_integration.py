#!/usr/bin/env python3
"""
Test script for LM Studio integration in AI IDE
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.lm_studio_manager import (
    get_lm_studio_manager, ModelRequest, ModelType, enhanced_call_llm
)

async def test_lm_studio_connection():
    """Test basic LM Studio connection"""
    print("Testing LM Studio connection...")
    
    try:
        manager = await get_lm_studio_manager()
        
        # Get model information
        models_info = manager.get_model_info()
        print(f"Connected to LM Studio with {len(models_info)} models:")
        
        for model_id, info in models_info.items():
            if 'error' not in info:
                print(f"  - {model_id}")
                print(f"    Type: {info.get('type', 'unknown')}")
                print(f"    Parameters: {info.get('parameters', 'unknown')}")
                print(f"    Context Length: {info.get('context_length', 'unknown')}")
        
        return len(models_info) > 0
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

async def test_model_selection():
    """Test intelligent model selection"""
    print("\nTesting model selection...")
    
    try:
        manager = await get_lm_studio_manager()
        
        test_cases = [
            (ModelType.REASONING, "Complex reasoning task"),
            (ModelType.CODE_GENERATION, "Code generation task"),
            (ModelType.INSTRUCT, "Simple instruction task"),
            (ModelType.CHAT, "Chat conversation")
        ]
        
        results = []
        
        for model_type, description in test_cases:
            selected_model = manager.select_best_model(model_type)
            print(f"  {description}: {selected_model or 'No suitable model'}")
            results.append(selected_model is not None)
        
        return all(results)
        
    except Exception as e:
        print(f"Model selection test failed: {e}")
        return False

async def test_code_generation():
    """Test code generation with LM Studio"""
    print("\nTesting code generation...")
    
    test_prompts = [
        {
            "prompt": "Create a function to calculate fibonacci numbers",
            "language": "python",
            "expected_keywords": ["def", "fibonacci", "return"]
        },
        {
            "prompt": "Create a simple HTTP server",
            "language": "javascript",
            "expected_keywords": ["function", "server", "http"]
        }
    ]
    
    results = []
    
    for test_case in test_prompts:
        print(f"  Generating {test_case['language']} code: '{test_case['prompt']}'")
        
        try:
            start_time = time.time()
            
            # Use enhanced call_llm function
            response = await enhanced_call_llm(
                prompt=test_case['prompt'],
                model_type="code",
                context={
                    'language': test_case['language'],
                    'task_type': 'code_generation'
                }
            )
            
            generation_time = time.time() - start_time
            
            print(f"    Generated in {generation_time:.2f}s")
            print(f"    Code preview: {response[:100]}...")
            
            # Check if expected keywords are present
            keywords_found = sum(1 for keyword in test_case['expected_keywords'] 
                                if keyword.lower() in response.lower())
            
            success = keywords_found >= len(test_case['expected_keywords']) // 2
            results.append(success)
            
            print(f"    Keywords found: {keywords_found}/{len(test_case['expected_keywords'])} - {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            print(f"    Generation failed: {e}")
            results.append(False)
    
    return all(results)

async def test_model_performance_tracking():
    """Test model performance tracking"""
    print("\nTesting performance tracking...")
    
    try:
        manager = await get_lm_studio_manager()
        
        # Make several requests to generate performance data
        test_requests = [
            "What is 2+2?",
            "Explain recursion",
            "Write a hello world function",
            "What is machine learning?",
            "Create a simple class"
        ]
        
        for i, prompt in enumerate(test_requests):
            print(f"  Request {i+1}: {prompt[:30]}...")
            
            request = ModelRequest(
                prompt=prompt,
                model_type=ModelType.INSTRUCT,
                max_tokens=100,
                temperature=0.7
            )
            
            response = await manager.generate_completion(request)
            print(f"    Success: {response.success}, Time: {response.response_time:.3f}s")
        
        # Get performance statistics
        stats = manager.get_performance_stats()
        
        print(f"\nPerformance Statistics:")
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"  Average response time: {stats.get('average_response_time', 0):.3f}s")
        print(f"  Active models: {stats.get('active_models', 0)}")
        print(f"  Best model: {stats.get('best_model', 'None')}")
        
        return stats.get('total_requests', 0) > 0
        
    except Exception as e:
        print(f"Performance tracking test failed: {e}")
        return False

async def test_connection_pooling():
    """Test connection pooling and concurrent requests"""
    print("\nTesting connection pooling...")
    
    try:
        manager = await get_lm_studio_manager()
        
        # Create multiple concurrent requests
        concurrent_requests = []
        
        for i in range(5):
            request = ModelRequest(
                prompt=f"Test request {i+1}: What is {i+1} + {i+1}?",
                model_type=ModelType.INSTRUCT,
                max_tokens=50,
                temperature=0.5
            )
            concurrent_requests.append(manager.generate_completion(request))
        
        # Execute all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*concurrent_requests, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_responses = sum(1 for r in responses if not isinstance(r, Exception) and r.success)
        
        print(f"  Concurrent requests: 5")
        print(f"  Successful responses: {successful_responses}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per request: {total_time/5:.3f}s")
        
        # Check connection pool status
        pool_stats = manager.get_performance_stats()
        print(f"  Connection pool active: {pool_stats.get('connection_pool_active', 0)}")
        
        return successful_responses >= 3  # At least 60% success rate
        
    except Exception as e:
        print(f"Connection pooling test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\nTesting error handling...")
    
    try:
        manager = await get_lm_studio_manager()
        
        # Test with invalid model type
        request = ModelRequest(
            prompt="Test prompt",
            model_type=ModelType.REASONING,  # May not be available
            max_tokens=10000,  # Very large token request
            temperature=2.0  # Invalid temperature
        )
        
        response = await manager.generate_completion(request)
        
        print(f"  Error handling test:")
        print(f"    Success: {response.success}")
        if not response.success:
            print(f"    Error: {response.error}")
        print(f"    Response time: {response.response_time:.3f}s")
        
        # Test fallback to enhanced_call_llm
        try:
            fallback_response = await enhanced_call_llm(
                "Simple test prompt",
                model_type="invalid_type"  # Should fallback
            )
            print(f"  Fallback test: {'PASS' if fallback_response else 'FAIL'}")
            return True
        except Exception as fallback_error:
            print(f"  Fallback test failed: {fallback_error}")
            return False
        
    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False

async def main():
    """Run all LM Studio integration tests"""
    print("AI IDE LM Studio Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("LM Studio Connection", test_lm_studio_connection),
        ("Model Selection", test_model_selection),
        ("Code Generation", test_code_generation),
        ("Performance Tracking", test_model_performance_tracking),
        ("Connection Pooling", test_connection_pooling),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            start_time = time.time()
            success = await test_func()
            test_time = time.time() - start_time
            
            results.append((test_name, success, test_time))
            print(f"\n{test_name}: {'PASS' if success else 'FAIL'} ({test_time:.2f}s)")
            
        except Exception as e:
            print(f"\n{test_name}: FAIL - Exception: {e}")
            results.append((test_name, False, 0))
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success, test_time in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name:<25} {status:>6} ({test_time:.2f}s)")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print(f"Total execution time: {total_time:.2f}s")
    
    # Cleanup
    try:
        manager = await get_lm_studio_manager()
        await manager.close()
        print("\nLM Studio manager closed successfully")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())