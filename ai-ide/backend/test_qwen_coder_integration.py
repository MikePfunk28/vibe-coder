#!/usr/bin/env python3
"""
Test script for Qwen Coder 3 integration
Tests the QwenCoderAgent functionality and API endpoints
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from qwen_coder_agent import (
    QwenCoderAgent, CodeRequest, CodeContext, CodeTaskType,
    get_qwen_coder_agent, complete_code, generate_code, 
    refactor_code, debug_code
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_qwen_coder')

async def test_code_completion():
    """Test code completion functionality"""
    logger.info("Testing code completion...")
    
    try:
        context = {
            'file_path': 'test.py',
            'surrounding_code': 'def calculate_sum(a, b):\n    # Calculate the sum of two numbers'
        }
        
        response = await complete_code(
            code="def calculate_sum(a, b):\n    return",
            language="python",
            context=context
        )
        
        logger.info(f"Code completion result:")
        logger.info(f"  Code: {response.code}")
        logger.info(f"  Language: {response.language}")
        logger.info(f"  Confidence: {response.confidence}")
        logger.info(f"  Execution time: {response.execution_time:.3f}s")
        
        if response.model_info:
            logger.info(f"  Model: {response.model_info.get('model_id', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Code completion test failed: {e}")
        return False

async def test_code_generation():
    """Test code generation functionality"""
    logger.info("Testing code generation...")
    
    try:
        context = {
            'file_path': 'calculator.py',
            'project_context': {
                'type': 'utility_library',
                'description': 'Mathematical utility functions'
            }
        }
        
        response = await generate_code(
            prompt="Create a function that calculates the factorial of a number with error handling",
            language="python",
            context=context,
            include_explanation=True
        )
        
        logger.info(f"Code generation result:")
        logger.info(f"  Code: {response.code}")
        logger.info(f"  Language: {response.language}")
        logger.info(f"  Confidence: {response.confidence}")
        logger.info(f"  Execution time: {response.execution_time:.3f}s")
        
        if response.explanation:
            logger.info(f"  Explanation: {response.explanation[:100]}...")
        
        if response.model_info:
            logger.info(f"  Model: {response.model_info.get('model_id', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Code generation test failed: {e}")
        return False

async def test_code_refactoring():
    """Test code refactoring functionality"""
    logger.info("Testing code refactoring...")
    
    try:
        code_to_refactor = """
def calc(x, y, op):
    if op == '+':
        return x + y
    elif op == '-':
        return x - y
    elif op == '*':
        return x * y
    elif op == '/':
        return x / y
    else:
        return None
"""
        
        context = {
            'file_path': 'calculator.py'
        }
        
        response = await refactor_code(
            code=code_to_refactor,
            language="python",
            refactoring_request="Improve this function with better error handling and documentation",
            context=context
        )
        
        logger.info(f"Code refactoring result:")
        logger.info(f"  Code: {response.code}")
        logger.info(f"  Language: {response.language}")
        logger.info(f"  Confidence: {response.confidence}")
        logger.info(f"  Execution time: {response.execution_time:.3f}s")
        
        if response.explanation:
            logger.info(f"  Explanation: {response.explanation[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Code refactoring test failed: {e}")
        return False

async def test_code_debugging():
    """Test code debugging functionality"""
    logger.info("Testing code debugging...")
    
    try:
        buggy_code = """
def divide_numbers(a, b):
    result = a / b
    return result

# This function has a potential division by zero error
"""
        
        context = {
            'file_path': 'math_utils.py',
            'surrounding_code': 'import math\n\n# Mathematical utility functions'
        }
        
        response = await debug_code(
            code=buggy_code,
            language="python",
            issue_description="This function crashes when b is zero",
            context=context
        )
        
        logger.info(f"Code debugging result:")
        logger.info(f"  Code: {response.code}")
        logger.info(f"  Language: {response.language}")
        logger.info(f"  Confidence: {response.confidence}")
        logger.info(f"  Execution time: {response.execution_time:.3f}s")
        
        if response.explanation:
            logger.info(f"  Explanation: {response.explanation[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Code debugging test failed: {e}")
        return False

async def test_streaming_generation():
    """Test streaming code generation"""
    logger.info("Testing streaming code generation...")
    
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = CodeContext(
            language="python",
            file_path="stream_test.py"
        )
        
        request = CodeRequest(
            prompt="Create a simple HTTP server using Flask",
            task_type=CodeTaskType.GENERATION,
            context=code_context,
            stream=True
        )
        
        logger.info("Streaming response:")
        full_response = ""
        
        async for chunk in agent.generate_code_stream(request):
            print(chunk, end='', flush=True)
            full_response += chunk
        
        print()  # New line after streaming
        logger.info(f"Full streaming response length: {len(full_response)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"Streaming generation test failed: {e}")
        return False

async def test_agent_performance():
    """Test agent performance and statistics"""
    logger.info("Testing agent performance...")
    
    try:
        agent = await get_qwen_coder_agent()
        
        # Get initial stats
        initial_stats = agent.get_performance_stats()
        logger.info(f"Initial performance stats: {initial_stats}")
        
        # Perform a few operations
        for i in range(3):
            await generate_code(
                prompt=f"Create a simple function number {i+1}",
                language="python"
            )
        
        # Get updated stats
        final_stats = agent.get_performance_stats()
        logger.info(f"Final performance stats: {final_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False

async def test_different_languages():
    """Test code generation for different programming languages"""
    logger.info("Testing different programming languages...")
    
    languages = [
        ("python", "Create a class for a simple calculator"),
        ("javascript", "Create a function to validate email addresses"),
        ("typescript", "Create an interface for a user profile"),
        ("java", "Create a method to sort an array of integers"),
        ("cpp", "Create a function to find the maximum element in an array")
    ]
    
    results = []
    
    for language, prompt in languages:
        try:
            logger.info(f"Testing {language}...")
            
            response = await generate_code(
                prompt=prompt,
                language=language,
                context={'file_path': f'test.{language}'}
            )
            
            results.append({
                'language': language,
                'success': True,
                'confidence': response.confidence,
                'code_length': len(response.code),
                'execution_time': response.execution_time
            })
            
            logger.info(f"  {language}: Success (confidence: {response.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"  {language}: Failed - {e}")
            results.append({
                'language': language,
                'success': False,
                'error': str(e)
            })
    
    logger.info("Language test results:")
    for result in results:
        if result['success']:
            logger.info(f"  {result['language']}: ✓ (conf: {result['confidence']:.2f}, "
                       f"len: {result['code_length']}, time: {result['execution_time']:.3f}s)")
        else:
            logger.info(f"  {result['language']}: ✗ ({result['error']})")
    
    return len([r for r in results if r['success']]) > 0

async def run_all_tests():
    """Run all tests"""
    logger.info("Starting Qwen Coder 3 integration tests...")
    
    tests = [
        ("Code Completion", test_code_completion),
        ("Code Generation", test_code_generation),
        ("Code Refactoring", test_code_refactoring),
        ("Code Debugging", test_code_debugging),
        ("Streaming Generation", test_streaming_generation),
        ("Agent Performance", test_agent_performance),
        ("Different Languages", test_different_languages)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✓ {test_name} passed")
            else:
                logger.error(f"✗ {test_name} failed")
                
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
        return False

async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        
        # Cleanup
        try:
            agent = await get_qwen_coder_agent()
            await agent.close()
        except:
            pass
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())