#!/usr/bin/env python3
"""
Simple test for Qwen Coder 3 integration
Basic functionality test without complex dependencies
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from qwen_coder_agent import (
    QwenCoderAgent, CodeRequest, CodeContext, CodeTaskType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_qwen_simple')

async def test_basic_functionality():
    """Test basic Qwen Coder functionality"""
    logger.info("Testing basic Qwen Coder functionality...")
    
    try:
        # Create agent instance
        agent = QwenCoderAgent()
        logger.info("✓ QwenCoderAgent created successfully")
        
        # Test code context creation
        context = CodeContext(
            language="python",
            file_path="test.py"
        )
        logger.info("✓ CodeContext created successfully")
        
        # Test code request creation
        request = CodeRequest(
            prompt="def hello_world():",
            task_type=CodeTaskType.COMPLETION,
            context=context
        )
        logger.info("✓ CodeRequest created successfully")
        
        # Test prompt template generation
        from qwen_coder_agent import CodePromptTemplates
        templates = CodePromptTemplates()
        prompt = templates.get_completion_prompt(request)
        logger.info(f"✓ Prompt template generated: {len(prompt)} characters")
        
        # Test fallback code generation
        fallback_code = agent._generate_fallback_code(request)
        logger.info(f"✓ Fallback code generated: {len(fallback_code)} characters")
        
        logger.info("🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    try:
        success = await test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())