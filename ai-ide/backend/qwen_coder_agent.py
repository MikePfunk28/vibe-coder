"""
Qwen Coder 3 Integration for AI IDE
Advanced code generation agent with streaming support and context awareness
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

# Import existing utilities
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from utils.call_llm import (
        call_llm, LMSTUDIO_URL, LMSTUDIO_MODELS,
        check_server_availability, get_available_lmstudio_models,
        find_best_lmstudio_model
    )
    from lm_studio_manager import (
        LMStudioManager, ModelRequest, ModelResponse, ModelType,
        get_lm_studio_manager
    )
except ImportError as e:
    logging.warning(f"Import error: {e}")
    # Fallback definitions
    LMSTUDIO_URL = "http://localhost:1234"
    LMSTUDIO_MODELS = {"default": "microsoft/phi-4-reasoning-plus"}

logger = logging.getLogger('qwen_coder_agent')

class CodeTaskType(Enum):
    """Types of code generation tasks"""
    COMPLETION = "completion"
    GENERATION = "generation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    OPTIMIZATION = "optimization"

@dataclass
class CodeContext:
    """Context information for code generation"""
    language: str
    file_path: Optional[str] = None
    selected_text: Optional[str] = None
    cursor_position: Optional[int] = None
    surrounding_code: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None
    imports: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    classes: Optional[List[str]] = None

@dataclass
class CodeRequest:
    """Request for code generation"""
    prompt: str
    task_type: CodeTaskType
    context: CodeContext
    max_tokens: int = 2048
    temperature: float = 0.3
    stream: bool = False
    include_explanation: bool = False
    format_output: bool = True

@dataclass
class CodeResponse:
    """Response from code generation"""
    code: str
    language: str
    confidence: float
    explanation: Optional[str] = None
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    model_info: Optional[Dict[str, Any]] = None

class CodePromptTemplates:
    """Templates for different code generation tasks"""
    
    @staticmethod
    def get_completion_prompt(request: CodeRequest) -> str:
        """Generate prompt for code completion"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Complete the following code based on the context.

Language: {context.language}
"""
        
        if context.file_path:
            prompt += f"File: {context.file_path}\n"
        
        if context.imports:
            prompt += f"Imports: {', '.join(context.imports)}\n"
        
        if context.surrounding_code:
            prompt += f"""
Context code:
```{context.language}
{context.surrounding_code}
```
"""
        
        prompt += f"""
Complete this code:
```{context.language}
{request.prompt}
```

Requirements:
1. Provide only the completion, not the entire code
2. Follow {context.language} best practices
3. Maintain consistency with existing code style
4. Include appropriate error handling if needed

Completion:"""
        
        return prompt
    
    @staticmethod
    def get_generation_prompt(request: CodeRequest) -> str:
        """Generate prompt for code generation"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Generate high-quality code based on the following request.

Language: {context.language}
Request: {request.prompt}
"""
        
        if context.file_path:
            prompt += f"Target file: {context.file_path}\n"
        
        if context.project_context:
            prompt += f"Project context: {json.dumps(context.project_context, indent=2)}\n"
        
        if context.selected_text:
            prompt += f"""
Selected code context:
```{context.language}
{context.selected_text}
```
"""
        
        prompt += f"""
Requirements:
1. Write clean, readable, and well-documented code
2. Follow {context.language} best practices and conventions
3. Include appropriate error handling
4. Add helpful comments explaining the logic
5. Ensure the code is production-ready
6. Use meaningful variable and function names

Generate the requested code:"""
        
        return prompt
    
    @staticmethod
    def get_refactoring_prompt(request: CodeRequest) -> str:
        """Generate prompt for code refactoring"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Refactor the following code to improve its quality.

Language: {context.language}
Refactoring request: {request.prompt}

Original code:
```{context.language}
{context.selected_text or "No code provided"}
```

Requirements:
1. Improve code readability and maintainability
2. Follow {context.language} best practices
3. Optimize performance where possible
4. Maintain the same functionality
5. Add or improve comments and documentation
6. Use better naming conventions if needed

Refactored code:"""
        
        return prompt
    
    @staticmethod
    def get_debugging_prompt(request: CodeRequest) -> str:
        """Generate prompt for debugging assistance"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} debugger. Help identify and fix issues in the following code.

Language: {context.language}
Issue description: {request.prompt}

Code to debug:
```{context.language}
{context.selected_text or "No code provided"}
```
"""
        
        if context.surrounding_code:
            prompt += f"""
Surrounding context:
```{context.language}
{context.surrounding_code}
```
"""
        
        prompt += f"""
Please:
1. Identify potential issues in the code
2. Explain what might be causing the problem
3. Provide a corrected version of the code
4. Suggest best practices to prevent similar issues

Analysis and fix:"""
        
        return prompt
    
    @staticmethod
    def get_documentation_prompt(request: CodeRequest) -> str:
        """Generate prompt for code documentation"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Generate comprehensive documentation for the following code.

Language: {context.language}
Documentation request: {request.prompt}

Code to document:
```{context.language}
{context.selected_text or "No code provided"}
```

Please provide:
1. Clear function/class docstrings following {context.language} conventions
2. Inline comments explaining complex logic
3. Parameter and return value descriptions
4. Usage examples if appropriate
5. Any important notes or warnings

Documented code:"""
        
        return prompt
    
    @staticmethod
    def get_explanation_prompt(request: CodeRequest) -> str:
        """Generate prompt for code explanation"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Explain the following code in detail.

Language: {context.language}
Explanation request: {request.prompt}

Code to explain:
```{context.language}
{context.selected_text or "No code provided"}
```

Please provide:
1. High-level overview of what the code does
2. Step-by-step breakdown of the logic
3. Explanation of key concepts and patterns used
4. Potential improvements or alternatives
5. Common use cases or applications

Explanation:"""
        
        return prompt
    
    @staticmethod
    def get_optimization_prompt(request: CodeRequest) -> str:
        """Generate prompt for code optimization"""
        context = request.context
        
        prompt = f"""You are an expert {context.language} programmer. Optimize the following code for better performance.

Language: {context.language}
Optimization request: {request.prompt}

Code to optimize:
```{context.language}
{context.selected_text or "No code provided"}
```

Please:
1. Identify performance bottlenecks
2. Suggest algorithmic improvements
3. Optimize data structures if needed
4. Provide the optimized version
5. Explain the improvements made
6. Estimate performance gains

Optimized code:"""
        
        return prompt

class QwenCoderAgent:
    """Advanced Qwen Coder 3 agent for code generation with streaming support"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or LMSTUDIO_URL
        self.lm_studio_manager: Optional[LMStudioManager] = None
        self.prompt_templates = CodePromptTemplates()
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.success_count = 0
        
    async def initialize(self):
        """Initialize the Qwen Coder agent"""
        try:
            logger.info("Initializing Qwen Coder Agent...")
            
            # Initialize LM Studio manager
            self.lm_studio_manager = await get_lm_studio_manager()
            
            # Initialize HTTP session for streaming
            connector = aiohttp.TCPConnector(
                limit=10,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=300)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            self.is_initialized = True
            logger.info("Qwen Coder Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen Coder Agent: {e}")
            raise
    
    async def close(self):
        """Close the agent and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.lm_studio_manager:
            await self.lm_studio_manager.close()
        
        self.is_initialized = False
    
    def _select_prompt_template(self, request: CodeRequest) -> str:
        """Select appropriate prompt template based on task type"""
        template_map = {
            CodeTaskType.COMPLETION: self.prompt_templates.get_completion_prompt,
            CodeTaskType.GENERATION: self.prompt_templates.get_generation_prompt,
            CodeTaskType.REFACTORING: self.prompt_templates.get_refactoring_prompt,
            CodeTaskType.DEBUGGING: self.prompt_templates.get_debugging_prompt,
            CodeTaskType.DOCUMENTATION: self.prompt_templates.get_documentation_prompt,
            CodeTaskType.EXPLANATION: self.prompt_templates.get_explanation_prompt,
            CodeTaskType.OPTIMIZATION: self.prompt_templates.get_optimization_prompt,
        }
        
        template_func = template_map.get(request.task_type, self.prompt_templates.get_generation_prompt)
        return template_func(request)
    
    def _extract_code_from_response(self, response: str, language: str) -> Tuple[str, Optional[str]]:
        """Extract code and explanation from LLM response"""
        # Look for code blocks first
        code_block_pattern = rf'```{language}?\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            code = matches[0].strip()
            # Extract explanation (text before or after code block)
            explanation_parts = re.split(code_block_pattern, response, flags=re.DOTALL | re.IGNORECASE)
            explanation = ' '.join([part.strip() for part in explanation_parts if part.strip() and part != code]).strip()
            return code, explanation if explanation else None
        
        # If no code blocks, try to separate code from explanation
        lines = response.split('\n')
        code_lines = []
        explanation_lines = []
        in_code_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip common explanation starters
            if any(starter in line_lower for starter in [
                'here is', 'here\'s', 'this code', 'explanation:', 'the code', 'solution:'
            ]):
                explanation_lines.append(line)
                continue
            
            # Detect code patterns
            if (line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'function ', 'const ', 'let ', 'var ')) or
                '=' in line or '(' in line or '{' in line or line.strip().endswith((';', ':', '{', '}'))):
                in_code_section = True
                code_lines.append(line)
            elif in_code_section and (line.strip() == '' or line.startswith('    ') or line.startswith('\t')):
                # Continue code section for indented lines or empty lines
                code_lines.append(line)
            else:
                if in_code_section and code_lines:
                    # End of code section
                    in_code_section = False
                explanation_lines.append(line)
        
        code = '\n'.join(code_lines).strip() if code_lines else response.strip()
        explanation = '\n'.join(explanation_lines).strip() if explanation_lines else None
        
        return code, explanation
    
    def _calculate_confidence(self, response_time: float, code_length: int, has_syntax_errors: bool) -> float:
        """Calculate confidence score for the generated code"""
        base_confidence = 0.9
        
        # Penalize slow responses
        if response_time > 10:
            base_confidence -= 0.1
        elif response_time > 5:
            base_confidence -= 0.05
        
        # Penalize very short or very long responses
        if code_length < 10:
            base_confidence -= 0.2
        elif code_length > 5000:
            base_confidence -= 0.1
        
        # Penalize syntax errors
        if has_syntax_errors:
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _detect_syntax_errors(self, code: str, language: str) -> bool:
        """Basic syntax error detection"""
        try:
            if language.lower() == 'python':
                import ast
                ast.parse(code)
                return False
            elif language.lower() in ['javascript', 'typescript']:
                # Basic bracket matching
                brackets = {'(': ')', '[': ']', '{': '}'}
                stack = []
                for char in code:
                    if char in brackets:
                        stack.append(brackets[char])
                    elif char in brackets.values():
                        if not stack or stack.pop() != char:
                            return True
                return len(stack) > 0
        except:
            return True
        
        return False
    
    async def generate_code(self, request: CodeRequest) -> CodeResponse:
        """Generate code using Qwen Coder 3"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Generate enhanced prompt
            enhanced_prompt = self._select_prompt_template(request)
            
            # Use LM Studio manager for generation
            if self.lm_studio_manager:
                model_request = ModelRequest(
                    prompt=enhanced_prompt,
                    model_type=ModelType.CODE_GENERATION,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=request.stream,
                    context={
                        'task_type': request.task_type.value,
                        'language': request.context.language,
                        'estimated_tokens': len(enhanced_prompt.split()) * 2
                    }
                )
                
                model_response = await self.lm_studio_manager.generate_completion(model_request)
                
                if model_response.success:
                    # Extract code and explanation
                    code, explanation = self._extract_code_from_response(
                        model_response.content, 
                        request.context.language
                    )
                    
                    # Calculate confidence
                    has_syntax_errors = self._detect_syntax_errors(code, request.context.language)
                    confidence = self._calculate_confidence(
                        model_response.response_time,
                        len(code),
                        has_syntax_errors
                    )
                    
                    execution_time = time.time() - start_time
                    self.total_response_time += execution_time
                    self.success_count += 1
                    
                    return CodeResponse(
                        code=code,
                        language=request.context.language,
                        confidence=confidence,
                        explanation=explanation if request.include_explanation else None,
                        metadata={
                            'task_type': request.task_type.value,
                            'has_syntax_errors': has_syntax_errors,
                            'prompt_length': len(enhanced_prompt),
                            'response_length': len(model_response.content)
                        },
                        execution_time=execution_time,
                        model_info={
                            'model_id': model_response.model_id,
                            'tokens_used': model_response.tokens_used,
                            'response_time': model_response.response_time
                        }
                    )
                else:
                    logger.warning(f"LM Studio generation failed: {model_response.error}")
            
            # Fallback to basic call_llm
            logger.info("Falling back to basic call_llm")
            response_text = call_llm(enhanced_prompt, model_name="code")
            
            code, explanation = self._extract_code_from_response(
                response_text, 
                request.context.language
            )
            
            execution_time = time.time() - start_time
            self.total_response_time += execution_time
            self.success_count += 1
            
            return CodeResponse(
                code=code,
                language=request.context.language,
                confidence=0.7,  # Lower confidence for fallback
                explanation=explanation if request.include_explanation else None,
                metadata={
                    'task_type': request.task_type.value,
                    'fallback_used': True
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_response_time += execution_time
            logger.error(f"Code generation failed: {e}")
            
            # Return fallback response
            fallback_code = self._generate_fallback_code(request)
            
            return CodeResponse(
                code=fallback_code,
                language=request.context.language,
                confidence=0.3,
                explanation="Generated using fallback method due to error",
                metadata={
                    'task_type': request.task_type.value,
                    'error': str(e),
                    'fallback_used': True
                },
                execution_time=execution_time
            )
    
    async def generate_code_stream(self, request: CodeRequest) -> AsyncGenerator[str, None]:
        """Generate code with streaming response"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Generate enhanced prompt
            enhanced_prompt = self._select_prompt_template(request)
            
            # Prepare streaming request
            payload = {
                "model": find_best_lmstudio_model("code"),
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                else:
                    # Fallback to non-streaming
                    response = await self.generate_code(
                        CodeRequest(
                            prompt=request.prompt,
                            task_type=request.task_type,
                            context=request.context,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            stream=False
                        )
                    )
                    yield response.code
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Fallback to non-streaming
            response = await self.generate_code(
                CodeRequest(
                    prompt=request.prompt,
                    task_type=request.task_type,
                    context=request.context,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=False
                )
            )
            yield response.code
    
    def _generate_fallback_code(self, request: CodeRequest) -> str:
        """Generate fallback code when all other methods fail"""
        language = request.context.language.lower()
        task_type = request.task_type
        prompt = request.prompt
        
        if language == 'python':
            if task_type == CodeTaskType.COMPLETION:
                return f"# Code completion for: {prompt}\npass  # TODO: Complete implementation"
            elif task_type == CodeTaskType.GENERATION:
                return f'''# Generated code for: {prompt}
def generated_function():
    """
    AI-generated function
    Request: {prompt}
    """
    # TODO: Implement the requested functionality
    pass

# Example usage:
# result = generated_function()
'''
            elif task_type == CodeTaskType.DEBUGGING:
                return f'''# Debugging assistance for: {prompt}
# TODO: Analyze the code and identify issues
# Common debugging steps:
# 1. Check for syntax errors
# 2. Verify variable names and types
# 3. Add print statements for debugging
# 4. Test with different inputs
'''
        
        elif language in ['javascript', 'typescript']:
            if task_type == CodeTaskType.COMPLETION:
                return f"// Code completion for: {prompt}\n// TODO: Complete implementation"
            elif task_type == CodeTaskType.GENERATION:
                return f'''// Generated code for: {prompt}
function generatedFunction() {{
    /**
     * AI-generated function
     * Request: {prompt}
     */
    // TODO: Implement the requested functionality
}}

// Example usage:
// const result = generatedFunction();
'''
        
        return f'''// Generated code for: {prompt}
// TODO: Implement the requested functionality
// Language: {language}
// Task: {task_type.value}
'''
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.request_count == 0:
            return {'no_data': True}
        
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'success_rate': self.success_count / self.request_count,
            'average_response_time': self.total_response_time / self.request_count,
            'total_response_time': self.total_response_time
        }

# Global instance
_qwen_coder_agent = None

async def get_qwen_coder_agent() -> QwenCoderAgent:
    """Get or create Qwen Coder agent instance"""
    global _qwen_coder_agent
    
    if _qwen_coder_agent is None:
        _qwen_coder_agent = QwenCoderAgent()
        await _qwen_coder_agent.initialize()
    
    return _qwen_coder_agent

# Convenience functions for different code tasks
async def complete_code(
    code: str,
    language: str,
    context: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> CodeResponse:
    """Complete code using Qwen Coder 3"""
    agent = await get_qwen_coder_agent()
    
    code_context = CodeContext(
        language=language,
        file_path=context.get('file_path') if context else None,
        selected_text=context.get('selected_text') if context else None,
        surrounding_code=context.get('surrounding_code') if context else None
    )
    
    request = CodeRequest(
        prompt=code,
        task_type=CodeTaskType.COMPLETION,
        context=code_context,
        stream=stream,
        temperature=0.2  # Lower temperature for completion
    )
    
    return await agent.generate_code(request)

async def generate_code(
    prompt: str,
    language: str,
    context: Optional[Dict[str, Any]] = None,
    include_explanation: bool = False,
    stream: bool = False
) -> CodeResponse:
    """Generate code using Qwen Coder 3"""
    agent = await get_qwen_coder_agent()
    
    code_context = CodeContext(
        language=language,
        file_path=context.get('file_path') if context else None,
        selected_text=context.get('selected_text') if context else None,
        project_context=context.get('project_context') if context else None
    )
    
    request = CodeRequest(
        prompt=prompt,
        task_type=CodeTaskType.GENERATION,
        context=code_context,
        include_explanation=include_explanation,
        stream=stream
    )
    
    return await agent.generate_code(request)

async def refactor_code(
    code: str,
    language: str,
    refactoring_request: str,
    context: Optional[Dict[str, Any]] = None
) -> CodeResponse:
    """Refactor code using Qwen Coder 3"""
    agent = await get_qwen_coder_agent()
    
    code_context = CodeContext(
        language=language,
        selected_text=code,
        file_path=context.get('file_path') if context else None
    )
    
    request = CodeRequest(
        prompt=refactoring_request,
        task_type=CodeTaskType.REFACTORING,
        context=code_context,
        include_explanation=True
    )
    
    return await agent.generate_code(request)

async def debug_code(
    code: str,
    language: str,
    issue_description: str,
    context: Optional[Dict[str, Any]] = None
) -> CodeResponse:
    """Debug code using Qwen Coder 3"""
    agent = await get_qwen_coder_agent()
    
    code_context = CodeContext(
        language=language,
        selected_text=code,
        surrounding_code=context.get('surrounding_code') if context else None,
        file_path=context.get('file_path') if context else None
    )
    
    request = CodeRequest(
        prompt=issue_description,
        task_type=CodeTaskType.DEBUGGING,
        context=code_context,
        include_explanation=True
    )
    
    return await agent.generate_code(request)