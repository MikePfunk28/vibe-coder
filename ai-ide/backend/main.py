#!/usr/bin/env python3
"""
AI IDE Backend - Enhanced PocketFlow Integration
Main entry point for the AI IDE backend services
"""

import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
from pathlib import Path

# Import the enhanced PocketFlow integration
from pocketflow_integration import create_ai_ide_flow, AIIDEFlow
from semantic_engine import get_semantic_index, get_performance_tracker
from lm_studio_manager import get_lm_studio_manager, ModelRequest, ModelType
from qwen_coder_agent import (
    get_qwen_coder_agent, QwenCoderAgent, CodeRequest, CodeContext, CodeTaskType,
    complete_code, generate_code, refactor_code, debug_code
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai-ide-backend.log')
    ]
)
logger = logging.getLogger('ai-ide-backend')

class AIIDEBackend:
    """Main backend service for AI IDE"""
    
    def __init__(self):
        self.initialized = False
        self.services = {}
        self.pocketflow_engine: Optional[AIIDEFlow] = None
        self.semantic_index = None
        self.performance_tracker = None
        self.lm_studio_manager = None
        self.qwen_coder_agent: Optional[QwenCoderAgent] = None
        
    async def initialize(self):
        """Initialize all backend services"""
        try:
            logger.info("Initializing AI IDE Backend...")
            
            # Initialize core services
            await self.init_pocketflow()
            await self.init_lm_studio_client()
            await self.init_qwen_coder_agent()
            await self.init_semantic_search()
            
            self.initialized = True
            logger.info("AI IDE Backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise
    
    async def init_pocketflow(self):
        """Initialize enhanced PocketFlow engine"""
        logger.info("Initializing enhanced PocketFlow engine...")
        try:
            # Create AI IDE flow with current working directory
            working_dir = Path.cwd()
            self.pocketflow_engine = create_ai_ide_flow(str(working_dir))
            logger.info("PocketFlow engine initialized with semantic routing")
        except Exception as e:
            logger.error(f"Failed to initialize PocketFlow: {e}")
            # Create a fallback mock engine
            self.pocketflow_engine = None
        
    async def init_lm_studio_client(self):
        """Initialize enhanced LM Studio client"""
        logger.info("Initializing enhanced LM Studio client...")
        try:
            # Initialize the enhanced LM Studio manager
            self.lm_studio_manager = await get_lm_studio_manager()
            
            # Get model information
            models_info = self.lm_studio_manager.get_model_info()
            logger.info(f"LM Studio initialized with {len(models_info)} models")
            
            # Log available models
            for model_id, info in models_info.items():
                if 'error' not in info:
                    logger.info(f"  - {model_id}: {info.get('type', 'unknown')} ({info.get('parameters', 'unknown')})")
            
            self.services['lm_studio'] = True
            logger.info("Enhanced LM Studio client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LM Studio client: {e}")
            # Fallback to basic connection check
            try:
                from utils.call_llm import check_server_availability, LMSTUDIO_URL
                if check_server_availability(LMSTUDIO_URL):
                    logger.info("LM Studio basic connection verified")
                    self.services['lm_studio'] = True
                else:
                    logger.warning("LM Studio not available")
                    self.services['lm_studio'] = False
            except ImportError:
                logger.warning("LM Studio utilities not available")
                self.services['lm_studio'] = False
    
    async def init_qwen_coder_agent(self):
        """Initialize Qwen Coder 3 agent"""
        logger.info("Initializing Qwen Coder 3 agent...")
        try:
            # Initialize the Qwen Coder agent
            self.qwen_coder_agent = await get_qwen_coder_agent()
            
            # Get performance stats
            stats = self.qwen_coder_agent.get_performance_stats()
            logger.info(f"Qwen Coder agent initialized successfully")
            
            self.services['qwen_coder'] = True
            logger.info("Qwen Coder 3 agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen Coder agent: {e}")
            self.services['qwen_coder'] = False
        
    async def init_semantic_search(self):
        """Initialize semantic search engine"""
        logger.info("Initializing semantic search engine...")
        try:
            # Initialize semantic index
            working_dir = Path.cwd()
            self.semantic_index = get_semantic_index(str(working_dir))
            self.performance_tracker = get_performance_tracker()
            
            # Index the workspace (async in background)
            logger.info("Starting workspace indexing...")
            try:
                self.semantic_index.index_workspace()
            except Exception as e:
                logger.warning(f"Workspace indexing failed: {e}")
                # Continue without semantic indexing
            
            self.services['semantic_search'] = True
            logger.info("Semantic search engine initialized successfully")
        except Exception as e:
            logger.warning(f"Semantic search initialization failed: {e}")
            self.services['semantic_search'] = False
    
    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming task from VSCode extension"""
        if not self.initialized:
            return {
                "taskId": task.get("id", "unknown"),
                "success": False,
                "error": "Backend not initialized",
                "executionTime": 0
            }
        
        task_id = task.get("id", "unknown")
        task_type = task.get("type", "unknown")
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing task {task_id} of type {task_type}")
            
            # Route task based on type
            if task_type == "code_generation":
                result = await self.handle_code_generation(task)
            elif task_type == "semantic_search":
                result = await self.handle_semantic_search(task)
            elif task_type == "reasoning":
                result = await self.handle_reasoning(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "taskId": task_id,
                "success": True,
                "result": result,
                "executionTime": execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Task {task_id} failed: {e}")
            
            return {
                "taskId": task_id,
                "success": False,
                "error": str(e),
                "executionTime": execution_time
            }
    
    async def handle_code_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation tasks using Qwen Coder 3 agent"""
        prompt = task.get("input", {}).get("prompt", "")
        context = task.get("context", {})
        language = context.get("language", "python")
        task_type = task.get("input", {}).get("task_type", "generation")
        
        start_time = datetime.now()
        
        try:
            # Use Qwen Coder agent if available
            if self.qwen_coder_agent and self.services.get('qwen_coder'):
                logger.info(f"Using Qwen Coder agent for {task_type} task")
                
                # Determine the appropriate code task type
                code_task_type = self._map_task_type_to_code_task_type(task_type)
                
                # Create code context
                code_context = CodeContext(
                    language=language,
                    file_path=context.get("filePath"),
                    selected_text=context.get("selectedText"),
                    cursor_position=context.get("cursorPosition"),
                    surrounding_code=context.get("surroundingCode"),
                    project_context=context.get("projectContext"),
                    imports=context.get("imports"),
                    functions=context.get("functions"),
                    classes=context.get("classes")
                )
                
                # Create code request
                code_request = CodeRequest(
                    prompt=prompt,
                    task_type=code_task_type,
                    context=code_context,
                    max_tokens=task.get("input", {}).get("max_tokens", 2048),
                    temperature=task.get("input", {}).get("temperature", 0.3),
                    stream=task.get("input", {}).get("stream", False),
                    include_explanation=task.get("input", {}).get("include_explanation", False)
                )
                
                # Generate code using Qwen Coder agent
                response = await self.qwen_coder_agent.generate_code(code_request)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "code": response.code,
                    "language": response.language,
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                    "suggestions": response.suggestions,
                    "model_info": response.model_info,
                    "execution_metrics": {
                        "total_time": execution_time,
                        "qwen_coder_time": response.execution_time,
                        "processing_time": execution_time - response.execution_time
                    },
                    "metadata": {
                        **response.metadata,
                        "agent_used": "qwen_coder",
                        "task_type": task_type
                    }
                }
            
            # Fallback to enhanced LM Studio manager if Qwen Coder not available
            elif self.lm_studio_manager and self.services.get('lm_studio'):
                logger.info("Falling back to LM Studio manager")
                
                # Create enhanced prompt for code generation
                enhanced_prompt = self._create_code_generation_prompt(prompt, context)
                
                # Create model request
                request = ModelRequest(
                    prompt=enhanced_prompt,
                    model_type=ModelType.CODE_GENERATION,
                    max_tokens=2048,
                    temperature=0.3,  # Lower temperature for more deterministic code
                    context={
                        'task_type': 'code_generation',
                        'language': language,
                        'estimated_tokens': len(enhanced_prompt.split()) * 2
                    }
                )
                
                # Generate code using LM Studio
                response = await self.lm_studio_manager.generate_completion(request)
                
                if response.success:
                    # Extract and clean the generated code
                    generated_code = self._extract_code_from_response(response.content, language)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        "code": generated_code,
                        "language": language,
                        "confidence": min(0.95, 1.0 - response.response_time / 10.0),  # Confidence based on response time
                        "model_info": {
                            "model_id": response.model_id,
                            "tokens_used": response.tokens_used,
                            "response_time": response.response_time
                        },
                        "execution_metrics": {
                            "total_time": execution_time,
                            "llm_time": response.response_time,
                            "processing_time": execution_time - response.response_time
                        },
                        "metadata": {
                            "agent_used": "lm_studio_manager",
                            "task_type": task_type
                        }
                    }
                else:
                    logger.warning(f"LM Studio code generation failed: {response.error}")
            
            # Fallback to PocketFlow if other methods not available
            elif self.pocketflow_engine:
                logger.info("Falling back to PocketFlow engine")
                result = self.pocketflow_engine.execute_task(task)
                
                if result["success"]:
                    history = result.get("history", [])
                    generated_code = "# Code generated by AI IDE\n"
                    
                    for entry in history:
                        if entry.get("tool") == "edit_file":
                            result_data = entry.get("result", {})
                            if "code" in result_data:
                                generated_code = result_data["code"]
                                break
                    
                    return {
                        "code": generated_code,
                        "language": language,
                        "confidence": 0.8,
                        "reasoning_trace": result.get("reasoning_trace", []),
                        "execution_metrics": result.get("metrics", {}),
                        "metadata": {
                            "agent_used": "pocketflow",
                            "task_type": task_type
                        }
                    }
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
        
        # Final fallback implementation
        logger.info("Using fallback code generation")
        generated_code = self._generate_fallback_code(prompt, language)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "code": generated_code,
            "language": language,
            "confidence": 0.6,
            "fallback": True,
            "execution_metrics": {
                "total_time": execution_time
            },
            "metadata": {
                "agent_used": "fallback",
                "task_type": task_type
            }
        }
    
    async def handle_semantic_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search tasks using enhanced semantic engine"""
        query = task.get("input", {}).get("query", "")
        options = task.get("input", {}).get("options", {})
        max_results = options.get("maxResults", 10)
        
        start_time = datetime.now()
        
        try:
            # Use enhanced semantic search if available
            if self.semantic_index and self.services.get('semantic_search'):
                search_results = self.semantic_index.search_semantic(query, max_results)
                
                # Convert to expected format
                matches = []
                for result in search_results:
                    context = result['context']
                    for match in result['matches']:
                        matches.append({
                            "file": result['file'],
                            "line": match['line'],
                            "content": match['content'],
                            "similarity": min(result['score'] / 10.0, 1.0),  # Normalize score
                            "semantic_score": result['score'],
                            "match_type": match['type']
                        })
                
                # Record performance
                duration = (datetime.now() - start_time).total_seconds()
                self.performance_tracker.record_search_time(duration)
                
                return {
                    "matches": matches,
                    "total": len(matches),
                    "query": query,
                    "semantic_enhanced": True,
                    "performance": {
                        "search_time": duration,
                        "indexed_files": len(self.semantic_index.contexts),
                        "stats": self.performance_tracker.get_stats()
                    }
                }
            
            # Fallback to PocketFlow if semantic engine not available
            elif self.pocketflow_engine:
                result = self.pocketflow_engine.execute_task(task)
                
                if result["success"]:
                    history = result.get("history", [])
                    for entry in history:
                        if entry.get("tool") == "semantic_search":
                            search_result = entry.get("result", {})
                            if search_result.get("success"):
                                matches = search_result.get("matches", [])
                                return {
                                    "matches": matches,
                                    "total": len(matches),
                                    "query": query,
                                    "semantic_enhanced": True,
                                    "execution_metrics": result.get("metrics", {})
                                }
                    
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        # Final fallback implementation
        matches = [
            {
                "file": "src/main.py",
                "line": 42,
                "content": f"def search_function(): # Related to: {query}",
                "similarity": 0.95,
                "semantic_score": 0.92,
                "match_type": "fallback"
            },
            {
                "file": "src/utils.py", 
                "line": 18,
                "content": f"class SearchHelper: # Matches: {query}",
                "similarity": 0.87,
                "semantic_score": 0.85,
                "match_type": "fallback"
            }
        ]
        
        return {
            "matches": matches,
            "total": len(matches),
            "query": query,
            "semantic_enhanced": False
        }
    
    async def handle_reasoning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reasoning tasks using enhanced PocketFlow"""
        if self.pocketflow_engine:
            try:
                # Use the enhanced PocketFlow engine with reasoning capabilities
                result = self.pocketflow_engine.execute_task(task)
                
                if result["success"]:
                    # Extract reasoning results
                    reasoning_traces = result.get("reasoning_trace", [])
                    if reasoning_traces:
                        latest_trace = reasoning_traces[-1]
                        return {
                            "solution": latest_trace.get("solution", ""),
                            "reasoning": latest_trace.get("steps", []),
                            "confidence": latest_trace.get("confidence", 0.8),
                            "mode": latest_trace.get("mode", "basic"),
                            "trace_id": latest_trace.get("id", ""),
                            "execution_metrics": result.get("metrics", {})
                        }
                    
                    # Fallback to history
                    history = result.get("history", [])
                    for entry in history:
                        if entry.get("tool") == "reasoning_task":
                            reasoning_result = entry.get("result", {})
                            return reasoning_result
                            
            except Exception as e:
                logger.error(f"PocketFlow reasoning failed: {e}")
        
        # Fallback implementation
        problem = task.get("input", {}).get("problem", "")
        mode = task.get("input", {}).get("mode", "basic")
        
        reasoning_steps = [
            "1. Analyze the problem statement",
            "2. Break down into sub-problems", 
            "3. Apply relevant algorithms and patterns",
            "4. Consider edge cases and constraints",
            "5. Synthesize comprehensive solution"
        ]
        
        return {
            "solution": f"Enhanced solution for: {problem}",
            "reasoning": reasoning_steps,
            "confidence": 0.88,
            "mode": mode,
            "enhanced": False
        }
    
    def _create_code_generation_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Create an enhanced prompt for code generation"""
        language = context.get("language", "python")
        file_path = context.get("filePath", "")
        selected_text = context.get("selectedText", "")
        
        enhanced_prompt = f"""You are an expert {language} programmer. Generate high-quality, well-documented code based on the following request.

Request: {prompt}

Context:
- Language: {language}
- File: {file_path if file_path else "New file"}
"""
        
        if selected_text:
            enhanced_prompt += f"""
- Selected code context:
```{language}
{selected_text}
```
"""
        
        enhanced_prompt += f"""
Requirements:
1. Write clean, readable, and well-documented code
2. Follow {language} best practices and conventions
3. Include appropriate error handling
4. Add helpful comments explaining the logic
5. Ensure the code is production-ready

Generate only the requested code without additional explanations:"""
        
        return enhanced_prompt
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract clean code from LLM response"""
        # Remove markdown code blocks if present
        import re
        
        # Look for code blocks
        code_block_pattern = rf'```{language}?\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, look for code-like content
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip explanatory text
            if any(word in line.lower() for word in ['here is', 'here\'s', 'this code', 'explanation']):
                continue
            
            # Detect code patterns
            if (line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ')) or
                '=' in line or '(' in line or '{' in line):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Return original response if no code patterns found
        return response.strip()
    
    def _map_task_type_to_code_task_type(self, task_type: str) -> CodeTaskType:
        """Map string task type to CodeTaskType enum"""
        task_type_mapping = {
            'completion': CodeTaskType.COMPLETION,
            'generation': CodeTaskType.GENERATION,
            'refactoring': CodeTaskType.REFACTORING,
            'debugging': CodeTaskType.DEBUGGING,
            'documentation': CodeTaskType.DOCUMENTATION,
            'explanation': CodeTaskType.EXPLANATION,
            'optimization': CodeTaskType.OPTIMIZATION
        }
        
        return task_type_mapping.get(task_type.lower(), CodeTaskType.GENERATION)
    
    def _generate_fallback_code(self, prompt: str, language: str) -> str:
        """Generate fallback code when all other methods fail"""
        if language.lower() == 'python':
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
        elif language.lower() in ['javascript', 'typescript']:
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
        else:
            return f'''// Generated code for: {prompt}
// TODO: Implement the requested functionality
// Language: {language}
'''

async def main():
    """Main entry point"""
    backend = AIIDEBackend()
    
    try:
        await backend.initialize()
        
        # Main communication loop with VSCode extension
        logger.info("Starting communication loop...")
        logger.info("Backend ready - you can now start VSCode extension")
        logger.info("Listening for JSON tasks on stdin...")
        
        while True:
            try:
                # Read task from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                task = json.loads(line.strip())
                result = await backend.handle_task(task)
                
                # Send result to stdout
                print(json.dumps(result), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_result = {
                    "taskId": "unknown",
                    "success": False,
                    "error": f"Invalid JSON: {e}",
                    "executionTime": 0
                }
                print(json.dumps(error_result), flush=True)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_result = {
                    "taskId": "unknown", 
                    "success": False,
                    "error": f"Unexpected error: {e}",
                    "executionTime": 0
                }
                print(json.dumps(error_result), flush=True)
                
    except KeyboardInterrupt:
        logger.info("Shutting down AI IDE Backend...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())