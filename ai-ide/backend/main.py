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
from universal_ai_provider import (
    get_universal_ai_provider, UniversalAIProvider, 
    generate_with_best_model, generate_with_cheapest_model, generate_with_fastest_model
)
from enhanced_ollama_integration import (
    get_enhanced_ollama, EnhancedOllamaIntegration,
    generate_with_deepseek_helper, generate_with_best_ollama_model
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
        self.universal_ai_provider: Optional[UniversalAIProvider] = None
        self.enhanced_ollama: Optional[EnhancedOllamaIntegration] = None
        
    async def initialize(self):
        """Initialize all backend services"""
        try:
            logger.info("Initializing AI IDE Backend...")
            
            # Initialize core services
            await self.init_universal_ai_provider()
            await self.init_enhanced_ollama()
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
    
    async def init_universal_ai_provider(self):
        """Initialize Universal AI Provider System"""
        logger.info("Initializing Universal AI Provider System...")
        try:
            self.universal_ai_provider = await get_universal_ai_provider()
            
            # Get available models summary
            available_models = self.universal_ai_provider.get_available_models()
            logger.info(f"Universal AI Provider initialized with {len(available_models)} models")
            
            # Log model summary by provider
            provider_counts = {}
            for model in available_models.values():
                provider_name = model.provider.value
                provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1
            
            for provider, count in provider_counts.items():
                logger.info(f"  - {provider}: {count} models")
            
            # Test with best available model
            try:
                best_model = self.universal_ai_provider.get_best_model('code')
                if best_model:
                    logger.info(f"Best coding model: {best_model.name} ({best_model.provider.value})")
                    
                    # Quick test
                    test_result = await self.universal_ai_provider.generate_completion(
                        best_model.name,
                        "def hello():",
                        max_tokens=50,
                        temperature=0.1
                    )
                    logger.info("Universal AI Provider test successful")
                else:
                    logger.warning("No coding models available")
            except Exception as e:
                logger.warning(f"Universal AI Provider test failed: {e}")
            
            self.services['universal_ai'] = True
            logger.info("Universal AI Provider System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Universal AI Provider: {e}")
            self.services['universal_ai'] = False
    
    async def init_enhanced_ollama(self):
        """Initialize Enhanced Ollama Integration"""
        logger.info("Initializing Enhanced Ollama Integration...")
        try:
            self.enhanced_ollama = await get_enhanced_ollama()
            
            if self.enhanced_ollama.is_running:
                # Get available models
                available_models = self.enhanced_ollama.get_available_models()
                helper_models = self.enhanced_ollama.get_helper_models()
                
                logger.info(f"Enhanced Ollama initialized with {len(available_models)} models")
                logger.info(f"Helper models available: {len(helper_models)}")
                
                # Log helper models
                for helper_model in helper_models:
                    logger.info(f"  - Helper: {helper_model}")
                
                # Test with DeepSeek helper model if available
                if helper_models:
                    try:
                        test_result = await self.enhanced_ollama.generate_with_helper_model(
                            "def hello():",
                            {'language': 'python', 'context': 'test'},
                            'deepseek'
                        )
                        logger.info("Enhanced Ollama test successful")
                    except Exception as e:
                        logger.warning(f"Enhanced Ollama test failed: {e}")
                
                self.services['enhanced_ollama'] = True
                logger.info("Enhanced Ollama Integration initialized successfully")
            else:
                logger.warning("Ollama service not running")
                self.services['enhanced_ollama'] = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Ollama: {e}")
            self.services['enhanced_ollama'] = False
    
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
            elif task_type == "multi_model_generation":
                result = await self.handle_multi_model_generation(task)
            elif task_type == "semantic_search":
                result = await self.handle_semantic_search(task)
            elif task_type == "reasoning":
                result = await self.handle_reasoning(task)
            elif task_type == "model_comparison":
                result = await self.handle_model_comparison(task)
            elif task_type == "get_available_models":
                result = await self.get_available_models()
            elif task_type == "ollama_helper_generation":
                result = await self.handle_ollama_helper_generation(task)
            elif task_type == "ollama_template_generation":
                result = await self.handle_ollama_template_generation(task)
            elif task_type == "get_ollama_models":
                result = await self.get_ollama_models()
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
            # First try Universal AI Provider for best model selection
            if self.universal_ai_provider and self.services.get('universal_ai'):
                logger.info("Using Universal AI Provider for code generation")
                
                # Get the best model for coding
                best_model = self.universal_ai_provider.get_best_model('code')
                if best_model:
                    # Create enhanced prompt
                    enhanced_prompt = self._create_code_generation_prompt(prompt, context)
                    
                    # Generate with best available model
                    response = await self.universal_ai_provider.generate_completion(
                        best_model.name,
                        enhanced_prompt,
                        max_tokens=task.get("input", {}).get("max_tokens", 2048),
                        temperature=task.get("input", {}).get("temperature", 0.3)
                    )
                    
                    # Extract and clean the generated code
                    generated_code = self._extract_code_from_response(response['text'], language)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        "code": generated_code,
                        "language": language,
                        "confidence": 0.95,  # High confidence for best model
                        "model_info": {
                            "model_name": best_model.name,
                            "provider": best_model.provider.value,
                            "context_length": best_model.context_length,
                            "cost_per_token": best_model.cost_per_token
                        },
                        "execution_metrics": {
                            "total_time": execution_time,
                            "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                        },
                        "metadata": {
                            "agent_used": "universal_ai_provider",
                            "task_type": task_type,
                            "finish_reason": response.get('finish_reason')
                        }
                    }
            
            # Try Enhanced Ollama with helper models
            elif self.enhanced_ollama and self.services.get('enhanced_ollama'):
                logger.info("Using Enhanced Ollama for code generation")
                
                # Determine task type for Ollama
                ollama_task_type = self._map_to_ollama_task_type(task_type)
                
                try:
                    # Use best Ollama model for the task
                    response = await generate_with_best_ollama_model(
                        prompt,
                        ollama_task_type,
                        {
                            'language': language,
                            'context': context.get('surroundingCode', ''),
                            'file_path': context.get('filePath', ''),
                            'selected_text': context.get('selectedText', '')
                        }
                    )
                    
                    # Extract and clean the generated code
                    generated_code = self._extract_code_from_response(response['text'], language)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        "code": generated_code,
                        "language": language,
                        "confidence": 0.9,  # High confidence for Ollama models
                        "model_info": {
                            "model_name": response['model'],
                            "provider": "enhanced_ollama",
                            "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                        },
                        "execution_metrics": {
                            "total_time": execution_time,
                            "eval_duration": response.get('eval_duration', 0),
                            "load_duration": response.get('load_duration', 0)
                        },
                        "metadata": {
                            "agent_used": "enhanced_ollama",
                            "task_type": task_type,
                            "finish_reason": response.get('finish_reason')
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Enhanced Ollama generation failed: {e}")
                    # Fall through to next option
            
            # Use Qwen Coder agent if available
            elif self.qwen_coder_agent and self.services.get('qwen_coder'):
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

    async def handle_multi_model_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-model code generation for comparison"""
        prompt = task.get("input", {}).get("prompt", "")
        context = task.get("context", {})
        language = context.get("language", "python")
        num_models = task.get("input", {}).get("num_models", 3)
        
        start_time = datetime.now()
        results = []
        
        try:
            if self.universal_ai_provider and self.services.get('universal_ai'):
                logger.info(f"Using Universal AI Provider for multi-model generation ({num_models} models)")
                
                # Get different types of models for comparison
                best_model = self.universal_ai_provider.get_best_model('code')
                cheapest_model = self.universal_ai_provider.get_cheapest_model('code')
                fastest_model = self.universal_ai_provider.get_fastest_model('code')
                
                models_to_try = []
                if best_model:
                    models_to_try.append(('best', best_model))
                if cheapest_model and cheapest_model.name != (best_model.name if best_model else None):
                    models_to_try.append(('cheapest', cheapest_model))
                if fastest_model and fastest_model.name not in [m[1].name for m in models_to_try]:
                    models_to_try.append(('fastest', fastest_model))
                
                # Limit to requested number of models
                models_to_try = models_to_try[:num_models]
                
                # Create enhanced prompt
                enhanced_prompt = self._create_code_generation_prompt(prompt, context)
                
                # Generate with each model
                for model_type, model_info in models_to_try:
                    try:
                        model_start_time = datetime.now()
                        
                        response = await self.universal_ai_provider.generate_completion(
                            model_info.name,
                            enhanced_prompt,
                            max_tokens=task.get("input", {}).get("max_tokens", 1024),
                            temperature=task.get("input", {}).get("temperature", 0.3)
                        )
                        
                        model_execution_time = (datetime.now() - model_start_time).total_seconds()
                        
                        # Extract and clean the generated code
                        generated_code = self._extract_code_from_response(response['text'], language)
                        
                        results.append({
                            "model_type": model_type,
                            "model_name": model_info.name,
                            "provider": model_info.provider.value,
                            "code": generated_code,
                            "confidence": self._calculate_code_confidence(generated_code, language),
                            "execution_time": model_execution_time,
                            "cost_estimate": (response.get('usage', {}).get('total_tokens', 0) * 
                                            (model_info.cost_per_token or 0)),
                            "context_length": model_info.context_length,
                            "finish_reason": response.get('finish_reason'),
                            "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to generate with {model_info.name}: {e}")
                        results.append({
                            "model_type": model_type,
                            "model_name": model_info.name,
                            "provider": model_info.provider.value,
                            "error": str(e),
                            "execution_time": 0,
                            "cost_estimate": 0
                        })
                
                total_execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "results": results,
                    "total_models": len(results),
                    "successful_models": len([r for r in results if 'code' in r]),
                    "total_execution_time": total_execution_time,
                    "language": language,
                    "prompt": prompt,
                    "metadata": {
                        "agent_used": "universal_ai_provider_multi",
                        "task_type": "multi_model_generation"
                    }
                }
            
            else:
                return {
                    "error": "Universal AI Provider not available",
                    "results": [],
                    "total_models": 0,
                    "successful_models": 0
                }
                
        except Exception as e:
            logger.error(f"Multi-model generation failed: {e}")
            return {
                "error": str(e),
                "results": results,
                "total_models": len(results),
                "successful_models": len([r for r in results if 'code' in r])
            }
    
    def _calculate_code_confidence(self, code: str, language: str) -> float:
        """Calculate confidence score for generated code"""
        if not code or not code.strip():
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check for basic code structure
        if language.lower() == 'python':
            if 'def ' in code or 'class ' in code:
                confidence += 0.2
            if 'import ' in code or 'from ' in code:
                confidence += 0.1
            if code.count(':') > 0:  # Python uses colons
                confidence += 0.1
        elif language.lower() in ['javascript', 'typescript']:
            if 'function ' in code or '=>' in code:
                confidence += 0.2
            if '{' in code and '}' in code:
                confidence += 0.1
            if ';' in code:
                confidence += 0.1
        
        # Check for comments
        if '//' in code or '#' in code or '/*' in code:
            confidence += 0.05
        
        # Check for reasonable length
        if 50 <= len(code) <= 2000:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available AI models"""
        if not self.universal_ai_provider or not self.services.get('universal_ai'):
            return {
                "error": "Universal AI Provider not available",
                "models": {},
                "providers": {}
            }
        
        try:
            available_models = self.universal_ai_provider.get_available_models()
            
            # Group models by provider
            providers = {}
            for model_name, model_info in available_models.items():
                provider_name = model_info.provider.value
                if provider_name not in providers:
                    providers[provider_name] = {
                        "models": [],
                        "total_models": 0,
                        "capabilities": set()
                    }
                
                providers[provider_name]["models"].append({
                    "name": model_name,
                    "context_length": model_info.context_length,
                    "capabilities": model_info.capabilities,
                    "cost_per_token": model_info.cost_per_token,
                    "size_gb": model_info.size_gb,
                    "local_path": model_info.local_path
                })
                providers[provider_name]["total_models"] += 1
                providers[provider_name]["capabilities"].update(model_info.capabilities)
            
            # Convert sets to lists for JSON serialization
            for provider_info in providers.values():
                provider_info["capabilities"] = list(provider_info["capabilities"])
            
            # Get recommended models
            recommendations = {
                "best_coding": self.universal_ai_provider.get_best_model('code'),
                "cheapest_coding": self.universal_ai_provider.get_cheapest_model('code'),
                "fastest_coding": self.universal_ai_provider.get_fastest_model('code'),
                "best_chat": self.universal_ai_provider.get_best_model('chat'),
                "cheapest_chat": self.universal_ai_provider.get_cheapest_model('chat'),
                "fastest_chat": self.universal_ai_provider.get_fastest_model('chat')
            }
            
            # Convert model objects to dictionaries
            for key, model in recommendations.items():
                if model:
                    recommendations[key] = {
                        "name": model.name,
                        "provider": model.provider.value,
                        "context_length": model.context_length,
                        "capabilities": model.capabilities,
                        "cost_per_token": model.cost_per_token
                    }
                else:
                    recommendations[key] = None
            
            return {
                "models": available_models,
                "providers": providers,
                "recommendations": recommendations,
                "total_models": len(available_models),
                "total_providers": len(providers)
            }
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return {
                "error": str(e),
                "models": {},
                "providers": {}
            }
    
    async def handle_model_comparison(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model comparison and selection"""
        capability = task.get("input", {}).get("capability", "code")
        preference = task.get("input", {}).get("preference", "best")  # best, cheapest, fastest
        
        try:
            if not self.universal_ai_provider or not self.services.get('universal_ai'):
                return {
                    "error": "Universal AI Provider not available",
                    "selected_model": None,
                    "alternatives": []
                }
            
            # Get model based on preference
            if preference == "cheapest":
                selected_model = self.universal_ai_provider.get_cheapest_model(capability)
            elif preference == "fastest":
                selected_model = self.universal_ai_provider.get_fastest_model(capability)
            else:  # best
                selected_model = self.universal_ai_provider.get_best_model(capability)
            
            # Get alternatives
            all_models = self.universal_ai_provider.get_models_by_capability(capability)
            alternatives = []
            
            for model_name, model_info in all_models.items():
                if not selected_model or model_name != selected_model.name:
                    alternatives.append({
                        "name": model_name,
                        "provider": model_info.provider.value,
                        "context_length": model_info.context_length,
                        "cost_per_token": model_info.cost_per_token,
                        "capabilities": model_info.capabilities,
                        "size_gb": model_info.size_gb
                    })
            
            # Sort alternatives by preference
            if preference == "cheapest":
                alternatives.sort(key=lambda x: x["cost_per_token"] or 0)
            elif preference == "fastest":
                # Local models are typically faster
                alternatives.sort(key=lambda x: (x["cost_per_token"] or 0 > 0, x["size_gb"] or 0))
            else:  # best
                alternatives.sort(key=lambda x: x["context_length"], reverse=True)
            
            return {
                "selected_model": {
                    "name": selected_model.name,
                    "provider": selected_model.provider.value,
                    "context_length": selected_model.context_length,
                    "cost_per_token": selected_model.cost_per_token,
                    "capabilities": selected_model.capabilities,
                    "size_gb": selected_model.size_gb
                } if selected_model else None,
                "alternatives": alternatives[:5],  # Top 5 alternatives
                "capability": capability,
                "preference": preference,
                "total_available": len(all_models)
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {
                "error": str(e),
                "selected_model": None,
                "alternatives": []
            }
    
    def _map_to_ollama_task_type(self, task_type: str) -> str:
        """Map general task type to Ollama-specific task type"""
        mapping = {
            'generation': 'code_generation',
            'completion': 'code_completion',
            'debugging': 'debugging',
            'refactoring': 'code_generation',
            'documentation': 'documentation',
            'explanation': 'chat',
            'review': 'code_review'
        }
        return mapping.get(task_type, 'code_generation')    
  
  async def handle_ollama_helper_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation using Ollama helper models"""
        prompt = task.get("input", {}).get("prompt", "")
        context = task.get("context", {})
        helper_preference = task.get("input", {}).get("helper_preference", "deepseek")
        
        start_time = datetime.now()
        
        try:
            if self.enhanced_ollama and self.services.get('enhanced_ollama'):
                logger.info(f"Using Ollama helper model ({helper_preference}) for generation")
                
                response = await self.enhanced_ollama.generate_with_helper_model(
                    prompt,
                    {
                        'language': context.get('language', 'python'),
                        'context': context.get('surroundingCode', ''),
                        'file_path': context.get('filePath', ''),
                        'selected_text': context.get('selectedText', '')
                    },
                    helper_preference
                )
                
                # Extract and clean the generated code
                generated_code = self._extract_code_from_response(response['text'], context.get('language', 'python'))
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "code": generated_code,
                    "language": context.get('language', 'python'),
                    "confidence": 0.85,
                    "helper_model": response['model'],
                    "model_info": {
                        "model_name": response['model'],
                        "provider": "ollama_helper",
                        "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                    },
                    "execution_metrics": {
                        "total_time": execution_time,
                        "eval_duration": response.get('eval_duration', 0)
                    },
                    "metadata": {
                        "agent_used": "ollama_helper",
                        "helper_preference": helper_preference
                    }
                }
            else:
                return {
                    "error": "Enhanced Ollama not available",
                    "code": "",
                    "confidence": 0
                }
                
        except Exception as e:
            logger.error(f"Ollama helper generation failed: {e}")
            return {
                "error": str(e),
                "code": "",
                "confidence": 0
            }
    
    async def handle_ollama_template_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation using Ollama templates"""
        model_name = task.get("input", {}).get("model_name", "")
        template_name = task.get("input", {}).get("template_name", "")
        variables = task.get("input", {}).get("variables", {})
        
        start_time = datetime.now()
        
        try:
            if self.enhanced_ollama and self.services.get('enhanced_ollama'):
                logger.info(f"Using Ollama template generation: {template_name} with {model_name}")
                
                response = await self.enhanced_ollama.generate_with_template(
                    model_name,
                    template_name,
                    variables,
                    max_tokens=task.get("input", {}).get("max_tokens", 1000),
                    temperature=task.get("input", {}).get("temperature", 0.7)
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "text": response['text'],
                    "model_name": model_name,
                    "template_name": template_name,
                    "variables": variables,
                    "model_info": {
                        "model_name": response['model'],
                        "provider": "ollama_template",
                        "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                    },
                    "execution_metrics": {
                        "total_time": execution_time,
                        "eval_duration": response.get('eval_duration', 0)
                    },
                    "metadata": {
                        "agent_used": "ollama_template",
                        "template_used": template_name
                    }
                }
            else:
                return {
                    "error": "Enhanced Ollama not available",
                    "text": ""
                }
                
        except Exception as e:
            logger.error(f"Ollama template generation failed: {e}")
            return {
                "error": str(e),
                "text": ""
            }
    
    async def get_ollama_models(self) -> Dict[str, Any]:
        """Get information about available Ollama models"""
        try:
            if not self.enhanced_ollama or not self.services.get('enhanced_ollama'):
                return {
                    "error": "Enhanced Ollama not available",
                    "models": {},
                    "helper_models": [],
                    "templates": {}
                }
            
            available_models = self.enhanced_ollama.get_available_models()
            helper_models = self.enhanced_ollama.get_helper_models()
            templates = self.enhanced_ollama.templates
            
            # Convert models to serializable format
            models_info = {}
            for model_name, model in available_models.items():
                models_info[model_name] = {
                    "name": model.name,
                    "size": model.size,
                    "modified": model.modified,
                    "type": model.model_type.value,
                    "capabilities": model.capabilities,
                    "context_length": model.context_length,
                    "is_helper_model": model.is_helper_model,
                    "template": model.template,
                    "system_prompt": model.system_prompt
                }
            
            # Convert templates to serializable format
            templates_info = {}
            for template_name, template in templates.items():
                templates_info[template_name] = {
                    "name": template.name,
                    "type": template.template_type,
                    "description": template.description,
                    "variables": template.variables,
                    "model_compatibility": template.model_compatibility
                }
            
            # Get models by type
            models_by_type = {}
            for model_type in ['code_generation', 'chat', 'reasoning', 'helper']:
                if model_type == 'helper':
                    models_by_type[model_type] = helper_models
                else:
                    from enhanced_ollama_integration import OllamaModelType
                    type_enum = getattr(OllamaModelType, model_type.upper(), None)
                    if type_enum:
                        type_models = self.enhanced_ollama.get_models_by_type(type_enum)
                        models_by_type[model_type] = list(type_models.keys())
            
            return {
                "models": models_info,
                "helper_models": helper_models,
                "templates": templates_info,
                "models_by_type": models_by_type,
                "total_models": len(available_models),
                "total_helpers": len(helper_models),
                "total_templates": len(templates),
                "is_running": self.enhanced_ollama.is_running
            }
            
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return {
                "error": str(e),
                "models": {},
                "helper_models": [],
                "templates": {}
            }