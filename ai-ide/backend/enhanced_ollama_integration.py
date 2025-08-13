#!/usr/bin/env python3
"""
Enhanced Ollama Integration
Provides advanced Ollama model management, template system, and helper model support
"""

import asyncio
import json
import logging
import subprocess
import aiohttp
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class OllamaModelType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    CHAT = "chat"
    REASONING = "reasoning"
    HELPER = "helper"

@dataclass
class OllamaModel:
    name: str
    size: str
    modified: str
    digest: str
    model_type: OllamaModelType
    capabilities: List[str]
    context_length: int = 4096
    is_helper_model: bool = False
    template: Optional[str] = None
    system_prompt: Optional[str] = None

@dataclass
class OllamaTemplate:
    name: str
    template_type: str  # 'xml', 'jinja2', 'simple'
    content: str
    variables: List[str]
    description: str
    model_compatibility: List[str]

class EnhancedOllamaIntegration:
    """Enhanced Ollama integration with template system and helper models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models: Dict[str, OllamaModel] = {}
        self.templates: Dict[str, OllamaTemplate] = {}
        self.helper_models: List[str] = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize Ollama integration"""
        logger.info("🚀 Initializing Enhanced Ollama Integration...")
        
        # Check if Ollama is running
        await self.check_ollama_status()
        
        if not self.is_running:
            # Try to start Ollama
            await self.start_ollama_service()
        
        if self.is_running:
            # Discover available models
            await self.discover_models()
            
            # Ensure helper models are available
            await self.setup_helper_models()
            
            # Load templates
            await self.load_templates()
            
            logger.info(f"✅ Enhanced Ollama initialized with {len(self.available_models)} models")
        else:
            logger.warning("❌ Ollama service not available")
    
    async def check_ollama_status(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as resp:
                    if resp.status == 200:
                        self.is_running = True
                        logger.info("✅ Ollama service is running")
                        return True
        except Exception as e:
            logger.warning(f"Ollama service check failed: {e}")
        
        self.is_running = False
        return False
    
    async def start_ollama_service(self):
        """Start Ollama service if not running"""
        try:
            logger.info("🔧 Starting Ollama service...")
            
            # Try to start Ollama
            if os.name == 'nt':  # Windows
                process = await asyncio.create_subprocess_exec(
                    'ollama.exe', 'serve',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            else:  # Unix-like
                process = await asyncio.create_subprocess_exec(
                    'ollama', 'serve',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            
            # Wait a bit for service to start
            await asyncio.sleep(3)
            
            # Check if it's running now
            if await self.check_ollama_status():
                logger.info("✅ Ollama service started successfully")
            else:
                logger.warning("❌ Failed to start Ollama service")
                
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
    
    async def discover_models(self):
        """Discover available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for model_data in data.get('models', []):
                            model_name = model_data['name']
                            
                            # Determine model type and capabilities
                            model_type, capabilities = self._classify_model(model_name)
                            
                            # Check if it's a helper model
                            is_helper = self._is_helper_model(model_name)
                            
                            model = OllamaModel(
                                name=model_name,
                                size=model_data.get('size', 'unknown'),
                                modified=model_data.get('modified', ''),
                                digest=model_data.get('digest', ''),
                                model_type=model_type,
                                capabilities=capabilities,
                                context_length=self._get_context_length(model_name),
                                is_helper_model=is_helper,
                                template=self._get_model_template(model_name),
                                system_prompt=self._get_system_prompt(model_name)
                            )
                            
                            self.available_models[model_name] = model
                            
                            if is_helper:
                                self.helper_models.append(model_name)
                        
                        logger.info(f"Discovered {len(self.available_models)} Ollama models")
                        
        except Exception as e:
            logger.error(f"Failed to discover Ollama models: {e}")
    
    def _classify_model(self, model_name: str) -> tuple[OllamaModelType, List[str]]:
        """Classify model type and capabilities based on name"""
        name_lower = model_name.lower()
        
        if any(keyword in name_lower for keyword in ['coder', 'code', 'deepseek', 'deepseek-r1', 'agent']):
            return OllamaModelType.CODE_GENERATION, ['code', 'completion', 'generation', 'debugging']
        elif any(keyword in name_lower for keyword in ['phi-4-mini', 'phi-4', 'devstral', 'gemma', 'llama','mistral', 'deepseek']):
            return OllamaModelType.CHAT, ['chat', 'reasoning', 'general', 'instruct', 'embedding', 'reasoning-plus']
        elif 'qwen' in name_lower:
            return OllamaModelType.CODE_GENERATION, ['code', 'chat', 'reasoning', 'multilingual']
        else:
            return OllamaModelType.CHAT, ['chat', 'general']
    
    def _is_helper_model(self, model_name: str) -> bool:
        """Check if model is a helper model"""
        helper_patterns = [
            'mikepfunk28/deepseekq3_coder',
            'deepseek-agent',
            'codellama:7b',
            'qwen2.5-coder:1.5b',
            'starcoder2:3b'
        ]
        
        return any(pattern in model_name.lower() for pattern in helper_patterns)
    
    def _get_context_length(self, model_name: str) -> int:
        """Get context length for model"""
        context_lengths = {
            'llama2': 4096,
            'llama3': 8192,
            'codellama': 16384,
            'deepseek': 16384,
            'qwen': 32768,
            'mistral': 8192,
            'gemma': 8192,
            'starcoder': 8192
        }
        
        for model_key, length in context_lengths.items():
            if model_key in model_name.lower():
                return length
        return 4096  # Default
    
    def _get_model_template(self, model_name: str) -> Optional[str]:
        """Get default template for model"""
        if 'deepseek' in model_name.lower():
            return 'deepseek_coder_template'
        elif 'qwen' in model_name.lower():
            return 'qwen_coder_template'
        elif 'codellama' in model_name.lower():
            return 'codellama_template'
        return None
    
    def _get_system_prompt(self, model_name: str) -> Optional[str]:
        """Get default system prompt for model"""
        if 'coder' in model_name.lower() or 'code' in model_name.lower():
            return """You are an expert software engineer and coding assistant. You write clean, efficient, and well-documented code. Always consider best practices, security, and performance. Provide explanations for complex logic."""
        elif 'deepseek' in model_name.lower():
            return """You are DeepSeek Coder, a helpful AI assistant specialized in programming and software development. You excel at code generation, debugging, optimization, and explaining complex programming concepts."""
        return None
    
    async def setup_helper_models(self):
        """Ensure helper models are available"""
        required_helper_models = [
            'mikepfunk28/deepseekq3_coder:latest',
            'deepseek-coder:6.7b',
            'qwen2.5-coder:1.5b'
        ]
        
        for model_name in required_helper_models:
            if not any(model_name in available for available in self.available_models.keys()):
                logger.info(f"🔧 Pulling helper model: {model_name}")
                await self.pull_model(model_name)
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"📥 Pulling model: {model_name}")
            
            async with aiohttp.ClientSession() as session:
                data = {'name': model_name}
                
                async with session.post(f"{self.base_url}/api/pull", json=data) as resp:
                    if resp.status == 200:
                        # Stream the response to show progress
                        async for line in resp.content:
                            if line:
                                try:
                                    progress_data = json.loads(line.decode())
                                    if 'status' in progress_data:
                                        logger.info(f"Pull progress: {progress_data['status']}")
                                except:
                                    pass
                        
                        logger.info(f"✅ Successfully pulled model: {model_name}")
                        
                        # Rediscover models to include the new one
                        await self.discover_models()
                        return True
                    else:
                        logger.error(f"Failed to pull model {model_name}: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def load_templates(self):
        """Load XML and Jinja2 templates"""
        # Built-in templates
        self.templates.update({
            'deepseek_coder_template': OllamaTemplate(
                name='deepseek_coder_template',
                template_type='xml',
                content='''<system>
You are DeepSeek Coder, an expert programming assistant.
Task: {{task_type}}
Language: {{language}}
Context: {{context}}
</system>

<user_request>
{{user_prompt}}
</user_request>

<code_context>
{{code_context}}
</code_context>

<instructions>
1. Analyze the request carefully
2. Consider the programming language and context
3. Generate clean, efficient code
4. Include commentsmpogic
5. est prces and conventions
</instructions>''',
               iables=['task_type', 'language', 'context', 'user_prompt', 'code_context'],
                description='Template for DeepSeek Coder models',
                model_compatibility=['deepseek', 'mikepfunk28/deepseekq3_coder']
            ),
            
   'qwen_codlate': Ollamplate(           nameoder_template',
                template_type='xml',
                content='''<system>
You are Qwen Coder, a multilingual programming assistant.
Programming Language: {{language}}
Task Type: {{task_type}}
</system>

<context>
{{context}}
</context>

<request>
{{user_prompt}}
</request>

<guidelines>
- Write clean, readable code
- Include appropriate error handling
- Add helpful comments
- Follow language-specific conventions
- Coder performance ecurit''',
                variables=['language', 'task_type', 'context', 'user_prompt'],
                description='Template for Qwen Coder models',
                model_compatibility=['qwen']
            ),
            
            'codellama_template': OllamaTemplate(
                name='codellama_template',
                template_ty'simple',            content='''[INST] You are Code Llama, a helpful coding assistant.

Task: {{t_type}}
Language: {{language}}
Context: {{context}
Please provide clel-commented code that follows best practices. [/INST]''',
                variables=['task_type', 'language', 'context', 'user_prompt'],           description='Template for Code Llama models',
                model_compatibility=['coama']
            ),
            
            'general_xml_template': OllamaTemplate(
                name='gene_xml_template',
                template_type='xml',
                content='''<system>
{{system_prompt}}
</system>

<task>
Type: {{task_type}}
Context: {{context}}
</task>

<user_input>
{{user_prompt}}
</user_input>

<response_format>
{{response_forma
</respo>''',
          variables=ys'task_type', 'context', 'user_prompt', 'response_format'],
                description='General XML template for structured prompts',
                model_compatibility=['*']
            )
        })
        
        logger.info(f"Loaded {len(self.templates)} templates")
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
       ent
        
        # Simple variable substitution (can be enhanc with Jinja2)
var_name,ables.itemacehol{{var_name}}}}}"
            content = content.replace(placeholder, str(var_value))
 
        return content
    
  derate_with_template(self,                     model_nam,
         
                                   variables: Dict[str, Any],
                        **kwargs) -> Dict[str, Any]:
        """Generate completion using a template"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        # Render template
        prompt = self.render_template(template_name, variables)
        
        # Generate completion    return aetion(modept, **kwargs)
    
    async def generate_complion(se               model_name: str,
                                prompt: str,
                                max_tokens: int = 1000,
                                temperatureat = 0.7,
                                st: bool = False,
                                ) -> Dict[str, Any]:
        """Generate completion with Ollama model       try:
            async with aiohttp.ClientSession() as session:
                data = {
                    'model': model_name,
                    'prompt': prompt,
                    'options': {
                        'num_predict': max_tokens,
                    mperature': temperature,
                        **kwargs
                    },
                    'stream': stream
                }
                
                # Add system prompt if available
                model = self.available_models.get(model_name)
                if model and model.system_prompt:
                    data['system'] = model.system_prompt
                
                async with session.post(f"{self.base_url}/api/generate", json=data) as resp:
                    if resp.status == 200:
                        if stream:
                            # Handle streaming response
                            full_response = ""
                            async for line in resp                          
                               chunk_a = json.ne.decode())
                                  'response' in a:
                     _data['response']
                                 if chundone', Fa                                  break
                                    except:
                                        pass
                            
                            return {
                                'text': full_response,
                                'model': model_name,
                                'provider': 'ollama',
                                'stream': True,
                                'finish_reason': 'stop'
                            }
                        else:
                            # Handle non-streaming r                           result = awason()
      return {
                                'text': result['response'],
                                'model': model_name,
                                'provider': 'ollama',
                                'usage': {
                                    'prompt_tokens': result.get('prompt_eval_count', 0),
                                    'completion_tokens': result.get('eval_count', 0),
                              l_tokempt_eval_count', 0) + result.get('eval_count', 0)
                                  'fish_reason': 'stop else ength'                        'eval_duration': result.get('eval_duration', 0),
                                'load_duration': result.get('load_duration', 0)
                            }
                 else:
                        error_tex= awaitt()
         raise Exception(f"Ollama APr: {resp.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to generate completion with {model_name}: {e}")
            rai   
    async def generate_with_helpelf,
                                       task: str,
                                       context: Dict[str, Any],
                                       helper_preference: str = 'deepseek') -> Dict[str, Any]:
        """Generate completion using a helper model"""
        # Find appropriate helper model
        helper_model = None
        
        if helper_preference == 'deepseek':
            helper_model = next((name for name in self.helper_models 
                               if 'deepseek' in name.lower()), None)
        elif helper_preference == 'qwen':
            helper_model = next((name for name in self.helper_models 
                               if 'qwen' in name.lower()), None)
        
        if not helper_model and self.helper_models:
            helper_model = self.helper_models[0]  # Use first available
        
        if not helper_model:
            raise Exception("No helper models available")
        
        # Use appropriate template
        template_name = self.available_models[helper_model].template or 'general_xml_template'
        
        variables = {
            'task_type': 'code_assistance',
            'language': context.get('language', 'python'),
            'context': context.get('context', ''),
            'user_prompt': task,
            'system_prompt': 'You are a helpful coding assistant.',
            'response_format': 'Provide clean code with explanations'
        }
        
        return await self.generate_with_template(helper_model, template_name, variables)
    
    def get_available_models(self) -> Dict[str, OllamaModel]:
        """Get all available models"""
        return self.available_models.copy()
    
    def get_helper_models(self) -> List[str]:
        """Get list of helper models"""
        return self.helper_models.copy()
    
    def get_models_by_type(self, model_type: OllamaModelType) -> Dict[str, OllamaModel]:
        """Get models by type"""
        return {
            name: model for name, model in self.available_models.items()
            if model.model_type == model_type
        }
    
    def get_models_by_capability(self, capability: str) -> Dict[str, OllamaModel]:
        """Get models by capability"""
        return {
            name: model for name, model in self.available_models.items()
            if capability in model.capabilities
        }
    
    def get_best_model_for_task(self, task_type: str) -> Optional[OllamaModel]:
        """Get the best model for a specific task"""
        task_preferences = {
            'code_generation': ['deepseek', 'qwen', 'codellama'],
            'code_completion': ['deepseek', 'codellama', 'qwen'],
            'debugging': ['deepseek', 'qwen', 'codellama'],
            'chat': ['qwen', 'llama', 'mistral'],
            'reasoning': ['qwen', 'llama', 'deepseek']
        }
        
        preferences = task_preferences.get(task_type, ['deepseek', 'qwen', 'llama'])
        
        for preference in preferences:
            for model_name, model in .available_m():
                if perence il_name.lower():
              return model
        
        # Return first available model if no preference match
        return next(iter(self.available_models.values())) if self.available_models else None
    
    async def chat_with_model(self,
                            model_name: str,
                            messages: List[Dict[str, str]],
                            **kwargs) -> Dict[str, Any]:
        """Chat with model using conversation format"""
        # Convert messages to prompt format
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        return await self.generate_completion(model_name, prompt, **kwargs)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        if model_name not in self.available_models:
            return None
        
        model = self.available_models[model_name]
        return {
            'name': model.name,
            'size': model.size,
            'modified': model.modified,
            'digest': model.digest,
            'type': model.model_type.value,
            'capabilities': model.capabilities,
            'context_length': model.context_length,
            'is_helper_model': model.is_helper_model,
            'template': model.template,
            'system_prompt': model.system_prompt
        }

# Global instance
_enhanced_ollama = None

async def get_enhanced_ollama() -> EnhancedOllamaIntegration:
    """Get the global enhanced Ollama instance"""
    global _enhanced_ollama
    if _enhanced_ollama is None:
        _enhanced_ollama = EnhancedOllamaIntegration()
        await _enhanced_ollama.initialize()
    return _enhanced_ollama

async def generate_with_deepseek_helper(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate using DeepSeek helper model"""
    ollama = await get_enhanced_ollama()
    return await ollama.generate_with_helper_model(task, context, 'deepseek')

async def generate_with_best_ollama_model(task: str, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate using the best Ollama model for the task"""
    ollama = await get_enhanced_ollama()
    best_model = ollama.get_best_model_for_task(task_type)
    
    if not best_model:
        raise Exception("No suitable Ollama models available")
    
    # Use template if available
    if best_model.template:
        variables = {
            'task_type': task_type,
            'language': context.get('language', 'python'),
            'context': context.get('context', ''),
            'user_prompt': task
        }
        return await ollama.generate_with_template(best_model.name, best_model.template, variables)
    else:
        return await ollama.generate_completion(best_model.name, task)