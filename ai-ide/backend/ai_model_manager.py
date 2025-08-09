#!/usr/bin/env python3
"""
AI Model Manager - Universal AI Model Interface
Handles ALL AI models with intelligent routing and optimization
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
from universal_ai_provider import UniversalAIProvider, ProviderType, ModelInfo
from model_installer import ModelInstaller

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CODE_COMPLETION = "code_completion"
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    CODE_REFACTORING = "code_refactoring"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    CHAT = "chat"
    AGENT_TASK = "agent_task"
    WEB_SEARCH = "web_search"
    REASONING = "reasoning"

@dataclass
class AIRequest:
    task_type: TaskType
    prompt: str
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    stream: bool = False
    model_preference: Optional[str] = None

@dataclass
class AIResponse:
    content: str
    model_used: str
    provider: ProviderType
    tokens_used: int
    response_time: float
    cost: Optional[float] = None

class AIModelManager:
    """Universal AI Model Manager - The brain of AI IDE"""
    
    def __init__(self):
        self.provider = UniversalAIProvider()
        self.installer = ModelInstaller()
        self.model_routing = {}
        self.performance_cache = {}
        self.active_models = {}
        
    async def initialize(self):
        """Initialize the AI Model Manager"""
        logger.info("🧠 Initializing AI Model Manager...")
        
        # Initialize universal provider
        await self.provider.initialize()
        
        # Setup optimal environment if needed
        if len(self.provider.available_models) == 0:
            logger.info("🚀 No models found. Setting up optimal coding environment...")
            await self.installer.setup_optimal_coding_environment()
            await self.provider.initialize()  # Re-scan after installation
        
        # Load model routing configuration
        await self.load_model_routing()
        
        # Warm up best models
        await self.warm_up_models()
        
        logger.info(f"✅ AI Model Manager ready with {len(self.provider.available_models)} models")
    
    async def load_model_routing(self):
        """Load intelligent model routing configuration"""
        try:
            from pathlib import Path
            config_file = Path.home() / ".ai-ide" / "model_routing.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.model_routing = config.get('model_routing', {})
                    logger.info("✅ Model routing configuration loaded")
            else:
                # Use default routing
                self.model_routing = self.get_default_routing()
                logger.info("📋 Using default model routing")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model routing: {e}")
            self.model_routing = self.get_default_routing()
    
    def get_default_routing(self) -> Dict:
        """Get default model routing configuration"""
        return {
            "code_completion": {
                "primary": "qwen2.5-coder:7b",
                "fallback": "deepseek-coder:6.7b", 
                "cloud_fallback": "gpt-4-turbo"
            },
            "code_generation": {
                "primary": "qwen2.5-coder:7b",
                "fallback": "starcoder2:7b",
                "cloud_fallback": "gpt-4-turbo"
            },
            "code_explanation": {
                "primary": "deepseek-coder:6.7b",
                "fallback": "qwen2.5-coder:7b",
                "cloud_fallback": "claude-3-5-sonnet"
            },
            "agent_task": {
                "primary": "mikepfunk28/deepseekq3_agent:latest",
                "fallback": "qwen2.5-coder:7b",
                "cloud_fallback": "gpt-4-turbo"
            },
            "web_search": {
                "primary": "gpt-4-turbo",
                "fallback": "claude-3-5-sonnet",
                "local_fallback": "qwen2.5-coder:7b"
            },
            "reasoning": {
                "primary": "claude-3-5-sonnet",
                "fallback": "gpt-4-turbo",
                "local_fallback": "deepseek-coder:6.7b"
            }
        }
    
    async def warm_up_models(self):
        """Warm up the most commonly used models"""
        logger.info("🔥 Warming up models...")
        
        # Get the most important models
        important_models = set()
        for task_routing in self.model_routing.values():
            important_models.add(task_routing.get('primary'))
            important_models.add(task_routing.get('fallback'))
        
        # Remove None values
        important_models = {model for model in important_models if model}
        
        # Warm up each model with a simple request
        for model_name in important_models:
            if model_name in self.provider.available_models:
                try:
                    await self.make_simple_request(model_name)
                    logger.info(f"✅ Warmed up {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to warm up {model_name}: {e}")
    
    async def make_simple_request(self, model_name: str):
        """Make a simple request to warm up a model"""
        model_info = self.provider.available_models[model_name]
        
        if model_info.provider == ProviderType.OLLAMA:
            async with aiohttp.ClientSession() as session:
                data = {
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False
                }
                async with session.post('http://localhost:11434/api/generate', json=data) as resp:
                    await resp.json()
        
        # Add other provider warm-up logic as needed
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process an AI request with intelligent model selection"""
        start_time = asyncio.get_event_loop().time()
        
        # Select the best model for this task
        selected_model = await self.select_best_model(request)
        
        if not selected_model:
            raise Exception("No suitable model available for this request")
        
        # Make the request
        try:
            response_content = await self.make_model_request(selected_model, request)
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # Create response
            model_info = self.provider.available_models[selected_model]
            
            response = AIResponse(
                content=response_content,
                model_used=selected_model,
                provider=model_info.provider,
                tokens_used=self.estimate_tokens(request.prompt + response_content),
                response_time=response_time,
                cost=self.calculate_cost(model_info, request.prompt + response_content)
            )
            
            # Update performance cache
            await self.update_performance_cache(selected_model, request.task_type, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Request failed with {selected_model}: {e}")
            
            # Try fallback model
            fallback_model = await self.get_fallback_model(request, selected_model)
            if fallback_model:
                logger.info(f"🔄 Trying fallback model: {fallback_model}")
                return await self.process_request_with_model(request, fallback_model)
            
            raise Exception(f"All models failed for request: {e}")
    
    async def select_best_model(self, request: AIRequest) -> Optional[str]:
        """Select the best model for a given request"""
        
        # If user specified a model preference, try that first
        if request.model_preference and request.model_preference in self.provider.available_models:
            return request.model_preference
        
        # Get routing for this task type
        task_routing = self.model_routing.get(request.task_type.value, {})
        
        # Try primary model
        primary_model = task_routing.get('primary')
        if primary_model and primary_model in self.provider.available_models:
            if await self.is_model_suitable(primary_model, request):
                return primary_model
        
        # Try fallback model
        fallback_model = task_routing.get('fallback')
        if fallback_model and fallback_model in self.provider.available_models:
            if await self.is_model_suitable(fallback_model, request):
                return fallback_model
        
        # Try cloud fallback
        cloud_fallback = task_routing.get('cloud_fallback')
        if cloud_fallback and cloud_fallback in self.provider.available_models:
            return cloud_fallback
        
        # Try local fallback
        local_fallback = task_routing.get('local_fallback')
        if local_fallback and local_fallback in self.provider.available_models:
            return local_fallback
        
        # If nothing else works, try any available model
        for model_name in self.provider.available_models:
            if await self.is_model_suitable(model_name, request):
                return model_name
        
        return None
    
    async def is_model_suitable(self, model_name: str, request: AIRequest) -> bool:
        """Check if a model is suitable for a request"""
        model_info = self.provider.available_models[model_name]
        
        # Check context length
        prompt_tokens = self.estimate_tokens(request.prompt + (request.context or ""))
        if prompt_tokens > model_info.context_length:
            return False
        
        # Check capabilities
        task_capability_map = {
            TaskType.CODE_COMPLETION: 'code',
            TaskType.CODE_GENERATION: 'code',
            TaskType.CODE_EXPLANATION: 'code',
            TaskType.CHAT: 'chat',
            TaskType.AGENT_TASK: 'agent'
        }
        
        required_capability = task_capability_map.get(request.task_type)
        if required_capability and required_capability not in model_info.capabilities:
            return False
        
        return True
    
    async def make_model_request(self, model_name: str, request: AIRequest) -> str:
        """Make a request to a specific model"""
        model_info = self.provider.available_models[model_name]
        
        if model_info.provider == ProviderType.OLLAMA:
            return await self.make_ollama_request(model_name, request)
        elif model_info.provider == ProviderType.OPENAI:
            return await self.make_openai_request(model_name, request)
        elif model_info.provider == ProviderType.ANTHROPIC:
            return await self.make_anthropic_request(model_name, request)
        elif model_info.provider == ProviderType.LMSTUDIO:
            return await self.make_lmstudio_request(model_name, request)
        elif model_info.provider == ProviderType.OPENROUTER:
            return await self.make_openrouter_request(model_name, request)
        else:
            raise Exception(f"Unsupported provider: {model_info.provider}")
    
    async def make_ollama_request(self, model_name: str, request: AIRequest) -> str:
        """Make request to Ollama"""
        async with aiohttp.ClientSession() as session:
            data = {
                "model": model_name,
                "prompt": self.format_prompt(request),
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens or -1
                }
            }
            
            async with session.post('http://localhost:11434/api/generate', json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get('response', '')
                else:
                    raise Exception(f"Ollama request failed: {resp.status}")
    
    async def make_openai_request(self, model_name: str, request: AIRequest) -> str:
        """Make request to OpenAI"""
        provider_config = self.provider.providers[ProviderType.OPENAI]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {provider_config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": self.format_prompt(request)}],
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            if request.max_tokens:
                data["max_tokens"] = request.max_tokens
            
            async with session.post(f'{provider_config.base_url}/chat/completions', 
                                  headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception(f"OpenAI request failed: {resp.status}")
    
    async def make_anthropic_request(self, model_name: str, request: AIRequest) -> str:
        """Make request to Anthropic"""
        provider_config = self.provider.providers[ProviderType.ANTHROPIC]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'x-api-key': provider_config.api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": self.format_prompt(request)}],
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature
            }
            
            async with session.post(f'{provider_config.base_url}/v1/messages',
                                  headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result['content'][0]['text']
                else:
                    raise Exception(f"Anthropic request failed: {resp.status}")
    
    async def make_lmstudio_request(self, model_name: str, request: AIRequest) -> str:
        """Make request to LM Studio"""
        provider_config = self.provider.providers[ProviderType.LMSTUDIO]
        
        async with aiohttp.ClientSession() as session:
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": self.format_prompt(request)}],
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            if request.max_tokens:
                data["max_tokens"] = request.max_tokens
            
            async with session.post(f'{provider_config.base_url}/chat/completions', json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception(f"LM Studio request failed: {resp.status}")
    
    async def make_openrouter_request(self, model_name: str, request: AIRequest) -> str:
        """Make request to OpenRouter"""
        provider_config = self.provider.providers[ProviderType.OPENROUTER]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {provider_config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": self.format_prompt(request)}],
                "temperature": request.temperature
            }
            
            if request.max_tokens:
                data["max_tokens"] = request.max_tokens
            
            async with session.post(f'{provider_config.base_url}/chat/completions',
                                  headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception(f"OpenRouter request failed: {resp.status}")
    
    def format_prompt(self, request: AIRequest) -> str:
        """Format prompt based on task type"""
        if request.context:
            return f"Context:\n{request.context}\n\nTask:\n{request.prompt}"
        return request.prompt
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text.split()) * 1.3  # Rough estimate
    
    def calculate_cost(self, model_info: ModelInfo, text: str) -> Optional[float]:
        """Calculate cost for the request"""
        if model_info.cost_per_token:
            tokens = self.estimate_tokens(text)
            return tokens * model_info.cost_per_token
        return None
    
    async def update_performance_cache(self, model_name: str, task_type: TaskType, response_time: float):
        """Update performance cache for model selection optimization"""
        key = f"{model_name}:{task_type.value}"
        
        if key not in self.performance_cache:
            self.performance_cache[key] = []
        
        self.performance_cache[key].append(response_time)
        
        # Keep only last 10 measurements
        if len(self.performance_cache[key]) > 10:
            self.performance_cache[key] = self.performance_cache[key][-10:]
    
    async def get_fallback_model(self, request: AIRequest, failed_model: str) -> Optional[str]:
        """Get fallback model when primary fails"""
        task_routing = self.model_routing.get(request.task_type.value, {})
        
        # Try different fallback options
        fallback_options = [
            task_routing.get('fallback'),
            task_routing.get('cloud_fallback'),
            task_routing.get('local_fallback')
        ]
        
        for fallback in fallback_options:
            if fallback and fallback != failed_model and fallback in self.provider.available_models:
                if await self.is_model_suitable(fallback, request):
                    return fallback
        
        return None
    
    async def process_request_with_model(self, request: AIRequest, model_name: str) -> AIResponse:
        """Process request with specific model (used for fallbacks)"""
        original_preference = request.model_preference
        request.model_preference = model_name
        
        try:
            return await self.process_request(request)
        finally:
            request.model_preference = original_preference
    
    async def get_available_models(self) -> Dict[str, Dict]:
        """Get all available models with their info"""
        models_info = {}
        
        for model_name, model_info in self.provider.available_models.items():
            models_info[model_name] = {
                'provider': model_info.provider.value,
                'context_length': model_info.context_length,
                'capabilities': model_info.capabilities,
                'cost_per_token': model_info.cost_per_token,
                'size_gb': model_info.size_gb
            }
        
        return models_info
    
    async def get_model_recommendations(self, task_type: TaskType) -> List[str]:
        """Get recommended models for a specific task type"""
        task_routing = self.model_routing.get(task_type.value, {})
        
        recommendations = []
        for key in ['primary', 'fallback', 'cloud_fallback', 'local_fallback']:
            model = task_routing.get(key)
            if model and model in self.provider.available_models:
                recommendations.append(model)
        
        return recommendations

# Example usage
async def main():
    manager = AIModelManager()
    await manager.initialize()
    
    # Test code completion
    request = AIRequest(
        task_type=TaskType.CODE_COMPLETION,
        prompt="def fibonacci(n):",
        max_tokens=100
    )
    
    response = await manager.process_request(request)
    print(f"Response: {response.content}")
    print(f"Model used: {response.model_used}")
    print(f"Response time: {response.response_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())