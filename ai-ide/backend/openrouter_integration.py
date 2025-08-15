#!/usr/bin/env python3
"""
OpenRouter.ai Integration for Universal AI Provider
Provides access to multiple AI models through OpenRouter.ai API
"""

import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger('openrouter-integration')

@dataclass
class OpenRouterModel:
    """Represents an OpenRouter model"""
    id: str
    name: str
    description: str
    context_length: int
    pricing: Dict[str, float]
    top_provider: Dict[str, Any]

class OpenRouterIntegration:
    """OpenRouter.ai integration for accessing multiple AI models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.models_cache = {}
        self.session = None
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        else:
            logger.info("OpenRouter integration initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_available_models(self) -> List[OpenRouterModel]:
        """Get all available models from OpenRouter"""
        if not self.api_key:
            return []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('data', []):
                        model = OpenRouterModel(
                            id=model_data.get('id', ''),
                            name=model_data.get('name', ''),
                            description=model_data.get('description', ''),
                            context_length=model_data.get('context_length', 4096),
                            pricing=model_data.get('pricing', {}),
                            top_provider=model_data.get('top_provider', {})
                        )
                        models.append(model)
                    
                    self.models_cache = {model.id: model for model in models}
                    logger.info(f"Retrieved {len(models)} models from OpenRouter")
                    return models
                else:
                    logger.error(f"Failed to get models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting OpenRouter models: {e}")
            return []
    
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using OpenRouter"""
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mike-ai-ide.local",
                "X-Title": "Mike-AI-IDE"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    choice = data.get('choices', [{}])[0]
                    message = choice.get('message', {})
                    usage = data.get('usage', {})
                    
                    return {
                        'success': True,
                        'text': message.get('content', ''),
                        'model': model,
                        'provider': 'openrouter',
                        'usage': {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0)
                        },
                        'finish_reason': choice.get('finish_reason', 'stop')
                    }
                else:
                    error_data = await response.json()
                    logger.error(f"OpenRouter API error: {response.status} - {error_data}")
                    return {
                        'success': False,
                        'error': f"API error: {response.status}",
                        'details': error_data
                    }
                    
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best model for a specific task"""
        
        # Task-specific model recommendations
        task_models = {
            'code': [
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4-turbo',
                'deepseek/deepseek-coder',
                'codellama/codellama-70b-instruct'
            ],
            'chat': [
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4-turbo',
                'meta-llama/llama-3.1-70b-instruct',
                'google/gemini-pro-1.5'
            ],
            'analysis': [
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4-turbo',
                'google/gemini-pro-1.5'
            ],
            'creative': [
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4-turbo',
                'meta-llama/llama-3.1-70b-instruct'
            ],
            'fast': [
                'openai/gpt-3.5-turbo',
                'anthropic/claude-3-haiku',
                'meta-llama/llama-3.1-8b-instruct',
                'google/gemini-flash-1.5'
            ],
            'cheap': [
                'openai/gpt-3.5-turbo',
                'anthropic/claude-3-haiku',
                'meta-llama/llama-3.1-8b-instruct',
                'google/gemini-flash-1.5'
            ]
        }
        
        models = task_models.get(task_type, task_models['chat'])
        
        # Return first available model
        for model in models:
            if model in self.models_cache:
                return model
        
        # Fallback to any available model
        if self.models_cache:
            return list(self.models_cache.keys())[0]
        
        return None
    
    def get_cheapest_model(self) -> Optional[str]:
        """Get the cheapest available model"""
        if not self.models_cache:
            return None
        
        cheapest_model = None
        lowest_cost = float('inf')
        
        for model_id, model in self.models_cache.items():
            if model.pricing and 'completion' in model.pricing:
                cost = float(model.pricing['completion'])
                if cost < lowest_cost:
                    lowest_cost = cost
                    cheapest_model = model_id
        
        return cheapest_model or list(self.models_cache.keys())[0]
    
    def get_fastest_model(self) -> Optional[str]:
        """Get the fastest available model (typically smaller models)"""
        fast_models = [
            'openai/gpt-3.5-turbo',
            'anthropic/claude-3-haiku',
            'meta-llama/llama-3.1-8b-instruct',
            'google/gemini-flash-1.5'
        ]
        
        for model in fast_models:
            if model in self.models_cache:
                return model
        
        return list(self.models_cache.keys())[0] if self.models_cache else None
    
    async def generate_with_multiple_models(
        self,
        models: List[str],
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Generate completions with multiple models for comparison"""
        
        tasks = []
        for model in models:
            if model in self.models_cache:
                task = self.generate_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                tasks.append(task)
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and add model info
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('success'):
                result['model_id'] = models[i]
                valid_results.append(result)
        
        return valid_results
    
    def get_model_info(self, model_id: str) -> Optional[OpenRouterModel]:
        """Get detailed information about a specific model"""
        return self.models_cache.get(model_id)
    
    def is_available(self) -> bool:
        """Check if OpenRouter integration is available"""
        return bool(self.api_key)

# Global instance
_openrouter_integration = None

async def get_openrouter_integration() -> OpenRouterIntegration:
    """Get or create the global OpenRouter integration instance"""
    global _openrouter_integration
    if _openrouter_integration is None:
        _openrouter_integration = OpenRouterIntegration()
        # Initialize models cache
        async with _openrouter_integration:
            await _openrouter_integration.get_available_models()
    return _openrouter_integration

# Convenience functions
async def generate_with_openrouter(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate text using OpenRouter with automatic model selection"""
    
    integration = await get_openrouter_integration()
    
    if not integration.is_available():
        return {
            'success': False,
            'error': 'OpenRouter API key not configured'
        }
    
    # Auto-select model if not specified
    if not model:
        model = integration.get_best_model_for_task('chat')
        if not model:
            return {
                'success': False,
                'error': 'No models available'
            }
    
    messages = [{"role": "user", "content": prompt}]
    
    async with integration:
        return await integration.generate_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

async def generate_with_best_openrouter_model(
    prompt: str,
    task_type: str = 'chat',
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate text using the best OpenRouter model for the task"""
    
    integration = await get_openrouter_integration()
    
    if not integration.is_available():
        return {
            'success': False,
            'error': 'OpenRouter API key not configured'
        }
    
    model = integration.get_best_model_for_task(task_type)
    if not model:
        return {
            'success': False,
            'error': f'No models available for task: {task_type}'
        }
    
    messages = [{"role": "user", "content": prompt}]
    
    async with integration:
        return await integration.generate_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )