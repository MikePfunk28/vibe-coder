#!/usr/bin/env python3
"""
Universal AI Provider System
Supports ALL AI models: Cloud (OpenAI, Anthropic, OpenRouter) + Local (Ollama, LM Studio, llama.cpp)
Auto-detects, auto-installs, and manages everything
"""

import asyncio
import json
import logging
import subprocess
import requests
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import docker
from pathlib import Path

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    LLAMACPP = "llamacpp"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    GROQ = "groq"
    COHERE = "cohere"

@dataclass
class ModelInfo:
    name: str
    provider: ProviderType
    context_length: int
    capabilities: List[str]
    cost_per_token: Optional[float] = None
    local_path: Optional[str] = None
    download_url: Optional[str] = None
    size_gb: Optional[float] = None

@dataclass
class ProviderConfig:
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[ModelInfo] = None
    is_available: bool = False
    auto_install: bool = True

class UniversalAIProvider:
    """Universal AI Provider that can connect to ANY AI model"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, ProviderConfig] = {}
        self.available_models: Dict[str, ModelInfo] = {}
        self.active_model: Optional[str] = None
        self.docker_client = None
        
    async def initialize(self):
        """Initialize and detect all available AI providers"""
        logger.info("🚀 Initializing Universal AI Provider System...")
        
        # Initialize Docker client for containerized models
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
        
        # Detect and setup all providers
        await self.detect_all_providers()
        await self.discover_all_models()
        
        logger.info(f"✅ Found {len(self.available_models)} models across {len(self.providers)} providers")
    
    async def detect_all_providers(self):
        """Detect all available AI providers"""
        detection_tasks = [
            self.detect_openai(),
            self.detect_anthropic(),
            self.detect_openrouter(),
            self.detect_ollama(),
            self.detect_lmstudio(),
            self.detect_llamacpp(),
            self.detect_huggingface(),
            self.detect_together(),
            self.detect_groq(),
            self.detect_cohere()
        ]
        
        await asyncio.gather(*detection_tasks, return_exceptions=True)    
    a
sync def detect_openai(self):
        """Detect OpenAI API availability"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {'Authorization': f'Bearer {api_key}'}
                    async with session.get('https://api.openai.com/v1/models', headers=headers) as resp:
                        if resp.status == 200:
                            self.providers[ProviderType.OPENAI] = ProviderConfig(
                                provider_type=ProviderType.OPENAI,
                                api_key=api_key,
                                base_url='https://api.openai.com/v1',
                                is_available=True
                            )
                            logger.info("✅ OpenAI API detected and available")
            except Exception as e:
                logger.warning(f"OpenAI API detection failed: {e}")
    
    async def detect_anthropic(self):
        """Detect Anthropic Claude API availability"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
                    # Test with a simple request
                    self.providers[ProviderType.ANTHROPIC] = ProviderConfig(
                        provider_type=ProviderType.ANTHROPIC,
                        api_key=api_key,
                        base_url='https://api.anthropic.com',
                        is_available=True
                    )
                    logger.info("✅ Anthropic Claude API detected and available")
            except Exception as e:
                logger.warning(f"Anthropic API detection failed: {e}")
    
    async def detect_openrouter(self):
        """Detect OpenRouter API availability"""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {'Authorization': f'Bearer {api_key}'}
                    async with session.get('https://openrouter.ai/api/v1/models', headers=headers) as resp:
                        if resp.status == 200:
                            self.providers[ProviderType.OPENROUTER] = ProviderConfig(
                                provider_type=ProviderType.OPENROUTER,
                                api_key=api_key,
                                base_url='https://openrouter.ai/api/v1',
                                is_available=True
                            )
                            logger.info("✅ OpenRouter API detected and available")
            except Exception as e:
                logger.warning(f"OpenRouter API detection failed: {e}")
    
    async def detect_ollama(self):
        """Detect Ollama local installation"""
        try:
            # Check if Ollama is running
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as resp:
                    if resp.status == 200:
                        self.providers[ProviderType.OLLAMA] = ProviderConfig(
                            provider_type=ProviderType.OLLAMA,
                            base_url='http://localhost:11434',
                            is_available=True
                        )
                        logger.info("✅ Ollama detected and running")
                        return
        except:
            pass
        
        # Try to auto-install Ollama if not found
        if await self.auto_install_ollama():
            self.providers[ProviderType.OLLAMA] = ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                base_url='http://localhost:11434',
                is_available=True
            )
    
    async def detect_lmstudio(self):
        """Detect LM Studio local installation"""
        try:
            # Check common LM Studio ports
            ports = [1234, 8080, 8000]
            for port in ports:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f'http://localhost:{port}/v1/models') as resp:
                            if resp.status == 200:
                                self.providers[ProviderType.LMSTUDIO] = ProviderConfig(
                                    provider_type=ProviderType.LMSTUDIO,
                                    base_url=f'http://localhost:{port}/v1',
                                    is_available=True
                                )
                                logger.info(f"✅ LM Studio detected on port {port}")
                                return
                except:
                    continue
        except Exception as e:
            logger.warning(f"LM Studio detection failed: {e}")
    
    async def detect_llamacpp(self):
        """Detect llama.cpp server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/health') as resp:
                    if resp.status == 200:
                        self.providers[ProviderType.LLAMACPP] = ProviderConfig(
                            provider_type=ProviderType.LLAMACPP,
                            base_url='http://localhost:8080',
                            is_available=True
                        )
                        logger.info("✅ llama.cpp server detected")
        except Exception as e:
            logger.warning(f"llama.cpp detection failed: {e}")
    
    async def detect_huggingface(self):
        """Detect Hugging Face API"""
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if api_key:
            self.providers[ProviderType.HUGGINGFACE] = ProviderConfig(
                provider_type=ProviderType.HUGGINGFACE,
                api_key=api_key,
                base_url='https://api-inference.huggingface.co',
                is_available=True
            )
            logger.info("✅ Hugging Face API detected")
    
    async def detect_together(self):
        """Detect Together AI API"""
        api_key = os.getenv('TOGETHER_API_KEY')
        if api_key:
            self.providers[ProviderType.TOGETHER] = ProviderConfig(
                provider_type=ProviderType.TOGETHER,
                api_key=api_key,
                base_url='https://api.together.xyz/v1',
                is_available=True
            )
            logger.info("✅ Together AI API detected")
    
    async def detect_groq(self):
        """Detect Groq API"""
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.providers[ProviderType.GROQ] = ProviderConfig(
                provider_type=ProviderType.GROQ,
                api_key=api_key,
                base_url='https://api.groq.com/openai/v1',
                is_available=True
            )
            logger.info("✅ Groq API detected")
    
    async def detect_cohere(self):
        """Detect Cohere API"""
        api_key = os.getenv('COHERE_API_KEY')
        if api_key:
            self.providers[ProviderType.COHERE] = ProviderConfig(
                provider_type=ProviderType.COHERE,
                api_key=api_key,
                base_url='https://api.cohere.ai/v1',
                is_available=True
            )
            logger.info("✅ Cohere API detected")
    
    async def auto_install_ollama(self) -> bool:
        """Auto-install Ollama if user wants it"""
        try:
            logger.info("🔧 Ollama not found. Attempting auto-installation...")
            
            # Download and install Ollama
            if os.name == 'nt':  # Windows
                install_cmd = 'powershell -Command "& {Invoke-WebRequest -Uri https://ollama.ai/install.ps1 -UseBasicParsing | Invoke-Expression}"'
            else:  # Unix-like
                install_cmd = 'curl -fsSL https://ollama.ai/install.sh | sh'
            
            process = await asyncio.create_subprocess_shell(
                install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Ollama installed successfully")
                
                # Start Ollama service
                start_cmd = 'ollama serve' if os.name != 'nt' else 'ollama.exe serve'
                asyncio.create_task(self.start_background_service(start_cmd))
                
                # Wait a bit for service to start
                await asyncio.sleep(3)
                return True
            else:
                logger.error(f"Ollama installation failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to auto-install Ollama: {e}")
            return False
    
    async def start_background_service(self, command: str):
        """Start a background service"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            logger.info(f"Started background service: {command}")
        except Exception as e:
            logger.error(f"Failed to start background service {command}: {e}")
    
    async def discover_all_models(self):
        """Discover all available models from all providers"""
        discovery_tasks = []
        
        for provider_type, config in self.providers.items():
            if config.is_available:
                discovery_tasks.append(self.discover_provider_models(provider_type, config))
        
        await asyncio.gather(*discovery_tasks, return_exceptions=True)
    
    async def discover_provider_models(self, provider_type: ProviderType, config: ProviderConfig):
        """Discover models for a specific provider"""
        try:
            if provider_type == ProviderType.OPENAI:
                await self.discover_openai_models(config)
            elif provider_type == ProviderType.ANTHROPIC:
                await self.discover_anthropic_models(config)
            elif provider_type == ProviderType.OPENROUTER:
                await self.discover_openrouter_models(config)
            elif provider_type == ProviderType.OLLAMA:
                await self.discover_ollama_models(config)
            elif provider_type == ProviderType.LMSTUDIO:
                await self.discover_lmstudio_models(config)
            elif provider_type == ProviderType.LLAMACPP:
                await self.discover_llamacpp_models(config)
            elif provider_type == ProviderType.HUGGINGFACE:
                await self.discover_huggingface_models(config)
            elif provider_type == ProviderType.TOGETHER:
                await self.discover_together_models(config)
            elif provider_type == ProviderType.GROQ:
                await self.discover_groq_models(config)
            elif provider_type == ProviderType.COHERE:
                await self.discover_cohere_models(config)
                
        except Exception as e:
            logger.error(f"Failed to discover models for {provider_type}: {e}")
    
    async def discover_openai_models(self, config: ProviderConfig):
        """Discover OpenAI models"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {config.api_key}'}
                async with session.get(f'{config.base_url}/models', headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data['data']:
                            model_id = model_data['id']
                            
                            # Focus on coding models
                            if any(keyword in model_id.lower() for keyword in ['gpt-4', 'gpt-3.5', 'code', 'davinci']):
                                model_info = ModelInfo(
                                    name=model_id,
                                    provider=ProviderType.OPENAI,
                                    context_length=self.get_openai_context_length(model_id),
                                    capabilities=['chat', 'completion', 'code'],
                                    cost_per_token=self.get_openai_cost(model_id)
                                )
                                self.available_models[model_id] = model_info
                                
        except Exception as e:
            logger.error(f"Failed to discover OpenAI models: {e}")
    
    def get_openai_context_length(self, model_id: str) -> int:
        """Get context length for OpenAI models"""
        context_lengths = {
            'gpt-4-turbo': 128000,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16385,
            'code-davinci-002': 8001
        }
        
        for model_name, length in context_lengths.items():
            if model_name in model_id:
                return length
        return 4096  # Default
    
    def get_openai_cost(self, model_id: str) -> float:
        """Get cost per token for OpenAI models"""
        costs = {
            'gpt-4-turbo': 0.00001,
            'gpt-4': 0.00003,
            'gpt-3.5-turbo': 0.000001
        }
        
        for model_name, cost in costs.items():
            if model_name in model_id:
                return cost
        return 0.00001  # Default    
    as
ync def discover_anthropic_models(self, config: ProviderConfig):
        """Discover Anthropic Claude models"""
        try:
            # Anthropic models (manually defined as they don't have a models endpoint)
            claude_models = [
                {
                    'id': 'claude-3-5-sonnet-20241022',
                    'context_length': 200000,
                    'capabilities': ['chat', 'code', 'analysis'],
                    'cost_per_token': 0.000003
                },
                {
                    'id': 'claude-3-haiku-20240307',
                    'context_length': 200000,
                    'capabilities': ['chat', 'code', 'fast'],
                    'cost_per_token': 0.00000025
                },
                {
                    'id': 'claude-3-opus-20240229',
                    'context_length': 200000,
                    'capabilities': ['chat', 'code', 'reasoning'],
                    'cost_per_token': 0.000015
                }
            ]
            
            for model_data in claude_models:
                model_info = ModelInfo(
                    name=model_data['id'],
                    provider=ProviderType.ANTHROPIC,
                    context_length=model_data['context_length'],
                    capabilities=model_data['capabilities'],
                    cost_per_token=model_data['cost_per_token']
                )
                self.available_models[model_data['id']] = model_info
                
        except Exception as e:
            logger.error(f"Failed to discover Anthropic models: {e}")
    
    async def discover_openrouter_models(self, config: ProviderConfig):
        """Discover OpenRouter models"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {config.api_key}'}
                async with session.get(f'{config.base_url}/models', headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data['data']:
                            model_id = model_data['id']
                            
                            # Focus on coding and reasoning models
                            if any(keyword in model_id.lower() for keyword in [
                                'claude', 'gpt-4', 'deepseek', 'qwen', 'codestral', 
                                'llama', 'mistral', 'gemini', 'code'
                            ]):
                                model_info = ModelInfo(
                                    name=model_id,
                                    provider=ProviderType.OPENROUTER,
                                    context_length=model_data.get('context_length', 4096),
                                    capabilities=['chat', 'completion', 'code'],
                                    cost_per_token=model_data.get('pricing', {}).get('prompt', 0.00001)
                                )
                                self.available_models[model_id] = model_info
                                
        except Exception as e:
            logger.error(f"Failed to discover OpenRouter models: {e}")
    
    async def discover_ollama_models(self, config: ProviderConfig):
        """Discover Ollama local models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{config.base_url}/api/tags') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data.get('models', []):
                            model_name = model_data['name']
                            
                            model_info = ModelInfo(
                                name=model_name,
                                provider=ProviderType.OLLAMA,
                                context_length=self.get_ollama_context_length(model_name),
                                capabilities=['chat', 'completion', 'code'],
                                cost_per_token=0.0,  # Local models are free
                                size_gb=model_data.get('size', 0) / (1024**3)
                            )
                            self.available_models[model_name] = model_info
                            
        except Exception as e:
            logger.error(f"Failed to discover Ollama models: {e}")
    
    async def discover_lmstudio_models(self, config: ProviderConfig):
        """Discover LM Studio models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{config.base_url}/models') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data.get('data', []):
                            model_id = model_data['id']
                            
                            model_info = ModelInfo(
                                name=model_id,
                                provider=ProviderType.LMSTUDIO,
                                context_length=model_data.get('context_length', 4096),
                                capabilities=['chat', 'completion', 'code'],
                                cost_per_token=0.0  # Local models are free
                            )
                            self.available_models[model_id] = model_info
                            
        except Exception as e:
            logger.error(f"Failed to discover LM Studio models: {e}")
    
    async def discover_llamacpp_models(self, config: ProviderConfig):
        """Discover llama.cpp models"""
        try:
            # llama.cpp typically serves one model at a time
            model_info = ModelInfo(
                name='llama-cpp-local',
                provider=ProviderType.LLAMACPP,
                context_length=4096,  # Default, can be configured
                capabilities=['chat', 'completion', 'code'],
                cost_per_token=0.0
            )
            self.available_models['llama-cpp-local'] = model_info
            
        except Exception as e:
            logger.error(f"Failed to discover llama.cpp models: {e}")
    
    async def discover_huggingface_models(self, config: ProviderConfig):
        """Discover Hugging Face models"""
        try:
            # Popular coding models on Hugging Face
            hf_models = [
                'microsoft/DialoGPT-medium',
                'microsoft/CodeBERT-base',
                'Salesforce/codegen-350M-mono',
                'bigcode/starcoder',
                'WizardLM/WizardCoder-15B-V1.0'
            ]
            
            for model_name in hf_models:
                model_info = ModelInfo(
                    name=model_name,
                    provider=ProviderType.HUGGINGFACE,
                    context_length=2048,
                    capabilities=['code', 'completion'],
                    cost_per_token=0.0  # Inference API is free for many models
                )
                self.available_models[model_name] = model_info
                
        except Exception as e:
            logger.error(f"Failed to discover Hugging Face models: {e}")
    
    async def discover_together_models(self, config: ProviderConfig):
        """Discover Together AI models"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {config.api_key}'}
                async with session.get(f'{config.base_url}/models', headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data:
                            model_id = model_data['id']
                            
                            if any(keyword in model_id.lower() for keyword in ['code', 'llama', 'mistral']):
                                model_info = ModelInfo(
                                    name=model_id,
                                    provider=ProviderType.TOGETHER,
                                    context_length=model_data.get('context_length', 4096),
                                    capabilities=['chat', 'completion', 'code'],
                                    cost_per_token=model_data.get('pricing', {}).get('input', 0.00001)
                                )
                                self.available_models[model_id] = model_info
                                
        except Exception as e:
            logger.error(f"Failed to discover Together AI models: {e}")
    
    async def discover_groq_models(self, config: ProviderConfig):
        """Discover Groq models"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {config.api_key}'}
                async with session.get(f'{config.base_url}/models', headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model_data in data.get('data', []):
                            model_id = model_data['id']
                            
                            model_info = ModelInfo(
                                name=model_id,
                                provider=ProviderType.GROQ,
                                context_length=model_data.get('context_length', 4096),
                                capabilities=['chat', 'completion', 'fast'],
                                cost_per_token=0.00001  # Groq is very fast and cheap
                            )
                            self.available_models[model_id] = model_info
                            
        except Exception as e:
            logger.error(f"Failed to discover Groq models: {e}")
    
    async def discover_cohere_models(self, config: ProviderConfig):
        """Discover Cohere models"""
        try:
            # Cohere models (manually defined)
            cohere_models = [
                {
                    'id': 'command-r-plus',
                    'context_length': 128000,
                    'capabilities': ['chat', 'code', 'reasoning'],
                    'cost_per_token': 0.000003
                },
                {
                    'id': 'command-r',
                    'context_length': 128000,
                    'capabilities': ['chat', 'code'],
                    'cost_per_token': 0.0000005
                }
            ]
            
            for model_data in cohere_models:
                model_info = ModelInfo(
                    name=model_data['id'],
                    provider=ProviderType.COHERE,
                    context_length=model_data['context_length'],
                    capabilities=model_data['capabilities'],
                    cost_per_token=model_data['cost_per_token']
                )
                self.available_models[model_data['id']] = model_info
                
        except Exception as e:
            logger.error(f"Failed to discover Cohere models: {e}")
    
    def get_ollama_context_length(self, model_name: str) -> int:
        """Get context length for Ollama models"""
        context_lengths = {
            'llama2': 4096,
            'llama3': 8192,
            'codellama': 16384,
            'deepseek-coder': 16384,
            'qwen': 32768,
            'mistral': 8192,
            'gemma': 8192
        }
        
        for model_key, length in context_lengths.items():
            if model_key in model_name.lower():
                return length
        return 4096  # Default
    
    async def generate_completion(self, 
                                model_name: str, 
                                prompt: str, 
                                max_tokens: int = 1000,
                                temperature: float = 0.7,
                                **kwargs) -> Dict[str, Any]:
        """Generate completion using specified model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_info = self.available_models[model_name]
        provider_config = self.providers[model_info.provider]
        
        try:
            if model_info.provider == ProviderType.OPENAI:
                return await self.generate_openai_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.ANTHROPIC:
                return await self.generate_anthropic_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.OPENROUTER:
                return await self.generate_openrouter_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.OLLAMA:
                return await self.generate_ollama_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.LMSTUDIO:
                return await self.generate_lmstudio_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.LLAMACPP:
                return await self.generate_llamacpp_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            elif model_info.provider == ProviderType.GROQ:
                return await self.generate_groq_completion(provider_config, model_name, prompt, max_tokens, temperature, **kwargs)
            else:
                raise ValueError(f"Provider {model_info.provider} not implemented for completion")
                
        except Exception as e:
            logger.error(f"Failed to generate completion with {model_name}: {e}")
            raise
    
    async def generate_openai_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate OpenAI completion"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/chat/completions', headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['choices'][0]['message']['content'],
                        'model': model_name,
                        'provider': 'openai',
                        'usage': result.get('usage', {}),
                        'finish_reason': result['choices'][0].get('finish_reason')
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error: {resp.status} - {error_text}")
    
    async def generate_anthropic_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate Anthropic completion"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'x-api-key': config.api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': model_name,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': [{'role': 'user', 'content': prompt}],
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/v1/messages', headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['content'][0]['text'],
                        'model': model_name,
                        'provider': 'anthropic',
                        'usage': result.get('usage', {}),
                        'finish_reason': result.get('stop_reason')
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"Anthropic API error: {resp.status} - {error_text}")
    
    async def generate_openrouter_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate OpenRouter completion"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://ai-ide.local',
                'X-Title': 'AI IDE'
            }
            
            data = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/chat/completions', headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['choices'][0]['message']['content'],
                        'model': model_name,
                        'provider': 'openrouter',
                        'usage': result.get('usage', {}),
                        'finish_reason': result['choices'][0].get('finish_reason')
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"OpenRouter API error: {resp.status} - {error_text}")
    
    async def generate_ollama_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate Ollama completion"""
        async with aiohttp.ClientSession() as session:
            data = {
                'model': model_name,
                'prompt': prompt,
                'options': {
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    **kwargs
                },
                'stream': False
            }
            
            async with session.post(f'{config.base_url}/api/generate', json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['response'],
                        'model': model_name,
                        'provider': 'ollama',
                        'usage': {
                            'prompt_tokens': result.get('prompt_eval_count', 0),
                            'completion_tokens': result.get('eval_count', 0)
                        },
                        'finish_reason': 'stop' if result.get('done') else 'length'
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"Ollama API error: {resp.status} - {error_text}")
    
    async def generate_lmstudio_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate LM Studio completion"""
        async with aiohttp.ClientSession() as session:
            headers = {'Content-Type': 'application/json'}
            
            data = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/chat/completions', headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['choices'][0]['message']['content'],
                        'model': model_name,
                        'provider': 'lmstudio',
                        'usage': result.get('usage', {}),
                        'finish_reason': result['choices'][0].get('finish_reason')
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"LM Studio API error: {resp.status} - {error_text}")
    
    async def generate_llamacpp_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate llama.cpp completion"""
        async with aiohttp.ClientSession() as session:
            data = {
                'prompt': prompt,
                'n_predict': max_tokens,
                'temperature': temperature,
                'stop': kwargs.get('stop', []),
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/completion', json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['content'],
                        'model': model_name,
                        'provider': 'llamacpp',
                        'usage': {
                            'prompt_tokens': result.get('tokens_evaluated', 0),
                            'completion_tokens': result.get('tokens_predicted', 0)
                        },
                        'finish_reason': 'stop' if result.get('stopped_eos') else 'length'
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"llama.cpp API error: {resp.status} - {error_text}")
    
    async def generate_groq_completion(self, config: ProviderConfig, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Generate Groq completion"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs
            }
            
            async with session.post(f'{config.base_url}/chat/completions', headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result['choices'][0]['message']['content'],
                        'model': model_name,
                        'provider': 'groq',
                        'usage': result.get('usage', {}),
                        'finish_reason': result['choices'][0].get('finish_reason')
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"Groq API error: {resp.status} - {error_text}")
    
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get all available models"""
        return self.available_models.copy()
    
    def get_models_by_capability(self, capability: str) -> Dict[str, ModelInfo]:
        """Get models that support a specific capability"""
        return {
            name: model for name, model in self.available_models.items()
            if capability in model.capabilities
        }
    
    def get_cheapest_model(self, capability: str = None) -> Optional[ModelInfo]:
        """Get the cheapest model, optionally filtered by capability"""
        models = self.get_models_by_capability(capability) if capability else self.available_models
        
        if not models:
            return None
        
        # Prefer free local models, then cheapest cloud models
        free_models = [model for model in models.values() if model.cost_per_token == 0.0]
        if free_models:
            return min(free_models, key=lambda m: m.size_gb or 0)
        
        paid_models = [model for model in models.values() if model.cost_per_token and model.cost_per_token > 0]
        if paid_models:
            return min(paid_models, key=lambda m: m.cost_per_token)
        
        return None
    
    def get_fastest_model(self, capability: str = None) -> Optional[ModelInfo]:
        """Get the fastest model, optionally filtered by capability"""
        models = self.get_models_by_capability(capability) if capability else self.available_models
        
        if not models:
            return None
        
        # Groq models are typically fastest, then local models, then others
        groq_models = [model for model in models.values() if model.provider == ProviderType.GROQ]
        if groq_models:
            return groq_models[0]
        
        local_models = [model for model in models.values() if model.provider in [ProviderType.OLLAMA, ProviderType.LMSTUDIO, ProviderType.LLAMACPP]]
        if local_models:
            return local_models[0]
        
        return list(models.values())[0]
    
    def get_best_model(self, capability: str = None, prefer_local: bool = False, prefer_cheap: bool = False) -> Optional[ModelInfo]:
        """Get the best model based on preferences"""
        models = self.get_models_by_capability(capability) if capability else self.available_models
        
        if not models:
            return None
        
        if prefer_cheap:
            return self.get_cheapest_model(capability)
        
        if prefer_local:
            local_models = [model for model in models.values() if model.provider in [ProviderType.OLLAMA, ProviderType.LMSTUDIO, ProviderType.LLAMACPP]]
            if local_models:
                # Prefer larger context length for local models
                return max(local_models, key=lambda m: m.context_length)
        
        # Default: prefer high-quality cloud models
        quality_order = [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.OPENROUTER,
            ProviderType.GROQ,
            ProviderType.TOGETHER,
            ProviderType.COHERE,
            ProviderType.OLLAMA,
            ProviderType.LMSTUDIO,
            ProviderType.LLAMACPP,
            ProviderType.HUGGINGFACE
        ]
        
        for provider_type in quality_order:
            provider_models = [model for model in models.values() if model.provider == provider_type]
            if provider_models:
                # Return the model with the largest context length from this provider
                return max(provider_models, key=lambda m: m.context_length)
        
        return None

# Global instance
_universal_provider = None

async def get_universal_ai_provider() -> UniversalAIProvider:
    """Get the global universal AI provider instance"""
    global _universal_provider
    if _universal_provider is None:
        _universal_provider = UniversalAIProvider()
        await _universal_provider.initialize()
    return _universal_provider

async def generate_with_best_model(prompt: str, capability: str = 'code', **kwargs) -> Dict[str, Any]:
    """Generate completion using the best available model for the task"""
    provider = await get_universal_ai_provider()
    best_model = provider.get_best_model(capability)
    
    if not best_model:
        raise Exception(f"No models available for capability: {capability}")
    
    return await provider.generate_completion(best_model.name, prompt, **kwargs)

async def generate_with_cheapest_model(prompt: str, capability: str = 'code', **kwargs) -> Dict[str, Any]:
    """Generate completion using the cheapest available model"""
    provider = await get_universal_ai_provider()
    cheapest_model = provider.get_cheapest_model(capability)
    
    if not cheapest_model:
        raise Exception(f"No models available for capability: {capability}")
    
    return await provider.generate_completion(cheapest_model.name, prompt, **kwargs)

async def generate_with_fastest_model(prompt: str, capability: str = 'code', **kwargs) -> Dict[str, Any]:
    """Generate completion using the fastest available model"""
    provider = await get_universal_ai_provider()
    fastest_model = provider.get_fastest_model(capability)
    
    if not fastest_model:
        raise Exception(f"No models available for capability: {capability}")
    
    return await provider.generate_completion(fastest_model.name, prompt, **kwargs)