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