#!/usr/bin/env python3
"""
AI Model Auto-Installer
Automatically installs and configures the best coding models
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
import aiohttp
import docker

logger = logging.getLogger(__name__)

class ModelInstaller:
    """Auto-installs and configures AI models for optimal coding experience"""
    
    def __init__(self):
        self.recommended_models = {
            'qwen2.5-coder:7b': {
                'provider': 'ollama',
                'description': 'Qwen 2.5 Coder 7B - Excellent for code generation',
                'size_gb': 4.1,
                'capabilities': ['code', 'chat', 'completion'],
                'install_command': 'ollama pull qwen2.5-coder:7b'
            },
            'deepseek-coder:6.7b': {
                'provider': 'ollama', 
                'description': 'DeepSeek Coder 6.7B - Great for code understanding',
                'size_gb': 3.8,
                'capabilities': ['code', 'chat', 'completion'],
                'install_command': 'ollama pull deepseek-coder:6.7b'
            },
            'codellama:7b-code': {
                'provider': 'ollama',
                'description': 'Code Llama 7B - Meta\'s coding model',
                'size_gb': 3.8,
                'capabilities': ['code', 'completion'],
                'install_command': 'ollama pull codellama:7b-code'
            },
            'mikepfunk28/deepseekq3_agent:latest': {
                'provider': 'ollama',
                'description': 'DeepSeek Q3 Agent - Optimized for AI IDE',
                'size_gb': 2.1,
                'capabilities': ['code', 'chat', 'agent'],
                'install_command': 'ollama pull mikepfunk28/deepseekq3_agent:latest'
            },
            'starcoder2:7b': {
                'provider': 'ollama',
                'description': 'StarCoder2 7B - Advanced code generation',
                'size_gb': 4.0,
                'capabilities': ['code', 'completion'],
                'install_command': 'ollama pull starcoder2:7b'
            }
        }
    
    async def setup_optimal_coding_environment(self):
        """Set up the optimal coding environment with best models"""
        logger.info("🚀 Setting up optimal AI coding environment...")
        
        # Check available disk space
        available_space = self.get_available_disk_space()
        logger.info(f"💾 Available disk space: {available_space:.1f} GB")
        
        # Install Ollama if not present
        if not await self.check_ollama_installed():
            await self.install_ollama()
        
        # Install recommended models based on available space
        await self.install_recommended_models(available_space)
        
        # Setup LM Studio integration
        await self.setup_lmstudio_integration()
        
        # Configure model routing
        await self.configure_model_routing()
        
        logger.info("✅ Optimal AI coding environment setup complete!")
    
    def get_available_disk_space(self) -> float:
        """Get available disk space in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return free / (1024**3)  # Convert to GB
        except:
            return 50.0  # Default assumption
    
    async def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags', timeout=5) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def install_ollama(self):
        """Install Ollama"""
        logger.info("📦 Installing Ollama...")
        
        try:
            if os.name == 'nt':  # Windows
                install_cmd = [
                    'powershell', '-Command',
                    'Invoke-WebRequest -Uri https://ollama.ai/install.ps1 -UseBasicParsing | Invoke-Expression'
                ]
            else:  # Unix-like
                install_cmd = ['curl', '-fsSL', 'https://ollama.ai/install.sh']
                
            process = await asyncio.create_subprocess_exec(
                *install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Ollama installed successfully")
                
                # Start Ollama service
                await self.start_ollama_service()
                return True
            else:
                logger.error(f"❌ Ollama installation failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to install Ollama: {e}")
            return False
    
    async def start_ollama_service(self):
        """Start Ollama service"""
        try:
            if os.name == 'nt':  # Windows
                cmd = ['ollama.exe', 'serve']
            else:
                cmd = ['ollama', 'serve']
            
            # Start as background process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            # Wait a bit for service to start
            await asyncio.sleep(3)
            logger.info("✅ Ollama service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Ollama service: {e}")
    
    async def install_recommended_models(self, available_space: float):
        """Install recommended models based on available space"""
        logger.info("🤖 Installing recommended coding models...")
        
        # Sort models by priority and size
        models_to_install = []
        total_size = 0
        
        # Always install the lightweight agent model first
        if available_space > 3:
            models_to_install.append('mikepfunk28/deepseekq3_agent:latest')
            total_size += self.recommended_models['mikepfunk28/deepseekq3_agent:latest']['size_gb']
        
        # Add Qwen 2.5 Coder if space allows
        if available_space > 8:
            models_to_install.append('qwen2.5-coder:7b')
            total_size += self.recommended_models['qwen2.5-coder:7b']['size_gb']
        
        # Add DeepSeek Coder if more space
        if available_space > 12:
            models_to_install.append('deepseek-coder:6.7b')
            total_size += self.recommended_models['deepseek-coder:6.7b']['size_gb']
        
        # Add StarCoder2 if plenty of space
        if available_space > 16:
            models_to_install.append('starcoder2:7b')
            total_size += self.recommended_models['starcoder2:7b']['size_gb']
        
        logger.info(f"📊 Installing {len(models_to_install)} models ({total_size:.1f} GB total)")
        
        # Install models
        for model_name in models_to_install:
            await self.install_model(model_name)
    
    async def install_model(self, model_name: str):
        """Install a specific model"""
        model_info = self.recommended_models[model_name]
        logger.info(f"⬇️ Installing {model_name} ({model_info['size_gb']} GB)...")
        
        try:
            cmd = model_info['install_command'].split()
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ {model_name} installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install {model_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing {model_name}: {e}")
            return False
    
    async def setup_lmstudio_integration(self):
        """Setup LM Studio integration"""
        logger.info("🔧 Setting up LM Studio integration...")
        
        # Check if LM Studio is installed
        lmstudio_paths = [
            os.path.expanduser("~/Applications/LM Studio.app"),  # macOS
            "C:\\Users\\%USERNAME%\\AppData\\Local\\LM Studio\\LM Studio.exe",  # Windows
            "/opt/lmstudio/lmstudio",  # Linux
        ]
        
        lmstudio_found = False
        for path in lmstudio_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                lmstudio_found = True
                logger.info(f"✅ LM Studio found at {expanded_path}")
                break
        
        if not lmstudio_found:
            logger.info("💡 LM Studio not found. You can download it from https://lmstudio.ai/")
            logger.info("💡 LM Studio provides a great UI for managing local models")
        
        # Create LM Studio configuration
        await self.create_lmstudio_config()
    
    async def create_lmstudio_config(self):
        """Create LM Studio configuration for AI IDE"""
        config = {
            "ai_ide_integration": True,
            "auto_load_models": True,
            "preferred_models": [
                "qwen2.5-coder:7b",
                "deepseek-coder:6.7b",
                "mikepfunk28/deepseekq3_agent:latest"
            ],
            "server_settings": {
                "host": "127.0.0.1",
                "port": 1234,
                "cors_enabled": True,
                "api_key_required": False
            }
        }
        
        # Save config to user directory
        config_dir = Path.home() / ".ai-ide"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "lmstudio_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ LM Studio config saved to {config_file}")
    
    async def configure_model_routing(self):
        """Configure intelligent model routing"""
        logger.info("🧠 Configuring intelligent model routing...")
        
        routing_config = {
            "model_routing": {
                "code_completion": {
                    "primary": "qwen2.5-coder:7b",
                    "fallback": "deepseek-coder:6.7b",
                    "local_fallback": "mikepfunk28/deepseekq3_agent:latest"
                },
                "code_generation": {
                    "primary": "qwen2.5-coder:7b", 
                    "fallback": "starcoder2:7b",
                    "local_fallback": "deepseek-coder:6.7b"
                },
                "code_explanation": {
                    "primary": "deepseek-coder:6.7b",
                    "fallback": "qwen2.5-coder:7b",
                    "local_fallback": "mikepfunk28/deepseekq3_agent:latest"
                },
                "agent_tasks": {
                    "primary": "mikepfunk28/deepseekq3_agent:latest",
                    "fallback": "qwen2.5-coder:7b",
                    "cloud_fallback": "gpt-4-turbo"
                },
                "web_search": {
                    "primary": "gpt-4-turbo",
                    "fallback": "claude-3-sonnet",
                    "local_fallback": "qwen2.5-coder:7b"
                }
            },
            "performance_optimization": {
                "context_length_limits": {
                    "qwen2.5-coder:7b": 32768,
                    "deepseek-coder:6.7b": 16384,
                    "mikepfunk28/deepseekq3_agent:latest": 8192
                },
                "batch_processing": True,
                "caching_enabled": True,
                "load_balancing": True
            }
        }
        
        # Save routing config
        config_dir = Path.home() / ".ai-ide"
        config_file = config_dir / "model_routing.json"
        
        with open(config_file, 'w') as f:
            json.dump(routing_config, f, indent=2)
        
        logger.info(f"✅ Model routing configured and saved to {config_file}")
    
    async def get_model_recommendations(self) -> Dict:
        """Get personalized model recommendations"""
        return {
            "quick_start": [
                {
                    "name": "mikepfunk28/deepseekq3_agent:latest",
                    "reason": "Lightweight, optimized for AI IDE, quick to download",
                    "size": "2.1 GB",
                    "install_time": "2-5 minutes"
                }
            ],
            "best_coding": [
                {
                    "name": "qwen2.5-coder:7b", 
                    "reason": "State-of-the-art coding model, excellent performance",
                    "size": "4.1 GB",
                    "install_time": "5-10 minutes"
                }
            ],
            "balanced": [
                {
                    "name": "deepseek-coder:6.7b",
                    "reason": "Great balance of size and performance",
                    "size": "3.8 GB", 
                    "install_time": "4-8 minutes"
                }
            ],
            "cloud_alternatives": [
                {
                    "name": "OpenAI GPT-4 Turbo",
                    "reason": "Most capable, requires API key",
                    "cost": "$0.01 per 1K tokens"
                },
                {
                    "name": "Anthropic Claude 3.5 Sonnet",
                    "reason": "Excellent reasoning, requires API key", 
                    "cost": "$0.003 per 1K tokens"
                }
            ]
        }

# Example usage
async def main():
    installer = ModelInstaller()
    await installer.setup_optimal_coding_environment()

if __name__ == "__main__":
    asyncio.run(main())