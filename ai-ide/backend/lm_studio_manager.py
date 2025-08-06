"""
Enhanced LM Studio Manager for AI IDE
Advanced model management, connection pooling, and intelligent model selection
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib

# Import existing LLM utilities
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from utils.call_llm import (
        call_llm, LMSTUDIO_URL, LMSTUDIO_MODELS, 
        check_server_availability, get_available_lmstudio_models
    )
except ImportError:
    # Fallback definitions
    LMSTUDIO_URL = "http://localhost:1234"
    LMSTUDIO_MODELS = {"default": "microsoft/phi-4-reasoning-plus"}
    def call_llm(prompt, **kwargs): return "Mock response"
    def check_server_availability(url): return True
    def get_available_lmstudio_models(): return ["microsoft/phi-4-reasoning-plus"]

logger = logging.getLogger('lm_studio_manager')

class ModelType(Enum):
    """Types of models for different tasks"""
    REASONING = "reasoning"
    INSTRUCT = "instruct"
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    EMBEDDING = "embedding"

@dataclass
class ModelInfo:
    """Information about a model"""
    id: str
    name: str
    type: ModelType
    context_length: int
    parameters: str
    capabilities: List[str]
    performance_score: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0

@dataclass
class ModelRequest:
    """Request for model inference"""
    prompt: str
    model_type: ModelType
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False
    context: Dict[str, Any] = None
    priority: int = 1  # 1=low, 2=medium, 3=high

@dataclass
class ModelResponse:
    """Response from model inference"""
    content: str
    model_id: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ConnectionPool:
    """Manages HTTP connections to LM Studio"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_requests = 0
        self.request_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize the connection pool"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        logger.info(f"Connection pool initialized with {self.max_connections} connections")
    
    async def close(self):
        """Close the connection pool"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a request through the connection pool"""
        if not self.session:
            await self.initialize()
        
        self.active_requests += 1
        try:
            async with self.session.request(method, url, **kwargs) as response:
                return response
        finally:
            self.active_requests -= 1

class ModelPerformanceTracker:
    """Tracks model performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.request_history = []
        
    def record_request(self, model_id: str, response_time: float, success: bool, tokens: int = 0):
        """Record a model request"""
        if model_id not in self.metrics:
            self.metrics[model_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0.0,
                'total_tokens': 0,
                'error_count': 0,
                'last_used': None
            }
        
        metrics = self.metrics[model_id]
        metrics['total_requests'] += 1
        metrics['total_response_time'] += response_time
        metrics['total_tokens'] += tokens
        metrics['last_used'] = datetime.now()
        
        if success:
            metrics['successful_requests'] += 1
        else:
            metrics['error_count'] += 1
        
        # Keep request history for analysis
        self.request_history.append({
            'model_id': model_id,
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': success,
            'tokens': tokens
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model"""
        if model_id not in self.metrics:
            return {}
        
        metrics = self.metrics[model_id]
        total_requests = metrics['total_requests']
        
        if total_requests == 0:
            return {'model_id': model_id, 'no_data': True}
        
        return {
            'model_id': model_id,
            'total_requests': total_requests,
            'success_rate': metrics['successful_requests'] / total_requests,
            'error_rate': metrics['error_count'] / total_requests,
            'average_response_time': metrics['total_response_time'] / total_requests,
            'tokens_per_second': metrics['total_tokens'] / metrics['total_response_time'] if metrics['total_response_time'] > 0 else 0,
            'last_used': metrics['last_used'].isoformat() if metrics['last_used'] else None
        }
    
    def get_best_model(self, model_type: ModelType = None) -> Optional[str]:
        """Get the best performing model for a given type"""
        if not self.metrics:
            return None
        
        best_model = None
        best_score = -1
        
        for model_id, metrics in self.metrics.items():
            if metrics['total_requests'] < 5:  # Need minimum requests for reliable stats
                continue
            
            # Calculate performance score
            success_rate = metrics['successful_requests'] / metrics['total_requests']
            avg_response_time = metrics['total_response_time'] / metrics['total_requests']
            
            # Score: prioritize success rate, penalize slow response times
            score = success_rate * 100 - (avg_response_time * 10)
            
            if score > best_score:
                best_score = score
                best_model = model_id
        
        return best_model

class LMStudioManager:
    """Enhanced LM Studio manager with advanced capabilities"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or LMSTUDIO_URL
        self.connection_pool = ConnectionPool()
        self.performance_tracker = ModelPerformanceTracker()
        self.available_models: Dict[str, ModelInfo] = {}
        self.model_cache = {}
        self.request_queue = asyncio.Queue()
        self.is_initialized = False
        
        # Model type mappings
        self.model_type_mapping = {
            'reasoning': ModelType.REASONING,
            'instruct': ModelType.INSTRUCT,
            'code': ModelType.CODE_GENERATION,
            'chat': ModelType.CHAT
        }
        
    async def initialize(self):
        """Initialize the LM Studio manager"""
        try:
            logger.info("Initializing LM Studio manager...")
            
            # Initialize connection pool
            await self.connection_pool.initialize()
            
            # Check server availability
            if not await self._check_server_health():
                raise ConnectionError("LM Studio server is not available")
            
            # Discover available models
            await self._discover_models()
            
            # Start background tasks
            asyncio.create_task(self._process_request_queue())
            asyncio.create_task(self._periodic_health_check())
            
            self.is_initialized = True
            logger.info(f"LM Studio manager initialized with {len(self.available_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize LM Studio manager: {e}")
            raise
    
    async def _check_server_health(self) -> bool:
        """Check if LM Studio server is healthy"""
        try:
            async with self.connection_pool.session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _discover_models(self):
        """Discover available models from LM Studio"""
        try:
            async with self.connection_pool.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models_data = data.get('data', [])
                    
                    for model_data in models_data:
                        model_id = model_data.get('id', '')
                        
                        # Determine model type based on name
                        model_type = self._classify_model_type(model_id)
                        
                        model_info = ModelInfo(
                            id=model_id,
                            name=model_data.get('object', model_id),
                            type=model_type,
                            context_length=self._estimate_context_length(model_id),
                            parameters=self._estimate_parameters(model_id),
                            capabilities=self._get_model_capabilities(model_id, model_type)
                        )
                        
                        self.available_models[model_id] = model_info
                        logger.info(f"Discovered model: {model_id} ({model_type.value})")
                
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            # Fallback to known models
            self._load_fallback_models()
    
    def _classify_model_type(self, model_id: str) -> ModelType:
        """Classify model type based on model ID"""
        model_id_lower = model_id.lower()
        
        if 'reasoning' in model_id_lower:
            return ModelType.REASONING
        elif 'instruct' in model_id_lower:
            return ModelType.INSTRUCT
        elif 'code' in model_id_lower or 'coder' in model_id_lower:
            return ModelType.CODE_GENERATION
        elif 'chat' in model_id_lower:
            return ModelType.CHAT
        else:
            return ModelType.INSTRUCT  # Default
    
    def _estimate_context_length(self, model_id: str) -> int:
        """Estimate context length based on model ID"""
        # Common context lengths for different models
        if 'phi-4' in model_id.lower():
            return 16384
        elif 'llama' in model_id.lower():
            return 8192
        elif 'mistral' in model_id.lower():
            return 8192
        else:
            return 4096  # Conservative default
    
    def _estimate_parameters(self, model_id: str) -> str:
        """Estimate parameter count based on model ID"""
        model_id_lower = model_id.lower()
        
        if 'phi-4' in model_id_lower:
            return "14B"
        elif '7b' in model_id_lower:
            return "7B"
        elif '13b' in model_id_lower:
            return "13B"
        elif '70b' in model_id_lower:
            return "70B"
        else:
            return "Unknown"
    
    def _get_model_capabilities(self, model_id: str, model_type: ModelType) -> List[str]:
        """Get model capabilities based on type"""
        base_capabilities = ["text_generation", "completion"]
        
        if model_type == ModelType.REASONING:
            return base_capabilities + ["reasoning", "problem_solving", "analysis"]
        elif model_type == ModelType.CODE_GENERATION:
            return base_capabilities + ["code_generation", "code_completion", "debugging"]
        elif model_type == ModelType.CHAT:
            return base_capabilities + ["conversation", "qa", "assistance"]
        else:
            return base_capabilities + ["instruction_following"]
    
    def _load_fallback_models(self):
        """Load fallback models when discovery fails"""
        fallback_models = [
            {
                'id': 'microsoft/phi-4-reasoning-plus',
                'type': ModelType.REASONING,
                'context_length': 16384,
                'parameters': '14B'
            },
            {
                'id': 'phi-4-mini-instruct',
                'type': ModelType.INSTRUCT,
                'context_length': 16384,
                'parameters': '14B'
            }
        ]
        
        for model_data in fallback_models:
            model_info = ModelInfo(
                id=model_data['id'],
                name=model_data['id'],
                type=model_data['type'],
                context_length=model_data['context_length'],
                parameters=model_data['parameters'],
                capabilities=self._get_model_capabilities(model_data['id'], model_data['type'])
            )
            self.available_models[model_data['id']] = model_info
    
    async def _process_request_queue(self):
        """Process queued requests"""
        while True:
            try:
                request = await self.request_queue.get()
                await self._execute_request(request)
                self.request_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing request queue: {e}")
                await asyncio.sleep(1)
    
    async def _periodic_health_check(self):
        """Periodic health check of LM Studio server"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                if not await self._check_server_health():
                    logger.warning("LM Studio server health check failed")
                    # Could implement reconnection logic here
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def select_best_model(self, model_type: ModelType, task_context: Dict[str, Any] = None) -> Optional[str]:
        """Select the best model for a given task"""
        # First, try performance-based selection
        best_model = self.performance_tracker.get_best_model(model_type)
        if best_model and best_model in self.available_models:
            return best_model
        
        # Fallback to type-based selection
        candidates = [
            model_id for model_id, model_info in self.available_models.items()
            if model_info.type == model_type
        ]
        
        if not candidates:
            # Fallback to any available model
            candidates = list(self.available_models.keys())
        
        if not candidates:
            return None
        
        # Select based on context requirements
        if task_context:
            estimated_tokens = task_context.get('estimated_tokens', 0)
            if estimated_tokens > 0:
                # Prefer models with larger context windows for long tasks
                candidates.sort(
                    key=lambda x: self.available_models[x].context_length,
                    reverse=True
                )
        
        return candidates[0]
    
    async def generate_completion(self, request: ModelRequest) -> ModelResponse:
        """Generate completion using the best available model"""
        if not self.is_initialized:
            await self.initialize()
        
        # Select best model
        model_id = self.select_best_model(request.model_type, request.context)
        if not model_id:
            return ModelResponse(
                content="",
                model_id="none",
                tokens_used=0,
                response_time=0,
                success=False,
                error="No suitable model available"
            )
        
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            # Make request
            async with self.connection_pool.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    tokens_used = data.get('usage', {}).get('total_tokens', 0)
                    
                    # Record performance
                    self.performance_tracker.record_request(
                        model_id, response_time, True, tokens_used
                    )
                    
                    return ModelResponse(
                        content=content,
                        model_id=model_id,
                        tokens_used=tokens_used,
                        response_time=response_time,
                        success=True,
                        metadata={
                            'model_info': self.available_models[model_id].__dict__,
                            'request_context': request.context
                        }
                    )
                else:
                    error_text = await response.text()
                    self.performance_tracker.record_request(
                        model_id, response_time, False
                    )
                    
                    return ModelResponse(
                        content="",
                        model_id=model_id,
                        tokens_used=0,
                        response_time=response_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_tracker.record_request(model_id, response_time, False)
            
            return ModelResponse(
                content="",
                model_id=model_id,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """Get information about models"""
        if model_id:
            if model_id in self.available_models:
                model_info = self.available_models[model_id]
                stats = self.performance_tracker.get_model_stats(model_id)
                return {
                    **model_info.__dict__,
                    'performance': stats
                }
            else:
                return {'error': f'Model {model_id} not found'}
        else:
            # Return all models
            return {
                model_id: {
                    **model_info.__dict__,
                    'performance': self.performance_tracker.get_model_stats(model_id)
                }
                for model_id, model_info in self.available_models.items()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        total_requests = sum(
            metrics['total_requests'] 
            for metrics in self.performance_tracker.metrics.values()
        )
        
        if total_requests == 0:
            return {'no_data': True}
        
        successful_requests = sum(
            metrics['successful_requests']
            for metrics in self.performance_tracker.metrics.values()
        )
        
        total_response_time = sum(
            metrics['total_response_time']
            for metrics in self.performance_tracker.metrics.values()
        )
        
        return {
            'total_requests': total_requests,
            'success_rate': successful_requests / total_requests,
            'average_response_time': total_response_time / total_requests,
            'active_models': len(self.available_models),
            'best_model': self.performance_tracker.get_best_model(),
            'connection_pool_active': self.connection_pool.active_requests
        }
    
    async def close(self):
        """Close the LM Studio manager"""
        await self.connection_pool.close()
        self.is_initialized = False

# Global instance
_lm_studio_manager = None

async def get_lm_studio_manager() -> LMStudioManager:
    """Get or create LM Studio manager instance"""
    global _lm_studio_manager
    
    if _lm_studio_manager is None:
        _lm_studio_manager = LMStudioManager()
        await _lm_studio_manager.initialize()
    
    return _lm_studio_manager

# Convenience functions for backward compatibility
async def enhanced_call_llm(
    prompt: str,
    model_type: str = "reasoning",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    context: Dict[str, Any] = None
) -> str:
    """Enhanced LLM call with model management"""
    manager = await get_lm_studio_manager()
    
    # Map string type to enum
    type_mapping = {
        'reasoning': ModelType.REASONING,
        'instruct': ModelType.INSTRUCT,
        'code': ModelType.CODE_GENERATION,
        'chat': ModelType.CHAT
    }
    
    model_type_enum = type_mapping.get(model_type, ModelType.INSTRUCT)
    
    request = ModelRequest(
        prompt=prompt,
        model_type=model_type_enum,
        max_tokens=max_tokens,
        temperature=temperature,
        context=context
    )
    
    response = await manager.generate_completion(request)
    
    if response.success:
        return response.content
    else:
        # Fallback to original call_llm
        logger.warning(f"Enhanced LLM call failed: {response.error}, falling back to original")
        return call_llm(prompt, model_name=model_type)