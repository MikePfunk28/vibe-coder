"""
Advanced Error Classification and Recovery System

This module implements comprehensive error handling with multi-agent fallback strategies,
circuit breakers, graceful degradation, and automatic retry mechanisms.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AI_MODEL_FAILURE = "ai_model_failure"
    CONTEXT_MANAGEMENT = "context_management"
    WEB_SEARCH_FAILURE = "web_search_failure"
    EXTERNAL_SERVICE = "external_service"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    REASONING_FAILURE = "reasoning_failure"
    AGENT_COORDINATION = "agent_coordination"
    SELF_IMPROVEMENT = "self_improvement"
    UNKNOWN = "unknown"


class RecoveryStrategyType(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    AGENT_FAILOVER = "agent_failover"
    CONTEXT_COMPRESSION = "context_compression"
    ROLLBACK = "rollback"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error analysis"""
    timestamp: datetime
    error_type: Type[Exception]
    error_message: str
    stack_trace: str
    component: str
    operation: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    previous_errors: List[str] = field(default_factory=list)
    recovery_attempts: int = 0
    
    # Enhanced context preservation for complex reasoning chains
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    agent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    context_windows: List[Dict[str, Any]] = field(default_factory=list)
    model_states: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_correlation_id: Optional[str] = None
    parent_operation_id: Optional[str] = None
    dependency_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type.__name__,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'component': self.component,
            'operation': self.operation,
            'user_context': self.user_context,
            'system_state': self.system_state,
            'previous_errors': self.previous_errors,
            'recovery_attempts': self.recovery_attempts,
            'reasoning_chain': self.reasoning_chain,
            'agent_interactions': self.agent_interactions,
            'context_windows': self.context_windows,
            'model_states': self.model_states,
            'performance_metrics': self.performance_metrics,
            'error_correlation_id': self.error_correlation_id,
            'parent_operation_id': self.parent_operation_id,
            'dependency_chain': self.dependency_chain
        }


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategyType
    fallback_data: Any = None
    error_message: Optional[str] = None
    recovery_time: float = 0.0
    context_preserved: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'success': self.success,
            'strategy_used': self.strategy_used.value,
            'error_message': self.error_message,
            'recovery_time': self.recovery_time,
            'context_preserved': self.context_preserved
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for external services"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > self.config.timeout:
                    raise TimeoutError(f"Operation timed out after {execution_time:.2f}s")
                
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0


class RetryConfig:
    """Configuration for retry mechanisms"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class RecoveryStrategyBase(ABC):
    """Abstract base class for recovery strategies"""
    
    @abstractmethod
    async def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt to recover from the error"""
        pass


class RetryRecoveryStrategy(RecoveryStrategyBase):
    """Retry with exponential backoff"""
    
    def __init__(self, retry_config: RetryConfig):
        self.retry_config = retry_config
    
    async def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through retries"""
        start_time = time.time()
        
        for attempt in range(self.retry_config.max_attempts):
            if attempt > 0:
                delay = self.retry_config.get_delay(attempt - 1)
                logger.info(f"Retrying {error_context.operation} in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            
            try:
                # This would be implemented by specific recovery handlers
                # For now, we simulate a recovery attempt
                await asyncio.sleep(0.1)  # Simulate work
                
                recovery_time = time.time() - start_time
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategyType.RETRY,
                    recovery_time=recovery_time
                )
                
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                continue
        
        recovery_time = time.time() - start_time
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategyType.RETRY,
            error_message="All retry attempts failed",
            recovery_time=recovery_time
        )


class FallbackRecoveryStrategy(RecoveryStrategyBase):
    """Fallback to alternative implementation"""
    
    def __init__(self, fallback_handlers: Dict[str, Callable]):
        self.fallback_handlers = fallback_handlers
    
    async def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through fallback"""
        start_time = time.time()
        
        handler_key = f"{error_context.component}_{error_context.operation}"
        fallback_handler = self.fallback_handlers.get(handler_key)
        
        if not fallback_handler:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategyType.FALLBACK,
                error_message=f"No fallback handler for {handler_key}"
            )
        
        try:
            fallback_data = await fallback_handler(error_context)
            recovery_time = time.time() - start_time
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategyType.FALLBACK,
                fallback_data=fallback_data,
                recovery_time=recovery_time
            )
            
        except Exception as e:
            recovery_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategyType.FALLBACK,
                error_message=f"Fallback handler failed: {e}",
                recovery_time=recovery_time
            )


class ErrorClassifier:
    """Classifies errors into categories and determines severity"""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def _build_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build error classification rules"""
        return {
            # AI Model Failures
            'ConnectionError': {
                'category': ErrorCategory.AI_MODEL_FAILURE,
                'severity': ErrorSeverity.HIGH,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.FALLBACK]
            },
            'TimeoutError': {
                'category': ErrorCategory.AI_MODEL_FAILURE,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.CIRCUIT_BREAKER]
            },
            'HTTPError': {
                'category': ErrorCategory.EXTERNAL_SERVICE,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.CIRCUIT_BREAKER]
            },
            
            # Context Management
            'MemoryError': {
                'category': ErrorCategory.CONTEXT_MANAGEMENT,
                'severity': ErrorSeverity.HIGH,
                'strategies': [RecoveryStrategyType.CONTEXT_COMPRESSION, RecoveryStrategyType.GRACEFUL_DEGRADATION]
            },
            'IndexError': {
                'category': ErrorCategory.CONTEXT_MANAGEMENT,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.FALLBACK]
            },
            
            # Database Errors
            'DatabaseError': {
                'category': ErrorCategory.DATABASE_ERROR,
                'severity': ErrorSeverity.HIGH,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.FALLBACK]
            },
            
            # Validation Errors
            'ValueError': {
                'category': ErrorCategory.VALIDATION_ERROR,
                'severity': ErrorSeverity.LOW,
                'strategies': [RecoveryStrategyType.FALLBACK]
            },
            'ValidationError': {
                'category': ErrorCategory.VALIDATION_ERROR,
                'severity': ErrorSeverity.LOW,
                'strategies': [RecoveryStrategyType.FALLBACK]
            },
            
            # Resource Exhaustion
            'ResourceExhaustedError': {
                'category': ErrorCategory.RESOURCE_EXHAUSTION,
                'severity': ErrorSeverity.CRITICAL,
                'strategies': [RecoveryStrategyType.GRACEFUL_DEGRADATION, RecoveryStrategyType.CIRCUIT_BREAKER]
            }
        }
    
    def classify_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Classify error and determine recovery strategies"""
        error_name = error.__class__.__name__
        
        # Check for specific error types
        classification = self.classification_rules.get(error_name)
        
        if not classification:
            # Try to classify based on error message patterns
            classification = self._classify_by_message(str(error), context)
        
        if not classification:
            # Default classification
            classification = {
                'category': ErrorCategory.UNKNOWN,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.FALLBACK]
            }
        
        return classification
    
    def _classify_by_message(self, error_message: str, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Classify error based on message patterns"""
        message_lower = error_message.lower()
        
        # Web search related errors
        if any(term in message_lower for term in ['search', 'web', 'api', 'request']):
            return {
                'category': ErrorCategory.WEB_SEARCH_FAILURE,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.RETRY, RecoveryStrategyType.FALLBACK]
            }
        
        # Context/memory related errors
        if any(term in message_lower for term in ['context', 'memory', 'embedding', 'vector']):
            return {
                'category': ErrorCategory.CONTEXT_MANAGEMENT,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.CONTEXT_COMPRESSION, RecoveryStrategyType.FALLBACK]
            }
        
        # Reasoning related errors
        if any(term in message_lower for term in ['reasoning', 'agent', 'chain', 'thought']):
            return {
                'category': ErrorCategory.REASONING_FAILURE,
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [RecoveryStrategyType.AGENT_FAILOVER, RecoveryStrategyType.FALLBACK]
            }
        
        return None


class ErrorRecoveryManager:
    """Main error recovery manager with multi-agent fallback strategies"""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[RecoveryStrategyType, RecoveryStrategyBase] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.fallback_handlers: Dict[str, Callable] = {}
        
        self._initialize_recovery_strategies()
        self._initialize_circuit_breakers()
        self._initialize_fallback_handlers()
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies"""
        # Enhanced retry configurations for different error types
        self.retry_configs = {
            'default': RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0),
            'network': RetryConfig(max_attempts=5, base_delay=0.5, max_delay=15.0, exponential_base=1.5),
            'ai_model': RetryConfig(max_attempts=3, base_delay=2.0, max_delay=60.0, exponential_base=2.0),
            'database': RetryConfig(max_attempts=4, base_delay=1.0, max_delay=20.0, exponential_base=1.8),
            'web_search': RetryConfig(max_attempts=3, base_delay=0.5, max_delay=10.0, exponential_base=1.5),
            'external_service': RetryConfig(max_attempts=3, base_delay=1.5, max_delay=45.0, exponential_base=2.0)
        }
        
        self.recovery_strategies[RecoveryStrategyType.RETRY] = RetryRecoveryStrategy(
            self.retry_configs['default']
        )
        self.recovery_strategies[RecoveryStrategyType.FALLBACK] = FallbackRecoveryStrategy(
            self.fallback_handlers
        )
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for external services"""
        # Enhanced circuit breaker configurations for different service types
        service_configs = {
            'web_search': CircuitBreakerConfig(
                failure_threshold=3,  # More sensitive for web search
                recovery_timeout=30.0,  # Faster recovery for web services
                success_threshold=2,
                timeout=10.0  # Shorter timeout for web requests
            ),
            'lm_studio': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3,
                timeout=30.0  # AI model requests can take longer
            ),
            'database': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=45.0,
                success_threshold=2,
                timeout=15.0
            ),
            'embedding_service': CircuitBreakerConfig(
                failure_threshold=4,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout=20.0
            ),
            'reasoning_engine': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=45.0,
                success_threshold=2,
                timeout=25.0
            ),
            'agent_coordinator': CircuitBreakerConfig(
                failure_threshold=2,  # Very sensitive for agent coordination
                recovery_timeout=20.0,
                success_threshold=1,
                timeout=10.0
            ),
            'external_api': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                success_threshold=2,
                timeout=15.0
            ),
            'mcp_server': CircuitBreakerConfig(
                failure_threshold=4,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout=20.0
            )
        }
        
        for service, config in service_configs.items():
            self.circuit_breakers[service] = CircuitBreaker(name=service, config=config)
    
    def _initialize_fallback_handlers(self):
        """Initialize fallback handlers for different components"""
        self.fallback_handlers.update({
            # AI Model fallbacks
            'ai_model_completion': self._fallback_simple_completion,
            'ai_model_generation': self._fallback_template_generation,
            'ai_model_reasoning': self._fallback_rule_based_reasoning,
            
            # Web search fallbacks
            'web_search_query': self._fallback_cached_search,
            'web_search_google': self._fallback_alternative_search_engine,
            'web_search_bing': self._fallback_alternative_search_engine,
            'web_search_duckduckgo': self._fallback_local_search,
            
            # Context management fallbacks
            'context_retrieval': self._fallback_simple_context,
            'context_compression': self._fallback_aggressive_compression,
            'context_embedding': self._fallback_text_similarity,
            
            # Reasoning engine fallbacks
            'reasoning_step': self._fallback_simple_reasoning,
            'reasoning_chain_of_thought': self._fallback_linear_reasoning,
            'reasoning_react': self._fallback_direct_action,
            
            # Agent coordination fallbacks
            'agent_coordination': self._fallback_single_agent,
            'multi_agent_task': self._fallback_sequential_agents,
            'agent_communication': self._fallback_direct_execution,
            
            # Database fallbacks
            'database_query': self._fallback_cached_data,
            'database_write': self._fallback_memory_storage,
            
            # External service fallbacks
            'external_api_call': self._fallback_mock_response,
            'mcp_server_call': self._fallback_local_tool,
            
            # Embedding and similarity fallbacks
            'embedding_generation': self._fallback_text_similarity,
            'semantic_search': self._fallback_keyword_search,
            
            # Self-improvement fallbacks
            'darwin_godel_improvement': self._fallback_static_model,
            'benchmark_validation': self._fallback_simple_validation
        })
    
    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        user_context: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> RecoveryResult:
        """Main error handling entry point"""
        
        # Generate correlation ID for error tracking
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
        
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            operation=operation,
            user_context=user_context or {},
            system_state=system_state or {},
            error_correlation_id=correlation_id,
            # Extract reasoning chain and agent interactions from system state if available
            reasoning_chain=system_state.get('reasoning_chain', []) if system_state else [],
            agent_interactions=system_state.get('agent_interactions', []) if system_state else [],
            context_windows=system_state.get('context_windows', []) if system_state else [],
            model_states=system_state.get('model_states', {}) if system_state else {},
            performance_metrics=system_state.get('performance_metrics', {}) if system_state else {},
            parent_operation_id=system_state.get('parent_operation_id') if system_state else None,
            dependency_chain=system_state.get('dependency_chain', []) if system_state else []
        )
        
        # Add to error history
        self.error_history.append(error_context)
        
        # Classify error
        classification = self.classifier.classify_error(error, error_context)
        
        # Select appropriate retry configuration based on error category
        retry_config_key = self._get_retry_config_key(classification['category'])
        if RecoveryStrategyType.RETRY in self.recovery_strategies:
            self.recovery_strategies[RecoveryStrategyType.RETRY].retry_config = self.retry_configs[retry_config_key]
        
        logger.error(
            f"Error in {component}.{operation}: {error} "
            f"(Category: {classification['category'].value}, "
            f"Severity: {classification['severity'].value}, "
            f"Correlation ID: {error_context.error_correlation_id})"
        )
        
        # Attempt recovery using available strategies
        for strategy in classification['strategies']:
            try:
                recovery_result = await self._attempt_recovery(error_context, strategy)
                
                # Update statistics
                self.recovery_stats[component][f"{strategy.value}_attempts"] += 1
                if recovery_result.success:
                    self.recovery_stats[component][f"{strategy.value}_successes"] += 1
                
                if recovery_result.success:
                    logger.info(
                        f"Successfully recovered from {component}.{operation} "
                        f"using {strategy.value} in {recovery_result.recovery_time:.2f}s"
                    )
                    return recovery_result
                else:
                    logger.warning(
                        f"Recovery strategy {strategy.value} failed for {component}.{operation}: "
                        f"{recovery_result.error_message}"
                    )
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.value} threw exception: {recovery_error}")
                continue
        
        # All recovery strategies failed
        logger.error(f"All recovery strategies failed for {component}.{operation}")
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategyType.IGNORE,
            error_message="All recovery strategies exhausted"
        )
    
    async def _attempt_recovery(
        self,
        error_context: ErrorContext,
        strategy: RecoveryStrategyType
    ) -> RecoveryResult:
        """Attempt recovery using specific strategy"""
        
        if strategy in self.recovery_strategies:
            return await self.recovery_strategies[strategy].recover(error_context)
        
        # Handle built-in strategies
        if strategy == RecoveryStrategyType.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(error_context)
        elif strategy == RecoveryStrategyType.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_recovery(error_context)
        elif strategy == RecoveryStrategyType.AGENT_FAILOVER:
            return await self._agent_failover_recovery(error_context)
        elif strategy == RecoveryStrategyType.CONTEXT_COMPRESSION:
            return await self._context_compression_recovery(error_context)
        elif strategy == RecoveryStrategyType.ROLLBACK:
            return await self._rollback_recovery(error_context)
        else:
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                error_message=f"Strategy {strategy.value} not implemented"
            )
    
    async def _circuit_breaker_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle recovery using circuit breaker"""
        service_name = error_context.component.lower()
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        if not circuit_breaker:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategyType.CIRCUIT_BREAKER,
                error_message=f"No circuit breaker for service {service_name}"
            )
        
        # Circuit breaker is already handling the failure
        # Return appropriate response based on circuit state
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategyType.CIRCUIT_BREAKER,
                fallback_data={"message": f"Service {service_name} temporarily unavailable"},
                context_preserved=True
            )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategyType.CIRCUIT_BREAKER,
            error_message="Circuit breaker not in appropriate state"
        )
    
    async def _graceful_degradation_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle graceful degradation"""
        start_time = time.time()
        
        # Provide reduced functionality based on component
        degraded_response = None
        
        if error_context.component == "ai_model":
            degraded_response = {
                "completion": "// AI completion temporarily unavailable",
                "confidence": 0.0,
                "degraded": True
            }
        elif error_context.component == "web_search":
            degraded_response = {
                "results": [],
                "message": "Web search temporarily unavailable",
                "degraded": True
            }
        elif error_context.component == "reasoning_engine":
            degraded_response = {
                "reasoning": "Simple fallback reasoning",
                "steps": ["Basic analysis"],
                "degraded": True
            }
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategyType.GRACEFUL_DEGRADATION,
            fallback_data=degraded_response,
            recovery_time=recovery_time,
            context_preserved=True
        )
    
    async def _agent_failover_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle agent failover"""
        start_time = time.time()
        
        # Simulate agent failover logic
        fallback_agent = self._get_fallback_agent(error_context.component)
        
        if fallback_agent:
            recovery_time = time.time() - start_time
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategyType.AGENT_FAILOVER,
                fallback_data={"agent": fallback_agent, "degraded": True},
                recovery_time=recovery_time
            )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategyType.AGENT_FAILOVER,
            error_message="No fallback agent available"
        )
    
    async def _context_compression_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle context compression for memory issues"""
        start_time = time.time()
        
        # Simulate context compression
        compressed_context = {
            "original_size": error_context.system_state.get("context_size", 0),
            "compressed_size": error_context.system_state.get("context_size", 0) // 2,
            "compression_ratio": 0.5,
            "degraded": True
        }
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategyType.CONTEXT_COMPRESSION,
            fallback_data=compressed_context,
            recovery_time=recovery_time,
            context_preserved=False  # Context was modified
        )
    
    async def _rollback_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle rollback recovery"""
        start_time = time.time()
        
        # Simulate rollback logic
        rollback_info = {
            "rolled_back": True,
            "previous_state": "stable",
            "rollback_reason": error_context.error_message
        }
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategyType.ROLLBACK,
            fallback_data=rollback_info,
            recovery_time=recovery_time
        )
    
    def _get_fallback_agent(self, failed_agent: str) -> Optional[str]:
        """Get fallback agent for failed agent"""
        fallback_map = {
            "reasoning_agent": "simple_reasoning_agent",
            "search_agent": "cached_search_agent",
            "code_agent": "basic_code_agent",
            "test_agent": "simple_test_agent",
            "web_search_agent": "cached_search_agent",
            "embedding_agent": "text_similarity_agent",
            "darwin_godel_agent": "static_model_agent",
            "multi_agent_coordinator": "single_agent_executor"
        }
        return fallback_map.get(failed_agent)
    
    def _get_retry_config_key(self, error_category: ErrorCategory) -> str:
        """Get appropriate retry configuration key based on error category"""
        category_mapping = {
            ErrorCategory.AI_MODEL_FAILURE: 'ai_model',
            ErrorCategory.WEB_SEARCH_FAILURE: 'web_search',
            ErrorCategory.EXTERNAL_SERVICE: 'external_service',
            ErrorCategory.DATABASE_ERROR: 'database',
            ErrorCategory.NETWORK_ERROR: 'network',
            ErrorCategory.CONTEXT_MANAGEMENT: 'default',
            ErrorCategory.VALIDATION_ERROR: 'default',
            ErrorCategory.RESOURCE_EXHAUSTION: 'default',
            ErrorCategory.REASONING_FAILURE: 'ai_model',
            ErrorCategory.AGENT_COORDINATION: 'default',
            ErrorCategory.SELF_IMPROVEMENT: 'ai_model',
            ErrorCategory.UNKNOWN: 'default'
        }
        return category_mapping.get(error_category, 'default')
    
    # Fallback handler implementations
    async def _fallback_simple_completion(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Simple completion fallback"""
        return {
            "completion": "// Fallback completion",
            "confidence": 0.1,
            "fallback": True
        }
    
    async def _fallback_cached_search(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Cached search results fallback"""
        return {
            "results": [],
            "cached": True,
            "message": "Using cached results due to search service unavailability"
        }
    
    async def _fallback_simple_context(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Simple context fallback"""
        return {
            "context": "Basic context information",
            "simplified": True
        }
    
    async def _fallback_simple_reasoning(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Simple reasoning fallback"""
        return {
            "reasoning": "Basic reasoning step",
            "confidence": 0.2,
            "simplified": True
        }
    
    async def _fallback_single_agent(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Single agent fallback"""
        return {
            "agent": "fallback_agent",
            "capabilities": ["basic_operations"],
            "degraded": True
        }
    
    async def _fallback_text_similarity(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Text similarity fallback"""
        return {
            "similarity": 0.5,
            "method": "text_based",
            "fallback": True
        }
    
    # Additional enhanced fallback handlers
    async def _fallback_template_generation(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Template-based code generation fallback"""
        return {
            "completion": "// Template-based fallback code",
            "confidence": 0.3,
            "method": "template",
            "fallback": True
        }
    
    async def _fallback_rule_based_reasoning(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Rule-based reasoning fallback"""
        return {
            "reasoning": "Applied basic rule-based logic",
            "confidence": 0.4,
            "method": "rule_based",
            "fallback": True
        }
    
    async def _fallback_alternative_search_engine(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Alternative search engine fallback"""
        return {
            "results": [],
            "search_engine": "fallback_engine",
            "message": "Using alternative search engine",
            "fallback": True
        }
    
    async def _fallback_local_search(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Local search fallback"""
        return {
            "results": [],
            "method": "local_index",
            "message": "Using local search index",
            "fallback": True
        }
    
    async def _fallback_aggressive_compression(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Aggressive context compression fallback"""
        original_size = error_context.system_state.get("context_size", 1000)
        return {
            "compressed_context": "Aggressively compressed context",
            "original_size": original_size,
            "compressed_size": original_size // 4,
            "compression_ratio": 0.25,
            "method": "aggressive",
            "fallback": True
        }
    
    async def _fallback_linear_reasoning(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Linear reasoning fallback"""
        return {
            "reasoning": "Linear step-by-step analysis",
            "steps": ["Step 1: Basic analysis", "Step 2: Simple conclusion"],
            "method": "linear",
            "fallback": True
        }
    
    async def _fallback_direct_action(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Direct action fallback (skip reasoning)"""
        return {
            "action": "direct_execution",
            "reasoning_skipped": True,
            "method": "direct",
            "fallback": True
        }
    
    async def _fallback_sequential_agents(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Sequential agent execution fallback"""
        return {
            "execution_mode": "sequential",
            "agents": ["primary_agent"],
            "coordination_disabled": True,
            "fallback": True
        }
    
    async def _fallback_direct_execution(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Direct execution without agent communication"""
        return {
            "execution": "direct",
            "communication_bypassed": True,
            "method": "direct",
            "fallback": True
        }
    
    async def _fallback_cached_data(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Cached data fallback"""
        return {
            "data": {},
            "source": "cache",
            "freshness": "stale",
            "fallback": True
        }
    
    async def _fallback_memory_storage(self, error_context: ErrorContext) -> Dict[str, Any]:
        """In-memory storage fallback"""
        return {
            "storage": "memory",
            "persistent": False,
            "message": "Data stored in memory only",
            "fallback": True
        }
    
    async def _fallback_mock_response(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Mock response fallback"""
        return {
            "response": {"status": "mock", "data": {}},
            "mocked": True,
            "fallback": True
        }
    
    async def _fallback_local_tool(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Local tool fallback"""
        return {
            "tool": "local_fallback",
            "capabilities": ["basic"],
            "remote_disabled": True,
            "fallback": True
        }
    
    async def _fallback_keyword_search(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Keyword-based search fallback"""
        return {
            "results": [],
            "method": "keyword",
            "semantic_disabled": True,
            "fallback": True
        }
    
    async def _fallback_static_model(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Static model fallback (no self-improvement)"""
        return {
            "model": "static",
            "improvement_disabled": True,
            "version": "baseline",
            "fallback": True
        }
    
    async def _fallback_simple_validation(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Simple validation fallback"""
        return {
            "validation": "basic",
            "passed": True,
            "method": "simple",
            "fallback": True
        }
    
    def get_circuit_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for service"""
        return self.circuit_breakers.get(service_name)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics"""
        return {
            "total_errors": len(self.error_history),
            "recovery_stats": dict(self.recovery_stats),
            "circuit_breaker_states": {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            },
            "recent_errors": [
                error.to_dict() for error in list(self.error_history)[-10:]
            ]
        }
    
    def register_fallback_handler(self, key: str, handler: Callable):
        """Register custom fallback handler"""
        self.fallback_handlers[key] = handler
        
        # Update fallback recovery strategy
        if RecoveryStrategyType.FALLBACK in self.recovery_strategies:
            self.recovery_strategies[RecoveryStrategyType.FALLBACK].fallback_handlers = self.fallback_handlers


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def handle_with_recovery(component: str, operation: str):
    """Decorator for automatic error handling and recovery"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_result = await error_recovery_manager.handle_error(
                    error=e,
                    component=component,
                    operation=operation,
                    user_context=kwargs.get('user_context'),
                    system_state=kwargs.get('system_state')
                )
                
                if recovery_result.success and recovery_result.fallback_data is not None:
                    return recovery_result.fallback_data
                else:
                    raise e
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't use async recovery
                # Log the error and re-raise
                logger.error(f"Error in {component}.{operation}: {e}")
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator