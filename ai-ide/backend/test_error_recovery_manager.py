"""
Tests for the Error Recovery Manager

This module tests the comprehensive error handling and recovery system.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from error_recovery_manager import (
    ErrorRecoveryManager,
    ErrorClassifier,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryConfig,
    RetryRecoveryStrategy,
    FallbackRecoveryStrategy,
    ErrorContext,
    RecoveryResult,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategyType,
    handle_with_recovery
)


class TestErrorClassifier:
    """Test error classification functionality"""
    
    def setup_method(self):
        self.classifier = ErrorClassifier()
    
    def test_classify_connection_error(self):
        """Test classification of connection errors"""
        error = ConnectionError("Failed to connect to service")
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=ConnectionError,
            error_message=str(error),
            stack_trace="",
            component="ai_model",
            operation="generate"
        )
        
        classification = self.classifier.classify_error(error, context)
        
        assert classification['category'] == ErrorCategory.AI_MODEL_FAILURE
        assert classification['severity'] == ErrorSeverity.HIGH
        assert RecoveryStrategyType.RETRY in classification['strategies']
        assert RecoveryStrategyType.FALLBACK in classification['strategies']
    
    def test_classify_memory_error(self):
        """Test classification of memory errors"""
        error = MemoryError("Out of memory")
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=MemoryError,
            error_message=str(error),
            stack_trace="",
            component="context_manager",
            operation="load_context"
        )
        
        classification = self.classifier.classify_error(error, context)
        
        assert classification['category'] == ErrorCategory.CONTEXT_MANAGEMENT
        assert classification['severity'] == ErrorSeverity.HIGH
        assert RecoveryStrategyType.CONTEXT_COMPRESSION in classification['strategies']
    
    def test_classify_by_message_pattern(self):
        """Test classification based on error message patterns"""
        error = Exception("Web search API request failed")
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message=str(error),
            stack_trace="",
            component="web_search",
            operation="search"
        )
        
        classification = self.classifier.classify_error(error, context)
        
        assert classification['category'] == ErrorCategory.WEB_SEARCH_FAILURE
        assert classification['severity'] == ErrorSeverity.MEDIUM
    
    def test_classify_unknown_error(self):
        """Test classification of unknown errors"""
        error = RuntimeError("Unknown error")
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=RuntimeError,
            error_message=str(error),
            stack_trace="",
            component="unknown",
            operation="unknown"
        )
        
        classification = self.classifier.classify_error(error, context)
        
        assert classification['category'] == ErrorCategory.UNKNOWN
        assert classification['severity'] == ErrorSeverity.MEDIUM


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def setup_method(self):
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2
        )
        self.circuit_breaker = CircuitBreaker("test_service", self.config)
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        def successful_operation():
            return "success"
        
        result = self.circuit_breaker.call(successful_operation)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures"""
        def failing_operation():
            raise Exception("Operation failed")
        
        # Trigger failures to open circuit breaker
        for i in range(self.config.failure_threshold):
            with pytest.raises(Exception):
                self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_prevents_calls_when_open(self):
        """Test circuit breaker prevents calls when open"""
        # Force circuit breaker to open state
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = datetime.now()
        
        def any_operation():
            return "should not execute"
        
        with pytest.raises(Exception, match="Circuit breaker test_service is OPEN"):
            self.circuit_breaker.call(any_operation)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to half-open after timeout"""
        # Set circuit breaker to open state with old failure time
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        def successful_operation():
            return "success"
        
        result = self.circuit_breaker.call(successful_operation)
        assert result == "success"
        # Should transition to half-open, then potentially to closed
    
    def test_circuit_breaker_timeout_handling(self):
        """Test circuit breaker timeout handling"""
        def slow_operation():
            time.sleep(0.1)  # Simulate slow operation
            return "success"
        
        # Set a very short timeout
        self.circuit_breaker.config.timeout = 0.05
        
        with pytest.raises(TimeoutError):
            self.circuit_breaker.call(slow_operation)


class TestRetryRecoveryStrategy:
    """Test retry recovery strategy"""
    
    def setup_method(self):
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,  # Very short delay for testing
            max_delay=0.1
        )
        self.strategy = RetryRecoveryStrategy(self.retry_config)
    
    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test successful retry recovery"""
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message="Test error",
            stack_trace="",
            component="test",
            operation="test_op"
        )
        
        # Mock the retry logic to succeed on first attempt
        with patch.object(self.strategy, 'recover') as mock_recover:
            mock_recover.return_value = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategyType.RETRY,
                recovery_time=0.1
            )
            
            result = await mock_recover(context)
            assert result.success is True
            assert result.strategy_used == RecoveryStrategyType.RETRY
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation with exponential backoff"""
        # Test without jitter for predictable results
        config_no_jitter = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.1,
            jitter=False
        )
        
        delay_0 = config_no_jitter.get_delay(0)
        delay_1 = config_no_jitter.get_delay(1)
        delay_2 = config_no_jitter.get_delay(2)
        
        assert delay_0 == config_no_jitter.base_delay
        assert delay_1 == config_no_jitter.base_delay * 2
        assert delay_2 == config_no_jitter.base_delay * 4
        
        # Test with jitter - should be within expected range
        delay_with_jitter = self.retry_config.get_delay(0)
        expected_min = self.retry_config.base_delay * 0.5
        expected_max = self.retry_config.base_delay
        assert expected_min <= delay_with_jitter <= expected_max
    
    def test_retry_max_delay_limit(self):
        """Test retry delay respects maximum limit"""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, exponential_base=10.0)
        
        # This should exceed max_delay without the limit
        delay = config.get_delay(10)
        assert delay <= config.max_delay


class TestFallbackRecoveryStrategy:
    """Test fallback recovery strategy"""
    
    def setup_method(self):
        self.fallback_handlers = {
            'test_component_test_operation': self.mock_fallback_handler
        }
        self.strategy = FallbackRecoveryStrategy(self.fallback_handlers)
    
    async def mock_fallback_handler(self, error_context):
        """Mock fallback handler"""
        return {"fallback": True, "data": "fallback_data"}
    
    @pytest.mark.asyncio
    async def test_successful_fallback(self):
        """Test successful fallback recovery"""
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message="Test error",
            stack_trace="",
            component="test_component",
            operation="test_operation"
        )
        
        result = await self.strategy.recover(context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategyType.FALLBACK
        assert result.fallback_data == {"fallback": True, "data": "fallback_data"}
    
    @pytest.mark.asyncio
    async def test_fallback_no_handler(self):
        """Test fallback when no handler exists"""
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message="Test error",
            stack_trace="",
            component="unknown_component",
            operation="unknown_operation"
        )
        
        result = await self.strategy.recover(context)
        
        assert result.success is False
        assert "No fallback handler" in result.error_message


class TestErrorRecoveryManager:
    """Test the main error recovery manager"""
    
    def setup_method(self):
        self.manager = ErrorRecoveryManager()
    
    @pytest.mark.asyncio
    async def test_handle_connection_error(self):
        """Test handling of connection errors"""
        error = ConnectionError("Failed to connect")
        
        result = await self.manager.handle_error(
            error=error,
            component="ai_model",
            operation="generate",
            user_context={"user_id": "test_user"},
            system_state={"model": "test_model"}
        )
        
        # Should attempt recovery
        assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_handle_memory_error(self):
        """Test handling of memory errors"""
        error = MemoryError("Out of memory")
        
        result = await self.manager.handle_error(
            error=error,
            component="context_manager",
            operation="load_context"
        )
        
        assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_recovery(self):
        """Test graceful degradation recovery"""
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message="Service unavailable",
            stack_trace="",
            component="ai_model",
            operation="generate"
        )
        
        result = await self.manager._graceful_degradation_recovery(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategyType.GRACEFUL_DEGRADATION
        assert result.fallback_data is not None
        assert result.fallback_data.get("degraded") is True
    
    @pytest.mark.asyncio
    async def test_agent_failover_recovery(self):
        """Test agent failover recovery"""
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_type=Exception,
            error_message="Agent failed",
            stack_trace="",
            component="reasoning_agent",
            operation="reason"
        )
        
        result = await self.manager._agent_failover_recovery(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategyType.AGENT_FAILOVER
        assert "agent" in result.fallback_data
    
    @pytest.mark.asyncio
    async def test_context_compression_recovery(self):
        """Test context compression recovery"""
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_type=MemoryError,
            error_message="Context too large",
            stack_trace="",
            component="context_manager",
            operation="load",
            system_state={"context_size": 1000}
        )
        
        result = await self.manager._context_compression_recovery(error_context)
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategyType.CONTEXT_COMPRESSION
        assert result.context_preserved is False  # Context was modified
        assert result.fallback_data["compressed_size"] < result.fallback_data["original_size"]
    
    def test_circuit_breaker_access(self):
        """Test circuit breaker access"""
        cb = self.manager.get_circuit_breaker("web_search")
        assert cb is not None
        assert cb.name == "web_search"
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        stats = self.manager.get_error_statistics()
        
        assert "total_errors" in stats
        assert "recovery_stats" in stats
        assert "circuit_breaker_states" in stats
        assert "recent_errors" in stats
        
        # Check circuit breaker states
        for service in ["web_search", "lm_studio", "database"]:
            assert service in stats["circuit_breaker_states"]
    
    def test_register_fallback_handler(self):
        """Test registering custom fallback handlers"""
        def custom_handler(error_context):
            return {"custom": True}
        
        self.manager.register_fallback_handler("custom_key", custom_handler)
        
        assert "custom_key" in self.manager.fallback_handlers
        assert self.manager.fallback_handlers["custom_key"] == custom_handler
    
    @pytest.mark.asyncio
    async def test_error_history_tracking(self):
        """Test error history tracking"""
        initial_count = len(self.manager.error_history)
        
        error = ValueError("Test error")
        await self.manager.handle_error(
            error=error,
            component="test",
            operation="test_op"
        )
        
        assert len(self.manager.error_history) == initial_count + 1
        
        latest_error = self.manager.error_history[-1]
        assert latest_error.component == "test"
        assert latest_error.operation == "test_op"
        assert latest_error.error_message == "Test error"


class TestErrorRecoveryDecorator:
    """Test the error recovery decorator"""
    
    @pytest.mark.asyncio
    async def test_async_function_with_recovery(self):
        """Test async function with recovery decorator"""
        
        @handle_with_recovery("test_component", "test_operation")
        async def failing_async_function():
            raise ConnectionError("Connection failed")
        
        # Mock the error recovery manager
        with patch('error_recovery_manager.error_recovery_manager') as mock_manager:
            mock_manager.handle_error = AsyncMock(return_value=RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategyType.FALLBACK,
                fallback_data={"recovered": True}
            ))
            
            result = await failing_async_function()
            assert result == {"recovered": True}
            
            # Verify error handling was called
            mock_manager.handle_error.assert_called_once()
    
    def test_sync_function_with_recovery(self):
        """Test sync function with recovery decorator"""
        
        @handle_with_recovery("test_component", "test_operation")
        def failing_sync_function():
            raise ValueError("Value error")
        
        # Sync functions should re-raise the error after logging
        with pytest.raises(ValueError):
            failing_sync_function()
    
    @pytest.mark.asyncio
    async def test_successful_function_no_recovery(self):
        """Test successful function doesn't trigger recovery"""
        
        @handle_with_recovery("test_component", "test_operation")
        async def successful_function():
            return "success"
        
        result = await successful_function()
        assert result == "success"


class TestErrorContextSerialization:
    """Test error context serialization"""
    
    def test_error_context_to_dict(self):
        """Test error context serialization to dictionary"""
        context = ErrorContext(
            timestamp=datetime.now(),
            error_type=ValueError,
            error_message="Test error",
            stack_trace="test stack trace",
            component="test_component",
            operation="test_operation",
            user_context={"user_id": "123"},
            system_state={"state": "active"},
            previous_errors=["error1", "error2"],
            recovery_attempts=2
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["error_type"] == "ValueError"
        assert context_dict["error_message"] == "Test error"
        assert context_dict["component"] == "test_component"
        assert context_dict["operation"] == "test_operation"
        assert context_dict["user_context"] == {"user_id": "123"}
        assert context_dict["system_state"] == {"state": "active"}
        assert context_dict["previous_errors"] == ["error1", "error2"]
        assert context_dict["recovery_attempts"] == 2
        assert "timestamp" in context_dict
    
    def test_recovery_result_to_dict(self):
        """Test recovery result serialization to dictionary"""
        result = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategyType.RETRY,
            fallback_data={"data": "test"},
            error_message=None,
            recovery_time=1.5,
            context_preserved=True
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["strategy_used"] == "retry"
        assert result_dict["error_message"] is None
        assert result_dict["recovery_time"] == 1.5
        assert result_dict["context_preserved"] is True


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components"""
    
    def setup_method(self):
        self.manager = ErrorRecoveryManager()
    
    @pytest.mark.asyncio
    async def test_web_search_failure_scenario(self):
        """Test complete web search failure scenario"""
        # Simulate web search service failure
        error = ConnectionError("Web search API unavailable")
        
        result = await self.manager.handle_error(
            error=error,
            component="web_search",
            operation="search_query",
            user_context={"query": "test search"},
            system_state={"search_engine": "google"}
        )
        
        # Should attempt recovery and potentially succeed with fallback
        assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_ai_model_cascade_failure(self):
        """Test AI model cascade failure scenario"""
        # Simulate multiple AI model failures
        errors = [
            TimeoutError("Model timeout"),
            ConnectionError("Model connection failed"),
            MemoryError("Model out of memory")
        ]
        
        results = []
        for error in errors:
            result = await self.manager.handle_error(
                error=error,
                component="ai_model",
                operation="generate",
                user_context={"prompt": "test prompt"}
            )
            results.append(result)
        
        # Should handle each error appropriately
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RecoveryResult)
    
    @pytest.mark.asyncio
    async def test_context_memory_pressure_scenario(self):
        """Test context memory pressure scenario"""
        # Simulate increasing memory pressure
        for context_size in [1000, 5000, 10000, 50000]:
            error = MemoryError(f"Context size {context_size} too large")
            
            result = await self.manager.handle_error(
                error=error,
                component="context_manager",
                operation="load_context",
                system_state={"context_size": context_size}
            )
            
            # Should handle with context compression
            assert isinstance(result, RecoveryResult)
            if result.success and result.fallback_data:
                assert "compressed_size" in result.fallback_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])