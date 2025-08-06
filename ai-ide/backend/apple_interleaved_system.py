"""
Apple Interleaved System Integration

Combines the InterleaveContextManager and InterleaveReasoningEngine
to provide a complete Apple-inspired interleaved context sliding windows
and reasoning system for the AI IDE.
"""

import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
import numpy as np

from interleaved_context_manager import InterleaveContextManager
from interleaved_reasoning_engine import InterleaveReasoningEngine, ReasoningMode

logger = logging.getLogger(__name__)


@dataclass
class AppleInterleaveConfig:
    """Configuration for the Apple Interleaved System."""
    # Context Manager Settings
    max_context_length: int = 32768
    similarity_threshold: float = 0.7
    max_windows: int = 50
    compression_enabled: bool = True
    
    # Reasoning Engine Settings
    max_reasoning_steps: int = 10
    ttft_target_ms: int = 200
    enable_progressive: bool = True
    
    # Integration Settings
    auto_context_refresh: bool = True
    context_relevance_feedback: bool = True
    adaptive_reasoning_mode: bool = True


class AppleInterleaveSystem:
    """
    Integrated Apple Interleaved System combining context management
    and reasoning patterns for optimal AI-powered code assistance.
    
    This system implements Apple's research on:
    - Interleaved context sliding windows for efficient memory management
    - Think-answer interleaving for better reasoning quality
    - TTFT optimization for responsive user experience
    - Progressive response generation with intermediate signals
    """
    
    def __init__(
        self,
        llm_client,
        config: Optional[AppleInterleaveConfig] = None
    ):
        self.config = config or AppleInterleaveConfig()
        self.llm_client = llm_client
        
        # Initialize context manager
        self.context_manager = InterleaveContextManager(
            max_context_length=self.config.max_context_length,
            similarity_threshold=self.config.similarity_threshold,
            max_windows=self.config.max_windows,
            compression_enabled=self.config.compression_enabled
        )
        
        # Initialize reasoning engine
        self.reasoning_engine = InterleaveReasoningEngine(
            context_manager=self.context_manager,
            llm_client=llm_client,
            max_reasoning_steps=self.config.max_reasoning_steps,
            ttft_target_ms=self.config.ttft_target_ms,
            enable_progressive=self.config.enable_progressive
        )
        
        # System state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics = {
            'total_requests': 0,
            'context_hits': 0,
            'reasoning_improvements': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("AppleInterleaveSystem initialized with advanced context and reasoning capabilities")
    
    async def process_code_assistance_request(
        self,
        query: str,
        code_context: Dict[str, Any],
        session_id: Optional[str] = None,
        reasoning_mode: Optional[ReasoningMode] = None,
        stream_response: bool = True
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Process a code assistance request using the integrated system.
        
        Args:
            query: The user's code assistance query
            code_context: Current code context (file content, cursor position, etc.)
            session_id: Optional session identifier for context continuity
            reasoning_mode: Specific reasoning mode to use
            stream_response: Whether to stream the response progressively
            
        Returns:
            Complete response or async generator for streaming
        """
        self.performance_metrics['total_requests'] += 1
        
        # Add current code context to context manager
        await self._update_context_from_code(code_context)
        
        # Determine optimal reasoning mode if not specified
        if reasoning_mode is None and self.config.adaptive_reasoning_mode:
            reasoning_mode = self._determine_optimal_reasoning_mode(query, code_context)
        elif reasoning_mode is None:
            reasoning_mode = ReasoningMode.BALANCED
        
        # Process the request through the reasoning engine
        response = await self.reasoning_engine.reason_and_respond(
            query=query,
            context=code_context,
            mode=reasoning_mode,
            stream=stream_response
        )
        
        # Update context relevance based on usage if enabled
        if self.config.context_relevance_feedback:
            await self._update_context_relevance_feedback(query, response)
        
        # Maintain session state if provided
        if session_id:
            await self._update_session_state(session_id, query, response, reasoning_mode)
        
        return response
    
    async def add_code_context(
        self,
        content: str,
        context_type: str = "code",
        source_file: Optional[str] = None,
        line_range: Optional[tuple] = None,
        priority: int = 1,
        semantic_embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Add code context to the interleaved context management system.
        
        Args:
            content: The code content
            context_type: Type of context ('code', 'documentation', etc.)
            source_file: Source file path
            line_range: Line range in source file
            priority: Priority level (1-5)
            semantic_embedding: Pre-computed semantic embedding
            
        Returns:
            Context ID
        """
        context_id = self.context_manager.add_context(
            content=content,
            context_type=context_type,
            source_file=source_file,
            line_range=line_range,
            priority=priority,
            semantic_embedding=semantic_embedding
        )
        
        logger.debug(f"Added code context {context_id} from {source_file}")
        return context_id
    
    async def search_semantic_context(
        self,
        query: str,
        context_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically relevant context.
        
        Args:
            query: Search query
            context_types: Filter by context types
            max_results: Maximum number of results
            
        Returns:
            List of relevant context windows with metadata
        """
        # Get query embedding if available
        query_embedding = None
        try:
            # This would typically use the same embedding model as the context manager
            # For now, we'll let the context manager handle it
            pass
        except Exception as e:
            logger.warning(f"Could not generate query embedding: {e}")
        
        # Search using context manager
        relevant_windows = self.context_manager.get_relevant_context(
            query=query,
            query_embedding=query_embedding,
            max_tokens=max_results * 200,  # Rough estimate
            context_types=context_types
        )
        
        # Convert to dictionary format with metadata
        results = []
        for window in relevant_windows[:max_results]:
            result = {
                'id': window.id,
                'content': window.content,
                'relevance_score': window.relevance_score,
                'context_type': window.window_type,
                'source_file': window.source_file,
                'line_range': window.line_range,
                'last_accessed': window.last_accessed.isoformat(),
                'tokens': window.tokens
            }
            results.append(result)
        
        self.performance_metrics['context_hits'] += len(results)
        return results
    
    def update_context_relevance(
        self,
        context_id: str,
        relevance_delta: float,
        feedback_type: str = "user"
    ) -> None:
        """
        Update context relevance based on feedback.
        
        Args:
            context_id: Context window ID
            relevance_delta: Change in relevance (-1.0 to 1.0)
            feedback_type: Type of feedback ('user', 'system', 'implicit')
        """
        self.context_manager.update_context_relevance(context_id, relevance_delta)
        logger.debug(f"Updated context {context_id} relevance by {relevance_delta} ({feedback_type})")
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """
        Optimize system performance by reorganizing context and analyzing patterns.
        
        Returns:
            Optimization results and statistics
        """
        # Optimize context windows
        self.context_manager.optimize_windows()
        
        # Get performance statistics
        context_stats = self.context_manager.get_context_stats()
        reasoning_stats = self.reasoning_engine.get_performance_stats()
        
        # Combine statistics
        optimization_results = {
            'context_optimization': {
                'windows_optimized': context_stats['total_windows'],
                'memory_utilization': context_stats['total_tokens'] / context_stats['max_tokens'],
                'cache_hit_rate': context_stats['cache_hit_rate'],
                'compressions': context_stats['compressions'],
                'evictions': context_stats['evictions']
            },
            'reasoning_optimization': {
                'total_queries': reasoning_stats['total_queries'],
                'avg_ttft': reasoning_stats['avg_ttft'],
                'avg_reasoning_time': reasoning_stats['avg_reasoning_time'],
                'avg_confidence': reasoning_stats.get('avg_confidence', 0.0),
                'mode_usage': reasoning_stats['mode_usage']
            },
            'system_metrics': self.performance_metrics.copy()
        }
        
        logger.info("System performance optimization completed")
        return optimization_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and health metrics.
        
        Returns:
            System status information
        """
        context_stats = self.context_manager.get_context_stats()
        reasoning_stats = self.reasoning_engine.get_performance_stats()
        
        return {
            'status': 'healthy',
            'context_manager': {
                'total_windows': context_stats['total_windows'],
                'memory_usage': f"{context_stats['total_tokens']}/{context_stats['max_tokens']} tokens",
                'cache_performance': f"{context_stats['cache_hit_rate']:.2%}",
                'active_slots': {
                    slot_name: {
                        'windows': slot_info['windows'],
                        'utilization': f"{slot_info['utilization']:.1%}"
                    }
                    for slot_name, slot_info in context_stats['slots'].items()
                }
            },
            'reasoning_engine': {
                'total_queries': reasoning_stats['total_queries'],
                'performance': {
                    'avg_ttft': f"{reasoning_stats['avg_ttft']:.1f}ms",
                    'avg_reasoning_time': f"{reasoning_stats['avg_reasoning_time']:.2f}s",
                    'avg_confidence': f"{reasoning_stats.get('avg_confidence', 0.0):.2f}"
                },
                'mode_distribution': reasoning_stats['mode_usage']
            },
            'system_performance': self.performance_metrics,
            'configuration': {
                'max_context_length': self.config.max_context_length,
                'ttft_target': f"{self.config.ttft_target_ms}ms",
                'adaptive_reasoning': self.config.adaptive_reasoning_mode,
                'progressive_responses': self.config.enable_progressive
            }
        }
    
    def clear_context(
        self,
        context_types: Optional[List[str]] = None,
        keep_recent_hours: Optional[int] = None
    ) -> None:
        """
        Clear context with optional filtering.
        
        Args:
            context_types: Specific context types to clear
            keep_recent_hours: Keep context from last N hours
        """
        if keep_recent_hours is not None:
            # This would require additional logic to filter by time
            logger.warning("Time-based context clearing not yet implemented")
        
        self.context_manager.clear_context(context_types)
        logger.info(f"Cleared context: {context_types or 'all types'}")
    
    def export_context_data(self) -> Dict[str, Any]:
        """
        Export context data for analysis or backup.
        
        Returns:
            Exportable context data
        """
        context_stats = self.context_manager.get_context_stats()
        
        # Export non-sensitive context metadata
        export_data = {
            'timestamp': self.context_manager.access_history[-10:] if self.context_manager.access_history else [],
            'statistics': context_stats,
            'configuration': {
                'max_context_length': self.config.max_context_length,
                'similarity_threshold': self.config.similarity_threshold,
                'max_windows': self.config.max_windows
            },
            'performance_metrics': self.performance_metrics.copy()
        }
        
        return export_data
    
    # Private helper methods
    
    async def _update_context_from_code(self, code_context: Dict[str, Any]) -> None:
        """Update context manager with current code context."""
        if 'current_file' in code_context:
            file_info = code_context['current_file']
            await self.add_code_context(
                content=file_info.get('content', ''),
                context_type='code',
                source_file=file_info.get('path'),
                priority=3  # High priority for current file
            )
        
        if 'selected_text' in code_context and code_context['selected_text']:
            await self.add_code_context(
                content=code_context['selected_text'],
                context_type='selection',
                priority=4  # Very high priority for selected text
            )
        
        if 'project_files' in code_context:
            for file_info in code_context['project_files'][:5]:  # Limit to 5 files
                await self.add_code_context(
                    content=file_info.get('content', ''),
                    context_type='project',
                    source_file=file_info.get('path'),
                    priority=1  # Lower priority for project files
                )
    
    def _determine_optimal_reasoning_mode(
        self,
        query: str,
        code_context: Dict[str, Any]
    ) -> ReasoningMode:
        """Determine optimal reasoning mode based on query and context."""
        # Simple heuristics for mode selection
        query_lower = query.lower()
        
        # Fast mode for simple queries
        if any(word in query_lower for word in ['what is', 'define', 'syntax', 'how to']):
            if len(query.split()) < 10:
                return ReasoningMode.FAST
        
        # Deep mode for complex analysis
        if any(word in query_lower for word in [
            'analyze', 'optimize', 'refactor', 'architecture', 'design pattern',
            'performance', 'security', 'best practice'
        ]):
            return ReasoningMode.DEEP
        
        # Progressive mode for step-by-step tasks
        if any(word in query_lower for word in [
            'implement', 'create', 'build', 'step by step', 'tutorial'
        ]):
            return ReasoningMode.PROGRESSIVE
        
        # Default to balanced
        return ReasoningMode.BALANCED
    
    async def _update_context_relevance_feedback(
        self,
        query: str,
        response: Union[str, AsyncGenerator[str, None]]
    ) -> None:
        """Update context relevance based on query-response patterns."""
        # This would analyze which contexts were most useful
        # and update their relevance scores accordingly
        # For now, we'll implement a simple version
        
        if isinstance(response, str):
            # Simple keyword matching to boost relevant contexts
            response_words = set(response.lower().split())
            query_words = set(query.lower().split())
            
            # Find contexts that match response content
            for context_id, window in self.context_manager.context_cache.items():
                window_words = set(window.content.lower().split())
                
                # Calculate overlap with response
                response_overlap = len(response_words & window_words) / max(1, len(response_words))
                query_overlap = len(query_words & window_words) / max(1, len(query_words))
                
                # Boost relevance for contexts that contributed to good responses
                if response_overlap > 0.1 or query_overlap > 0.2:
                    relevance_boost = min(0.1, (response_overlap + query_overlap) / 2)
                    self.context_manager.update_context_relevance(context_id, relevance_boost)
    
    async def _update_session_state(
        self,
        session_id: str,
        query: str,
        response: Union[str, AsyncGenerator[str, None]],
        reasoning_mode: ReasoningMode
    ) -> None:
        """Update session state for context continuity."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'queries': [],
                'reasoning_modes': [],
                'context_usage': {},
                'created_at': self.context_manager.access_history[-1][1] if self.context_manager.access_history else None
            }
        
        session = self.active_sessions[session_id]
        session['queries'].append(query)
        session['reasoning_modes'].append(reasoning_mode.value)
        
        # Keep only recent queries (last 10)
        if len(session['queries']) > 10:
            session['queries'] = session['queries'][-10:]
            session['reasoning_modes'] = session['reasoning_modes'][-10:]