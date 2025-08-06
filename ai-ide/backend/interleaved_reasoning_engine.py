"""
Interleaved Reasoning Engine

Implementation of Apple's interleaved reasoning patterns for code assistance.
Includes think-answer interleaving, intermediate signal processing, and TTFT optimization.

Based on Apple's research on interleaved reasoning and progressive response generation.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Different modes of reasoning."""
    FAST = "fast"  # Quick responses with minimal reasoning
    BALANCED = "balanced"  # Balance between speed and depth
    DEEP = "deep"  # Thorough reasoning with multiple steps
    PROGRESSIVE = "progressive"  # Progressive disclosure of reasoning


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    step_id: str
    step_type: str  # 'think', 'answer', 'intermediate', 'verification'
    content: str
    confidence: float
    timestamp: datetime
    processing_time: float
    context_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    trace_id: str
    query: str
    mode: ReasoningMode
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_time: float = 0.0
    confidence_score: float = 0.0
    context_windows_used: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IntermediateSignal:
    """Intermediate signal for reasoning guidance."""
    signal_type: str  # 'confidence', 'direction', 'context_need', 'complexity'
    value: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


class InterleaveReasoningEngine:
    """
    Implements Apple's interleaved reasoning patterns for code assistance.
    
    Key features:
    - Think-answer interleaving for better reasoning quality
    - Intermediate signal processing for reasoning guidance
    - TTFT (Time To First Token) optimization
    - Progressive response generation for better UX
    """
    
    def __init__(
        self,
        context_manager,
        llm_client,
        max_reasoning_steps: int = 10,
        ttft_target_ms: int = 200,
        enable_progressive: bool = True
    ):
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.max_reasoning_steps = max_reasoning_steps
        self.ttft_target_ms = ttft_target_ms
        self.enable_progressive = enable_progressive
        
        # Reasoning state
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self.reasoning_history: List[ReasoningTrace] = []
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'avg_ttft': 0.0,
            'avg_reasoning_time': 0.0,
            'confidence_scores': [],
            'mode_usage': {mode.value: 0 for mode in ReasoningMode}
        }
        
        # Intermediate signal processors
        self.signal_processors = {
            'confidence': self._process_confidence_signal,
            'direction': self._process_direction_signal,
            'context_need': self._process_context_need_signal,
            'complexity': self._process_complexity_signal
        }
        
        logger.info(f"InterleaveReasoningEngine initialized with TTFT target: {ttft_target_ms}ms")
    
    async def reason_and_respond(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode = ReasoningMode.BALANCED,
        stream: bool = True
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Main entry point for interleaved reasoning and response generation.
        
        Args:
            query: The user query or code assistance request
            context: Additional context information
            mode: Reasoning mode to use
            stream: Whether to stream the response progressively
            
        Returns:
            Complete response or async generator for streaming
        """
        trace_id = f"trace_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Create reasoning trace
        trace = ReasoningTrace(
            trace_id=trace_id,
            query=query,
            mode=mode
        )
        self.active_traces[trace_id] = trace
        
        try:
            if stream and self.enable_progressive:
                return self._progressive_reasoning_stream(trace, query, context)
            else:
                response = await self._complete_reasoning(trace, query, context)
                return response
        finally:
            # Update performance stats
            trace.total_time = time.time() - start_time
            self._update_performance_stats(trace)
            
            # Move to history
            self.reasoning_history.append(trace)
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]
    
    async def _progressive_reasoning_stream(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generate progressive reasoning stream with interleaved think-answer pattern.
        """
        start_time = time.time()
        
        # Step 1: Quick initial response for TTFT optimization
        initial_step = await self._generate_initial_response(trace, query, context)
        if initial_step:
            yield f"🤔 {initial_step.content}\n\n"
            
            # Check if we hit TTFT target
            ttft = (time.time() - start_time) * 1000
            if ttft <= self.ttft_target_ms:
                logger.debug(f"TTFT target met: {ttft:.1f}ms")
        
        # Step 2: Interleaved reasoning process
        reasoning_complete = False
        step_count = 0
        
        while not reasoning_complete and step_count < self.max_reasoning_steps:
            # Think step
            think_step = await self._generate_think_step(trace, query, context, step_count)
            if think_step:
                trace.steps.append(think_step)
                if trace.mode == ReasoningMode.DEEP:
                    yield f"💭 **Thinking**: {think_step.content}\n\n"
            
            # Process intermediate signals
            signals = await self._generate_intermediate_signals(trace, think_step)
            for signal in signals:
                await self._process_signal(signal, trace)
            
            # Answer step
            answer_step = await self._generate_answer_step(trace, query, context, step_count)
            if answer_step:
                trace.steps.append(answer_step)
                yield f"💡 {answer_step.content}\n\n"
                
                # Check if reasoning is complete
                if answer_step.confidence > 0.8 or step_count >= self.max_reasoning_steps - 1:
                    reasoning_complete = True
                    trace.final_answer = answer_step.content
                    trace.confidence_score = answer_step.confidence
            
            step_count += 1
            
            # Brief pause for better UX
            await asyncio.sleep(0.1)
        
        # Final verification step
        if trace.mode in [ReasoningMode.BALANCED, ReasoningMode.DEEP]:
            verification_step = await self._generate_verification_step(trace, query, context)
            if verification_step:
                trace.steps.append(verification_step)
                yield f"✅ **Verification**: {verification_step.content}\n\n"
    
    async def _complete_reasoning(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Complete reasoning process without streaming.
        """
        # Get relevant context
        relevant_context = self.context_manager.get_relevant_context(
            query,
            max_tokens=4096 if trace.mode == ReasoningMode.DEEP else 2048
        )
        
        trace.context_windows_used = [ctx.id for ctx in relevant_context]
        
        # Build reasoning prompt based on mode
        reasoning_prompt = self._build_reasoning_prompt(query, relevant_context, trace.mode)
        
        # Generate response
        start_time = time.time()
        response = await self.llm_client.generate(
            prompt=reasoning_prompt,
            max_tokens=1024 if trace.mode == ReasoningMode.FAST else 2048,
            temperature=0.1 if trace.mode == ReasoningMode.FAST else 0.3
        )
        processing_time = time.time() - start_time
        
        # Create final step
        final_step = ReasoningStep(
            step_id=f"{trace.trace_id}_final",
            step_type="answer",
            content=response,
            confidence=0.8,  # Default confidence
            timestamp=datetime.now(),
            processing_time=processing_time,
            context_used=trace.context_windows_used
        )
        
        trace.steps.append(final_step)
        trace.final_answer = response
        trace.confidence_score = final_step.confidence
        
        return response
    
    async def _generate_initial_response(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[ReasoningStep]:
        """
        Generate quick initial response for TTFT optimization.
        """
        start_time = time.time()
        
        # Use minimal context for speed
        relevant_context = self.context_manager.get_relevant_context(
            query,
            max_tokens=512
        )
        
        # Simple prompt for quick response
        prompt = f"""
        Quick initial response to: {query}
        
        Context: {' '.join([ctx.content[:100] for ctx in relevant_context[:2]])}
        
        Provide a brief initial thought or direction:
        """
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            return ReasoningStep(
                step_id=f"{trace.trace_id}_initial",
                step_type="initial",
                content=response.strip(),
                confidence=0.6,
                timestamp=datetime.now(),
                processing_time=processing_time,
                context_used=[ctx.id for ctx in relevant_context]
            )
        except Exception as e:
            logger.error(f"Error generating initial response: {e}")
            return None
    
    async def _generate_think_step(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any],
        step_number: int
    ) -> Optional[ReasoningStep]:
        """
        Generate a thinking step in the reasoning process.
        """
        start_time = time.time()
        
        # Get context based on previous steps
        previous_content = " ".join([step.content for step in trace.steps[-2:]])
        relevant_context = self.context_manager.get_relevant_context(
            f"{query} {previous_content}",
            max_tokens=1024
        )
        
        # Build thinking prompt
        prompt = f"""
        Query: {query}
        Previous reasoning: {previous_content}
        
        Think about this step by step. What should we consider next?
        Focus on: analysis, potential approaches, edge cases, or verification needs.
        
        Thinking step {step_number + 1}:
        """
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            processing_time = time.time() - start_time
            
            return ReasoningStep(
                step_id=f"{trace.trace_id}_think_{step_number}",
                step_type="think",
                content=response.strip(),
                confidence=0.7,
                timestamp=datetime.now(),
                processing_time=processing_time,
                context_used=[ctx.id for ctx in relevant_context]
            )
        except Exception as e:
            logger.error(f"Error generating think step: {e}")
            return None
    
    async def _generate_answer_step(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any],
        step_number: int
    ) -> Optional[ReasoningStep]:
        """
        Generate an answer step based on previous thinking.
        """
        start_time = time.time()
        
        # Get comprehensive context
        all_reasoning = " ".join([step.content for step in trace.steps])
        relevant_context = self.context_manager.get_relevant_context(
            f"{query} {all_reasoning}",
            max_tokens=2048
        )
        
        # Build answer prompt
        prompt = f"""
        Query: {query}
        Reasoning so far: {all_reasoning}
        
        Based on the thinking above, provide a concrete answer or next step.
        Be specific and actionable.
        
        Answer step {step_number + 1}:
        """
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            processing_time = time.time() - start_time
            
            # Calculate confidence based on reasoning depth and consistency
            confidence = self._calculate_step_confidence(trace, response)
            
            return ReasoningStep(
                step_id=f"{trace.trace_id}_answer_{step_number}",
                step_type="answer",
                content=response.strip(),
                confidence=confidence,
                timestamp=datetime.now(),
                processing_time=processing_time,
                context_used=[ctx.id for ctx in relevant_context]
            )
        except Exception as e:
            logger.error(f"Error generating answer step: {e}")
            return None
    
    async def _generate_verification_step(
        self,
        trace: ReasoningTrace,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[ReasoningStep]:
        """
        Generate verification step to validate the reasoning.
        """
        start_time = time.time()
        
        if not trace.final_answer:
            return None
        
        prompt = f"""
        Original query: {query}
        Final answer: {trace.final_answer}
        
        Verify this answer by checking:
        1. Does it address the original query?
        2. Is it technically correct?
        3. Are there any obvious issues or improvements?
        
        Verification:
        """
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            return ReasoningStep(
                step_id=f"{trace.trace_id}_verify",
                step_type="verification",
                content=response.strip(),
                confidence=0.9,
                timestamp=datetime.now(),
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Error generating verification step: {e}")
            return None
    
    async def _generate_intermediate_signals(
        self,
        trace: ReasoningTrace,
        current_step: Optional[ReasoningStep]
    ) -> List[IntermediateSignal]:
        """
        Generate intermediate signals for reasoning guidance.
        """
        signals = []
        
        if not current_step:
            return signals
        
        # Confidence signal
        confidence_signal = IntermediateSignal(
            signal_type="confidence",
            value=current_step.confidence,
            description=f"Confidence in current step: {current_step.confidence:.2f}"
        )
        signals.append(confidence_signal)
        
        # Direction signal (based on content analysis)
        direction_value = self._analyze_reasoning_direction(current_step.content)
        direction_signal = IntermediateSignal(
            signal_type="direction",
            value=direction_value,
            description=f"Reasoning direction clarity: {direction_value:.2f}"
        )
        signals.append(direction_signal)
        
        # Context need signal
        context_need = self._analyze_context_need(trace, current_step)
        context_signal = IntermediateSignal(
            signal_type="context_need",
            value=context_need,
            description=f"Additional context needed: {context_need:.2f}"
        )
        signals.append(context_signal)
        
        # Complexity signal
        complexity = self._analyze_complexity(current_step.content)
        complexity_signal = IntermediateSignal(
            signal_type="complexity",
            value=complexity,
            description=f"Problem complexity: {complexity:.2f}"
        )
        signals.append(complexity_signal)
        
        return signals
    
    async def _process_signal(self, signal: IntermediateSignal, trace: ReasoningTrace) -> None:
        """
        Process an intermediate signal and adjust reasoning accordingly.
        """
        processor = self.signal_processors.get(signal.signal_type)
        if processor:
            await processor(signal, trace)
    
    async def _process_confidence_signal(self, signal: IntermediateSignal, trace: ReasoningTrace) -> None:
        """Process confidence signal."""
        if signal.value < 0.5:
            # Low confidence - need more reasoning steps
            logger.debug(f"Low confidence detected: {signal.value:.2f}")
            # Could trigger additional context retrieval or deeper reasoning
    
    async def _process_direction_signal(self, signal: IntermediateSignal, trace: ReasoningTrace) -> None:
        """Process direction signal."""
        if signal.value < 0.4:
            # Unclear direction - might need to refocus
            logger.debug(f"Unclear reasoning direction: {signal.value:.2f}")
    
    async def _process_context_need_signal(self, signal: IntermediateSignal, trace: ReasoningTrace) -> None:
        """Process context need signal."""
        if signal.value > 0.7:
            # High context need - retrieve more context
            logger.debug(f"High context need detected: {signal.value:.2f}")
            # Could trigger additional context retrieval
    
    async def _process_complexity_signal(self, signal: IntermediateSignal, trace: ReasoningTrace) -> None:
        """Process complexity signal."""
        if signal.value > 0.8:
            # High complexity - might need to switch to deeper reasoning mode
            logger.debug(f"High complexity detected: {signal.value:.2f}")
    
    def _build_reasoning_prompt(
        self,
        query: str,
        context_windows: List,
        mode: ReasoningMode
    ) -> str:
        """
        Build reasoning prompt based on mode and context.
        """
        context_text = "\n".join([f"Context {i+1}: {ctx.content}" for i, ctx in enumerate(context_windows)])
        
        if mode == ReasoningMode.FAST:
            return f"""
            Query: {query}
            Context: {context_text[:500]}
            
            Provide a quick, direct answer:
            """
        elif mode == ReasoningMode.BALANCED:
            return f"""
            Query: {query}
            Context: {context_text[:1500]}
            
            Think through this step by step and provide a well-reasoned answer:
            """
        elif mode == ReasoningMode.DEEP:
            return f"""
            Query: {query}
            Context: {context_text}
            
            Analyze this thoroughly:
            1. Break down the problem
            2. Consider multiple approaches
            3. Evaluate trade-offs
            4. Provide detailed reasoning
            5. Give a comprehensive answer
            
            Deep analysis:
            """
        else:  # PROGRESSIVE
            return f"""
            Query: {query}
            Context: {context_text[:1000]}
            
            Start with initial thoughts and build up the reasoning progressively:
            """
    
    def _calculate_step_confidence(self, trace: ReasoningTrace, response: str) -> float:
        """
        Calculate confidence score for a reasoning step.
        """
        base_confidence = 0.7
        
        # Boost confidence based on response characteristics
        if len(response) > 50:  # Detailed response
            base_confidence += 0.1
        
        if any(word in response.lower() for word in ['because', 'therefore', 'since', 'due to']):
            base_confidence += 0.1  # Explanatory language
        
        if any(word in response.lower() for word in ['function', 'class', 'method', 'variable']):
            base_confidence += 0.05  # Code-specific terms
        
        # Reduce confidence for uncertainty markers
        if any(word in response.lower() for word in ['maybe', 'might', 'possibly', 'unclear']):
            base_confidence -= 0.2
        
        # Consider reasoning depth
        reasoning_depth = len(trace.steps) / self.max_reasoning_steps
        base_confidence += reasoning_depth * 0.1
        
        return min(1.0, max(0.1, base_confidence))
    
    def _analyze_reasoning_direction(self, content: str) -> float:
        """
        Analyze how clear the reasoning direction is.
        """
        direction_indicators = ['next', 'then', 'therefore', 'so', 'thus', 'consequently']
        question_indicators = ['what', 'how', 'why', 'when', 'where']
        
        direction_count = sum(1 for word in direction_indicators if word in content.lower())
        question_count = sum(1 for word in question_indicators if word in content.lower())
        
        # More direction indicators = clearer direction
        # More questions = less clear direction
        direction_score = (direction_count - question_count * 0.5) / max(1, len(content.split()) / 10)
        
        return max(0.0, min(1.0, direction_score + 0.5))
    
    def _analyze_context_need(self, trace: ReasoningTrace, current_step: ReasoningStep) -> float:
        """
        Analyze how much additional context is needed.
        """
        need_indicators = ['need', 'require', 'missing', 'unclear', 'more info', 'additional']
        context_indicators = ['given', 'provided', 'available', 'known', 'clear']
        
        need_count = sum(1 for phrase in need_indicators if phrase in current_step.content.lower())
        context_count = sum(1 for phrase in context_indicators if phrase in current_step.content.lower())
        
        # More need indicators = higher context need
        # More context indicators = lower context need
        need_score = (need_count - context_count * 0.5) / max(1, len(current_step.content.split()) / 20)
        
        return max(0.0, min(1.0, need_score + 0.3))
    
    def _analyze_complexity(self, content: str) -> float:
        """
        Analyze the complexity of the current reasoning.
        """
        complexity_indicators = [
            'complex', 'complicated', 'multiple', 'various', 'several',
            'depends', 'consider', 'analyze', 'evaluate', 'trade-off'
        ]
        
        simple_indicators = ['simple', 'easy', 'straightforward', 'obvious', 'clear']
        
        complex_count = sum(1 for word in complexity_indicators if word in content.lower())
        simple_count = sum(1 for word in simple_indicators if word in content.lower())
        
        # More complexity indicators = higher complexity
        complexity_score = (complex_count - simple_count * 0.5) / max(1, len(content.split()) / 15)
        
        return max(0.0, min(1.0, complexity_score + 0.4))
    
    def _update_performance_stats(self, trace: ReasoningTrace) -> None:
        """
        Update performance statistics.
        """
        self.performance_stats['total_queries'] += 1
        self.performance_stats['mode_usage'][trace.mode.value] += 1
        
        if trace.steps:
            # Calculate TTFT (time to first meaningful step)
            first_step_time = trace.steps[0].processing_time * 1000  # Convert to ms
            current_avg = self.performance_stats['avg_ttft']
            total_queries = self.performance_stats['total_queries']
            self.performance_stats['avg_ttft'] = (current_avg * (total_queries - 1) + first_step_time) / total_queries
        
        # Update average reasoning time
        current_avg_time = self.performance_stats['avg_reasoning_time']
        self.performance_stats['avg_reasoning_time'] = (
            current_avg_time * (self.performance_stats['total_queries'] - 1) + trace.total_time
        ) / self.performance_stats['total_queries']
        
        # Track confidence scores
        if trace.confidence_score > 0:
            self.performance_stats['confidence_scores'].append(trace.confidence_score)
            # Keep only last 100 scores
            if len(self.performance_stats['confidence_scores']) > 100:
                self.performance_stats['confidence_scores'] = self.performance_stats['confidence_scores'][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        """
        stats = self.performance_stats.copy()
        
        if stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['confidence_std'] = np.std(stats['confidence_scores'])
        else:
            stats['avg_confidence'] = 0.0
            stats['confidence_std'] = 0.0
        
        return stats
    
    def get_reasoning_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """
        Get a specific reasoning trace.
        """
        # Check active traces first
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        # Check history
        for trace in self.reasoning_history:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def clear_history(self, keep_recent: int = 10) -> None:
        """
        Clear reasoning history, optionally keeping recent traces.
        """
        if keep_recent > 0:
            self.reasoning_history = self.reasoning_history[-keep_recent:]
        else:
            self.reasoning_history.clear()
        
        logger.info(f"Cleared reasoning history, kept {len(self.reasoning_history)} recent traces")