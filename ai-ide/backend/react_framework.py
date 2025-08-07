"""
ReAct (Reasoning + Acting) Framework

Implementation of the ReAct pattern for dynamic tool usage during reasoning.
Combines reasoning and acting in an interleaved manner to solve complex problems
by dynamically selecting and using tools based on reasoning context.

Based on the ReAct paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from react_core import ReActTrace, ReActStep, ReasoningStep, ActionType, Tool
from react_tools import ToolRegistry, ToolSelector
from react_strategy import AdaptiveReasoningStrategy
from react_execution import ReActExecutor
from react_default_tools import DefaultToolImplementations

logger = logging.getLogger(__name__)


class ReActFramework:
    """
    Main ReAct framework implementation.

    Combines reasoning and acting in an interleaved manner to solve complex problems
    by dynamically selecting and using tools based on reasoning context.
    """

    def __init__(
        self,
        llm_client,
        context_manager,
        max_iterations: int = 20,
        confidence_threshold: float = 0.8
    ):
        self.llm_client = llm_client
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Core components
        self.tool_registry = ToolRegistry()
        self.tool_selector = ToolSelector(self.tool_registry, llm_client)
        self.reasoning_strategy = AdaptiveReasoningStrategy()
        self.executor = ReActExecutor(
            llm_client, context_manager, self.tool_registry,
            self.tool_selector, self.reasoning_strategy
        )

        # Default tool implementations
        self.default_tools = DefaultToolImplementations(
            llm_client, context_manager)

        # State tracking
        self.active_traces: Dict[str, ReActTrace] = {}
        self.completed_traces: List[ReActTrace] = []

        # Initialize default tools
        self._initialize_default_tools()

        logger.info("ReActFramework initialized")

    def _initialize_default_tools(self) -> None:
        """Initialize default tools for the framework."""

        # Search tool
        search_tool = Tool(
            name="semantic_search",
            description="Search for relevant code, documentation, or information",
            action_type=ActionType.SEARCH,
            parameters={"query": "string", "max_results": "int"},
            execute_func=self.default_tools.execute_search_tool
        )
        self.tool_registry.register_tool(search_tool)

        # Code analysis tool
        analysis_tool = Tool(
            name="code_analyzer",
            description="Analyze code for quality, bugs, and improvements",
            action_type=ActionType.CODE_ANALYSIS,
            parameters={"code": "string", "analysis_type": "string"},
            execute_func=self.default_tools.execute_analysis_tool
        )
        self.tool_registry.register_tool(analysis_tool)

        # Code generation tool
        generation_tool = Tool(
            name="code_generator",
            description="Generate code based on specifications",
            action_type=ActionType.CODE_GENERATION,
            parameters={"description": "string", "language": "string"},
            execute_func=self.default_tools.execute_generation_tool
        )
        self.tool_registry.register_tool(generation_tool)

        # Reasoning tool
        reasoning_tool = Tool(
            name="deep_reasoner",
            description="Perform deep reasoning about complex problems",
            action_type=ActionType.REASONING,
            parameters={"problem": "string", "context": "dict"},
            execute_func=self.default_tools.execute_reasoning_tool
        )
        self.tool_registry.register_tool(reasoning_tool)

    async def solve_problem(
        self,
        problem: str,
        task_complexity: str = "moderate",
        context: Optional[Dict[str, Any]] = None
    ) -> ReActTrace:
        """
        Main entry point for solving problems using ReAct pattern.

        Args:
            problem: The problem statement to solve
            task_complexity: Complexity level (simple, moderate, complex, expert)
            context: Additional context information

        Returns:
            Complete ReAct trace with reasoning and actions
        """
        trace_id = f"react_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Create trace
        trace = ReActTrace(
            trace_id=trace_id,
            problem_statement=problem,
            context=context or {}
        )

        self.active_traces[trace_id] = trace

        try:
            # Get reasoning strategy
            strategy = self.reasoning_strategy.get_strategy(task_complexity)

            # Execute ReAct loop using executor
            await self.executor.execute_react_loop(trace, strategy)

            # Finalize trace
            trace.total_time = time.time() - start_time
            trace.confidence_score = self._calculate_overall_confidence(trace)
            trace.success = trace.confidence_score >= self.confidence_threshold

            return trace

        finally:
            # Move to completed traces
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]
            self.completed_traces.append(trace)

    def register_custom_tool(self, tool: Tool) -> None:
        """Register a custom tool with the framework."""
        self.tool_registry.register_tool(tool)

    def get_trace(self, trace_id: str) -> Optional[ReActTrace]:
        """Get a trace by ID."""
        return self.active_traces.get(trace_id) or next(
            (trace for trace in self.completed_traces if trace.trace_id == trace_id),
            None
        )

    def _calculate_overall_confidence(self, trace: ReActTrace) -> float:
        """Calculate overall confidence for the trace."""
        if not trace.steps:
            return 0.0

        # Consider different factors
        step_confidences = [step.confidence for step in trace.steps]
        avg_confidence = sum(step_confidences) / len(step_confidences)

        # Bonus for successful tool usage
        successful_actions = len(
            [s for s in trace.steps if s.step_type == ReasoningStep.ACTION and not s.error])
        total_actions = len(
            [s for s in trace.steps if s.step_type == ReasoningStep.ACTION])

        success_rate = successful_actions / total_actions if total_actions > 0 else 1.0

        # Bonus for having final answer
        final_answer_bonus = 0.1 if trace.final_answer else 0.0

        overall_confidence = (avg_confidence * 0.7 +
                              success_rate * 0.2 + final_answer_bonus)

        return max(0.0, min(1.0, overall_confidence))

    def visualize_trace(self, trace: ReActTrace) -> str:
        """Generate a visualization of the ReAct trace."""
        visualization = []
        visualization.append(f"🤖 ReAct Reasoning Trace: {trace.trace_id}")
        visualization.append(f"📋 Problem: {trace.problem_statement}")
        visualization.append(f"⏱️  Total Time: {trace.total_time:.2f}s")
        visualization.append(f"🎯 Confidence: {trace.confidence_score:.2f}")
        visualization.append(f"✅ Success: {trace.success}")
        visualization.append(f"🔧 Tools Used: {', '.join(trace.tools_used)}")
        visualization.append("")

        # Step-by-step breakdown
        for i, step in enumerate(trace.steps):
            step_icon = self._get_step_icon(step.step_type)
            tool_info = f" ({step.tool_name})" if step.tool_name else ""

            visualization.append(
                f"{step_icon} Step {i+1}: {step.step_type.value.title()}{tool_info}")
            visualization.append(f"   Confidence: {step.confidence:.2f}")

            if step.execution_time > 0:
                visualization.append(
                    f"   Execution Time: {step.execution_time:.2f}s")

            if step.error:
                visualization.append(f"   ❌ Error: {step.error}")

            # Show content (truncated)
            content = step.content[:100] + \
                "..." if len(step.content) > 100 else step.content
            visualization.append(f"   {content}")
            visualization.append("")

        # Final answer
        if trace.final_answer:
            visualization.append("🎯 Final Answer:")
            answer = trace.final_answer[:200] + "..." if len(
                trace.final_answer) > 200 else trace.final_answer
            visualization.append(f"   {answer}")

        return '\n'.join(visualization)

    def _get_step_icon(self, step_type: ReasoningStep) -> str:
        """Get emoji icon for step type."""
        icons = {
            ReasoningStep.THOUGHT: "💭",
            ReasoningStep.ACTION: "⚡",
            ReasoningStep.OBSERVATION: "👁️",
            ReasoningStep.REFLECTION: "🤔"
        }
        return icons.get(step_type, "📝")

    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces),
            "registered_tools": len(self.tool_registry.tools),
            "tool_categories": {
                action_type.value: len(tools)
                for action_type, tools in self.tool_registry.tool_categories.items()
            },
            "average_confidence": (
                sum(trace.confidence_score for trace in self.completed_traces) /
                len(self.completed_traces) if self.completed_traces else 0.0
            ),
            "success_rate": (
                sum(1 for trace in self.completed_traces if trace.success) /
                len(self.completed_traces) if self.completed_traces else 0.0
            )
        }
