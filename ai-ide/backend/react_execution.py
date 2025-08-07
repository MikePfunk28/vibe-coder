"""
ReAct Framework Execution Engine

Core execution logic for the ReAct reasoning loop.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Any

from react_core import ReActTrace, ReActStep, ReasoningStep, ActionType, Tool
from react_tools import ToolRegistry, ToolSelector
from react_strategy import AdaptiveReasoningStrategy

logger = logging.getLogger(__name__)


class ReActExecutor:
    """Handles the execution of ReAct reasoning loops."""

    def __init__(self, llm_client, context_manager, tool_registry: ToolRegistry, 
                 tool_selector: ToolSelector, reasoning_strategy: AdaptiveReasoningStrategy):
        self.llm_client = llm_client
        self.context_manager = context_manager
        self.tool_registry = tool_registry
        self.tool_selector = tool_selector
        self.reasoning_strategy = reasoning_strategy

    async def execute_react_loop(self, trace: ReActTrace, strategy: Dict[str, Any]) -> None:
        """Execute the main ReAct reasoning and acting loop."""
        max_steps = strategy.get("max_steps", 10)
        current_context = trace.problem_statement

        for iteration in range(max_steps):
            # Check if we should continue
            current_confidence = self._calculate_step_confidence(trace)
            if not self.reasoning_strategy.should_continue_reasoning(
                len(trace.steps), strategy, current_confidence
            ):
                break

            # Reasoning step (Thought)
            thought_step = await self._generate_thought_step(trace, current_context, iteration)
            if thought_step:
                trace.steps.append(thought_step)
                current_context = thought_step.content

            # Check if we need reflection
            if self.reasoning_strategy.should_reflect(len(trace.steps), strategy):
                reflection_step = await self._generate_reflection_step(trace, current_context)
                if reflection_step:
                    trace.steps.append(reflection_step)
                    current_context = reflection_step.content

            # Action step
            action_step = await self._generate_action_step(trace, current_context)
            if action_step:
                trace.steps.append(action_step)

                # Execute the action if it's not a final answer
                if action_step.action_type != ActionType.FINAL_ANSWER:
                    observation_step = await self._execute_action_step(trace, action_step)
                    if observation_step:
                        trace.steps.append(observation_step)
                        current_context = observation_step.content
                else:
                    # Final answer reached
                    trace.final_answer = action_step.content
                    break

        # If no final answer was generated, create one
        if not trace.final_answer:
            trace.final_answer = await self._generate_final_answer(trace)

    async def _generate_thought_step(
        self,
        trace: ReActTrace,
        current_context: str,
        iteration: int
    ) -> Optional[ReActStep]:
        """Generate a reasoning/thought step."""

        # Build context from previous steps
        previous_steps = self._format_previous_steps(trace.steps[-3:])  # Last 3 steps

        prompt = f"""
        Problem: {trace.problem_statement}
        
        Previous steps:
        {previous_steps}
        
        Current context: {current_context}
        
        Think step by step about what to do next. Consider:
        1. What have we learned so far?
        2. What information do we still need?
        3. What should be our next action?
        
        Thought:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3
            )

            step = ReActStep(
                step_id=f"{trace.trace_id}_thought_{iteration}",
                step_type=ReasoningStep.THOUGHT,
                content=response.strip(),
                confidence=self._estimate_thought_confidence(response)
            )

            return step

        except Exception as e:
            logger.error(f"Error generating thought step: {e}")
            return None

    async def _generate_action_step(
        self,
        trace: ReActTrace,
        current_context: str
    ) -> Optional[ReActStep]:
        """Generate an action step with tool selection."""

        # Get available tools based on context
        available_tools = self.tool_selector.get_contextual_tools(current_context)

        # Select best tool
        selected_tool, tool_confidence = await self.tool_selector.select_tool(
            current_context, available_tools
        )

        if not selected_tool:
            # Generate final answer if no tool is appropriate
            return await self._generate_final_answer_step(trace, current_context)

        # Generate action with selected tool
        tool = self.tool_registry.get_tool(selected_tool)
        if not tool:
            return None

        # Generate tool input
        tool_input = await self._generate_tool_input(trace, tool, current_context)

        step = ReActStep(
            step_id=f"{trace.trace_id}_action_{len(trace.steps)}",
            step_type=ReasoningStep.ACTION,
            content=f"I will use {tool.name} to {tool.description}",
            action_type=tool.action_type,
            tool_name=tool.name,
            tool_input=tool_input,
            confidence=tool_confidence
        )

        # Track tool usage
        if tool.name not in trace.tools_used:
            trace.tools_used.append(tool.name)

        return step

    async def _execute_action_step(
        self,
        trace: ReActTrace,
        action_step: ReActStep
    ) -> Optional[ReActStep]:
        """Execute an action step and generate observation."""

        if not action_step.tool_name or not action_step.tool_input:
            return None

        tool = self.tool_registry.get_tool(action_step.tool_name)
        if not tool:
            return None

        start_time = time.time()

        try:
            # Execute tool
            result = await tool.execute_func(action_step.tool_input)
            execution_time = time.time() - start_time

            # Update action step with result
            action_step.tool_output = result
            action_step.execution_time = execution_time

            # Create observation step
            observation_step = ReActStep(
                step_id=f"{trace.trace_id}_obs_{len(trace.steps)}",
                step_type=ReasoningStep.OBSERVATION,
                content=self._format_tool_output(result),
                confidence=0.9,  # Observations are generally reliable
                execution_time=execution_time
            )

            return observation_step

        except Exception as e:
            error_msg = f"Error executing {tool.name}: {str(e)}"
            logger.error(error_msg)

            action_step.error = error_msg

            # Create error observation
            observation_step = ReActStep(
                step_id=f"{trace.trace_id}_obs_{len(trace.steps)}",
                step_type=ReasoningStep.OBSERVATION,
                content=f"Error: {error_msg}",
                confidence=0.1,
                error=error_msg
            )

            return observation_step

    async def _generate_reflection_step(
        self,
        trace: ReActTrace,
        current_context: str
    ) -> Optional[ReActStep]:
        """Generate a reflection step to assess progress."""

        recent_steps = self._format_previous_steps(trace.steps[-5:])

        prompt = f"""
        Problem: {trace.problem_statement}
        
        Recent progress:
        {recent_steps}
        
        Reflect on the progress so far:
        1. Are we making progress toward solving the problem?
        2. What has worked well and what hasn't?
        3. Should we change our approach?
        4. What are the key insights gained?
        
        Reflection:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=250,
                temperature=0.2
            )

            step = ReActStep(
                step_id=f"{trace.trace_id}_reflect_{len(trace.steps)}",
                step_type=ReasoningStep.REFLECTION,
                content=response.strip(),
                confidence=0.8
            )

            return step

        except Exception as e:
            logger.error(f"Error generating reflection step: {e}")
            return None

    async def _generate_final_answer_step(
        self,
        trace: ReActTrace,
        current_context: str
    ) -> ReActStep:
        """Generate final answer step."""

        all_steps = self._format_previous_steps(trace.steps)

        prompt = f"""
        Problem: {trace.problem_statement}
        
        All reasoning and actions taken:
        {all_steps}
        
        Based on all the reasoning and observations above, provide a final answer to the problem:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.1
            )

            final_answer = response.strip()

        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            final_answer = "Unable to generate final answer due to error."

        step = ReActStep(
            step_id=f"{trace.trace_id}_final",
            step_type=ReasoningStep.ACTION,
            content=final_answer,
            action_type=ActionType.FINAL_ANSWER,
            confidence=0.8
        )

        return step

    async def _generate_final_answer(self, trace: ReActTrace) -> str:
        """Generate final answer from trace."""
        final_steps = [step for step in trace.steps if step.step_type ==
                       ReasoningStep.ACTION and step.action_type == ActionType.FINAL_ANSWER]

        if final_steps:
            return final_steps[-1].content

        # Generate from all steps
        all_steps = self._format_previous_steps(trace.steps)

        prompt = f"""
        Problem: {trace.problem_statement}
        
        Complete reasoning trace:
        {all_steps}
        
        Synthesize a final answer:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "Unable to provide final answer."

    async def _generate_tool_input(
        self,
        trace: ReActTrace,
        tool: Tool,
        context: str
    ) -> Dict[str, Any]:
        """Generate appropriate input for a tool based on context."""

        prompt = f"""
        Tool: {tool.name}
        Description: {tool.description}
        Parameters: {json.dumps(tool.parameters, indent=2)}
        
        Context: {context}
        Problem: {trace.problem_statement}
        
        Generate appropriate input parameters for this tool as JSON:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2
            )

            # Try to parse as JSON
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback to simple parsing
                return self._parse_tool_input_fallback(response, tool)

        except Exception as e:
            logger.error(f"Error generating tool input: {e}")
            return {}

    def _parse_tool_input_fallback(self, response: str, tool: Tool) -> Dict[str, Any]:
        """Fallback parsing for tool input."""
        input_data = {}

        # Simple keyword extraction based on tool parameters
        for param_name in tool.parameters.keys():
            if param_name.lower() in response.lower():
                # Extract value after parameter name
                lines = response.split('\n')
                for line in lines:
                    if param_name.lower() in line.lower():
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            input_data[param_name] = parts[1].strip().strip('"\'')
                        break

        return input_data

    def _format_previous_steps(self, steps: List[ReActStep]) -> str:
        """Format previous steps for prompt context."""
        if not steps:
            return "No previous steps."

        formatted = []
        for step in steps:
            step_type = step.step_type.value.title()
            content = step.content[:200] + "..." if len(step.content) > 200 else step.content

            if step.step_type == ReasoningStep.ACTION and step.tool_name:
                formatted.append(f"{step_type} ({step.tool_name}): {content}")
            else:
                formatted.append(f"{step_type}: {content}")

        return '\n'.join(formatted)

    def _format_tool_output(self, output: Any) -> str:
        """Format tool output for observation step."""
        if isinstance(output, dict):
            # Format key results
            if 'results' in output:
                results = output['results']
                if isinstance(results, list) and len(results) > 0:
                    return f"Found {len(results)} results. First result: {str(results[0])[:100]}..."

            # Format other dict outputs
            key_items = []
            for key, value in output.items():
                if key in ['error', 'success', 'message', 'result']:
                    key_items.append(f"{key}: {str(value)[:100]}")

            if key_items:
                return '; '.join(key_items)

        # Default string representation
        output_str = str(output)
        return output_str[:300] + "..." if len(output_str) > 300 else output_str

    def _estimate_thought_confidence(self, thought: str) -> float:
        """Estimate confidence of a thought step."""
        confidence = 0.7  # Base confidence

        # Increase confidence for structured thinking
        if any(marker in thought for marker in ['1.', '2.', '3.', 'First', 'Next', 'Then']):
            confidence += 0.1

        # Increase confidence for specific plans
        if any(word in thought.lower() for word in ['will', 'should', 'need to', 'plan to']):
            confidence += 0.1

        # Decrease confidence for uncertainty
        if any(word in thought.lower() for word in ['maybe', 'might', 'not sure', 'unclear']):
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def _calculate_step_confidence(self, trace: ReActTrace) -> float:
        """Calculate current confidence based on recent steps."""
        if not trace.steps:
            return 0.0

        # Weight recent steps more heavily
        recent_steps = trace.steps[-5:]
        weights = [i + 1 for i in range(len(recent_steps))]

        weighted_sum = sum(step.confidence * weight for step, weight in zip(recent_steps, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0