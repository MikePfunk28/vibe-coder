"""
ReAct (Reasoning + Acting) Framework

Implementation of the ReAct pattern for dynamic tool usage during reasoning.
Combines reasoning and acting in an interleaved manner to solve complex problems
by dynamically selecting and using tools based on reasoning context.

Based on the ReAct paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be performed."""
    SEARCH = "search"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    TEST_EXECUTION = "test_execution"
    FILE_OPERATION = "file_operation"
    REASONING = "reasoning"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ReasoningStep(Enum):
    """Types of reasoning steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


@dataclass
class Tool:
    """Represents a tool that can be used by the ReAct framework."""
    name: str
    description: str
    action_type: ActionType
    parameters: Dict[str, Any]
    execute_func: Callable
    confidence_threshold: float = 0.7
    max_retries: int = 3
    timeout: int = 30


@dataclass
class ReActStep:
    """Represents a single step in the ReAct reasoning process."""
    step_id: str
    step_type: ReasoningStep
    content: str
    action_type: Optional[ActionType] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ReActTrace:
    """Complete ReAct reasoning trace with interleaved reasoning and acting."""
    trace_id: str
    problem_statement: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    total_time: float = 0.0
    confidence_score: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[ActionType, List[str]] = {}
        
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
        
        if tool.action_type not in self.tool_categories:
            self.tool_categories[tool.action_type] = []
        self.tool_categories[tool.action_type].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} ({tool.action_type.value})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tools_by_type(self, action_type: ActionType) -> List[Tool]:
        """Get all tools of a specific type."""
        tool_names = self.tool_categories.get(action_type, [])
        return [self.tools[name] for name in tool_names]
    
    def list_available_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())
    
    def get_tool_description(self, name: str) -> str:
        """Get tool description for prompt generation."""
        tool = self.tools.get(name)
        if not tool:
            return f"Tool '{name}' not found"
        
        return f"{tool.name}: {tool.description}"


class ToolSelector:
    """Intelligent tool selection based on reasoning context."""
    
    def __init__(self, tool_registry: ToolRegistry, llm_client):
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.selection_history: List[Dict[str, Any]] = []
    
    async def select_tool(
        self,
        reasoning_context: str,
        available_tools: List[str],
        task_complexity: str = "moderate"
    ) -> Tuple[Optional[str], float]:
        """
        Select the most appropriate tool based on reasoning context.
        
        Returns:
            Tuple of (tool_name, confidence_score)
        """
        if not available_tools:
            return None, 0.0
        
        # Build tool selection prompt
        tool_descriptions = []
        for tool_name in available_tools:
            description = self.tool_registry.get_tool_description(tool_name)
            tool_descriptions.append(description)
        
        prompt = f"""
        Given the following reasoning context, select the most appropriate tool to use:
        
        Context: {reasoning_context}
        Task Complexity: {task_complexity}
        
        Available Tools:
        {chr(10).join(tool_descriptions)}
        
        Consider:
        1. Which tool best addresses the current reasoning need?
        2. What is the expected effectiveness of each tool?
        3. Are there any dependencies or prerequisites?
        
        Respond with just the tool name and confidence (0-1):
        Tool: [tool_name]
        Confidence: [0.0-1.0]
        Reasoning: [brief explanation]
        """
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2
            )
            
            # Parse response
            tool_name, confidence = self._parse_tool_selection(response, available_tools)
            
            # Record selection for learning
            self.selection_history.append({
                "context": reasoning_context,
                "selected_tool": tool_name,
                "confidence": confidence,
                "available_tools": available_tools,
                "timestamp": datetime.now()
            })
            
            return tool_name, confidence
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            # Fallback to first available tool
            return available_tools[0] if available_tools else None, 0.5
    
    def _parse_tool_selection(self, response: str, available_tools: List[str]) -> Tuple[Optional[str], float]:
        """Parse LLM response for tool selection."""
        lines = response.strip().split('\n')
        tool_name = None
        confidence = 0.5
        
        for line in lines:
            line = line.strip().lower()
            if line.startswith('tool:'):
                tool_candidate = line.replace('tool:', '').strip()
                # Find matching tool (case insensitive)
                for available_tool in available_tools:
                    if available_tool.lower() in tool_candidate or tool_candidate in available_tool.lower():
                        tool_name = available_tool
                        break
            elif line.startswith('confidence:'):
                try:
                    confidence_str = line.replace('confidence:', '').strip()
                    confidence = float(confidence_str)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except ValueError:
                    confidence = 0.5
        
        return tool_name, confidence
    
    def get_contextual_tools(self, reasoning_context: str) -> List[str]:
        """Get tools that are contextually relevant."""
        context_lower = reasoning_context.lower()
        relevant_tools = []
        
        # Keyword-based tool suggestion
        tool_keywords = {
            'search': ['find', 'search', 'look for', 'discover'],
            'code_analysis': ['analyze', 'review', 'examine', 'inspect'],
            'code_generation': ['generate', 'create', 'write', 'implement'],
            'test_execution': ['test', 'verify', 'validate', 'check'],
            'file_operation': ['file', 'read', 'write', 'save', 'load']
        }
        
        for action_type_str, keywords in tool_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                try:
                    action_type = ActionType(action_type_str)
                    tools = self.tool_registry.get_tools_by_type(action_type)
                    relevant_tools.extend([tool.name for tool in tools])
                except ValueError:
                    continue
        
        # If no contextual tools found, return all tools
        if not relevant_tools:
            relevant_tools = self.tool_registry.list_available_tools()
        
        return list(set(relevant_tools))  # Remove duplicates


class AdaptiveReasoningStrategy:
    """Adaptive reasoning strategies based on task complexity."""
    
    def __init__(self):
        self.strategies = {
            "simple": {
                "max_steps": 5,
                "reasoning_depth": "shallow",
                "tool_usage": "minimal",
                "reflection_frequency": 0.2
            },
            "moderate": {
                "max_steps": 10,
                "reasoning_depth": "medium",
                "tool_usage": "balanced",
                "reflection_frequency": 0.3
            },
            "complex": {
                "max_steps": 20,
                "reasoning_depth": "deep",
                "tool_usage": "extensive",
                "reflection_frequency": 0.4
            },
            "expert": {
                "max_steps": 30,
                "reasoning_depth": "very_deep",
                "tool_usage": "comprehensive",
                "reflection_frequency": 0.5
            }
        }
    
    def get_strategy(self, task_complexity: str) -> Dict[str, Any]:
        """Get reasoning strategy for given complexity."""
        return self.strategies.get(task_complexity, self.strategies["moderate"])
    
    def should_reflect(self, step_count: int, strategy: Dict[str, Any]) -> bool:
        """Determine if reflection step is needed."""
        reflection_freq = strategy.get("reflection_frequency", 0.3)
        return (step_count > 0 and 
                (step_count % max(1, int(1 / reflection_freq)) == 0))
    
    def should_continue_reasoning(
        self,
        step_count: int,
        strategy: Dict[str, Any],
        current_confidence: float
    ) -> bool:
        """Determine if reasoning should continue."""
        max_steps = strategy.get("max_steps", 10)
        
        # Continue if under step limit and confidence is not high enough
        if step_count >= max_steps:
            return False
        
        # Stop early if very confident
        if current_confidence > 0.9:
            return False
        
        # Continue if confidence is low
        if current_confidence < 0.7:
            return True
        
        # Default: continue for moderate confidence
        return step_count < max_steps * 0.8


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
            execute_func=self._execute_search_tool
        )
        self.tool_registry.register_tool(search_tool)
        
        # Code analysis tool
        analysis_tool = Tool(
            name="code_analyzer",
            description="Analyze code for quality, bugs, and improvements",
            action_type=ActionType.CODE_ANALYSIS,
            parameters={"code": "string", "analysis_type": "string"},
            execute_func=self._execute_analysis_tool
        )
        self.tool_registry.register_tool(analysis_tool)
        
        # Code generation tool
        generation_tool = Tool(
            name="code_generator",
            description="Generate code based on specifications",
            action_type=ActionType.CODE_GENERATION,
            parameters={"description": "string", "language": "string"},
            execute_func=self._execute_generation_tool
        )
        self.tool_registry.register_tool(generation_tool)
        
        # Reasoning tool
        reasoning_tool = Tool(
            name="deep_reasoner",
            description="Perform deep reasoning about complex problems",
            action_type=ActionType.REASONING,
            parameters={"problem": "string", "context": "dict"},
            execute_func=self._execute_reasoning_tool
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
            
            # Execute ReAct loop
            await self._execute_react_loop(trace, strategy)
            
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
    
    async def _execute_react_loop(self, trace: ReActTrace, strategy: Dict[str, Any]) -> None:
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
        final_steps = [step for step in trace.steps if step.step_type == ReasoningStep.ACTION and step.action_type == ActionType.FINAL_ANSWER]
        
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
    
    def _calculate_overall_confidence(self, trace: ReActTrace) -> float:
        """Calculate overall confidence for the trace."""
        if not trace.steps:
            return 0.0
        
        # Consider different factors
        step_confidences = [step.confidence for step in trace.steps]
        avg_confidence = sum(step_confidences) / len(step_confidences)
        
        # Bonus for successful tool usage
        successful_actions = len([s for s in trace.steps if s.step_type == ReasoningStep.ACTION and not s.error])
        total_actions = len([s for s in trace.steps if s.step_type == ReasoningStep.ACTION])
        
        success_rate = successful_actions / total_actions if total_actions > 0 else 1.0
        
        # Bonus for having final answer
        final_answer_bonus = 0.1 if trace.final_answer else 0.0
        
        overall_confidence = (avg_confidence * 0.7 + success_rate * 0.2 + final_answer_bonus)
        
        return max(0.0, min(1.0, overall_confidence))
    
    # Default tool implementations
    async def _execute_search_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search tool."""
        query = input_data.get("query", "")
        max_results = input_data.get("max_results", 5)
        
        try:
            # Use context manager for search
            context = self.context_manager.get_relevant_context(query, max_tokens=1024)
            
            results = []
            for ctx in context[:max_results]:
                results.append({
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                    "source": getattr(ctx, 'source', 'unknown'),
                    "relevance": getattr(ctx, 'relevance', 0.5)
                })
            
            return {
                "results": results,
                "query": query,
                "total_found": len(results)
            }
            
        except Exception as e:
            return {"error": str(e), "results": []}
    
    async def _execute_analysis_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code analysis tool."""
        code = input_data.get("code", "")
        analysis_type = input_data.get("analysis_type", "general")
        
        if not code:
            return {"error": "No code provided for analysis"}
        
        try:
            prompt = f"""
            Analyze the following code for {analysis_type}:
            
            {code}
            
            Provide analysis including:
            1. Code quality assessment
            2. Potential issues
            3. Suggestions for improvement
            
            Analysis:"""
            
            analysis = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.2
            )
            
            return {
                "analysis": analysis.strip(),
                "code_length": len(code),
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_generation_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation tool."""
        description = input_data.get("description", "")
        language = input_data.get("language", "python")
        
        if not description:
            return {"error": "No description provided for code generation"}
        
        try:
            prompt = f"""
            Generate {language} code for:
            {description}
            
            Provide clean, well-documented code:
            """
            
            code = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            return {
                "generated_code": code.strip(),
                "language": language,
                "description": description
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_reasoning_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deep reasoning tool."""
        problem = input_data.get("problem", "")
        context = input_data.get("context", {})
        
        if not problem:
            return {"error": "No problem provided for reasoning"}
        
        try:
            prompt = f"""
            Think deeply about this problem:
            {problem}
            
            Context: {json.dumps(context, indent=2)}
            
            Provide step-by-step reasoning:
            1. Problem understanding
            2. Key considerations
            3. Potential approaches
            4. Recommended solution
            
            Reasoning:"""
            
            reasoning = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.2
            )
            
            return {
                "reasoning": reasoning.strip(),
                "problem": problem,
                "context": context
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def register_custom_tool(self, tool: Tool) -> None:
        """Register a custom tool with the framework."""
        self.tool_registry.register_tool(tool)
    
    def get_trace(self, trace_id: str) -> Optional[ReActTrace]:
        """Get a trace by ID."""
        return self.active_traces.get(trace_id) or next(
            (trace for trace in self.completed_traces if trace.trace_id == trace_id),
            None
        )
    
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
            
            visualization.append(f"{step_icon} Step {i+1}: {step.step_type.value.title()}{tool_info}")
            visualization.append(f"   Confidence: {step.confidence:.2f}")
            
            if step.execution_time > 0:
                visualization.append(f"   Execution Time: {step.execution_time:.2f}s")
            
            if step.error:
                visualization.append(f"   ❌ Error: {step.error}")
            
            # Show content (truncated)
            content = step.content[:100] + "..." if len(step.content) > 100 else step.content
            visualization.append(f"   {content}")
            visualization.append("")
        
        # Final answer
        if trace.final_answer:
            visualization.append("🎯 Final Answer:")
            answer = trace.final_answer[:200] + "..." if len(trace.final_answer) > 200 else trace.final_answer
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