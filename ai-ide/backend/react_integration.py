"""
ReAct Framework Integration

Integration of the ReAct framework with the existing multi-agent system,
chain-of-thought engine, and other AI IDE components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from react_framework import ReActFramework
from react_core import Tool, ActionType
from multi_agent_system import MultiAgentSystem, AgentTask, TaskPriority
from chain_of_thought_engine import ChainOfThoughtEngine

logger = logging.getLogger(__name__)


@dataclass
class ReActIntegrationConfig:
    """Configuration for ReAct framework integration."""
    enable_multi_agent_tools: bool = True
    enable_cot_reasoning_tool: bool = True
    enable_context_aware_selection: bool = True
    max_reasoning_depth: int = 5
    tool_timeout: int = 30


class ReActAgentIntegration:
    """Integration layer between ReAct framework and multi-agent system."""
    
    def __init__(
        self,
        react_framework: ReActFramework,
        multi_agent_system: MultiAgentSystem,
        cot_engine: ChainOfThoughtEngine,
        config: Optional[ReActIntegrationConfig] = None
    ):
        self.react_framework = react_framework
        self.multi_agent_system = multi_agent_system
        self.cot_engine = cot_engine
        self.config = config or ReActIntegrationConfig()
        
        # Register integrated tools
        self._register_agent_tools()
        self._register_reasoning_tools()
        
        logger.info("ReAct-Agent integration initialized")
    
    def _register_agent_tools(self) -> None:
        """Register tools that delegate to specialized agents."""
        if not self.config.enable_multi_agent_tools:
            return
        
        # Code agent tool
        code_agent_tool = Tool(
            name="code_agent",
            description="Delegate code-related tasks to specialized code agent",
            action_type=ActionType.CODE_GENERATION,
            parameters={
                "task_type": "string",
                "input_data": "dict",
                "priority": "string"
            },
            execute_func=self._execute_code_agent_tool,
            timeout=self.config.tool_timeout
        )
        self.react_framework.register_custom_tool(code_agent_tool)
        
        # Search agent tool
        search_agent_tool = Tool(
            name="search_agent",
            description="Delegate search tasks to specialized search agent",
            action_type=ActionType.SEARCH,
            parameters={
                "task_type": "string",
                "input_data": "dict",
                "priority": "string"
            },
            execute_func=self._execute_search_agent_tool,
            timeout=self.config.tool_timeout
        )
        self.react_framework.register_custom_tool(search_agent_tool)
        
        # Reasoning agent tool
        reasoning_agent_tool = Tool(
            name="reasoning_agent",
            description="Delegate complex reasoning tasks to specialized reasoning agent",
            action_type=ActionType.REASONING,
            parameters={
                "task_type": "string",
                "input_data": "dict",
                "priority": "string"
            },
            execute_func=self._execute_reasoning_agent_tool,
            timeout=self.config.tool_timeout
        )
        self.react_framework.register_custom_tool(reasoning_agent_tool)
        
        # Test agent tool
        test_agent_tool = Tool(
            name="test_agent",
            description="Delegate testing tasks to specialized test agent",
            action_type=ActionType.TEST_EXECUTION,
            parameters={
                "task_type": "string",
                "input_data": "dict",
                "priority": "string"
            },
            execute_func=self._execute_test_agent_tool,
            timeout=self.config.tool_timeout
        )
        self.react_framework.register_custom_tool(test_agent_tool)
    
    def _register_reasoning_tools(self) -> None:
        """Register tools that use chain-of-thought reasoning."""
        if not self.config.enable_cot_reasoning_tool:
            return
        
        # Chain-of-thought reasoning tool
        cot_tool = Tool(
            name="chain_of_thought",
            description="Use structured chain-of-thought reasoning for complex problems",
            action_type=ActionType.REASONING,
            parameters={
                "problem": "string",
                "complexity": "string",
                "problem_type": "string"
            },
            execute_func=self._execute_cot_tool,
            timeout=self.config.tool_timeout * 2  # CoT may take longer
        )
        self.react_framework.register_custom_tool(cot_tool)
    
    async def _execute_code_agent_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code agent tool."""
        try:
            task_type = input_data.get("task_type", "code_generation")
            task_input = input_data.get("input_data", {})
            priority_str = input_data.get("priority", "normal")
            
            # Map priority string to enum
            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "urgent": TaskPriority.URGENT
            }
            priority = priority_map.get(priority_str, TaskPriority.NORMAL)
            
            # Create agent task
            task = AgentTask(
                task_id=f"react_code_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}",
                task_type=task_type,
                description=f"ReAct delegated task: {task_type}",
                input_data=task_input,
                priority=priority
            )
            
            # Get code agent and execute task
            code_agent = self.multi_agent_system.get_agent_by_type("code")
            if not code_agent:
                return {"error": "Code agent not available"}
            
            # Assign and wait for completion
            if await code_agent.assign_task(task):
                await code_agent.process_tasks()
                
                # Wait for task completion
                max_wait = self.config.tool_timeout
                wait_time = 0
                while task.status.value not in ["completed", "failed"] and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1
                
                if task.result:
                    return {
                        "result": task.result,
                        "status": task.status.value,
                        "agent_id": code_agent.agent_id
                    }
                else:
                    return {
                        "error": task.error or "Task failed without result",
                        "status": task.status.value
                    }
            else:
                return {"error": "Failed to assign task to code agent"}
                
        except Exception as e:
            logger.error(f"Error executing code agent tool: {e}")
            return {"error": str(e)}
    
    async def _execute_search_agent_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search agent tool."""
        try:
            task_type = input_data.get("task_type", "semantic_search")
            task_input = input_data.get("input_data", {})
            priority_str = input_data.get("priority", "normal")
            
            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "urgent": TaskPriority.URGENT
            }
            priority = priority_map.get(priority_str, TaskPriority.NORMAL)
            
            task = AgentTask(
                task_id=f"react_search_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}",
                task_type=task_type,
                description=f"ReAct delegated task: {task_type}",
                input_data=task_input,
                priority=priority
            )
            
            search_agent = self.multi_agent_system.get_agent_by_type("search")
            if not search_agent:
                return {"error": "Search agent not available"}
            
            if await search_agent.assign_task(task):
                await search_agent.process_tasks()
                
                max_wait = self.config.tool_timeout
                wait_time = 0
                while task.status.value not in ["completed", "failed"] and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1
                
                if task.result:
                    return {
                        "result": task.result,
                        "status": task.status.value,
                        "agent_id": search_agent.agent_id
                    }
                else:
                    return {
                        "error": task.error or "Task failed without result",
                        "status": task.status.value
                    }
            else:
                return {"error": "Failed to assign task to search agent"}
                
        except Exception as e:
            logger.error(f"Error executing search agent tool: {e}")
            return {"error": str(e)}
    
    async def _execute_reasoning_agent_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning agent tool."""
        try:
            task_type = input_data.get("task_type", "logical_analysis")
            task_input = input_data.get("input_data", {})
            priority_str = input_data.get("priority", "normal")
            
            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "urgent": TaskPriority.URGENT
            }
            priority = priority_map.get(priority_str, TaskPriority.NORMAL)
            
            task = AgentTask(
                task_id=f"react_reasoning_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}",
                task_type=task_type,
                description=f"ReAct delegated task: {task_type}",
                input_data=task_input,
                priority=priority
            )
            
            reasoning_agent = self.multi_agent_system.get_agent_by_type("reasoning")
            if not reasoning_agent:
                return {"error": "Reasoning agent not available"}
            
            if await reasoning_agent.assign_task(task):
                await reasoning_agent.process_tasks()
                
                max_wait = self.config.tool_timeout * 2  # Reasoning may take longer
                wait_time = 0
                while task.status.value not in ["completed", "failed"] and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1
                
                if task.result:
                    return {
                        "result": task.result,
                        "status": task.status.value,
                        "agent_id": reasoning_agent.agent_id
                    }
                else:
                    return {
                        "error": task.error or "Task failed without result",
                        "status": task.status.value
                    }
            else:
                return {"error": "Failed to assign task to reasoning agent"}
                
        except Exception as e:
            logger.error(f"Error executing reasoning agent tool: {e}")
            return {"error": str(e)}
    
    async def _execute_test_agent_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test agent tool."""
        try:
            task_type = input_data.get("task_type", "test_generation")
            task_input = input_data.get("input_data", {})
            priority_str = input_data.get("priority", "normal")
            
            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "urgent": TaskPriority.URGENT
            }
            priority = priority_map.get(priority_str, TaskPriority.NORMAL)
            
            task = AgentTask(
                task_id=f"react_test_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}",
                task_type=task_type,
                description=f"ReAct delegated task: {task_type}",
                input_data=task_input,
                priority=priority
            )
            
            test_agent = self.multi_agent_system.get_agent_by_type("test")
            if not test_agent:
                return {"error": "Test agent not available"}
            
            if await test_agent.assign_task(task):
                await test_agent.process_tasks()
                
                max_wait = self.config.tool_timeout
                wait_time = 0
                while task.status.value not in ["completed", "failed"] and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1
                
                if task.result:
                    return {
                        "result": task.result,
                        "status": task.status.value,
                        "agent_id": test_agent.agent_id
                    }
                else:
                    return {
                        "error": task.error or "Task failed without result",
                        "status": task.status.value
                    }
            else:
                return {"error": "Failed to assign task to test agent"}
                
        except Exception as e:
            logger.error(f"Error executing test agent tool: {e}")
            return {"error": str(e)}
    
    async def _execute_cot_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning tool."""
        try:
            problem = input_data.get("problem", "")
            complexity_str = input_data.get("complexity", "moderate")
            problem_type = input_data.get("problem_type", "coding_problem")
            
            if not problem:
                return {"error": "No problem provided for chain-of-thought reasoning"}
            
            # Map complexity string to enum
            from chain_of_thought_engine import CoTComplexity
            complexity_map = {
                "simple": CoTComplexity.SIMPLE,
                "moderate": CoTComplexity.MODERATE,
                "complex": CoTComplexity.COMPLEX,
                "expert": CoTComplexity.EXPERT
            }
            complexity = complexity_map.get(complexity_str, CoTComplexity.MODERATE)
            
            # Execute chain-of-thought reasoning
            trace = await self.cot_engine.reason_through_problem(
                problem=problem,
                problem_type=problem_type,
                complexity=complexity
            )
            
            return {
                "reasoning_trace": {
                    "trace_id": trace.trace_id,
                    "steps": len(trace.steps),
                    "final_solution": trace.final_solution,
                    "confidence": trace.confidence_score,
                    "quality": trace.quality_score,
                    "total_time": trace.total_time
                },
                "problem": problem,
                "complexity": complexity_str,
                "success": trace.quality_score >= 0.7
            }
            
        except Exception as e:
            logger.error(f"Error executing chain-of-thought tool: {e}")
            return {"error": str(e)}
    
    async def solve_with_integrated_reasoning(
        self,
        problem: str,
        task_complexity: str = "moderate",
        use_agents: bool = True,
        use_cot: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using integrated ReAct reasoning with agents and CoT.
        
        Args:
            problem: The problem to solve
            task_complexity: Complexity level for ReAct reasoning
            use_agents: Whether to enable agent delegation tools
            use_cot: Whether to enable chain-of-thought reasoning tools
            context: Additional context
            
        Returns:
            Combined results from ReAct reasoning with agent integration
        """
        try:
            # Temporarily configure tool availability
            original_agent_config = self.config.enable_multi_agent_tools
            original_cot_config = self.config.enable_cot_reasoning_tool
            
            self.config.enable_multi_agent_tools = use_agents
            self.config.enable_cot_reasoning_tool = use_cot
            
            # Execute ReAct reasoning
            react_trace = await self.react_framework.solve_problem(
                problem=problem,
                task_complexity=task_complexity,
                context=context
            )
            
            # Collect agent usage statistics
            agent_tools_used = [
                tool for tool in react_trace.tools_used 
                if tool in ["code_agent", "search_agent", "reasoning_agent", "test_agent"]
            ]
            
            cot_tools_used = [
                tool for tool in react_trace.tools_used 
                if tool == "chain_of_thought"
            ]
            
            # Restore original configuration
            self.config.enable_multi_agent_tools = original_agent_config
            self.config.enable_cot_reasoning_tool = original_cot_config
            
            return {
                "react_trace": react_trace,
                "integration_stats": {
                    "agent_tools_used": agent_tools_used,
                    "cot_tools_used": cot_tools_used,
                    "total_tools_used": len(react_trace.tools_used),
                    "agents_enabled": use_agents,
                    "cot_enabled": use_cot
                },
                "success": react_trace.success,
                "confidence": react_trace.confidence_score,
                "final_answer": react_trace.final_answer
            }
            
        except Exception as e:
            logger.error(f"Error in integrated reasoning: {e}")
            return {
                "error": str(e),
                "success": False,
                "confidence": 0.0
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the ReAct integration."""
        return {
            "react_framework_status": self.react_framework.get_framework_status(),
            "multi_agent_status": self.multi_agent_system.get_system_status(),
            "integration_config": {
                "multi_agent_tools_enabled": self.config.enable_multi_agent_tools,
                "cot_reasoning_enabled": self.config.enable_cot_reasoning_tool,
                "context_aware_selection": self.config.enable_context_aware_selection,
                "max_reasoning_depth": self.config.max_reasoning_depth,
                "tool_timeout": self.config.tool_timeout
            },
            "available_integrated_tools": [
                tool_name for tool_name in self.react_framework.tool_registry.list_available_tools()
                if tool_name in ["code_agent", "search_agent", "reasoning_agent", "test_agent", "chain_of_thought"]
            ]
        }


class ContextAwareToolSelector:
    """Enhanced tool selector that considers agent availability and load."""
    
    def __init__(self, react_integration: ReActAgentIntegration):
        self.react_integration = react_integration
        self.selection_history: List[Dict[str, Any]] = []
    
    async def select_optimal_tool(
        self,
        reasoning_context: str,
        available_tools: List[str],
        consider_agent_load: bool = True
    ) -> tuple[Optional[str], float, Dict[str, Any]]:
        """
        Select optimal tool considering agent availability and load.
        
        Returns:
            Tuple of (tool_name, confidence, selection_metadata)
        """
        if not available_tools:
            return None, 0.0, {"reason": "no_tools_available"}
        
        # Get agent tools and their current load
        agent_tools = ["code_agent", "search_agent", "reasoning_agent", "test_agent"]
        agent_tool_loads = {}
        
        if consider_agent_load:
            for tool_name in agent_tools:
                if tool_name in available_tools:
                    agent_type = tool_name.replace("_agent", "")
                    agent = self.react_integration.multi_agent_system.get_agent_by_type(agent_type)
                    if agent:
                        load_factor = len(agent.active_tasks) / agent.max_concurrent_tasks
                        agent_tool_loads[tool_name] = load_factor
                    else:
                        agent_tool_loads[tool_name] = 1.0  # Agent not available
        
        # Use base tool selector
        base_tool, base_confidence = await self.react_integration.react_framework.tool_selector.select_tool(
            reasoning_context, available_tools
        )
        
        # Adjust selection based on agent load
        if base_tool in agent_tool_loads and consider_agent_load:
            load_factor = agent_tool_loads[base_tool]
            
            # Reduce confidence if agent is heavily loaded
            adjusted_confidence = base_confidence * (1.0 - load_factor * 0.3)
            
            # Consider alternative if load is too high
            if load_factor > 0.8:
                # Find alternative tool with lower load
                alternatives = [
                    (tool, load) for tool, load in agent_tool_loads.items()
                    if tool != base_tool and tool in available_tools and load < 0.6
                ]
                
                if alternatives:
                    # Select tool with lowest load
                    alt_tool, alt_load = min(alternatives, key=lambda x: x[1])
                    alt_confidence = base_confidence * 0.8 * (1.0 - alt_load * 0.2)
                    
                    if alt_confidence > adjusted_confidence:
                        return alt_tool, alt_confidence, {
                            "reason": "load_balancing",
                            "original_tool": base_tool,
                            "original_load": load_factor,
                            "selected_load": alt_load
                        }
            
            return base_tool, adjusted_confidence, {
                "reason": "load_adjusted",
                "agent_load": load_factor
            }
        
        return base_tool, base_confidence, {"reason": "base_selection"}
    
    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get tool selection history for analysis."""
        return self.selection_history.copy()