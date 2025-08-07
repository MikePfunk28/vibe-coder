"""
ReAct Framework Tool Management

Tool registry and selection components for the ReAct framework.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from react_core import Tool, ActionType

logger = logging.getLogger(__name__)


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
            tool_name, confidence = self._parse_tool_selection(
                response, available_tools)

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