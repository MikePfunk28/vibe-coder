"""
Tool Chain Builder for AI IDE
Dynamically constructs tool sequences for complex tasks
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.schema import BaseMessage
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    # Mock classes for development
    class BaseTool: pass
    class StructuredTool: pass
    class RunnablePassthrough: pass
    class RunnableLambda: pass
    class RunnableSequence: pass
    class ChatPromptTemplate: pass
    class BaseMessage: pass

logger = logging.getLogger('tool_chain_builder')

class ToolType(Enum):
    """Types of tools available in the system"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    SEMANTIC_SEARCH = "semantic_search"
    WEB_SEARCH = "web_search"
    FILE_OPERATION = "file_operation"
    REASONING = "reasoning"
    MULTI_AGENT = "multi_agent"
    REACT = "react"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"

@dataclass
class ToolDefinition:
    """Definition of a tool that can be used in chains"""
    name: str
    tool_type: ToolType
    description: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    execution_function: Optional[Callable] = None
    estimated_time: float = 1.0
    complexity: str = "medium"  # low, medium, high
    reliability: float = 0.9  # 0.0 to 1.0

@dataclass
class ToolChainStep:
    """A single step in a tool chain"""
    id: str
    tool_name: str
    tool_type: ToolType
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    execution_time: float = 0.0
    error: Optional[str] = None

@dataclass
class ToolChain:
    """A complete tool chain with multiple steps"""
    id: str
    name: str
    description: str
    steps: List[ToolChainStep]
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    total_estimated_time: float = 0.0
    actual_execution_time: float = 0.0

class ToolChainBuilder:
    """Builds dynamic tool chains for complex AI tasks"""
    
    def __init__(self):
        self.tool_registry: Dict[str, ToolDefinition] = {}
        self.chain_templates: Dict[str, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Initialize with default tools
        self._register_default_tools()
        self._register_chain_templates()
    
    def _register_default_tools(self):
        """Register default tools available in the AI IDE"""
        
        # Code generation tools
        self.register_tool(ToolDefinition(
            name="qwen_coder_generation",
            tool_type=ToolType.CODE_GENERATION,
            description="Generate code using Qwen Coder 3 model",
            inputs=["prompt", "language", "context"],
            outputs=["code", "explanation", "confidence"],
            estimated_time=3.0,
            complexity="medium",
            reliability=0.9
        ))
        
        self.register_tool(ToolDefinition(
            name="code_completion",
            tool_type=ToolType.CODE_GENERATION,
            description="Complete partial code snippets",
            inputs=["partial_code", "language", "context"],
            outputs=["completed_code", "confidence"],
            estimated_time=1.5,
            complexity="low",
            reliability=0.85
        ))
        
        # Search tools
        self.register_tool(ToolDefinition(
            name="semantic_code_search",
            tool_type=ToolType.SEMANTIC_SEARCH,
            description="Search code using semantic similarity",
            inputs=["query", "max_results"],
            outputs=["search_results", "relevance_scores"],
            estimated_time=2.0,
            complexity="medium",
            reliability=0.88
        ))
        
        self.register_tool(ToolDefinition(
            name="web_search",
            tool_type=ToolType.WEB_SEARCH,
            description="Search the web for information",
            inputs=["query", "max_results"],
            outputs=["search_results", "summaries"],
            estimated_time=4.0,
            complexity="high",
            reliability=0.75
        ))
        
        # Analysis tools
        self.register_tool(ToolDefinition(
            name="code_analysis",
            tool_type=ToolType.CODE_ANALYSIS,
            description="Analyze code structure and quality",
            inputs=["code", "language"],
            outputs=["analysis", "suggestions", "metrics"],
            estimated_time=2.5,
            complexity="medium",
            reliability=0.9
        ))
        
        self.register_tool(ToolDefinition(
            name="complexity_analysis",
            tool_type=ToolType.CODE_ANALYSIS,
            description="Analyze code complexity and maintainability",
            inputs=["code", "language"],
            outputs=["complexity_score", "maintainability", "recommendations"],
            estimated_time=1.8,
            complexity="low",
            reliability=0.92
        ))
        
        # Reasoning tools
        self.register_tool(ToolDefinition(
            name="chain_of_thought",
            tool_type=ToolType.REASONING,
            description="Perform step-by-step reasoning",
            inputs=["problem", "context"],
            outputs=["reasoning_steps", "solution", "confidence"],
            estimated_time=5.0,
            complexity="high",
            reliability=0.85
        ))
        
        self.register_tool(ToolDefinition(
            name="deep_reasoning",
            tool_type=ToolType.REASONING,
            description="Perform deep analysis and reasoning",
            inputs=["problem", "context", "constraints"],
            outputs=["analysis", "solution", "alternatives"],
            estimated_time=8.0,
            complexity="high",
            reliability=0.8
        ))
        
        # File operation tools
        self.register_tool(ToolDefinition(
            name="file_reader",
            tool_type=ToolType.FILE_OPERATION,
            description="Read file contents",
            inputs=["file_path"],
            outputs=["content", "metadata"],
            estimated_time=0.5,
            complexity="low",
            reliability=0.95
        ))
        
        self.register_tool(ToolDefinition(
            name="file_writer",
            tool_type=ToolType.FILE_OPERATION,
            description="Write content to file",
            inputs=["file_path", "content"],
            outputs=["success", "message"],
            estimated_time=0.8,
            complexity="low",
            reliability=0.93
        ))
        
        # Validation tools
        self.register_tool(ToolDefinition(
            name="code_validator",
            tool_type=ToolType.VALIDATION,
            description="Validate code syntax and structure",
            inputs=["code", "language"],
            outputs=["is_valid", "errors", "warnings"],
            estimated_time=1.2,
            complexity="low",
            reliability=0.95
        ))
        
        self.register_tool(ToolDefinition(
            name="output_validator",
            tool_type=ToolType.VALIDATION,
            description="Validate tool outputs against requirements",
            inputs=["output", "requirements"],
            outputs=["is_valid", "compliance_score", "issues"],
            estimated_time=1.0,
            complexity="low",
            reliability=0.9
        ))
        
        # Multi-agent tools
        self.register_tool(ToolDefinition(
            name="multi_agent_coordination",
            tool_type=ToolType.MULTI_AGENT,
            description="Coordinate multiple AI agents",
            inputs=["task", "agent_types"],
            outputs=["coordination_plan", "agent_assignments"],
            estimated_time=3.5,
            complexity="high",
            reliability=0.82
        ))
        
        # ReAct tools
        self.register_tool(ToolDefinition(
            name="react_framework",
            tool_type=ToolType.REACT,
            description="Reasoning and Acting framework",
            inputs=["task", "available_tools"],
            outputs=["reasoning_trace", "actions", "result"],
            estimated_time=6.0,
            complexity="high",
            reliability=0.78
        ))
    
    def _register_chain_templates(self):
        """Register common tool chain templates"""
        
        # Code generation chain
        self.chain_templates["code_generation"] = [
            "semantic_code_search",  # Find similar code
            "qwen_coder_generation",  # Generate code
            "code_validator",  # Validate generated code
            "code_analysis"  # Analyze quality
        ]
        
        # Research and development chain
        self.chain_templates["research_development"] = [
            "web_search",  # Research the topic
            "semantic_code_search",  # Find existing implementations
            "chain_of_thought",  # Reason about approach
            "qwen_coder_generation",  # Generate solution
            "code_validator",  # Validate result
            "complexity_analysis"  # Analyze complexity
        ]
        
        # Code analysis chain
        self.chain_templates["code_analysis"] = [
            "file_reader",  # Read code file
            "code_analysis",  # Analyze structure
            "complexity_analysis",  # Analyze complexity
            "code_validator"  # Validate syntax
        ]
        
        # Multi-step reasoning chain
        self.chain_templates["complex_reasoning"] = [
            "semantic_code_search",  # Gather context
            "web_search",  # Research background
            "deep_reasoning",  # Perform analysis
            "chain_of_thought",  # Step-by-step solution
            "output_validator"  # Validate reasoning
        ]
        
        # ReAct problem solving chain
        self.chain_templates["react_problem_solving"] = [
            "react_framework"  # Use ReAct for dynamic tool usage
        ]
        
        # Multi-agent collaboration chain
        self.chain_templates["multi_agent_task"] = [
            "multi_agent_coordination",  # Plan agent coordination
            "semantic_code_search",  # Gather context
            "qwen_coder_generation",  # Generate code
            "code_validator"  # Validate result
        ]
    
    def register_tool(self, tool_definition: ToolDefinition):
        """Register a new tool in the registry"""
        self.tool_registry[tool_definition.name] = tool_definition
        logger.info(f"Registered tool: {tool_definition.name} ({tool_definition.tool_type.value})")
    
    def build_chain_from_template(self, template_name: str, context: Dict[str, Any]) -> ToolChain:
        """Build a tool chain from a predefined template"""
        if template_name not in self.chain_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        tool_names = self.chain_templates[template_name]
        return self.build_chain(tool_names, context, f"Template: {template_name}")
    
    def build_chain(self, tool_names: List[str], context: Dict[str, Any], 
                   description: str = "") -> ToolChain:
        """Build a tool chain from a list of tool names"""
        chain_id = f"chain_{datetime.now().timestamp()}"
        
        # Validate all tools exist
        for tool_name in tool_names:
            if tool_name not in self.tool_registry:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        # Create chain steps
        steps = []
        total_estimated_time = 0.0
        
        for i, tool_name in enumerate(tool_names):
            tool_def = self.tool_registry[tool_name]
            
            # Determine dependencies (previous step)
            dependencies = [f"step_{i-1}"] if i > 0 else []
            
            step = ToolChainStep(
                id=f"step_{i}",
                tool_name=tool_name,
                tool_type=tool_def.tool_type,
                inputs=self._prepare_tool_inputs(tool_def, context, i),
                dependencies=dependencies
            )
            
            steps.append(step)
            total_estimated_time += tool_def.estimated_time
        
        chain = ToolChain(
            id=chain_id,
            name=f"Chain_{len(tool_names)}_steps",
            description=description or f"Chain with {len(tool_names)} tools",
            steps=steps,
            context=context,
            total_estimated_time=total_estimated_time
        )
        
        logger.info(f"Built tool chain {chain_id} with {len(steps)} steps")
        return chain
    
    def build_adaptive_chain(self, task_description: str, context: Dict[str, Any]) -> ToolChain:
        """Build a tool chain adaptively based on task analysis"""
        
        # Analyze the task to determine required tools
        required_tools = self._analyze_task_requirements(task_description, context)
        
        # Optimize tool sequence
        optimized_sequence = self._optimize_tool_sequence(required_tools, context)
        
        return self.build_chain(
            optimized_sequence, 
            context, 
            f"Adaptive chain for: {task_description[:50]}..."
        )
    
    def _analyze_task_requirements(self, task_description: str, context: Dict[str, Any]) -> List[str]:
        """Analyze task to determine required tools"""
        task_lower = task_description.lower()
        required_tools = []
        
        # Code generation indicators
        if any(keyword in task_lower for keyword in ['generate', 'create', 'write', 'implement']):
            if 'code' in task_lower or any(lang in task_lower for lang in ['python', 'javascript', 'java']):
                required_tools.extend(['semantic_code_search', 'qwen_coder_generation', 'code_validator'])
        
        # Search indicators
        if any(keyword in task_lower for keyword in ['find', 'search', 'locate', 'discover']):
            if 'web' in task_lower or 'internet' in task_lower:
                required_tools.append('web_search')
            else:
                required_tools.append('semantic_code_search')
        
        # Analysis indicators
        if any(keyword in task_lower for keyword in ['analyze', 'review', 'check', 'examine']):
            required_tools.extend(['code_analysis', 'complexity_analysis'])
        
        # Reasoning indicators
        if any(keyword in task_lower for keyword in ['explain', 'reason', 'think', 'solve']):
            if 'complex' in task_lower or 'difficult' in task_lower:
                required_tools.append('deep_reasoning')
            else:
                required_tools.append('chain_of_thought')
        
        # File operation indicators
        if any(keyword in task_lower for keyword in ['read', 'file', 'open']):
            required_tools.append('file_reader')
        
        if any(keyword in task_lower for keyword in ['save', 'write', 'store']):
            required_tools.append('file_writer')
        
        # Multi-agent indicators
        if any(keyword in task_lower for keyword in ['collaborate', 'multiple', 'team']):
            required_tools.append('multi_agent_coordination')
        
        # ReAct indicators
        if any(keyword in task_lower for keyword in ['dynamic', 'adaptive', 'flexible']):
            required_tools.append('react_framework')
        
        # Default fallback
        if not required_tools:
            required_tools = ['semantic_code_search', 'chain_of_thought']
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in required_tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _optimize_tool_sequence(self, tools: List[str], context: Dict[str, Any]) -> List[str]:
        """Optimize the sequence of tools for better performance"""
        
        # Group tools by type for better sequencing
        tool_groups = {
            'search': [],
            'generation': [],
            'analysis': [],
            'reasoning': [],
            'validation': [],
            'file_ops': [],
            'coordination': []
        }
        
        for tool_name in tools:
            if tool_name not in self.tool_registry:
                continue
                
            tool_def = self.tool_registry[tool_name]
            tool_type = tool_def.tool_type
            
            if tool_type in [ToolType.SEMANTIC_SEARCH, ToolType.WEB_SEARCH]:
                tool_groups['search'].append(tool_name)
            elif tool_type == ToolType.CODE_GENERATION:
                tool_groups['generation'].append(tool_name)
            elif tool_type == ToolType.CODE_ANALYSIS:
                tool_groups['analysis'].append(tool_name)
            elif tool_type == ToolType.REASONING:
                tool_groups['reasoning'].append(tool_name)
            elif tool_type == ToolType.VALIDATION:
                tool_groups['validation'].append(tool_name)
            elif tool_type == ToolType.FILE_OPERATION:
                tool_groups['file_ops'].append(tool_name)
            elif tool_type in [ToolType.MULTI_AGENT, ToolType.REACT]:
                tool_groups['coordination'].append(tool_name)
        
        # Optimal sequence: search -> reasoning -> generation -> analysis -> validation
        optimized_sequence = []
        
        # Add file reading first if needed
        if 'file_reader' in tool_groups['file_ops']:
            optimized_sequence.append('file_reader')
            tool_groups['file_ops'].remove('file_reader')
        
        # Add search tools
        optimized_sequence.extend(tool_groups['search'])
        
        # Add reasoning tools
        optimized_sequence.extend(tool_groups['reasoning'])
        
        # Add coordination tools (ReAct, multi-agent)
        optimized_sequence.extend(tool_groups['coordination'])
        
        # Add generation tools
        optimized_sequence.extend(tool_groups['generation'])
        
        # Add analysis tools
        optimized_sequence.extend(tool_groups['analysis'])
        
        # Add validation tools
        optimized_sequence.extend(tool_groups['validation'])
        
        # Add remaining file operations
        optimized_sequence.extend(tool_groups['file_ops'])
        
        return optimized_sequence
    
    def _prepare_tool_inputs(self, tool_def: ToolDefinition, context: Dict[str, Any], 
                           step_index: int) -> Dict[str, Any]:
        """Prepare inputs for a tool based on context and step position"""
        inputs = {}
        
        # Map common context keys to tool inputs
        for input_name in tool_def.inputs:
            if input_name in context:
                inputs[input_name] = context[input_name]
            elif input_name == "prompt" and "task_description" in context:
                inputs[input_name] = context["task_description"]
            elif input_name == "query" and "prompt" in context:
                inputs[input_name] = context["prompt"]
            elif input_name == "problem" and "prompt" in context:
                inputs[input_name] = context["prompt"]
            elif input_name == "task" and "prompt" in context:
                inputs[input_name] = context["prompt"]
            elif input_name == "max_results":
                inputs[input_name] = context.get("max_results", 10)
            elif input_name == "language":
                inputs[input_name] = context.get("language", "python")
        
        # Add step-specific context
        inputs["step_index"] = step_index
        inputs["tool_name"] = tool_def.name
        
        return inputs
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        if tool_name in self.tool_registry:
            tool_def = self.tool_registry[tool_name]
            return {
                "name": tool_def.name,
                "type": tool_def.tool_type.value,
                "description": tool_def.description,
                "inputs": tool_def.inputs,
                "outputs": tool_def.outputs,
                "dependencies": tool_def.dependencies,
                "estimated_time": tool_def.estimated_time,
                "complexity": tool_def.complexity,
                "reliability": tool_def.reliability
            }
        return None
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                "name": tool_def.name,
                "type": tool_def.tool_type.value,
                "description": tool_def.description,
                "complexity": tool_def.complexity,
                "reliability": tool_def.reliability
            }
            for tool_def in self.tool_registry.values()
        ]
    
    def list_chain_templates(self) -> Dict[str, List[str]]:
        """List all available chain templates"""
        return self.chain_templates.copy()
    
    def estimate_chain_performance(self, chain: ToolChain) -> Dict[str, Any]:
        """Estimate the performance characteristics of a tool chain"""
        total_time = 0.0
        total_reliability = 1.0
        complexity_scores = {"low": 1, "medium": 2, "high": 3}
        max_complexity = 0
        
        for step in chain.steps:
            if step.tool_name in self.tool_registry:
                tool_def = self.tool_registry[step.tool_name]
                total_time += tool_def.estimated_time
                total_reliability *= tool_def.reliability
                max_complexity = max(max_complexity, complexity_scores.get(tool_def.complexity, 2))
        
        complexity_names = {1: "low", 2: "medium", 3: "high"}
        
        return {
            "estimated_time": total_time,
            "estimated_reliability": total_reliability,
            "overall_complexity": complexity_names[max_complexity],
            "steps_count": len(chain.steps),
            "parallel_potential": self._analyze_parallelization_potential(chain)
        }
    
    def _analyze_parallelization_potential(self, chain: ToolChain) -> Dict[str, Any]:
        """Analyze which steps in the chain can be parallelized"""
        parallel_groups = []
        current_group = []
        
        for step in chain.steps:
            if not step.dependencies:
                # Can start immediately
                if not current_group:
                    current_group = [step.id]
                else:
                    parallel_groups.append(current_group)
                    current_group = [step.id]
            else:
                # Has dependencies, check if they're satisfied by current group
                deps_satisfied = all(dep in [s for group in parallel_groups for s in group] 
                                   for dep in step.dependencies)
                if deps_satisfied:
                    current_group.append(step.id)
                else:
                    if current_group:
                        parallel_groups.append(current_group)
                    current_group = [step.id]
        
        if current_group:
            parallel_groups.append(current_group)
        
        return {
            "parallel_groups": parallel_groups,
            "max_parallelism": max(len(group) for group in parallel_groups) if parallel_groups else 1,
            "sequential_groups": len(parallel_groups)
        }

# Global tool chain builder instance
_tool_chain_builder: Optional[ToolChainBuilder] = None

def get_tool_chain_builder() -> ToolChainBuilder:
    """Get the global tool chain builder instance"""
    global _tool_chain_builder
    if _tool_chain_builder is None:
        _tool_chain_builder = ToolChainBuilder()
    return _tool_chain_builder