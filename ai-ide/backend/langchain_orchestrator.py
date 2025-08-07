"""
LangChain Orchestration Layer for AI IDE
Provides workflow management and tool chain building capabilities
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.runnable import Runnable
    from langchain.schema.output_parser import BaseOutputParser
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    # Mock classes for development
    class BaseMessage: pass
    class HumanMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass
    class Runnable: pass
    class BaseOutputParser: pass
    class BaseTool: pass
    class BaseCallbackHandler: pass
    class RunnablePassthrough: pass
    class RunnableLambda: pass
    class ChatPromptTemplate: pass
    class PromptTemplate: pass
    class JsonOutputParser: pass
    class StrOutputParser: pass

# Import existing AI IDE components
from qwen_coder_agent import QwenCoderAgent, get_qwen_coder_agent
from semantic_search_engine import SemanticSearchEngine
from web_search_agent import WebSearchAgent
from multi_agent_system import MultiAgentSystem
from chain_of_thought_engine import ChainOfThoughtEngine
from react_framework import ReActFramework

logger = logging.getLogger('langchain_orchestrator')

class TaskType(Enum):
    """Types of tasks that can be orchestrated"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    SEMANTIC_SEARCH = "semantic_search"
    WEB_SEARCH = "web_search"
    REASONING = "reasoning"
    MULTI_STEP = "multi_step"
    REACT = "react"

class ModelType(Enum):
    """Types of models available for task execution"""
    QWEN_CODER = "qwen_coder"
    REASONING = "reasoning"
    GENERAL = "general"
    SPECIALIZED = "specialized"

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    id: str
    name: str
    tool: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    execution_time: float = 0.0
    error: Optional[str] = None

@dataclass
class Workflow:
    """Represents a complete workflow with multiple steps"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_execution_time: float = 0.0

@dataclass
class ToolChain:
    """Represents a chain of tools to be executed in sequence"""
    id: str
    name: str
    tools: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)

class AIIDECallbackHandler(BaseCallbackHandler):
    """Custom callback handler for AI IDE workflow tracking"""
    
    def __init__(self):
        self.events = []
        self.current_step = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts running"""
        event = {
            "type": "chain_start",
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "serialized": serialized
        }
        self.events.append(event)
        logger.info(f"Chain started: {serialized.get('name', 'Unknown')}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes running"""
        event = {
            "type": "chain_end",
            "timestamp": datetime.now().isoformat(),
            "outputs": outputs
        }
        self.events.append(event)
        logger.info("Chain completed successfully")
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error"""
        event = {
            "type": "chain_error",
            "timestamp": datetime.now().isoformat(),
            "error": str(error)
        }
        self.events.append(event)
        logger.error(f"Chain error: {error}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running"""
        event = {
            "type": "tool_start",
            "timestamp": datetime.now().isoformat(),
            "tool": serialized.get('name', 'Unknown'),
            "input": input_str
        }
        self.events.append(event)
        logger.info(f"Tool started: {serialized.get('name', 'Unknown')}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes running"""
        event = {
            "type": "tool_end",
            "timestamp": datetime.now().isoformat(),
            "output": output
        }
        self.events.append(event)
        logger.info("Tool completed successfully")
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error"""
        event = {
            "type": "tool_error",
            "timestamp": datetime.now().isoformat(),
            "error": str(error)
        }
        self.events.append(event)
        logger.error(f"Tool error: {error}")

class WorkflowManager:
    """Manages complex multi-step AI workflows"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, Workflow] = {}
        self.callback_handler = AIIDECallbackHandler()
        
        # Initialize AI components
        self.qwen_coder_agent: Optional[QwenCoderAgent] = None
        self.semantic_search: Optional[SemanticSearchEngine] = None
        self.web_search: Optional[WebSearchAgent] = None
        self.multi_agent: Optional[MultiAgentSystem] = None
        self.cot_engine: Optional[ChainOfThoughtEngine] = None
        self.react_framework: Optional[ReActFramework] = None
        
        # Tool registry
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    async def initialize(self):
        """Initialize all AI components"""
        try:
            logger.info("Initializing WorkflowManager components...")
            
            # Initialize Qwen Coder agent
            try:
                self.qwen_coder_agent = await get_qwen_coder_agent()
                logger.info("Qwen Coder agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Qwen Coder agent: {e}")
            
            # Initialize other components
            try:
                from semantic_search_engine import get_semantic_search_engine
                self.semantic_search = get_semantic_search_engine()
                logger.info("Semantic search engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic search: {e}")
            
            try:
                from web_search_agent import get_web_search_agent
                self.web_search = await get_web_search_agent()
                logger.info("Web search agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize web search: {e}")
            
            try:
                from multi_agent_system import get_multi_agent_system
                self.multi_agent = get_multi_agent_system()
                logger.info("Multi-agent system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize multi-agent system: {e}")
            
            try:
                from chain_of_thought_engine import get_chain_of_thought_engine
                self.cot_engine = get_chain_of_thought_engine()
                logger.info("Chain of thought engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CoT engine: {e}")
            
            try:
                from react_framework import get_react_framework
                self.react_framework = get_react_framework()
                logger.info("ReAct framework initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ReAct framework: {e}")
            
            logger.info("WorkflowManager initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            raise
    
    def _register_default_tools(self):
        """Register default tools for workflow execution"""
        self.tools.update({
            "code_generation": self._execute_code_generation,
            "semantic_search": self._execute_semantic_search,
            "web_search": self._execute_web_search,
            "reasoning": self._execute_reasoning,
            "multi_agent": self._execute_multi_agent,
            "react": self._execute_react,
            "file_operation": self._execute_file_operation,
            "analysis": self._execute_analysis
        })
    
    def create_workflow(self, task_type: TaskType, context: Dict[str, Any]) -> Workflow:
        """Create a new workflow based on task type and context"""
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        
        if task_type == TaskType.CODE_GENERATION:
            workflow = self._create_code_generation_workflow(workflow_id, context)
        elif task_type == TaskType.SEMANTIC_SEARCH:
            workflow = self._create_semantic_search_workflow(workflow_id, context)
        elif task_type == TaskType.WEB_SEARCH:
            workflow = self._create_web_search_workflow(workflow_id, context)
        elif task_type == TaskType.REASONING:
            workflow = self._create_reasoning_workflow(workflow_id, context)
        elif task_type == TaskType.MULTI_STEP:
            workflow = self._create_multi_step_workflow(workflow_id, context)
        elif task_type == TaskType.REACT:
            workflow = self._create_react_workflow(workflow_id, context)
        else:
            workflow = self._create_default_workflow(workflow_id, context)
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow {workflow_id} for task type {task_type}")
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a complete workflow"""
        workflow.status = "running"
        self.active_workflows[workflow.id] = workflow
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing workflow {workflow.id}: {workflow.name}")
            
            # Execute steps in dependency order
            executed_steps = set()
            results = {}
            
            while len(executed_steps) < len(workflow.steps):
                # Find steps that can be executed (dependencies satisfied)
                ready_steps = [
                    step for step in workflow.steps
                    if step.id not in executed_steps and
                    all(dep in executed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    raise RuntimeError("Circular dependency or missing dependencies in workflow")
                
                # Execute ready steps (can be parallelized in the future)
                for step in ready_steps:
                    step_result = await self._execute_workflow_step(step, workflow.context, results)
                    results[step.id] = step_result
                    executed_steps.add(step.id)
            
            workflow.status = "completed"
            workflow.completed_at = datetime.now()
            workflow.total_execution_time = (workflow.completed_at - start_time).total_seconds()
            
            logger.info(f"Workflow {workflow.id} completed successfully in {workflow.total_execution_time:.2f}s")
            
            return {
                "success": True,
                "workflow_id": workflow.id,
                "results": results,
                "execution_time": workflow.total_execution_time,
                "steps_executed": len(executed_steps)
            }
            
        except Exception as e:
            workflow.status = "failed"
            workflow.completed_at = datetime.now()
            workflow.total_execution_time = (workflow.completed_at - start_time).total_seconds()
            
            logger.error(f"Workflow {workflow.id} failed: {e}")
            
            return {
                "success": False,
                "workflow_id": workflow.id,
                "error": str(e),
                "execution_time": workflow.total_execution_time,
                "steps_executed": len(executed_steps)
            }
        
        finally:
            if workflow.id in self.active_workflows:
                del self.active_workflows[workflow.id]
    
    async def _execute_workflow_step(self, step: WorkflowStep, context: Dict[str, Any], 
                                   previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step.status = "running"
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing step {step.id}: {step.name}")
            
            # Prepare inputs with context and previous results
            inputs = step.inputs.copy()
            inputs.update(context)
            
            # Add results from dependency steps
            for dep_id in step.dependencies:
                if dep_id in previous_results:
                    inputs[f"dep_{dep_id}"] = previous_results[dep_id]
            
            # Execute the tool
            if step.tool in self.tools:
                result = await self.tools[step.tool](inputs)
            else:
                raise ValueError(f"Unknown tool: {step.tool}")
            
            step.status = "completed"
            step.outputs = result
            step.execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Step {step.id} completed in {step.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Step {step.id} failed: {e}")
            raise
    
    def _create_code_generation_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a workflow for code generation tasks"""
        steps = [
            WorkflowStep(
                id="analyze_request",
                name="Analyze Code Request",
                tool="analysis",
                inputs={"request": context.get("prompt", ""), "type": "code_generation"}
            ),
            WorkflowStep(
                id="generate_code",
                name="Generate Code",
                tool="code_generation",
                inputs=context,
                dependencies=["analyze_request"]
            ),
            WorkflowStep(
                id="validate_code",
                name="Validate Generated Code",
                tool="analysis",
                inputs={"type": "code_validation"},
                dependencies=["generate_code"]
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Code Generation Workflow",
            description="Generate and validate code based on user request",
            steps=steps,
            context=context
        )
    
    def _create_semantic_search_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a workflow for semantic search tasks"""
        steps = [
            WorkflowStep(
                id="semantic_search",
                name="Perform Semantic Search",
                tool="semantic_search",
                inputs=context
            ),
            WorkflowStep(
                id="rank_results",
                name="Rank Search Results",
                tool="analysis",
                inputs={"type": "result_ranking"},
                dependencies=["semantic_search"]
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Semantic Search Workflow",
            description="Search and rank code using semantic similarity",
            steps=steps,
            context=context
        )
    
    def _create_reasoning_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a workflow for reasoning tasks"""
        steps = [
            WorkflowStep(
                id="reasoning",
                name="Perform Reasoning",
                tool="reasoning",
                inputs=context
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Reasoning Workflow",
            description="Perform complex reasoning tasks",
            steps=steps,
            context=context
        )
    
    def _create_multi_step_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a complex multi-step workflow"""
        steps = [
            WorkflowStep(
                id="analyze_task",
                name="Analyze Complex Task",
                tool="analysis",
                inputs={"request": context.get("prompt", ""), "type": "task_analysis"}
            ),
            WorkflowStep(
                id="search_context",
                name="Search for Context",
                tool="semantic_search",
                inputs={"query": context.get("prompt", "")},
                dependencies=["analyze_task"]
            ),
            WorkflowStep(
                id="web_research",
                name="Web Research",
                tool="web_search",
                inputs={"query": context.get("prompt", "")},
                dependencies=["analyze_task"]
            ),
            WorkflowStep(
                id="reasoning",
                name="Complex Reasoning",
                tool="reasoning",
                inputs=context,
                dependencies=["search_context", "web_research"]
            ),
            WorkflowStep(
                id="generate_solution",
                name="Generate Solution",
                tool="code_generation",
                inputs=context,
                dependencies=["reasoning"]
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Multi-Step Complex Workflow",
            description="Complex workflow with multiple AI components",
            steps=steps,
            context=context
        )
    
    def _create_react_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a ReAct (Reasoning + Acting) workflow"""
        steps = [
            WorkflowStep(
                id="react_execution",
                name="ReAct Framework Execution",
                tool="react",
                inputs=context
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="ReAct Workflow",
            description="Reasoning and Acting framework execution",
            steps=steps,
            context=context
        )
    
    def _create_web_search_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a workflow for web search tasks"""
        steps = [
            WorkflowStep(
                id="web_search",
                name="Perform Web Search",
                tool="web_search",
                inputs=context
            ),
            WorkflowStep(
                id="analyze_results",
                name="Analyze Search Results",
                tool="analysis",
                inputs={"type": "web_result_analysis"},
                dependencies=["web_search"]
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Web Search Workflow",
            description="Search web and analyze results",
            steps=steps,
            context=context
        )
    
    def _create_default_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Workflow:
        """Create a default workflow for unknown task types"""
        steps = [
            WorkflowStep(
                id="analyze_request",
                name="Analyze Request",
                tool="analysis",
                inputs=context
            ),
            WorkflowStep(
                id="execute_task",
                name="Execute Task",
                tool="reasoning",
                inputs=context,
                dependencies=["analyze_request"]
            )
        ]
        
        return Workflow(
            id=workflow_id,
            name="Default Workflow",
            description="Default workflow for unknown task types",
            steps=steps,
            context=context
        )
    
    # Tool execution methods
    async def _execute_code_generation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation using Qwen Coder agent"""
        if self.qwen_coder_agent:
            try:
                from qwen_coder_agent import CodeRequest, CodeContext, CodeTaskType
                
                # Create code request
                code_request = CodeRequest(
                    prompt=inputs.get("prompt", ""),
                    task_type=CodeTaskType.GENERATION,
                    context=CodeContext(
                        language=inputs.get("language", "python"),
                        file_path=inputs.get("file_path"),
                        selected_text=inputs.get("selected_text"),
                        cursor_position=inputs.get("cursor_position"),
                        surrounding_code=inputs.get("surrounding_code")
                    ),
                    max_tokens=inputs.get("max_tokens", 2048),
                    temperature=inputs.get("temperature", 0.3)
                )
                
                response = await self.qwen_coder_agent.generate_code(code_request)
                
                return {
                    "success": True,
                    "code": response.code,
                    "language": response.language,
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                    "execution_time": response.execution_time
                }
                
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Qwen Coder agent not available"}
    
    async def _execute_semantic_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search"""
        if self.semantic_search:
            try:
                query = inputs.get("query", "")
                max_results = inputs.get("max_results", 10)
                
                results = await self.semantic_search.search_async(query, max_results)
                
                return {
                    "success": True,
                    "results": results,
                    "query": query,
                    "total_results": len(results)
                }
                
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Semantic search not available"}
    
    async def _execute_web_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search"""
        if self.web_search:
            try:
                query = inputs.get("query", "")
                max_results = inputs.get("max_results", 5)
                
                results = await self.web_search.search_async(query, max_results)
                
                return {
                    "success": True,
                    "results": results,
                    "query": query,
                    "total_results": len(results)
                }
                
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Web search not available"}
    
    async def _execute_reasoning(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning task"""
        if self.cot_engine:
            try:
                problem = inputs.get("problem", inputs.get("prompt", ""))
                mode = inputs.get("mode", "basic")
                
                result = await self.cot_engine.reason_async(problem, mode)
                
                return {
                    "success": True,
                    "solution": result.get("solution", ""),
                    "reasoning": result.get("reasoning", []),
                    "confidence": result.get("confidence", 0.8)
                }
                
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Chain of thought engine not available"}
    
    async def _execute_multi_agent(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-agent task"""
        if self.multi_agent:
            try:
                task = inputs.get("task", inputs.get("prompt", ""))
                
                result = await self.multi_agent.execute_task_async(task)
                
                return {
                    "success": True,
                    "result": result.get("result", ""),
                    "agents_used": result.get("agents_used", []),
                    "coordination_log": result.get("coordination_log", [])
                }
                
            except Exception as e:
                logger.error(f"Multi-agent execution failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Multi-agent system not available"}
    
    async def _execute_react(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ReAct framework task"""
        if self.react_framework:
            try:
                task = inputs.get("task", inputs.get("prompt", ""))
                
                result = await self.react_framework.execute_async(task)
                
                return {
                    "success": True,
                    "result": result.get("result", ""),
                    "reasoning_trace": result.get("reasoning_trace", []),
                    "actions_taken": result.get("actions_taken", [])
                }
                
            except Exception as e:
                logger.error(f"ReAct execution failed: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "ReAct framework not available"}
    
    async def _execute_file_operation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file operations"""
        try:
            operation = inputs.get("operation", "read")
            file_path = inputs.get("file_path", "")
            
            if operation == "read":
                from utils.read_file import read_file
                content, success = read_file(file_path)
                return {"success": success, "content": content}
            
            elif operation == "write":
                from utils.replace_file import replace_file
                content = inputs.get("content", "")
                success, message = replace_file(file_path, content)
                return {"success": success, "message": message}
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis tasks"""
        try:
            analysis_type = inputs.get("type", "general")
            
            if analysis_type == "code_generation":
                request = inputs.get("request", "")
                return {
                    "success": True,
                    "analysis": f"Code generation request analyzed: {request}",
                    "complexity": "medium",
                    "estimated_time": 5.0
                }
            
            elif analysis_type == "task_analysis":
                request = inputs.get("request", "")
                return {
                    "success": True,
                    "analysis": f"Task analyzed: {request}",
                    "subtasks": ["analyze", "research", "implement"],
                    "complexity": "high"
                }
            
            else:
                return {
                    "success": True,
                    "analysis": f"General analysis completed for type: {analysis_type}",
                    "type": analysis_type
                }
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow"""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            return {
                "id": workflow.id,
                "name": workflow.name,
                "status": workflow.status,
                "steps_total": len(workflow.steps),
                "steps_completed": len([s for s in workflow.steps if s.status == "completed"]),
                "steps_failed": len([s for s in workflow.steps if s.status == "failed"]),
                "execution_time": workflow.total_execution_time,
                "created_at": workflow.created_at.isoformat(),
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
            }
        return None
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "status": workflow.status,
                "created_at": workflow.created_at.isoformat()
            }
            for workflow in self.workflows.values()
        ]

# Global workflow manager instance
_workflow_manager: Optional[WorkflowManager] = None

async def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
        await _workflow_manager.initialize()
    return _workflow_manager