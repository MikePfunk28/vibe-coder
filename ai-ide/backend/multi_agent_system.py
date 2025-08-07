"""
Multi-Agent System Architecture

Implementation of specialized agents for different AI IDE tasks including:
- CodeAgent: Code generation, completion, and analysis
- SearchAgent: Semantic search and code discovery
- ReasoningAgent: Complex problem solving and analysis
- TestAgent: Test generation and validation

Includes agent communication protocols, task delegation, and performance monitoring.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    CODE = "code"
    SEARCH = "search"
    REASONING = "reasoning"
    TEST = "test"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Represents a task for agent execution."""
    task_id: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent."""
    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    last_activity: Optional[datetime] = None
    specialization_score: float = 0.0
    load_factor: float = 0.0


class BaseAgent(ABC):
    """Base class for all specialized agents."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        capabilities: List[str],
        max_concurrent_tasks: int = 3
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task management
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: List[AgentMessage] = []
        
        # Performance tracking
        self.metrics = AgentPerformanceMetrics(agent_id=agent_id)
        self.is_active = True
        
        logger.info(f"Initialized {agent_type.value} agent: {agent_id}")
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_handle_task(self, task: AgentTask) -> float:
        """Return confidence score (0-1) for handling this task type."""
        pass
    
    async def assign_task(self, task: AgentTask) -> bool:
        """Assign a task to this agent."""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return False
        
        if not self.can_handle_task(task):
            return False
        
        task.assigned_agent = self.agent_id
        task.status = TaskStatus.ASSIGNED
        self.task_queue.append(task)
        
        logger.debug(f"Task {task.task_id} assigned to agent {self.agent_id}")
        return True
    
    async def process_tasks(self) -> None:
        """Process tasks from the queue."""
        while self.task_queue and len(self.active_tasks) < self.max_concurrent_tasks:
            task = self.task_queue.pop(0)
            await self._execute_task_with_tracking(task)
    
    async def _execute_task_with_tracking(self, task: AgentTask) -> None:
        """Execute task with performance tracking."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        start_time = time.time()
        
        try:
            result = await self.execute_task(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update metrics
            self.metrics.tasks_completed += 1
            execution_time = time.time() - start_time
            self._update_avg_execution_time(execution_time)
            
            logger.debug(f"Task {task.task_id} completed by {self.agent_id} in {execution_time:.2f}s")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            self.metrics.tasks_failed += 1
            
            logger.error(f"Task {task.task_id} failed in agent {self.agent_id}: {e}")
        
        finally:
            # Move to completed tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            # Update success rate
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks > 0:
                self.metrics.success_rate = self.metrics.tasks_completed / total_tasks
            
            self.metrics.last_activity = datetime.now()
    
    def _update_avg_execution_time(self, execution_time: float) -> None:
        """Update average execution time."""
        if self.metrics.tasks_completed == 1:
            self.metrics.avg_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_execution_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.avg_execution_time
            )
    
    async def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any]) -> str:
        """Send message to another agent."""
        message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content
        )
        
        # This would be handled by the MultiAgentSystem coordinator
        logger.debug(f"Agent {self.agent_id} sending message to {recipient_id}: {message_type}")
        return message.message_id
    
    async def handle_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming message."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            return None
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "capabilities": self.capabilities,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": self.metrics.success_rate,
                "avg_execution_time": self.metrics.avg_execution_time,
                "load_factor": len(self.active_tasks) / self.max_concurrent_tasks
            }
        }


class CodeAgent(BaseAgent):
    """Specialized agent for code generation, completion, and analysis."""
    
    def __init__(self, agent_id: str, llm_client, context_manager):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CODE,
            capabilities=[
                "code_completion",
                "code_generation",
                "code_analysis",
                "refactoring",
                "bug_fixing",
                "documentation_generation"
            ],
            max_concurrent_tasks=5
        )
        self.llm_client = llm_client
        self.context_manager = context_manager
        
        # Register message handlers
        self.register_message_handler("code_review_request", self._handle_code_review)
        self.register_message_handler("refactor_request", self._handle_refactor_request)
    
    def can_handle_task(self, task: AgentTask) -> float:
        """Determine if this agent can handle the task."""
        code_task_types = {
            "code_completion": 0.95,
            "code_generation": 0.9,
            "code_analysis": 0.85,
            "refactoring": 0.8,
            "bug_fixing": 0.75,
            "documentation": 0.7
        }
        return code_task_types.get(task.task_type, 0.0)
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute code-related tasks."""
        task_type = task.task_type
        input_data = task.input_data
        
        if task_type == "code_completion":
            return await self._handle_code_completion(input_data)
        elif task_type == "code_generation":
            return await self._handle_code_generation(input_data)
        elif task_type == "code_analysis":
            return await self._handle_code_analysis(input_data)
        elif task_type == "refactoring":
            return await self._handle_refactoring(input_data)
        elif task_type == "bug_fixing":
            return await self._handle_bug_fixing(input_data)
        elif task_type == "documentation":
            return await self._handle_documentation(input_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_code_completion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code completion requests."""
        code_context = input_data.get("code_context", "")
        cursor_position = input_data.get("cursor_position", 0)
        
        # Get relevant context
        context = self.context_manager.get_relevant_context(code_context, max_tokens=1024)
        
        # Generate completion
        prompt = f"""
        Complete the following code:
        
        {code_context}
        
        Provide a natural completion that fits the context.
        """
        
        completion = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.2
        )
        
        return {
            "completion": completion.strip(),
            "confidence": 0.8,
            "context_used": len(context)
        }
    
    async def _handle_code_generation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation requests."""
        description = input_data.get("description", "")
        language = input_data.get("language", "python")
        requirements = input_data.get("requirements", [])
        
        prompt = f"""
        Generate {language} code for the following:
        
        Description: {description}
        Requirements: {', '.join(requirements)}
        
        Provide clean, well-documented code:
        """
        
        generated_code = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3
        )
        
        return {
            "generated_code": generated_code.strip(),
            "language": language,
            "confidence": 0.75
        }
    
    async def _handle_code_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code analysis requests."""
        code = input_data.get("code", "")
        analysis_type = input_data.get("analysis_type", "general")
        
        prompt = f"""
        Analyze the following code for {analysis_type}:
        
        {code}
        
        Provide analysis including:
        1. Code quality assessment
        2. Potential issues or bugs
        3. Suggestions for improvement
        4. Performance considerations
        """
        
        analysis = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.2
        )
        
        return {
            "analysis": analysis.strip(),
            "analysis_type": analysis_type,
            "confidence": 0.8
        }
    
    async def _handle_refactoring(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code refactoring requests."""
        code = input_data.get("code", "")
        refactor_goal = input_data.get("goal", "improve readability")
        
        prompt = f"""
        Refactor the following code to {refactor_goal}:
        
        {code}
        
        Provide the refactored code with explanations:
        """
        
        refactored = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2
        )
        
        return {
            "refactored_code": refactored.strip(),
            "refactor_goal": refactor_goal,
            "confidence": 0.7
        }
    
    async def _handle_bug_fixing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bug fixing requests."""
        code = input_data.get("code", "")
        error_message = input_data.get("error_message", "")
        
        prompt = f"""
        Fix the bug in the following code:
        
        Code:
        {code}
        
        Error: {error_message}
        
        Provide the fixed code with explanation:
        """
        
        fix = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.2
        )
        
        return {
            "fixed_code": fix.strip(),
            "error_message": error_message,
            "confidence": 0.75
        }
    
    async def _handle_documentation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation generation."""
        code = input_data.get("code", "")
        doc_type = input_data.get("doc_type", "docstring")
        
        prompt = f"""
        Generate {doc_type} documentation for:
        
        {code}
        
        Provide clear, comprehensive documentation:
        """
        
        documentation = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.2
        )
        
        return {
            "documentation": documentation.strip(),
            "doc_type": doc_type,
            "confidence": 0.8
        }
    
    async def _handle_code_review(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle code review requests from other agents."""
        code = message.content.get("code", "")
        
        analysis_result = await self._handle_code_analysis({
            "code": code,
            "analysis_type": "code_review"
        })
        
        return {
            "review_result": analysis_result,
            "reviewer": self.agent_id
        }
    
    async def _handle_refactor_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle refactoring requests from other agents."""
        code = message.content.get("code", "")
        goal = message.content.get("goal", "improve code quality")
        
        refactor_result = await self._handle_refactoring({
            "code": code,
            "goal": goal
        })
        
        return {
            "refactor_result": refactor_result,
            "refactorer": self.agent_id
        }


class SearchAgent(BaseAgent):
    """Specialized agent for semantic search and code discovery."""
    
    def __init__(self, agent_id: str, search_engine, embedding_generator):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.SEARCH,
            capabilities=[
                "semantic_search",
                "code_discovery",
                "pattern_matching",
                "similarity_analysis",
                "context_retrieval"
            ],
            max_concurrent_tasks=8
        )
        self.search_engine = search_engine
        self.embedding_generator = embedding_generator
        
        # Register message handlers
        self.register_message_handler("search_request", self._handle_search_request)
        self.register_message_handler("similarity_request", self._handle_similarity_request)
    
    def can_handle_task(self, task: AgentTask) -> float:
        """Determine if this agent can handle the task."""
        search_task_types = {
            "semantic_search": 0.95,
            "code_discovery": 0.9,
            "pattern_matching": 0.85,
            "similarity_analysis": 0.8,
            "context_retrieval": 0.75
        }
        return search_task_types.get(task.task_type, 0.0)
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute search-related tasks."""
        task_type = task.task_type
        input_data = task.input_data
        
        if task_type == "semantic_search":
            return await self._handle_semantic_search(input_data)
        elif task_type == "code_discovery":
            return await self._handle_code_discovery(input_data)
        elif task_type == "pattern_matching":
            return await self._handle_pattern_matching(input_data)
        elif task_type == "similarity_analysis":
            return await self._handle_similarity_analysis(input_data)
        elif task_type == "context_retrieval":
            return await self._handle_context_retrieval(input_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_semantic_search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search requests."""
        query = input_data.get("query", "")
        max_results = input_data.get("max_results", 10)
        
        results = await self.search_engine.search_similar(query, max_results=max_results)
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "confidence": 0.85
        }
    
    async def _handle_code_discovery(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code discovery requests."""
        pattern = input_data.get("pattern", "")
        language = input_data.get("language", "")
        
        # Use semantic search with code-specific context
        search_query = f"{pattern} {language} code example"
        results = await self.search_engine.search_similar(search_query, max_results=15)
        
        # Filter for code-specific results
        code_results = [r for r in results if self._is_code_result(r)]
        
        return {
            "code_results": code_results,
            "pattern": pattern,
            "language": language,
            "confidence": 0.8
        }
    
    async def _handle_pattern_matching(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pattern matching requests."""
        target_pattern = input_data.get("target_pattern", "")
        search_space = input_data.get("search_space", [])
        
        matches = []
        for item in search_space:
            similarity = await self._calculate_pattern_similarity(target_pattern, item)
            if similarity > 0.6:  # Threshold for matches
                matches.append({
                    "item": item,
                    "similarity": similarity
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "matches": matches,
            "target_pattern": target_pattern,
            "confidence": 0.75
        }
    
    async def _handle_similarity_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle similarity analysis requests."""
        item1 = input_data.get("item1", "")
        item2 = input_data.get("item2", "")
        
        similarity = await self._calculate_pattern_similarity(item1, item2)
        
        return {
            "similarity_score": similarity,
            "item1": item1,
            "item2": item2,
            "confidence": 0.9
        }
    
    async def _handle_context_retrieval(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context retrieval requests."""
        query = input_data.get("query", "")
        max_tokens = input_data.get("max_tokens", 2048)
        
        context = self.search_engine.get_relevant_context(query, max_tokens=max_tokens)
        
        return {
            "context": context,
            "query": query,
            "token_count": sum(len(c.content.split()) for c in context),
            "confidence": 0.85
        }
    
    def _is_code_result(self, result) -> bool:
        """Check if a search result contains code."""
        content = result.get("content", "").lower()
        code_indicators = ["def ", "class ", "function", "import", "return", "{", "}", "//", "/*"]
        return any(indicator in content for indicator in code_indicators)
    
    async def _calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two patterns."""
        # Generate embeddings
        embedding1 = await self.embedding_generator.generate_embedding(pattern1)
        embedding2 = await self.embedding_generator.generate_embedding(pattern2)
        
        # Calculate cosine similarity
        import numpy as np
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    async def _handle_search_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle search requests from other agents."""
        query = message.content.get("query", "")
        max_results = message.content.get("max_results", 10)
        
        search_result = await self._handle_semantic_search({
            "query": query,
            "max_results": max_results
        })
        
        return {
            "search_result": search_result,
            "searcher": self.agent_id
        }
    
    async def _handle_similarity_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle similarity requests from other agents."""
        item1 = message.content.get("item1", "")
        item2 = message.content.get("item2", "")
        
        similarity_result = await self._handle_similarity_analysis({
            "item1": item1,
            "item2": item2
        })
        
        return {
            "similarity_result": similarity_result,
            "analyzer": self.agent_id
        }

class ReasoningAgent(BaseAgent):
    """Specialized agent for complex problem solving and analysis."""
    
    def __init__(self, agent_id: str, cot_engine, interleaved_engine):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.REASONING,
            capabilities=[
                "chain_of_thought",
                "problem_decomposition",
                "logical_analysis",
                "decision_making",
                "strategy_planning"
            ],
            max_concurrent_tasks=3  # Reasoning is resource-intensive
        )
        self.cot_engine = cot_engine
        self.interleaved_engine = interleaved_engine
        
        # Register message handlers
        self.register_message_handler("reasoning_request", self._handle_reasoning_request)
        self.register_message_handler("analysis_request", self._handle_analysis_request)
    
    def can_handle_task(self, task: AgentTask) -> float:
        """Determine if this agent can handle the task."""
        reasoning_task_types = {
            "chain_of_thought": 0.95,
            "problem_decomposition": 0.9,
            "logical_analysis": 0.85,
            "decision_making": 0.8,
            "strategy_planning": 0.75,
            "complex_reasoning": 0.9
        }
        return reasoning_task_types.get(task.task_type, 0.0)
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute reasoning-related tasks."""
        task_type = task.task_type
        input_data = task.input_data
        
        if task_type == "chain_of_thought":
            return await self._handle_chain_of_thought(input_data)
        elif task_type == "problem_decomposition":
            return await self._handle_problem_decomposition(input_data)
        elif task_type == "logical_analysis":
            return await self._handle_logical_analysis(input_data)
        elif task_type == "decision_making":
            return await self._handle_decision_making(input_data)
        elif task_type == "strategy_planning":
            return await self._handle_strategy_planning(input_data)
        elif task_type == "complex_reasoning":
            return await self._handle_complex_reasoning(input_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_chain_of_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chain-of-thought reasoning requests."""
        problem = input_data.get("problem", "")
        complexity = input_data.get("complexity", "moderate")
        
        # Map string complexity to enum
        complexity_map = {
            "simple": "SIMPLE",
            "moderate": "MODERATE", 
            "complex": "COMPLEX",
            "expert": "EXPERT"
        }
        
        from chain_of_thought_engine import CoTComplexity
        cot_complexity = getattr(CoTComplexity, complexity_map.get(complexity, "MODERATE"))
        
        trace = await self.cot_engine.reason_through_problem(
            problem=problem,
            complexity=cot_complexity
        )
        
        return {
            "reasoning_trace": {
                "trace_id": trace.trace_id,
                "steps": len(trace.steps),
                "final_solution": trace.final_solution,
                "confidence": trace.confidence_score,
                "quality": trace.quality_score
            },
            "problem": problem,
            "confidence": trace.confidence_score
        }
    
    async def _handle_problem_decomposition(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle problem decomposition requests."""
        problem = input_data.get("problem", "")
        max_depth = input_data.get("max_depth", 3)
        
        # Use CoT engine for structured decomposition
        from chain_of_thought_engine import CoTComplexity
        trace = await self.cot_engine.reason_through_problem(
            problem=f"Decompose this problem: {problem}",
            problem_type="architecture",
            complexity=CoTComplexity.COMPLEX
        )
        
        # Extract decomposition from trace
        decomposition_steps = [
            step for step in trace.steps 
            if step.step_type.value == "decomposition"
        ]
        
        return {
            "decomposition": [step.content for step in decomposition_steps],
            "problem": problem,
            "depth": len(decomposition_steps),
            "confidence": trace.confidence_score
        }
    
    async def _handle_logical_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logical analysis requests."""
        statement = input_data.get("statement", "")
        analysis_type = input_data.get("analysis_type", "validity")
        
        # Use interleaved reasoning for logical analysis
        from interleaved_reasoning_engine import ReasoningMode
        
        analysis_result = await self.interleaved_engine.reason_and_respond(
            query=f"Analyze the logical {analysis_type} of: {statement}",
            context={"analysis_type": analysis_type},
            mode=ReasoningMode.DEEP,
            stream=False
        )
        
        return {
            "analysis": analysis_result,
            "statement": statement,
            "analysis_type": analysis_type,
            "confidence": 0.8
        }
    
    async def _handle_decision_making(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decision-making requests."""
        options = input_data.get("options", [])
        criteria = input_data.get("criteria", [])
        context = input_data.get("context", "")
        
        # Use CoT for structured decision making
        decision_problem = f"""
        Make a decision between these options: {options}
        Based on criteria: {criteria}
        Context: {context}
        """
        
        from chain_of_thought_engine import CoTComplexity
        trace = await self.cot_engine.reason_through_problem(
            problem=decision_problem,
            complexity=CoTComplexity.MODERATE
        )
        
        return {
            "decision": trace.final_solution,
            "reasoning": [step.content for step in trace.steps],
            "options": options,
            "criteria": criteria,
            "confidence": trace.confidence_score
        }
    
    async def _handle_strategy_planning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy planning requests."""
        goal = input_data.get("goal", "")
        constraints = input_data.get("constraints", [])
        resources = input_data.get("resources", [])
        
        planning_problem = f"""
        Create a strategy to achieve: {goal}
        Constraints: {constraints}
        Available resources: {resources}
        """
        
        from chain_of_thought_engine import CoTComplexity
        trace = await self.cot_engine.reason_through_problem(
            problem=planning_problem,
            problem_type="architecture",
            complexity=CoTComplexity.COMPLEX
        )
        
        return {
            "strategy": trace.final_solution,
            "planning_steps": [step.content for step in trace.steps],
            "goal": goal,
            "confidence": trace.confidence_score
        }
    
    async def _handle_complex_reasoning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complex reasoning requests."""
        query = input_data.get("query", "")
        reasoning_mode = input_data.get("mode", "balanced")
        
        # Map to interleaved reasoning mode
        mode_map = {
            "fast": "FAST",
            "balanced": "BALANCED", 
            "deep": "DEEP",
            "progressive": "PROGRESSIVE"
        }
        
        from interleaved_reasoning_engine import ReasoningMode
        mode = getattr(ReasoningMode, mode_map.get(reasoning_mode, "BALANCED"))
        
        reasoning_result = await self.interleaved_engine.reason_and_respond(
            query=query,
            context=input_data.get("context", {}),
            mode=mode,
            stream=False
        )
        
        return {
            "reasoning_result": reasoning_result,
            "query": query,
            "mode": reasoning_mode,
            "confidence": 0.8
        }
    
    async def _handle_reasoning_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle reasoning requests from other agents."""
        problem = message.content.get("problem", "")
        reasoning_type = message.content.get("type", "chain_of_thought")
        
        if reasoning_type == "chain_of_thought":
            result = await self._handle_chain_of_thought({
                "problem": problem,
                "complexity": message.content.get("complexity", "moderate")
            })
        else:
            result = await self._handle_complex_reasoning({
                "query": problem,
                "mode": message.content.get("mode", "balanced")
            })
        
        return {
            "reasoning_result": result,
            "reasoner": self.agent_id
        }
    
    async def _handle_analysis_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle analysis requests from other agents."""
        statement = message.content.get("statement", "")
        analysis_type = message.content.get("analysis_type", "validity")
        
        analysis_result = await self._handle_logical_analysis({
            "statement": statement,
            "analysis_type": analysis_type
        })
        
        return {
            "analysis_result": analysis_result,
            "analyzer": self.agent_id
        }


class TestAgent(BaseAgent):
    """Specialized agent for test generation and validation."""
    
    def __init__(self, agent_id: str, llm_client, code_analyzer):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.TEST,
            capabilities=[
                "test_generation",
                "test_validation",
                "coverage_analysis",
                "test_optimization",
                "mock_generation"
            ],
            max_concurrent_tasks=4
        )
        self.llm_client = llm_client
        self.code_analyzer = code_analyzer
        
        # Register message handlers
        self.register_message_handler("test_request", self._handle_test_request)
        self.register_message_handler("validation_request", self._handle_validation_request)
    
    def can_handle_task(self, task: AgentTask) -> float:
        """Determine if this agent can handle the task."""
        test_task_types = {
            "test_generation": 0.95,
            "test_validation": 0.9,
            "coverage_analysis": 0.85,
            "test_optimization": 0.8,
            "mock_generation": 0.75
        }
        return test_task_types.get(task.task_type, 0.0)
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute test-related tasks."""
        task_type = task.task_type
        input_data = task.input_data
        
        if task_type == "test_generation":
            return await self._handle_test_generation(input_data)
        elif task_type == "test_validation":
            return await self._handle_test_validation(input_data)
        elif task_type == "coverage_analysis":
            return await self._handle_coverage_analysis(input_data)
        elif task_type == "test_optimization":
            return await self._handle_test_optimization(input_data)
        elif task_type == "mock_generation":
            return await self._handle_mock_generation(input_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_test_generation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test generation requests."""
        code = input_data.get("code", "")
        test_type = input_data.get("test_type", "unit")
        language = input_data.get("language", "python")
        
        prompt = f"""
        Generate {test_type} tests for the following {language} code:
        
        {code}
        
        Include:
        1. Test cases for normal functionality
        2. Edge cases and boundary conditions
        3. Error handling tests
        4. Appropriate assertions
        
        Generated tests:
        """
        
        tests = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2
        )
        
        return {
            "generated_tests": tests.strip(),
            "test_type": test_type,
            "language": language,
            "confidence": 0.8
        }
    
    async def _handle_test_validation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test validation requests."""
        test_code = input_data.get("test_code", "")
        target_code = input_data.get("target_code", "")
        
        prompt = f"""
        Validate these tests against the target code:
        
        Target Code:
        {target_code}
        
        Test Code:
        {test_code}
        
        Check for:
        1. Test completeness
        2. Correctness of assertions
        3. Edge case coverage
        4. Test quality
        
        Validation report:
        """
        
        validation = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.1
        )
        
        return {
            "validation_report": validation.strip(),
            "confidence": 0.85
        }
    
    async def _handle_coverage_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coverage analysis requests."""
        code = input_data.get("code", "")
        tests = input_data.get("tests", "")
        
        # Analyze code structure
        analysis = await self.code_analyzer.analyze_code_structure(code)
        
        prompt = f"""
        Analyze test coverage for:
        
        Code structure: {analysis}
        Tests: {tests}
        
        Identify:
        1. Covered functionality
        2. Missing test coverage
        3. Coverage percentage estimate
        4. Recommendations for improvement
        
        Coverage analysis:
        """
        
        coverage = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1
        )
        
        return {
            "coverage_analysis": coverage.strip(),
            "code_structure": analysis,
            "confidence": 0.75
        }
    
    async def _handle_test_optimization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test optimization requests."""
        tests = input_data.get("tests", "")
        optimization_goal = input_data.get("goal", "performance")
        
        prompt = f"""
        Optimize these tests for {optimization_goal}:
        
        {tests}
        
        Provide optimized tests with explanations:
        """
        
        optimized = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=700,
            temperature=0.2
        )
        
        return {
            "optimized_tests": optimized.strip(),
            "optimization_goal": optimization_goal,
            "confidence": 0.7
        }
    
    async def _handle_mock_generation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mock generation requests."""
        interface = input_data.get("interface", "")
        mock_type = input_data.get("mock_type", "simple")
        language = input_data.get("language", "python")
        
        prompt = f"""
        Generate {mock_type} mocks for this {language} interface:
        
        {interface}
        
        Include appropriate mock behavior and setup:
        """
        
        mocks = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.2
        )
        
        return {
            "generated_mocks": mocks.strip(),
            "mock_type": mock_type,
            "language": language,
            "confidence": 0.75
        }
    
    async def _handle_test_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle test requests from other agents."""
        code = message.content.get("code", "")
        test_type = message.content.get("test_type", "unit")
        
        test_result = await self._handle_test_generation({
            "code": code,
            "test_type": test_type,
            "language": message.content.get("language", "python")
        })
        
        return {
            "test_result": test_result,
            "tester": self.agent_id
        }
    
    async def _handle_validation_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle validation requests from other agents."""
        test_code = message.content.get("test_code", "")
        target_code = message.content.get("target_code", "")
        
        validation_result = await self._handle_test_validation({
            "test_code": test_code,
            "target_code": target_code
        })
        
        return {
            "validation_result": validation_result,
            "validator": self.agent_id
        }


class MultiAgentSystem:
    """
    Coordinator for the multi-agent system.
    
    Manages agent lifecycle, task delegation, communication, and performance monitoring.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        self.message_queue: List[AgentMessage] = []
        
        # Performance tracking
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_task_time": 0.0,
            "agent_utilization": {}
        }
        
        # Task routing
        self.task_router = TaskRouter()
        
        logger.info("MultiAgentSystem initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the system."""
        self.agents[agent.agent_id] = agent
        self.system_metrics["agent_utilization"][agent.agent_id] = 0.0
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the system."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.system_metrics["agent_utilization"]:
                del self.system_metrics["agent_utilization"][agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def submit_task(
        self,
        task_type: str,
        description: str,
        input_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Submit a task to the system."""
        task = AgentTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=description,
            input_data=input_data,
            priority=priority
        )
        
        self.task_queue.append(task)
        self.system_metrics["total_tasks"] += 1
        
        logger.info(f"Task submitted: {task.task_id} ({task_type})")
        
        # Trigger task processing
        await self._process_task_queue()
        
        return task.task_id
    
    async def _process_task_queue(self) -> None:
        """Process tasks in the queue."""
        # Sort by priority
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        processed_tasks = []
        
        for task in self.task_queue:
            if task.status == TaskStatus.PENDING:
                # Find best agent for task
                best_agent = await self.task_router.find_best_agent(task, self.agents)
                
                if best_agent:
                    success = await best_agent.assign_task(task)
                    if success:
                        processed_tasks.append(task)
                        logger.debug(f"Task {task.task_id} assigned to {best_agent.agent_id}")
        
        # Remove processed tasks from queue
        for task in processed_tasks:
            self.task_queue.remove(task)
        
        # Process tasks in all agents
        await self._process_agent_tasks()
    
    async def _process_agent_tasks(self) -> None:
        """Process tasks in all agents."""
        tasks = []
        for agent in self.agents.values():
            if agent.is_active:
                tasks.append(agent.process_tasks())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: str,
        content: Dict[str, Any],
        requires_response: bool = False
    ) -> str:
        """Send message between agents."""
        message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            requires_response=requires_response
        )
        
        self.message_queue.append(message)
        
        # Deliver message immediately
        await self._deliver_message(message)
        
        return message.message_id
    
    async def _deliver_message(self, message: AgentMessage) -> None:
        """Deliver message to recipient agent."""
        recipient = self.agents.get(message.recipient_id)
        if recipient:
            try:
                response = await recipient.handle_message(message)
                
                if message.requires_response and response:
                    # Send response back
                    response_message = AgentMessage(
                        message_id=f"resp_{uuid.uuid4().hex[:8]}",
                        sender_id=message.recipient_id,
                        recipient_id=message.sender_id,
                        message_type=f"{message.message_type}_response",
                        content=response,
                        correlation_id=message.message_id
                    )
                    
                    sender = self.agents.get(message.sender_id)
                    if sender:
                        await sender.handle_message(response_message)
                
            except Exception as e:
                logger.error(f"Error delivering message {message.message_id}: {e}")
        else:
            logger.warning(f"Recipient agent not found: {message.recipient_id}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks in agents
        for agent in self.agents.values():
            if task_id in agent.active_tasks:
                task = agent.active_tasks[task_id]
                return self._task_to_dict(task)
            
            # Check completed tasks
            for task in agent.completed_tasks:
                if task.task_id == task_id:
                    return self._task_to_dict(task)
        
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return self._task_to_dict(task)
        
        return None
    
    def _task_to_dict(self, task: AgentTask) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status.value,
            "assigned_agent": task.assigned_agent,
            "priority": task.priority.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        return {
            "agents": agent_statuses,
            "task_queue_size": len(self.task_queue),
            "system_metrics": self.system_metrics,
            "active_tasks": sum(len(agent.active_tasks) for agent in self.agents.values()),
            "total_completed": sum(len(agent.completed_tasks) for agent in self.agents.values())
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = self.system_metrics.copy()
        
        # Calculate agent utilization
        for agent_id, agent in self.agents.items():
            utilization = len(agent.active_tasks) / agent.max_concurrent_tasks
            metrics["agent_utilization"][agent_id] = utilization
        
        # Calculate success rate
        total_completed = sum(agent.metrics.tasks_completed for agent in self.agents.values())
        total_failed = sum(agent.metrics.tasks_failed for agent in self.agents.values())
        
        if total_completed + total_failed > 0:
            metrics["success_rate"] = total_completed / (total_completed + total_failed)
        else:
            metrics["success_rate"] = 0.0
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the multi-agent system."""
        logger.info("Shutting down multi-agent system")
        
        # Mark all agents as inactive
        for agent in self.agents.values():
            agent.is_active = False
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            active_tasks = sum(len(agent.active_tasks) for agent in self.agents.values())
            if active_tasks == 0:
                break
            await asyncio.sleep(1)
        
        logger.info("Multi-agent system shutdown complete")


class TaskRouter:
    """Routes tasks to the most appropriate agents."""
    
    def __init__(self):
        self.routing_history: List[Dict[str, Any]] = []
    
    async def find_best_agent(
        self,
        task: AgentTask,
        agents: Dict[str, BaseAgent]
    ) -> Optional[BaseAgent]:
        """Find the best agent for a given task."""
        candidates = []
        
        for agent in agents.values():
            if not agent.is_active:
                continue
            
            # Check if agent can handle the task
            confidence = agent.can_handle_task(task)
            if confidence > 0:
                # Calculate agent load
                load_factor = len(agent.active_tasks) / agent.max_concurrent_tasks
                
                # Calculate score (confidence weighted by availability)
                availability_factor = 1.0 - load_factor
                score = confidence * availability_factor
                
                candidates.append({
                    "agent": agent,
                    "confidence": confidence,
                    "load_factor": load_factor,
                    "score": score
                })
        
        if not candidates:
            logger.warning(f"No suitable agent found for task {task.task_id} ({task.task_type})")
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        best_candidate = candidates[0]
        best_agent = best_candidate["agent"]
        
        # Record routing decision
        self.routing_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "selected_agent": best_agent.agent_id,
            "confidence": best_candidate["confidence"],
            "load_factor": best_candidate["load_factor"],
            "score": best_candidate["score"],
            "timestamp": datetime.now()
        })
        
        return best_agent
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {"message": "No routing history available"}
        
        # Agent usage statistics
        agent_usage = {}
        for record in self.routing_history:
            agent_id = record["selected_agent"]
            if agent_id not in agent_usage:
                agent_usage[agent_id] = 0
            agent_usage[agent_id] += 1
        
        # Task type distribution
        task_type_dist = {}
        for record in self.routing_history:
            task_type = record["task_type"]
            if task_type not in task_type_dist:
                task_type_dist[task_type] = 0
            task_type_dist[task_type] += 1
        
        return {
            "total_routings": len(self.routing_history),
            "agent_usage": agent_usage,
            "task_type_distribution": task_type_dist,
            "avg_confidence": sum(r["confidence"] for r in self.routing_history) / len(self.routing_history),
            "avg_load_factor": sum(r["load_factor"] for r in self.routing_history) / len(self.routing_history)
        }