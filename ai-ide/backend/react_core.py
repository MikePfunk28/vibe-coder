"""
ReAct Framework Core Components

Core data structures and enums for the ReAct framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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