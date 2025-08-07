"""
ReAct Framework Adaptive Strategies

Adaptive reasoning strategies based on task complexity.
"""

from typing import Dict, Any

from react_core import ReActTrace


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