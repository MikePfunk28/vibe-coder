"""
Chain-of-Thought Reasoning Engine

Implementation of step-by-step problem decomposition and reasoning for complex coding problems.
Includes reasoning trace visualization, debugging, and quality assessment.

This engine provides structured CoT reasoning capabilities that complement the interleaved
reasoning engine with explicit step-by-step problem solving.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
from collections import deque

logger = logging.getLogger(__name__)


class CoTStepType(Enum):
    """Types of Chain-of-Thought reasoning steps."""
    PROBLEM_ANALYSIS = "problem_analysis"
    DECOMPOSITION = "decomposition"
    SOLUTION_PLANNING = "solution_planning"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    REFLECTION = "reflection"


class CoTComplexity(Enum):
    """Complexity levels for CoT reasoning."""
    SIMPLE = "simple"      # 2-3 steps
    MODERATE = "moderate"  # 4-6 steps
    COMPLEX = "complex"    # 7-10 steps
    EXPERT = "expert"      # 10+ steps


@dataclass
class CoTStep:
    """Represents a single step in Chain-of-Thought reasoning."""
    step_id: str
    step_number: int
    step_type: CoTStepType
    title: str
    content: str
    reasoning: str
    confidence: float
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTTrace:
    """Complete Chain-of-Thought reasoning trace."""
    trace_id: str
    problem_statement: str
    complexity: CoTComplexity
    steps: List[CoTStep] = field(default_factory=list)
    final_solution: Optional[str] = None
    confidence_score: float = 0.0
    total_time: float = 0.0
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    context_used: List[str] = field(default_factory=list)


class ChainOfThoughtEngine:
    """
    Chain-of-Thought reasoning engine for complex coding problems.
    
    Provides structured step-by-step problem decomposition and solution development
    with explicit reasoning traces and quality assessment.
    """  
  
    def __init__(
        self,
        context_manager,
        llm_client,
        max_steps: int = 15,
        enable_visualization: bool = True,
        quality_threshold: float = 0.7
    ):
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.enable_visualization = enable_visualization
        self.quality_threshold = quality_threshold
        
        # Reasoning state
        self.active_traces: Dict[str, CoTTrace] = {}
        self.completed_traces: List[CoTTrace] = []
        
        # Step templates for different problem types
        self.step_templates = {
            'coding_problem': [
                CoTStepType.PROBLEM_ANALYSIS,
                CoTStepType.DECOMPOSITION,
                CoTStepType.SOLUTION_PLANNING,
                CoTStepType.IMPLEMENTATION,
                CoTStepType.VERIFICATION
            ],
            'debugging': [
                CoTStepType.PROBLEM_ANALYSIS,
                CoTStepType.DECOMPOSITION,
                CoTStepType.VERIFICATION,
                CoTStepType.SOLUTION_PLANNING,
                CoTStepType.IMPLEMENTATION
            ],
            'architecture': [
                CoTStepType.PROBLEM_ANALYSIS,
                CoTStepType.DECOMPOSITION,
                CoTStepType.SOLUTION_PLANNING,
                CoTStepType.VERIFICATION,
                CoTStepType.REFLECTION
            ]
        }
        
        # Quality assessment criteria
        self.quality_criteria = {
            'logical_flow': 0.3,
            'completeness': 0.25,
            'clarity': 0.2,
            'correctness': 0.25
        }
        
        logger.info("ChainOfThoughtEngine initialized")
    
    async def reason_through_problem(
        self,
        problem: str,
        problem_type: str = 'coding_problem',
        complexity: CoTComplexity = CoTComplexity.MODERATE,
        context: Optional[Dict[str, Any]] = None
    ) -> CoTTrace:
        """
        Main entry point for Chain-of-Thought reasoning.
        
        Args:
            problem: The problem statement to reason through
            problem_type: Type of problem (coding_problem, debugging, architecture)
            complexity: Expected complexity level
            context: Additional context information
            
        Returns:
            Complete CoT reasoning trace
        """
        trace_id = f"cot_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        # Create reasoning trace
        trace = CoTTrace(
            trace_id=trace_id,
            problem_statement=problem,
            complexity=complexity
        )
        
        self.active_traces[trace_id] = trace
        
        try:
            # Get relevant context
            relevant_context = self.context_manager.get_relevant_context(
                problem,
                max_tokens=self._get_context_tokens_for_complexity(complexity)
            )
            trace.context_used = [ctx.id for ctx in relevant_context]
            
            # Execute reasoning steps
            await self._execute_reasoning_steps(trace, problem_type, context or {})
            
            # Generate final solution
            trace.final_solution = await self._synthesize_solution(trace)
            
            # Assess quality
            trace.quality_score = await self._assess_reasoning_quality(trace)
            trace.confidence_score = self._calculate_overall_confidence(trace)
            
            return trace
            
        finally:
            trace.total_time = time.time() - start_time
            self.completed_traces.append(trace)
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]
    
    async def _execute_reasoning_steps(
        self,
        trace: CoTTrace,
        problem_type: str,
        context: Dict[str, Any]
    ) -> None:
        """Execute the sequence of reasoning steps."""
        step_sequence = self.step_templates.get(problem_type, self.step_templates['coding_problem'])
        
        for i, step_type in enumerate(step_sequence):
            if len(trace.steps) >= self.max_steps:
                break
                
            step = await self._generate_reasoning_step(
                trace, step_type, i + 1, context
            )
            
            if step:
                trace.steps.append(step)
                
                # Check if we need additional steps based on step output
                if await self._should_add_additional_steps(trace, step):
                    additional_steps = await self._generate_additional_steps(trace, step)
                    for additional_step in additional_steps:
                        if len(trace.steps) < self.max_steps:
                            trace.steps.append(additional_step)
    
    async def _generate_reasoning_step(
        self,
        trace: CoTTrace,
        step_type: CoTStepType,
        step_number: int,
        context: Dict[str, Any]
    ) -> Optional[CoTStep]:
        """Generate a single reasoning step."""
        start_time = time.time()
        
        # Build step-specific prompt
        prompt = self._build_step_prompt(trace, step_type, step_number, context)
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=self._get_max_tokens_for_step(step_type),
                temperature=0.2
            )
            
            processing_time = time.time() - start_time
            
            # Parse response into structured step
            step_content, reasoning = self._parse_step_response(response, step_type)
            
            step = CoTStep(
                step_id=f"{trace.trace_id}_step_{step_number}",
                step_number=step_number,
                step_type=step_type,
                title=self._generate_step_title(step_type, step_number),
                content=step_content,
                reasoning=reasoning,
                confidence=self._calculate_step_confidence(step_content, step_type),
                processing_time=processing_time
            )
            
            return step
            
        except Exception as e:
            logger.error(f"Error generating reasoning step {step_type}: {e}")
            return None   
 
    def _build_step_prompt(
        self,
        trace: CoTTrace,
        step_type: CoTStepType,
        step_number: int,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for specific reasoning step."""
        previous_steps = "\n".join([
            f"Step {s.step_number} ({s.step_type.value}): {s.content}"
            for s in trace.steps
        ])
        
        base_context = f"""
Problem: {trace.problem_statement}
Previous Steps:
{previous_steps}

Context: {json.dumps(context, indent=2)}
"""
        
        if step_type == CoTStepType.PROBLEM_ANALYSIS:
            return f"""{base_context}

Step {step_number}: Problem Analysis
Analyze the problem thoroughly:
1. What exactly needs to be solved?
2. What are the key requirements and constraints?
3. What information do we have and what's missing?
4. What are potential challenges or edge cases?

Provide your analysis:"""

        elif step_type == CoTStepType.DECOMPOSITION:
            return f"""{base_context}

Step {step_number}: Problem Decomposition
Break down the problem into smaller, manageable parts:
1. Identify the main components or sub-problems
2. Determine dependencies between components
3. Prioritize components by importance and complexity
4. Define clear boundaries for each component

Decomposition:"""

        elif step_type == CoTStepType.SOLUTION_PLANNING:
            return f"""{base_context}

Step {step_number}: Solution Planning
Plan the approach to solve each component:
1. Choose appropriate algorithms, patterns, or techniques
2. Consider alternative approaches and trade-offs
3. Plan the implementation sequence
4. Identify potential risks and mitigation strategies

Solution plan:"""

        elif step_type == CoTStepType.IMPLEMENTATION:
            return f"""{base_context}

Step {step_number}: Implementation
Provide concrete implementation details:
1. Write specific code or detailed pseudocode
2. Explain key implementation decisions
3. Handle edge cases and error conditions
4. Consider performance and maintainability

Implementation:"""

        elif step_type == CoTStepType.VERIFICATION:
            return f"""{base_context}

Step {step_number}: Verification
Verify the solution:
1. Check if the solution addresses all requirements
2. Test with example inputs and edge cases
3. Review for potential bugs or issues
4. Validate performance and efficiency

Verification:"""

        elif step_type == CoTStepType.REFLECTION:
            return f"""{base_context}

Step {step_number}: Reflection
Reflect on the solution:
1. What worked well in this approach?
2. What could be improved or done differently?
3. What lessons can be learned for similar problems?
4. Are there any remaining concerns or limitations?

Reflection:"""

        else:
            return f"""{base_context}

Step {step_number}: Continue reasoning about the problem.
Provide the next logical step in solving this problem:"""
    
    def _parse_step_response(self, response: str, step_type: CoTStepType) -> Tuple[str, str]:
        """Parse LLM response into content and reasoning."""
        lines = response.strip().split('\n')
        
        # Try to separate explicit reasoning from content
        reasoning_markers = ['reasoning:', 'because:', 'rationale:', 'explanation:']
        
        content_lines = []
        reasoning_lines = []
        in_reasoning = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if any(marker in line_lower for marker in reasoning_markers):
                in_reasoning = True
                reasoning_lines.append(line)
            elif in_reasoning:
                reasoning_lines.append(line)
            else:
                content_lines.append(line)
        
        content = '\n'.join(content_lines).strip()
        reasoning = '\n'.join(reasoning_lines).strip() if reasoning_lines else content
        
        return content, reasoning
    
    def _generate_step_title(self, step_type: CoTStepType, step_number: int) -> str:
        """Generate a descriptive title for the step."""
        titles = {
            CoTStepType.PROBLEM_ANALYSIS: f"Step {step_number}: Analyze the Problem",
            CoTStepType.DECOMPOSITION: f"Step {step_number}: Break Down the Problem",
            CoTStepType.SOLUTION_PLANNING: f"Step {step_number}: Plan the Solution",
            CoTStepType.IMPLEMENTATION: f"Step {step_number}: Implement the Solution",
            CoTStepType.VERIFICATION: f"Step {step_number}: Verify the Solution",
            CoTStepType.REFLECTION: f"Step {step_number}: Reflect on the Solution"
        }
        return titles.get(step_type, f"Step {step_number}: Reasoning Step")
    
    def _calculate_step_confidence(self, content: str, step_type: CoTStepType) -> float:
        """Calculate confidence score for a reasoning step."""
        base_confidence = 0.7
        
        # Adjust based on content characteristics
        if len(content) > 100:  # Detailed content
            base_confidence += 0.1
        
        if step_type == CoTStepType.IMPLEMENTATION and 'def ' in content:
            base_confidence += 0.1  # Contains actual code
        
        if any(word in content.lower() for word in ['because', 'therefore', 'since', 'due to']):
            base_confidence += 0.1  # Explanatory language
        
        # Reduce confidence for uncertainty
        if any(word in content.lower() for word in ['maybe', 'might', 'possibly', 'unclear', 'not sure']):
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    async def _should_add_additional_steps(self, trace: CoTTrace, step: CoTStep) -> bool:
        """Determine if additional steps are needed based on current step."""
        # Check if step indicates complexity that needs more breakdown
        complexity_indicators = ['complex', 'multiple', 'several', 'various', 'depends on']
        
        if any(indicator in step.content.lower() for indicator in complexity_indicators):
            return True
        
        # Check if step confidence is low
        if step.confidence < 0.6:
            return True
        
        # Check if we're in early steps and problem seems complex
        if step.step_number <= 3 and trace.complexity in [CoTComplexity.COMPLEX, CoTComplexity.EXPERT]:
            return True
        
        return False
    
    async def _generate_additional_steps(self, trace: CoTTrace, triggering_step: CoTStep) -> List[CoTStep]:
        """Generate additional reasoning steps when needed."""
        additional_steps = []
        
        if triggering_step.step_type == CoTStepType.DECOMPOSITION and triggering_step.confidence < 0.7:
            # Add more detailed decomposition
            detailed_step = await self._generate_reasoning_step(
                trace, CoTStepType.DECOMPOSITION, len(trace.steps) + 1, 
                {'focus': 'detailed_breakdown', 'previous_step': triggering_step.content}
            )
            if detailed_step:
                detailed_step.title = f"Step {detailed_step.step_number}: Detailed Decomposition"
                additional_steps.append(detailed_step)
        
        elif triggering_step.step_type == CoTStepType.SOLUTION_PLANNING and 'alternative' in triggering_step.content.lower():
            # Add alternative solution exploration
            alt_step = await self._generate_reasoning_step(
                trace, CoTStepType.SOLUTION_PLANNING, len(trace.steps) + 1,
                {'focus': 'alternatives', 'previous_step': triggering_step.content}
            )
            if alt_step:
                alt_step.title = f"Step {alt_step.step_number}: Explore Alternatives"
                additional_steps.append(alt_step)
        
        return additional_steps   
 
    async def _synthesize_solution(self, trace: CoTTrace) -> str:
        """Synthesize final solution from all reasoning steps."""
        if not trace.steps:
            return "No solution could be generated."
        
        # Find implementation steps
        implementation_steps = [s for s in trace.steps if s.step_type == CoTStepType.IMPLEMENTATION]
        
        if implementation_steps:
            # Combine implementation steps
            implementations = [step.content for step in implementation_steps]
            solution = "\n\n".join(implementations)
        else:
            # Synthesize from all steps
            all_content = "\n\n".join([f"{step.title}:\n{step.content}" for step in trace.steps])
            
            # Generate synthesis prompt
            synthesis_prompt = f"""
Based on the following reasoning steps, provide a final, comprehensive solution:

{all_content}

Final Solution:"""
            
            try:
                solution = await self.llm_client.generate(
                    prompt=synthesis_prompt,
                    max_tokens=800,
                    temperature=0.1
                )
            except Exception as e:
                logger.error(f"Error synthesizing solution: {e}")
                solution = "Error generating final solution."
        
        return solution.strip()
    
    async def _assess_reasoning_quality(self, trace: CoTTrace) -> float:
        """Assess the quality of the reasoning process."""
        if not trace.steps:
            return 0.0
        
        quality_scores = {}
        
        # Logical flow assessment
        quality_scores['logical_flow'] = self._assess_logical_flow(trace)
        
        # Completeness assessment
        quality_scores['completeness'] = self._assess_completeness(trace)
        
        # Clarity assessment
        quality_scores['clarity'] = self._assess_clarity(trace)
        
        # Correctness assessment (basic heuristics)
        quality_scores['correctness'] = self._assess_correctness(trace)
        
        # Calculate weighted score
        total_score = sum(
            score * self.quality_criteria[criterion]
            for criterion, score in quality_scores.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _assess_logical_flow(self, trace: CoTTrace) -> float:
        """Assess logical flow between reasoning steps."""
        if len(trace.steps) < 2:
            return 0.5
        
        flow_score = 0.0
        transitions = 0
        
        for i in range(1, len(trace.steps)):
            prev_step = trace.steps[i-1]
            curr_step = trace.steps[i]
            
            # Check if current step builds on previous
            if self._steps_are_connected(prev_step, curr_step):
                flow_score += 1.0
            else:
                flow_score += 0.3  # Partial credit for weak connection
            
            transitions += 1
        
        return flow_score / transitions if transitions > 0 else 0.5
    
    def _assess_completeness(self, trace: CoTTrace) -> float:
        """Assess completeness of the reasoning process."""
        expected_steps = self._get_expected_step_types(trace.complexity)
        present_steps = set(step.step_type for step in trace.steps)
        
        coverage = len(present_steps.intersection(expected_steps)) / len(expected_steps)
        
        # Bonus for having implementation
        if CoTStepType.IMPLEMENTATION in present_steps:
            coverage += 0.1
        
        # Bonus for verification
        if CoTStepType.VERIFICATION in present_steps:
            coverage += 0.1
        
        return min(1.0, coverage)
    
    def _assess_clarity(self, trace: CoTTrace) -> float:
        """Assess clarity of reasoning steps."""
        if not trace.steps:
            return 0.0
        
        clarity_scores = []
        
        for step in trace.steps:
            step_clarity = 0.7  # Base score
            
            # Check for clear structure
            if any(marker in step.content for marker in ['1.', '2.', '3.', '-', '*']):
                step_clarity += 0.1
            
            # Check for explanatory language
            if any(word in step.content.lower() for word in ['because', 'therefore', 'since', 'so that']):
                step_clarity += 0.1
            
            # Penalize vague language
            if any(word in step.content.lower() for word in ['somehow', 'maybe', 'probably']):
                step_clarity -= 0.2
            
            clarity_scores.append(max(0.0, min(1.0, step_clarity)))
        
        return sum(clarity_scores) / len(clarity_scores)
    
    def _assess_correctness(self, trace: CoTTrace) -> float:
        """Basic correctness assessment using heuristics."""
        correctness_score = 0.7  # Base assumption
        
        # Check for common correctness indicators
        implementation_steps = [s for s in trace.steps if s.step_type == CoTStepType.IMPLEMENTATION]
        
        for step in implementation_steps:
            content_lower = step.content.lower()
            
            # Positive indicators
            if any(indicator in content_lower for indicator in ['def ', 'class ', 'return', 'if ', 'for ', 'while ']):
                correctness_score += 0.05
            
            # Negative indicators
            if any(indicator in content_lower for indicator in ['error', 'wrong', 'incorrect', 'bug']):
                correctness_score -= 0.1
        
        # Check verification steps
        verification_steps = [s for s in trace.steps if s.step_type == CoTStepType.VERIFICATION]
        if verification_steps:
            correctness_score += 0.1
        
        return max(0.0, min(1.0, correctness_score))
    
    def _steps_are_connected(self, prev_step: CoTStep, curr_step: CoTStep) -> bool:
        """Check if two consecutive steps are logically connected."""
        # Check for explicit references
        if any(word in curr_step.content.lower() for word in ['based on', 'from above', 'previous', 'earlier']):
            return True
        
        # Check for logical progression
        step_progression = {
            CoTStepType.PROBLEM_ANALYSIS: [CoTStepType.DECOMPOSITION],
            CoTStepType.DECOMPOSITION: [CoTStepType.SOLUTION_PLANNING, CoTStepType.IMPLEMENTATION],
            CoTStepType.SOLUTION_PLANNING: [CoTStepType.IMPLEMENTATION],
            CoTStepType.IMPLEMENTATION: [CoTStepType.VERIFICATION],
            CoTStepType.VERIFICATION: [CoTStepType.REFLECTION]
        }
        
        expected_next = step_progression.get(prev_step.step_type, [])
        return curr_step.step_type in expected_next
    
    def _get_expected_step_types(self, complexity: CoTComplexity) -> set:
        """Get expected step types for given complexity."""
        base_steps = {CoTStepType.PROBLEM_ANALYSIS, CoTStepType.SOLUTION_PLANNING, CoTStepType.IMPLEMENTATION}
        
        if complexity in [CoTComplexity.MODERATE, CoTComplexity.COMPLEX, CoTComplexity.EXPERT]:
            base_steps.add(CoTStepType.DECOMPOSITION)
        
        if complexity in [CoTComplexity.COMPLEX, CoTComplexity.EXPERT]:
            base_steps.add(CoTStepType.VERIFICATION)
        
        if complexity == CoTComplexity.EXPERT:
            base_steps.add(CoTStepType.REFLECTION)
        
        return base_steps
    
    def _calculate_overall_confidence(self, trace: CoTTrace) -> float:
        """Calculate overall confidence from individual step confidences."""
        if not trace.steps:
            return 0.0
        
        step_confidences = [step.confidence for step in trace.steps]
        
        # Weighted average with more weight on later steps
        weights = [i + 1 for i in range(len(step_confidences))]
        weighted_sum = sum(conf * weight for conf, weight in zip(step_confidences, weights))
        weight_sum = sum(weights)
        
        base_confidence = weighted_sum / weight_sum
        
        # Adjust based on quality score
        if hasattr(trace, 'quality_score') and trace.quality_score > 0:
            base_confidence = (base_confidence + trace.quality_score) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _get_context_tokens_for_complexity(self, complexity: CoTComplexity) -> int:
        """Get context token limit based on complexity."""
        token_limits = {
            CoTComplexity.SIMPLE: 1024,
            CoTComplexity.MODERATE: 2048,
            CoTComplexity.COMPLEX: 4096,
            CoTComplexity.EXPERT: 6144
        }
        return token_limits.get(complexity, 2048)
    
    def _get_max_tokens_for_step(self, step_type: CoTStepType) -> int:
        """Get max tokens for different step types."""
        token_limits = {
            CoTStepType.PROBLEM_ANALYSIS: 400,
            CoTStepType.DECOMPOSITION: 500,
            CoTStepType.SOLUTION_PLANNING: 600,
            CoTStepType.IMPLEMENTATION: 800,
            CoTStepType.VERIFICATION: 400,
            CoTStepType.REFLECTION: 300
        }
        return token_limits.get(step_type, 500)    

    def visualize_reasoning_trace(self, trace: CoTTrace) -> str:
        """Generate visualization of reasoning trace for debugging."""
        if not self.enable_visualization:
            return "Visualization disabled"
        
        visualization = []
        visualization.append(f"🧠 Chain-of-Thought Reasoning Trace")
        visualization.append(f"📋 Problem: {trace.problem_statement}")
        visualization.append(f"⚡ Complexity: {trace.complexity.value}")
        visualization.append(f"⏱️  Total Time: {trace.total_time:.2f}s")
        visualization.append(f"🎯 Confidence: {trace.confidence_score:.2f}")
        visualization.append(f"⭐ Quality: {trace.quality_score:.2f}")
        visualization.append("")
        
        # Step-by-step breakdown
        for i, step in enumerate(trace.steps):
            step_icon = self._get_step_icon(step.step_type)
            visualization.append(f"{step_icon} {step.title}")
            visualization.append(f"   Confidence: {step.confidence:.2f} | Time: {step.processing_time:.2f}s")
            
            # Show content with indentation
            content_lines = step.content.split('\n')
            for line in content_lines[:3]:  # Show first 3 lines
                visualization.append(f"   {line}")
            
            if len(content_lines) > 3:
                visualization.append(f"   ... ({len(content_lines) - 3} more lines)")
            
            visualization.append("")
        
        # Final solution
        if trace.final_solution:
            visualization.append("🎯 Final Solution:")
            solution_lines = trace.final_solution.split('\n')
            for line in solution_lines[:5]:  # Show first 5 lines
                visualization.append(f"   {line}")
            
            if len(solution_lines) > 5:
                visualization.append(f"   ... ({len(solution_lines) - 5} more lines)")
        
        return '\n'.join(visualization)
    
    def _get_step_icon(self, step_type: CoTStepType) -> str:
        """Get emoji icon for step type."""
        icons = {
            CoTStepType.PROBLEM_ANALYSIS: "🔍",
            CoTStepType.DECOMPOSITION: "🧩",
            CoTStepType.SOLUTION_PLANNING: "📋",
            CoTStepType.IMPLEMENTATION: "⚙️",
            CoTStepType.VERIFICATION: "✅",
            CoTStepType.REFLECTION: "🤔"
        }
        return icons.get(step_type, "📝")
    
    def get_debugging_info(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed debugging information for a trace."""
        trace = self.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        debugging_info = {
            "trace_id": trace.trace_id,
            "problem": trace.problem_statement,
            "complexity": trace.complexity.value,
            "total_steps": len(trace.steps),
            "total_time": trace.total_time,
            "confidence_score": trace.confidence_score,
            "quality_score": trace.quality_score,
            "steps": []
        }
        
        for step in trace.steps:
            step_info = {
                "step_number": step.step_number,
                "step_type": step.step_type.value,
                "title": step.title,
                "confidence": step.confidence,
                "processing_time": step.processing_time,
                "content_length": len(step.content),
                "reasoning_length": len(step.reasoning),
                "dependencies": step.dependencies,
                "outputs": step.outputs
            }
            debugging_info["steps"].append(step_info)
        
        return debugging_info
    
    def validate_reasoning_quality(self, trace: CoTTrace) -> Dict[str, Any]:
        """Validate reasoning quality and provide improvement suggestions."""
        validation_result = {
            "overall_quality": trace.quality_score,
            "meets_threshold": trace.quality_score >= self.quality_threshold,
            "issues": [],
            "suggestions": []
        }
        
        # Check for common issues
        if len(trace.steps) < 3:
            validation_result["issues"].append("Too few reasoning steps")
            validation_result["suggestions"].append("Add more detailed problem analysis and decomposition")
        
        # Check step confidence
        low_confidence_steps = [s for s in trace.steps if s.confidence < 0.5]
        if low_confidence_steps:
            validation_result["issues"].append(f"{len(low_confidence_steps)} steps have low confidence")
            validation_result["suggestions"].append("Review and strengthen low-confidence reasoning steps")
        
        # Check for missing key step types
        step_types = set(s.step_type for s in trace.steps)
        if CoTStepType.PROBLEM_ANALYSIS not in step_types:
            validation_result["issues"].append("Missing problem analysis step")
            validation_result["suggestions"].append("Add explicit problem analysis")
        
        if CoTStepType.IMPLEMENTATION not in step_types and trace.complexity != CoTComplexity.SIMPLE:
            validation_result["issues"].append("Missing implementation step")
            validation_result["suggestions"].append("Add concrete implementation details")
        
        # Check logical flow
        flow_score = self._assess_logical_flow(trace)
        if flow_score < 0.6:
            validation_result["issues"].append("Poor logical flow between steps")
            validation_result["suggestions"].append("Improve connections between reasoning steps")
        
        return validation_result
    
    def get_trace(self, trace_id: str) -> Optional[CoTTrace]:
        """Get a specific reasoning trace."""
        # Check active traces
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the CoT engine."""
        if not self.completed_traces:
            return {"message": "No completed traces yet"}
        
        traces = self.completed_traces
        
        stats = {
            "total_traces": len(traces),
            "avg_steps": sum(len(t.steps) for t in traces) / len(traces),
            "avg_time": sum(t.total_time for t in traces) / len(traces),
            "avg_confidence": sum(t.confidence_score for t in traces) / len(traces),
            "avg_quality": sum(t.quality_score for t in traces) / len(traces),
            "complexity_distribution": {},
            "step_type_usage": {},
            "high_quality_traces": len([t for t in traces if t.quality_score >= self.quality_threshold])
        }
        
        # Complexity distribution
        for complexity in CoTComplexity:
            count = len([t for t in traces if t.complexity == complexity])
            stats["complexity_distribution"][complexity.value] = count
        
        # Step type usage
        for step_type in CoTStepType:
            count = sum(1 for t in traces for s in t.steps if s.step_type == step_type)
            stats["step_type_usage"][step_type.value] = count
        
        return stats
    
    def clear_traces(self, keep_recent: int = 5) -> None:
        """Clear completed traces, optionally keeping recent ones."""
        if keep_recent > 0:
            self.completed_traces = self.completed_traces[-keep_recent:]
        else:
            self.completed_traces.clear()
        
        logger.info(f"Cleared CoT traces, kept {len(self.completed_traces)} recent traces")