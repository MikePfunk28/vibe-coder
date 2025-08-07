"""
Context Router for AI IDE
Routes tasks to optimal models based on context analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

logger = logging.getLogger('context_router')

class ModelCapability(Enum):
    """Model capabilities for routing decisions"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    REASONING = "reasoning"
    SEARCH = "search"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class ModelType(Enum):
    """Available model types in the system"""
    QWEN_CODER = "qwen_coder"
    REASONING_PLUS = "reasoning_plus"
    GENERAL_LLM = "general_llm"
    SPECIALIZED = "specialized"
    MULTI_MODAL = "multi_modal"

@dataclass
class ModelProfile:
    """Profile of a model with its capabilities and characteristics"""
    name: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    max_context_length: int
    response_time_avg: float  # seconds
    accuracy_score: float  # 0.0 to 1.0
    cost_per_token: float
    specializations: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=list)
    max_output_tokens: int = 4096
    temperature_range: Tuple[float, float] = (0.0, 1.0)
    availability: float = 0.99  # uptime percentage

@dataclass
class RoutingContext:
    """Context information for routing decisions"""
    task_description: str
    task_type: str
    complexity: TaskComplexity
    required_capabilities: List[ModelCapability]
    context_length: int
    expected_output_length: int
    language: str = "python"
    priority: str = "normal"  # low, normal, high, critical
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: str
    model_type: ModelType
    confidence: float
    reasoning: str
    alternative_models: List[str] = field(default_factory=list)
    estimated_performance: Dict[str, Any] = field(default_factory=dict)
    routing_metadata: Dict[str, Any] = field(default_factory=dict)

class ContextRouter:
    """Routes tasks to optimal models based on context analysis"""
    
    def __init__(self):
        self.model_registry: Dict[str, ModelProfile] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize with default models
        self._register_default_models()
        
        # Routing rules and weights
        self.capability_weights = {
            ModelCapability.CODE_GENERATION: 1.0,
            ModelCapability.CODE_ANALYSIS: 0.8,
            ModelCapability.REASONING: 0.9,
            ModelCapability.SEARCH: 0.7,
            ModelCapability.CONVERSATION: 0.6,
            ModelCapability.TRANSLATION: 0.5,
            ModelCapability.SUMMARIZATION: 0.6,
            ModelCapability.CLASSIFICATION: 0.7
        }
        
        self.complexity_thresholds = {
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MODERATE: 0.6,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 1.0
        }
    
    def _register_default_models(self):
        """Register default models available in the AI IDE"""
        
        # Qwen Coder 3 - Specialized for code generation
        self.register_model(ModelProfile(
            name="qwen_coder_3",
            model_type=ModelType.QWEN_CODER,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_ANALYSIS,
                ModelCapability.REASONING
            ],
            max_context_length=32768,
            response_time_avg=2.5,
            accuracy_score=0.92,
            cost_per_token=0.0001,
            specializations=["python", "javascript", "typescript", "java", "cpp"],
            supported_languages=["python", "javascript", "typescript", "java", "cpp", "go", "rust"],
            max_output_tokens=8192,
            temperature_range=(0.0, 0.8)
        ))
        
        # Reasoning Plus - Advanced reasoning model
        self.register_model(ModelProfile(
            name="reasoning_plus",
            model_type=ModelType.REASONING_PLUS,
            capabilities=[
                ModelCapability.REASONING,
                ModelCapability.CODE_ANALYSIS,
                ModelCapability.CONVERSATION
            ],
            max_context_length=128000,
            response_time_avg=4.2,
            accuracy_score=0.95,
            cost_per_token=0.0003,
            specializations=["complex_reasoning", "problem_solving", "analysis"],
            supported_languages=["python", "javascript", "java", "cpp", "go"],
            max_output_tokens=16384,
            temperature_range=(0.0, 1.0)
        ))
        
        # General LLM - Balanced general purpose model
        self.register_model(ModelProfile(
            name="general_llm",
            model_type=ModelType.GENERAL_LLM,
            capabilities=[
                ModelCapability.CONVERSATION,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CLASSIFICATION,
                ModelCapability.TRANSLATION
            ],
            max_context_length=16384,
            response_time_avg=1.8,
            accuracy_score=0.85,
            cost_per_token=0.00005,
            specializations=["general_purpose", "conversation"],
            supported_languages=["python", "javascript", "java"],
            max_output_tokens=4096,
            temperature_range=(0.0, 1.0)
        ))
        
        # Specialized Search Model
        self.register_model(ModelProfile(
            name="search_specialist",
            model_type=ModelType.SPECIALIZED,
            capabilities=[
                ModelCapability.SEARCH,
                ModelCapability.CLASSIFICATION,
                ModelCapability.SUMMARIZATION
            ],
            max_context_length=8192,
            response_time_avg=1.2,
            accuracy_score=0.88,
            cost_per_token=0.00003,
            specializations=["semantic_search", "information_retrieval"],
            supported_languages=["python", "javascript"],
            max_output_tokens=2048,
            temperature_range=(0.0, 0.5)
        ))
    
    def register_model(self, model_profile: ModelProfile):
        """Register a new model in the router"""
        self.model_registry[model_profile.name] = model_profile
        self.performance_metrics[model_profile.name] = {
            "success_rate": 0.9,
            "avg_response_time": model_profile.response_time_avg,
            "user_satisfaction": 0.8,
            "total_requests": 0
        }
        logger.info(f"Registered model: {model_profile.name} ({model_profile.model_type.value})")
    
    def route_task(self, context: RoutingContext) -> RoutingDecision:
        """Route a task to the optimal model based on context"""
        
        logger.info(f"Routing task: {context.task_description[:100]}...")
        
        # Analyze task requirements
        task_analysis = self._analyze_task(context)
        
        # Score all available models
        model_scores = self._score_models(context, task_analysis)
        
        # Select the best model
        best_model = self._select_best_model(model_scores, context)
        
        # Create routing decision
        decision = self._create_routing_decision(best_model, model_scores, context, task_analysis)
        
        # Record routing decision
        self._record_routing_decision(decision, context)
        
        logger.info(f"Selected model: {decision.selected_model} (confidence: {decision.confidence:.2f})")
        
        return decision
    
    def _analyze_task(self, context: RoutingContext) -> Dict[str, Any]:
        """Analyze the task to understand its requirements"""
        
        task_lower = context.task_description.lower()
        
        # Detect task characteristics
        analysis = {
            "is_code_task": self._is_code_task(task_lower),
            "is_reasoning_task": self._is_reasoning_task(task_lower),
            "is_search_task": self._is_search_task(task_lower),
            "is_analysis_task": self._is_analysis_task(task_lower),
            "requires_creativity": self._requires_creativity(task_lower),
            "requires_precision": self._requires_precision(task_lower),
            "language_detected": self._detect_programming_language(task_lower, context.language),
            "estimated_complexity": self._estimate_complexity(context),
            "output_type": self._detect_output_type(task_lower)
        }
        
        return analysis
    
    def _is_code_task(self, task_text: str) -> bool:
        """Detect if task involves code generation or analysis"""
        code_keywords = [
            'code', 'function', 'class', 'method', 'implement', 'generate',
            'write', 'create', 'debug', 'fix', 'refactor', 'optimize',
            'algorithm', 'program', 'script', 'module'
        ]
        return any(keyword in task_text for keyword in code_keywords)
    
    def _is_reasoning_task(self, task_text: str) -> bool:
        """Detect if task requires complex reasoning"""
        reasoning_keywords = [
            'explain', 'analyze', 'reason', 'think', 'solve', 'understand',
            'compare', 'evaluate', 'assess', 'determine', 'conclude',
            'deduce', 'infer', 'logic', 'problem'
        ]
        return any(keyword in task_text for keyword in reasoning_keywords)
    
    def _is_search_task(self, task_text: str) -> bool:
        """Detect if task involves searching or finding information"""
        search_keywords = [
            'find', 'search', 'locate', 'discover', 'lookup', 'retrieve',
            'query', 'explore', 'investigate', 'research'
        ]
        return any(keyword in task_text for keyword in search_keywords)
    
    def _is_analysis_task(self, task_text: str) -> bool:
        """Detect if task involves analysis"""
        analysis_keywords = [
            'analyze', 'review', 'examine', 'inspect', 'check', 'audit',
            'evaluate', 'assess', 'study', 'investigate'
        ]
        return any(keyword in task_text for keyword in analysis_keywords)
    
    def _requires_creativity(self, task_text: str) -> bool:
        """Detect if task requires creative thinking"""
        creativity_keywords = [
            'creative', 'innovative', 'novel', 'unique', 'original',
            'brainstorm', 'design', 'invent', 'imagine'
        ]
        return any(keyword in task_text for keyword in creativity_keywords)
    
    def _requires_precision(self, task_text: str) -> bool:
        """Detect if task requires high precision"""
        precision_keywords = [
            'precise', 'exact', 'accurate', 'correct', 'specific',
            'detailed', 'thorough', 'comprehensive', 'complete'
        ]
        return any(keyword in task_text for keyword in precision_keywords)
    
    def _detect_programming_language(self, task_text: str, default_language: str) -> str:
        """Detect the programming language from task description"""
        language_patterns = {
            'python': r'\b(python|py|django|flask|pandas|numpy)\b',
            'javascript': r'\b(javascript|js|node|react|vue|angular)\b',
            'typescript': r'\b(typescript|ts)\b',
            'java': r'\b(java|spring|maven|gradle)\b',
            'cpp': r'\b(c\+\+|cpp|c plus plus)\b',
            'c': r'\b(c language|c programming)\b',
            'go': r'\b(golang|go language)\b',
            'rust': r'\b(rust|cargo)\b',
            'php': r'\b(php|laravel|symfony)\b',
            'ruby': r'\b(ruby|rails)\b',
            'swift': r'\b(swift|ios)\b',
            'kotlin': r'\b(kotlin|android)\b'
        }
        
        for language, pattern in language_patterns.items():
            if re.search(pattern, task_text, re.IGNORECASE):
                return language
        
        return default_language
    
    def _estimate_complexity(self, context: RoutingContext) -> TaskComplexity:
        """Estimate task complexity based on context"""
        
        # Use provided complexity if available
        if hasattr(context, 'complexity') and context.complexity:
            return context.complexity
        
        task_text = context.task_description.lower()
        
        # Simple task indicators
        simple_indicators = ['simple', 'basic', 'easy', 'quick', 'small']
        if any(indicator in task_text for indicator in simple_indicators):
            return TaskComplexity.SIMPLE
        
        # Complex task indicators
        complex_indicators = [
            'complex', 'advanced', 'sophisticated', 'comprehensive',
            'multi-step', 'intricate', 'detailed', 'thorough'
        ]
        if any(indicator in task_text for indicator in complex_indicators):
            return TaskComplexity.COMPLEX
        
        # Expert task indicators
        expert_indicators = [
            'expert', 'professional', 'enterprise', 'production',
            'scalable', 'optimized', 'high-performance'
        ]
        if any(indicator in task_text for indicator in expert_indicators):
            return TaskComplexity.EXPERT
        
        # Estimate based on context length and requirements
        if context.context_length > 10000 or len(context.required_capabilities) > 3:
            return TaskComplexity.COMPLEX
        elif context.context_length > 2000 or len(context.required_capabilities) > 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _detect_output_type(self, task_text: str) -> str:
        """Detect the expected output type"""
        if any(keyword in task_text for keyword in ['code', 'function', 'class', 'script']):
            return "code"
        elif any(keyword in task_text for keyword in ['explain', 'description', 'summary']):
            return "explanation"
        elif any(keyword in task_text for keyword in ['list', 'items', 'results']):
            return "list"
        elif any(keyword in task_text for keyword in ['analysis', 'report', 'assessment']):
            return "analysis"
        else:
            return "general"
    
    def _score_models(self, context: RoutingContext, task_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Score all available models for the given context"""
        
        model_scores = {}
        
        for model_name, model_profile in self.model_registry.items():
            score = self._calculate_model_score(model_profile, context, task_analysis)
            model_scores[model_name] = score
        
        return model_scores
    
    def _calculate_model_score(self, model: ModelProfile, context: RoutingContext, 
                             task_analysis: Dict[str, Any]) -> float:
        """Calculate a score for a specific model"""
        
        score = 0.0
        
        # Capability matching score (40% weight)
        capability_score = self._score_capabilities(model, context.required_capabilities)
        score += capability_score * 0.4
        
        # Task type matching score (25% weight)
        task_type_score = self._score_task_type(model, task_analysis)
        score += task_type_score * 0.25
        
        # Performance score (20% weight)
        performance_score = self._score_performance(model, context)
        score += performance_score * 0.2
        
        # Language support score (10% weight)
        language_score = self._score_language_support(model, task_analysis.get("language_detected", "python"))
        score += language_score * 0.1
        
        # Availability and reliability score (5% weight)
        reliability_score = model.availability * model.accuracy_score
        score += reliability_score * 0.05
        
        # Apply complexity penalty/bonus
        complexity_modifier = self._get_complexity_modifier(model, task_analysis.get("estimated_complexity", TaskComplexity.MODERATE))
        score *= complexity_modifier
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_capabilities(self, model: ModelProfile, required_capabilities: List[ModelCapability]) -> float:
        """Score model based on capability matching"""
        if not required_capabilities:
            return 0.8  # Default score if no specific capabilities required
        
        matched_capabilities = 0
        total_weight = 0
        
        for capability in required_capabilities:
            weight = self.capability_weights.get(capability, 0.5)
            total_weight += weight
            
            if capability in model.capabilities:
                matched_capabilities += weight
        
        return matched_capabilities / total_weight if total_weight > 0 else 0.0
    
    def _score_task_type(self, model: ModelProfile, task_analysis: Dict[str, Any]) -> float:
        """Score model based on task type matching"""
        score = 0.0
        
        # Code task scoring
        if task_analysis.get("is_code_task", False):
            if model.model_type == ModelType.QWEN_CODER:
                score += 0.9
            elif ModelCapability.CODE_GENERATION in model.capabilities:
                score += 0.7
            else:
                score += 0.3
        
        # Reasoning task scoring
        if task_analysis.get("is_reasoning_task", False):
            if model.model_type == ModelType.REASONING_PLUS:
                score += 0.9
            elif ModelCapability.REASONING in model.capabilities:
                score += 0.7
            else:
                score += 0.4
        
        # Search task scoring
        if task_analysis.get("is_search_task", False):
            if ModelCapability.SEARCH in model.capabilities:
                score += 0.8
            else:
                score += 0.3
        
        # Analysis task scoring
        if task_analysis.get("is_analysis_task", False):
            if ModelCapability.CODE_ANALYSIS in model.capabilities:
                score += 0.8
            else:
                score += 0.4
        
        # Normalize score
        task_count = sum([
            task_analysis.get("is_code_task", False),
            task_analysis.get("is_reasoning_task", False),
            task_analysis.get("is_search_task", False),
            task_analysis.get("is_analysis_task", False)
        ])
        
        return score / max(task_count, 1)
    
    def _score_performance(self, model: ModelProfile, context: RoutingContext) -> float:
        """Score model based on performance requirements"""
        score = 0.0
        
        # Response time scoring
        if context.priority == "critical":
            # Prefer faster models for critical tasks
            if model.response_time_avg < 2.0:
                score += 0.4
            elif model.response_time_avg < 5.0:
                score += 0.2
        else:
            # Balance speed and accuracy
            if model.response_time_avg < 10.0:
                score += 0.3
        
        # Accuracy scoring
        score += model.accuracy_score * 0.4
        
        # Context length support
        if context.context_length <= model.max_context_length:
            score += 0.3
        else:
            score += 0.1  # Penalty for insufficient context length
        
        return score
    
    def _score_language_support(self, model: ModelProfile, language: str) -> float:
        """Score model based on programming language support"""
        if language in model.supported_languages:
            if language in model.specializations:
                return 1.0  # Perfect match
            else:
                return 0.8  # Good support
        else:
            return 0.3  # Limited support
    
    def _get_complexity_modifier(self, model: ModelProfile, complexity: TaskComplexity) -> float:
        """Get complexity modifier for model scoring"""
        
        # High-capability models get bonus for complex tasks
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            if model.model_type in [ModelType.REASONING_PLUS, ModelType.QWEN_CODER]:
                return 1.1
            elif model.model_type == ModelType.GENERAL_LLM:
                return 0.9
        
        # Simple tasks don't need complex models
        elif complexity == TaskComplexity.SIMPLE:
            if model.model_type == ModelType.GENERAL_LLM:
                return 1.1
            elif model.model_type in [ModelType.REASONING_PLUS]:
                return 0.95  # Slight penalty for overkill
        
        return 1.0  # No modifier
    
    def _select_best_model(self, model_scores: Dict[str, float], context: RoutingContext) -> str:
        """Select the best model from scored options"""
        
        if not model_scores:
            raise ValueError("No models available for routing")
        
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply user preferences if available
        if context.user_preferences:
            preferred_model = context.user_preferences.get("preferred_model")
            if preferred_model and preferred_model in model_scores:
                # Boost preferred model score
                model_scores[preferred_model] *= 1.2
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select the top model
        best_model = sorted_models[0][0]
        
        # Fallback check - ensure minimum score threshold
        if sorted_models[0][1] < 0.3:
            logger.warning(f"Best model score is low: {sorted_models[0][1]:.2f}")
            # Use general LLM as fallback
            if "general_llm" in model_scores:
                best_model = "general_llm"
        
        return best_model
    
    def _create_routing_decision(self, selected_model: str, model_scores: Dict[str, float],
                               context: RoutingContext, task_analysis: Dict[str, Any]) -> RoutingDecision:
        """Create a routing decision object"""
        
        model_profile = self.model_registry[selected_model]
        confidence = model_scores[selected_model]
        
        # Get alternative models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [model for model, score in sorted_models[1:4]]  # Top 3 alternatives
        
        # Create reasoning explanation
        reasoning = self._generate_routing_reasoning(selected_model, model_profile, task_analysis, confidence)
        
        # Estimate performance
        estimated_performance = {
            "response_time": model_profile.response_time_avg,
            "accuracy": model_profile.accuracy_score,
            "cost_estimate": context.expected_output_length * model_profile.cost_per_token,
            "context_utilization": min(context.context_length / model_profile.max_context_length, 1.0)
        }
        
        return RoutingDecision(
            selected_model=selected_model,
            model_type=model_profile.model_type,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_performance=estimated_performance,
            routing_metadata={
                "task_analysis": task_analysis,
                "all_scores": model_scores,
                "routing_timestamp": datetime.now().isoformat()
            }
        )
    
    def _generate_routing_reasoning(self, selected_model: str, model_profile: ModelProfile,
                                  task_analysis: Dict[str, Any], confidence: float) -> str:
        """Generate human-readable reasoning for the routing decision"""
        
        reasons = []
        
        # Model type reasoning
        if task_analysis.get("is_code_task", False) and model_profile.model_type == ModelType.QWEN_CODER:
            reasons.append("Selected Qwen Coder for code generation task")
        elif task_analysis.get("is_reasoning_task", False) and model_profile.model_type == ModelType.REASONING_PLUS:
            reasons.append("Selected Reasoning Plus for complex reasoning task")
        
        # Capability reasoning
        if ModelCapability.CODE_GENERATION in model_profile.capabilities and task_analysis.get("is_code_task", False):
            reasons.append("Model has strong code generation capabilities")
        
        # Language support reasoning
        detected_lang = task_analysis.get("language_detected", "python")
        if detected_lang in model_profile.specializations:
            reasons.append(f"Model specializes in {detected_lang}")
        
        # Performance reasoning
        if model_profile.accuracy_score > 0.9:
            reasons.append("High accuracy model selected")
        
        if confidence > 0.8:
            reasons.append("High confidence routing decision")
        elif confidence < 0.5:
            reasons.append("Low confidence - consider manual review")
        
        return "; ".join(reasons) if reasons else "Default routing based on availability"
    
    def _record_routing_decision(self, decision: RoutingDecision, context: RoutingContext):
        """Record routing decision for analysis and improvement"""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "selected_model": decision.selected_model,
            "confidence": decision.confidence,
            "task_type": context.task_type,
            "complexity": context.complexity.value if hasattr(context.complexity, 'value') else str(context.complexity),
            "context_length": context.context_length,
            "language": context.language,
            "reasoning": decision.reasoning
        }
        
        self.routing_history.append(record)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def update_model_performance(self, model_name: str, success: bool, response_time: float,
                               user_satisfaction: Optional[float] = None):
        """Update model performance metrics based on actual usage"""
        
        if model_name not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[model_name]
        
        # Update success rate
        total_requests = metrics["total_requests"]
        current_success_rate = metrics["success_rate"]
        new_success_rate = (current_success_rate * total_requests + (1.0 if success else 0.0)) / (total_requests + 1)
        metrics["success_rate"] = new_success_rate
        
        # Update average response time
        current_avg_time = metrics["avg_response_time"]
        new_avg_time = (current_avg_time * total_requests + response_time) / (total_requests + 1)
        metrics["avg_response_time"] = new_avg_time
        
        # Update user satisfaction if provided
        if user_satisfaction is not None:
            current_satisfaction = metrics["user_satisfaction"]
            new_satisfaction = (current_satisfaction * total_requests + user_satisfaction) / (total_requests + 1)
            metrics["user_satisfaction"] = new_satisfaction
        
        # Update total requests
        metrics["total_requests"] = total_requests + 1
        
        logger.info(f"Updated performance metrics for {model_name}: "
                   f"success_rate={new_success_rate:.3f}, "
                   f"avg_time={new_avg_time:.2f}s")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.model_registry:
            model = self.model_registry[model_name]
            metrics = self.performance_metrics.get(model_name, {})
            
            return {
                "name": model.name,
                "type": model.model_type.value,
                "capabilities": [cap.value for cap in model.capabilities],
                "specializations": model.specializations,
                "supported_languages": model.supported_languages,
                "max_context_length": model.max_context_length,
                "max_output_tokens": model.max_output_tokens,
                "accuracy_score": model.accuracy_score,
                "availability": model.availability,
                "performance_metrics": metrics
            }
        return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their capabilities"""
        return [
            {
                "name": model.name,
                "type": model.model_type.value,
                "capabilities": [cap.value for cap in model.capabilities],
                "specializations": model.specializations,
                "accuracy_score": model.accuracy_score,
                "avg_response_time": model.response_time_avg
            }
            for model in self.model_registry.values()
        ]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        if not self.routing_history:
            return {"total_decisions": 0}
        
        total_decisions = len(self.routing_history)
        model_usage = {}
        avg_confidence = 0.0
        
        for record in self.routing_history:
            model = record["selected_model"]
            model_usage[model] = model_usage.get(model, 0) + 1
            avg_confidence += record["confidence"]
        
        avg_confidence /= total_decisions
        
        return {
            "total_decisions": total_decisions,
            "model_usage": model_usage,
            "average_confidence": avg_confidence,
            "most_used_model": max(model_usage, key=model_usage.get) if model_usage else None
        }

# Global context router instance
_context_router: Optional[ContextRouter] = None

def get_context_router() -> ContextRouter:
    """Get the global context router instance"""
    global _context_router
    if _context_router is None:
        _context_router = ContextRouter()
    return _context_router