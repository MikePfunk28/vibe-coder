"""
Repository for agent interactions and related data.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from .base_repository import BaseRepository, QueryFilter, QueryOptions
from ..caching import cache_manager

@dataclass
class AgentInteraction:
    """Agent interaction entity."""
    id: Optional[str] = None
    session_id: str = ""
    agent_type: str = ""
    interaction_type: str = ""
    input_data: Dict[str, Any] = None
    output_data: Dict[str, Any] = None
    context: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.input_data is None:
            self.input_data = {}
        if self.output_data is None:
            self.output_data = {}
        if self.context is None:
            self.context = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class ReasoningTrace:
    """Reasoning trace entity."""
    id: Optional[str] = None
    interaction_id: str = ""
    step_number: int = 0
    reasoning_type: str = ""
    thought_process: str = ""
    intermediate_results: Dict[str, Any] = None
    confidence_score: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.intermediate_results is None:
            self.intermediate_results = {}

class AgentInteractionRepository(BaseRepository[AgentInteraction]):
    """Repository for agent interactions."""
    
    def __init__(self):
        super().__init__("agent_interactions")
    
    def _row_to_entity(self, row: Dict[str, Any]) -> AgentInteraction:
        """Convert database row to AgentInteraction entity."""
        return AgentInteraction(
            id=str(row['id']) if row['id'] else None,
            session_id=str(row['session_id']),
            agent_type=row['agent_type'],
            interaction_type=row['interaction_type'],
            input_data=row['input_data'] or {},
            output_data=row['output_data'] or {},
            context=row['context'] or {},
            performance_metrics=row['performance_metrics'] or {},
            timestamp=row['timestamp'],
            duration_ms=row['duration_ms']
        )
    
    def _entity_to_dict(self, entity: AgentInteraction) -> Dict[str, Any]:
        """Convert AgentInteraction entity to dictionary."""
        return {
            'id': entity.id,
            'session_id': entity.session_id,
            'agent_type': entity.agent_type,
            'interaction_type': entity.interaction_type,
            'input_data': entity.input_data,
            'output_data': entity.output_data,
            'context': entity.context,
            'performance_metrics': entity.performance_metrics,
            'timestamp': entity.timestamp or datetime.now(),
            'duration_ms': entity.duration_ms
        }
    
    async def find_by_session(self, session_id: str, limit: Optional[int] = None) -> List[AgentInteraction]:
        """Find interactions by session ID."""
        filters = [QueryFilter("session_id", "=", session_id)]
        options = QueryOptions(
            filters=filters,
            order_by="timestamp",
            order_direction="DESC",
            limit=limit
        )
        return await self.find_all(options)
    
    async def find_by_agent_type(self, agent_type: str, limit: Optional[int] = None) -> List[AgentInteraction]:
        """Find interactions by agent type."""
        filters = [QueryFilter("agent_type", "=", agent_type)]
        options = QueryOptions(
            filters=filters,
            order_by="timestamp",
            order_direction="DESC",
            limit=limit
        )
        return await self.find_all(options)
    
    async def find_recent_interactions(self, hours: int = 24, limit: Optional[int] = None) -> List[AgentInteraction]:
        """Find recent interactions within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filters = [QueryFilter("timestamp", ">=", cutoff_time)]
        options = QueryOptions(
            filters=filters,
            order_by="timestamp",
            order_direction="DESC",
            limit=limit
        )
        return await self.find_all(options)
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        try:
            # Check cache first
            cache_key = f"session_stats:{session_id}"
            cached_stats = await cache_manager.get('agent', cache_key)
            if cached_stats:
                return cached_stats
            
            interactions = await self.find_by_session(session_id)
            
            if not interactions:
                return {}
            
            # Calculate statistics
            total_interactions = len(interactions)
            agent_types = {}
            interaction_types = {}
            total_duration = 0
            
            for interaction in interactions:
                # Count by agent type
                agent_types[interaction.agent_type] = agent_types.get(interaction.agent_type, 0) + 1
                
                # Count by interaction type
                interaction_types[interaction.interaction_type] = interaction_types.get(interaction.interaction_type, 0) + 1
                
                # Sum duration
                if interaction.duration_ms:
                    total_duration += interaction.duration_ms
            
            stats = {
                'session_id': session_id,
                'total_interactions': total_interactions,
                'agent_types': agent_types,
                'interaction_types': interaction_types,
                'total_duration_ms': total_duration,
                'average_duration_ms': total_duration / total_interactions if total_interactions > 0 else 0,
                'first_interaction': interactions[-1].timestamp if interactions else None,
                'last_interaction': interactions[0].timestamp if interactions else None
            }
            
            # Cache for 5 minutes
            await cache_manager.set('agent', cache_key, stats, ttl=300)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}

class ReasoningTraceRepository(BaseRepository[ReasoningTrace]):
    """Repository for reasoning traces."""
    
    def __init__(self):
        super().__init__("reasoning_traces")
    
    def _row_to_entity(self, row: Dict[str, Any]) -> ReasoningTrace:
        """Convert database row to ReasoningTrace entity."""
        return ReasoningTrace(
            id=str(row['id']) if row['id'] else None,
            interaction_id=str(row['interaction_id']),
            step_number=row['step_number'],
            reasoning_type=row['reasoning_type'],
            thought_process=row['thought_process'],
            intermediate_results=row['intermediate_results'] or {},
            confidence_score=row['confidence_score'],
            timestamp=row['timestamp']
        )
    
    def _entity_to_dict(self, entity: ReasoningTrace) -> Dict[str, Any]:
        """Convert ReasoningTrace entity to dictionary."""
        return {
            'id': entity.id,
            'interaction_id': entity.interaction_id,
            'step_number': entity.step_number,
            'reasoning_type': entity.reasoning_type,
            'thought_process': entity.thought_process,
            'intermediate_results': entity.intermediate_results,
            'confidence_score': entity.confidence_score,
            'timestamp': entity.timestamp or datetime.now()
        }
    
    async def find_by_interaction(self, interaction_id: str) -> List[ReasoningTrace]:
        """Find reasoning traces by interaction ID."""
        filters = [QueryFilter("interaction_id", "=", interaction_id)]
        options = QueryOptions(
            filters=filters,
            order_by="step_number",
            order_direction="ASC"
        )
        return await self.find_all(options)
    
    async def find_by_reasoning_type(self, reasoning_type: str, limit: Optional[int] = None) -> List[ReasoningTrace]:
        """Find traces by reasoning type."""
        filters = [QueryFilter("reasoning_type", "=", reasoning_type)]
        options = QueryOptions(
            filters=filters,
            order_by="timestamp",
            order_direction="DESC",
            limit=limit
        )
        return await self.find_all(options)
    
    async def get_interaction_trace(self, interaction_id: str) -> List[ReasoningTrace]:
        """Get complete reasoning trace for an interaction with caching."""
        try:
            # Check cache first
            cached_trace = await cache_manager.get_reasoning_trace(interaction_id)
            if cached_trace:
                return [ReasoningTrace(**trace) for trace in cached_trace]
            
            traces = await self.find_by_interaction(interaction_id)
            
            # Cache the traces
            if traces:
                trace_dicts = [asdict(trace) for trace in traces]
                await cache_manager.cache_reasoning_trace(interaction_id, trace_dicts)
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to get interaction trace: {e}")
            return []
    
    async def create_trace_step(self, interaction_id: str, step_number: int, reasoning_type: str, 
                              thought_process: str, intermediate_results: Dict[str, Any] = None,
                              confidence_score: Optional[float] = None) -> str:
        """Create a new reasoning trace step."""
        trace = ReasoningTrace(
            interaction_id=interaction_id,
            step_number=step_number,
            reasoning_type=reasoning_type,
            thought_process=thought_process,
            intermediate_results=intermediate_results or {},
            confidence_score=confidence_score
        )
        
        trace_id = await self.create(trace)
        
        # Invalidate cache for this interaction
        await cache_manager.delete('reasoning', interaction_id)
        
        return trace_id

# Global repository instances
agent_interaction_repo = AgentInteractionRepository()
reasoning_trace_repo = ReasoningTraceRepository()