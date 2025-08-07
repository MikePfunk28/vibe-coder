"""
Knowledge graph storage and management for RAG relationships.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from .connection import db_manager

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntity:
    """Represents a knowledge entity in the graph."""
    id: Optional[str] = None
    entity_type: str = ""
    entity_name: str = ""
    properties: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class KnowledgeRelationship:
    """Represents a relationship between knowledge entities."""
    id: Optional[str] = None
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: str = ""
    properties: Dict[str, Any] = None
    confidence_score: float = 1.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class KnowledgeGraphManager:
    """Manages knowledge graph storage and retrieval for RAG relationships."""
    
    def __init__(self):
        self.embedding_dimension = 1536
    
    async def create_entity(self, entity: KnowledgeEntity) -> str:
        """Create a new knowledge entity."""
        try:
            # Convert embedding to list if it's a numpy array
            embedding_list = None
            if entity.embedding is not None:
                embedding_list = entity.embedding.tolist() if isinstance(entity.embedding, np.ndarray) else entity.embedding
            
            entity_id = await db_manager.fetchval(
                """
                INSERT INTO knowledge_entities (entity_type, entity_name, properties, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (entity_type, entity_name) 
                DO UPDATE SET 
                    properties = EXCLUDED.properties,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                RETURNING id
                """,
                entity.entity_type,
                entity.entity_name,
                json.dumps(entity.properties),
                embedding_list
            )
            
            logger.info(f"Created/updated knowledge entity: {entity.entity_name} ({entity.entity_type})")
            return str(entity_id)
            
        except Exception as e:
            logger.error(f"Failed to create knowledge entity: {e}")
            raise
    
    async def create_relationship(self, relationship: KnowledgeRelationship) -> str:
        """Create a new knowledge relationship."""
        try:
            relationship_id = await db_manager.fetchval(
                """
                INSERT INTO knowledge_relationships 
                (source_entity_id, target_entity_id, relationship_type, properties, confidence_score)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relationship_type,
                json.dumps(relationship.properties),
                relationship.confidence_score
            )
            
            logger.info(f"Created knowledge relationship: {relationship.relationship_type}")
            return str(relationship_id)
            
        except Exception as e:
            logger.error(f"Failed to create knowledge relationship: {e}")
            raise
    
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get a knowledge entity by ID."""
        try:
            row = await db_manager.fetchrow(
                "SELECT * FROM knowledge_entities WHERE id = $1",
                entity_id
            )
            
            if not row:
                return None
            
            # Convert embedding back to numpy array
            embedding = None
            if row['embedding']:
                embedding = np.array(row['embedding'])
            
            return KnowledgeEntity(
                id=str(row['id']),
                entity_type=row['entity_type'],
                entity_name=row['entity_name'],
                properties=row['properties'] or {},
                embedding=embedding,
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            
        except Exception as e:
            logger.error(f"Failed to get knowledge entity: {e}")
            return None
    
    async def find_entities_by_type(self, entity_type: str, limit: int = 100) -> List[KnowledgeEntity]:
        """Find entities by type."""
        try:
            rows = await db_manager.fetch(
                "SELECT * FROM knowledge_entities WHERE entity_type = $1 LIMIT $2",
                entity_type, limit
            )
            
            entities = []
            for row in rows:
                embedding = None
                if row['embedding']:
                    embedding = np.array(row['embedding'])
                
                entities.append(KnowledgeEntity(
                    id=str(row['id']),
                    entity_type=row['entity_type'],
                    entity_name=row['entity_name'],
                    properties=row['properties'] or {},
                    embedding=embedding,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to find entities by type: {e}")
            return []
    
    async def find_similar_entities(self, embedding: np.ndarray, entity_type: Optional[str] = None, 
                                  limit: int = 10, threshold: float = 0.7) -> List[Tuple[KnowledgeEntity, float]]:
        """Find entities similar to the given embedding."""
        try:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            query = """
                SELECT *, (embedding <=> $1) as distance
                FROM knowledge_entities
                WHERE embedding IS NOT NULL
            """
            params = [embedding_list]
            
            if entity_type:
                query += " AND entity_type = $2"
                params.append(entity_type)
            
            query += " ORDER BY distance LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            rows = await db_manager.fetch(query, *params)
            
            results = []
            for row in rows:
                # Convert distance to similarity score (1 - distance for cosine distance)
                similarity = 1.0 - row['distance']
                
                if similarity >= threshold:
                    entity_embedding = np.array(row['embedding']) if row['embedding'] else None
                    
                    entity = KnowledgeEntity(
                        id=str(row['id']),
                        entity_type=row['entity_type'],
                        entity_name=row['entity_name'],
                        properties=row['properties'] or {},
                        embedding=entity_embedding,
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    
                    results.append((entity, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar entities: {e}")
            return []
    
    async def get_entity_relationships(self, entity_id: str, 
                                     relationship_type: Optional[str] = None) -> List[KnowledgeRelationship]:
        """Get all relationships for an entity."""
        try:
            query = """
                SELECT * FROM knowledge_relationships 
                WHERE source_entity_id = $1 OR target_entity_id = $1
            """
            params = [entity_id]
            
            if relationship_type:
                query += " AND relationship_type = $2"
                params.append(relationship_type)
            
            rows = await db_manager.fetch(query, *params)
            
            relationships = []
            for row in rows:
                relationships.append(KnowledgeRelationship(
                    id=str(row['id']),
                    source_entity_id=str(row['source_entity_id']),
                    target_entity_id=str(row['target_entity_id']),
                    relationship_type=row['relationship_type'],
                    properties=row['properties'] or {},
                    confidence_score=row['confidence_score'],
                    created_at=row['created_at']
                ))
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return []
    
    async def get_connected_entities(self, entity_id: str, max_depth: int = 2) -> List[KnowledgeEntity]:
        """Get entities connected to the given entity within max_depth hops."""
        try:
            # Use recursive CTE to find connected entities
            query = """
                WITH RECURSIVE connected_entities AS (
                    -- Base case: direct relationships
                    SELECT DISTINCT 
                        CASE 
                            WHEN source_entity_id = $1 THEN target_entity_id
                            ELSE source_entity_id
                        END as entity_id,
                        1 as depth
                    FROM knowledge_relationships
                    WHERE source_entity_id = $1 OR target_entity_id = $1
                    
                    UNION
                    
                    -- Recursive case: indirect relationships
                    SELECT DISTINCT
                        CASE 
                            WHEN kr.source_entity_id = ce.entity_id THEN kr.target_entity_id
                            ELSE kr.source_entity_id
                        END as entity_id,
                        ce.depth + 1
                    FROM knowledge_relationships kr
                    JOIN connected_entities ce ON (kr.source_entity_id = ce.entity_id OR kr.target_entity_id = ce.entity_id)
                    WHERE ce.depth < $2
                )
                SELECT ke.* FROM knowledge_entities ke
                JOIN connected_entities ce ON ke.id = ce.entity_id
                WHERE ke.id != $1
            """
            
            rows = await db_manager.fetch(query, entity_id, max_depth)
            
            entities = []
            for row in rows:
                embedding = None
                if row['embedding']:
                    embedding = np.array(row['embedding'])
                
                entities.append(KnowledgeEntity(
                    id=str(row['id']),
                    entity_type=row['entity_type'],
                    entity_name=row['entity_name'],
                    properties=row['properties'] or {},
                    embedding=embedding,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to get connected entities: {e}")
            return []
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete a knowledge entity and its relationships."""
        try:
            async with db_manager.transaction() as conn:
                # Delete relationships first
                await conn.execute(
                    "DELETE FROM knowledge_relationships WHERE source_entity_id = $1 OR target_entity_id = $1",
                    entity_id
                )
                
                # Delete entity
                result = await conn.execute(
                    "DELETE FROM knowledge_entities WHERE id = $1",
                    entity_id
                )
                
                logger.info(f"Deleted knowledge entity: {entity_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete knowledge entity: {e}")
            return False
    
    async def update_entity_embedding(self, entity_id: str, embedding: np.ndarray) -> bool:
        """Update the embedding for a knowledge entity."""
        try:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            await db_manager.execute(
                "UPDATE knowledge_entities SET embedding = $1, updated_at = NOW() WHERE id = $2",
                embedding_list, entity_id
            )
            
            logger.info(f"Updated embedding for entity: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entity embedding: {e}")
            return False
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            stats = {}
            
            # Entity counts by type
            entity_counts = await db_manager.fetch(
                "SELECT entity_type, COUNT(*) as count FROM knowledge_entities GROUP BY entity_type"
            )
            stats['entity_counts'] = {row['entity_type']: row['count'] for row in entity_counts}
            
            # Relationship counts by type
            relationship_counts = await db_manager.fetch(
                "SELECT relationship_type, COUNT(*) as count FROM knowledge_relationships GROUP BY relationship_type"
            )
            stats['relationship_counts'] = {row['relationship_type']: row['count'] for row in relationship_counts}
            
            # Total counts
            stats['total_entities'] = await db_manager.fetchval("SELECT COUNT(*) FROM knowledge_entities")
            stats['total_relationships'] = await db_manager.fetchval("SELECT COUNT(*) FROM knowledge_relationships")
            
            # Entities with embeddings
            stats['entities_with_embeddings'] = await db_manager.fetchval(
                "SELECT COUNT(*) FROM knowledge_entities WHERE embedding IS NOT NULL"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}

# Global knowledge graph manager instance
knowledge_graph = KnowledgeGraphManager()