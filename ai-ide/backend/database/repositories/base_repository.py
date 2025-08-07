"""
Base repository class implementing common data access patterns.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from ..connection import db_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class QueryFilter:
    """Represents a query filter condition."""
    field: str
    operator: str  # =, !=, >, <, >=, <=, IN, LIKE, ILIKE
    value: Any
    
@dataclass
class QueryOptions:
    """Query options for pagination, sorting, etc."""
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    order_direction: str = "ASC"  # ASC or DESC
    filters: List[QueryFilter] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = []

class BaseRepository(ABC, Generic[T]):
    """Base repository class with common CRUD operations."""
    
    def __init__(self, table_name: str, id_field: str = "id"):
        self.table_name = table_name
        self.id_field = id_field
    
    @abstractmethod
    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Convert database row to entity object."""
        pass
    
    @abstractmethod
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity object to dictionary for database operations."""
        pass
    
    def _build_where_clause(self, filters: List[QueryFilter]) -> tuple[str, List[Any]]:
        """Build WHERE clause from filters."""
        if not filters:
            return "", []
        
        conditions = []
        params = []
        param_count = 1
        
        for filter_item in filters:
            if filter_item.operator == "IN":
                placeholders = ",".join([f"${param_count + i}" for i in range(len(filter_item.value))])
                conditions.append(f"{filter_item.field} IN ({placeholders})")
                params.extend(filter_item.value)
                param_count += len(filter_item.value)
            else:
                conditions.append(f"{filter_item.field} {filter_item.operator} ${param_count}")
                params.append(filter_item.value)
                param_count += 1
        
        where_clause = " WHERE " + " AND ".join(conditions)
        return where_clause, params
    
    def _build_order_clause(self, order_by: Optional[str], order_direction: str) -> str:
        """Build ORDER BY clause."""
        if not order_by:
            return ""
        return f" ORDER BY {order_by} {order_direction}"
    
    def _build_limit_clause(self, limit: Optional[int], offset: Optional[int]) -> tuple[str, List[Any]]:
        """Build LIMIT and OFFSET clause."""
        clause_parts = []
        params = []
        
        if limit is not None:
            clause_parts.append(f"LIMIT ${len(params) + 1}")
            params.append(limit)
        
        if offset is not None:
            clause_parts.append(f"OFFSET ${len(params) + 1}")
            params.append(offset)
        
        return " " + " ".join(clause_parts) if clause_parts else "", params
    
    async def create(self, entity: T) -> str:
        """Create a new entity and return its ID."""
        try:
            entity_dict = self._entity_to_dict(entity)
            
            # Remove ID field if it's None (for auto-generated IDs)
            if self.id_field in entity_dict and entity_dict[self.id_field] is None:
                del entity_dict[self.id_field]
            
            fields = list(entity_dict.keys())
            values = list(entity_dict.values())
            placeholders = [f"${i+1}" for i in range(len(values))]
            
            query = f"""
                INSERT INTO {self.table_name} ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                RETURNING {self.id_field}
            """
            
            entity_id = await db_manager.fetchval(query, *values)
            logger.debug(f"Created entity in {self.table_name} with ID: {entity_id}")
            return str(entity_id)
            
        except Exception as e:
            logger.error(f"Failed to create entity in {self.table_name}: {e}")
            raise
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        try:
            query = f"SELECT * FROM {self.table_name} WHERE {self.id_field} = $1"
            row = await db_manager.fetchrow(query, entity_id)
            
            if row:
                return self._row_to_entity(dict(row))
            return None
            
        except Exception as e:
            logger.error(f"Failed to get entity from {self.table_name}: {e}")
            return None
    
    async def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity by ID."""
        try:
            if not updates:
                return True
            
            # Add updated_at timestamp if the table has this field
            updates = updates.copy()
            updates['updated_at'] = datetime.now()
            
            fields = list(updates.keys())
            values = list(updates.values())
            set_clause = ", ".join([f"{field} = ${i+1}" for i, field in enumerate(fields)])
            
            query = f"""
                UPDATE {self.table_name} 
                SET {set_clause}
                WHERE {self.id_field} = ${len(values) + 1}
            """
            
            result = await db_manager.execute(query, *values, entity_id)
            success = "UPDATE 1" in result
            
            if success:
                logger.debug(f"Updated entity in {self.table_name} with ID: {entity_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update entity in {self.table_name}: {e}")
            return False
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        try:
            query = f"DELETE FROM {self.table_name} WHERE {self.id_field} = $1"
            result = await db_manager.execute(query, entity_id)
            success = "DELETE 1" in result
            
            if success:
                logger.debug(f"Deleted entity from {self.table_name} with ID: {entity_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete entity from {self.table_name}: {e}")
            return False
    
    async def find_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """Find all entities with optional filtering and pagination."""
        try:
            options = options or QueryOptions()
            
            base_query = f"SELECT * FROM {self.table_name}"
            
            where_clause, where_params = self._build_where_clause(options.filters)
            order_clause = self._build_order_clause(options.order_by, options.order_direction)
            limit_clause, limit_params = self._build_limit_clause(options.limit, options.offset)
            
            query = base_query + where_clause + order_clause + limit_clause
            params = where_params + limit_params
            
            rows = await db_manager.fetch(query, *params)
            return [self._row_to_entity(dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to find entities in {self.table_name}: {e}")
            return []
    
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Count entities with optional filtering."""
        try:
            base_query = f"SELECT COUNT(*) FROM {self.table_name}"
            
            where_clause, params = self._build_where_clause(filters or [])
            query = base_query + where_clause
            
            count = await db_manager.fetchval(query, *params)
            return count or 0
            
        except Exception as e:
            logger.error(f"Failed to count entities in {self.table_name}: {e}")
            return 0
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists by ID."""
        try:
            query = f"SELECT 1 FROM {self.table_name} WHERE {self.id_field} = $1 LIMIT 1"
            result = await db_manager.fetchval(query, entity_id)
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to check entity existence in {self.table_name}: {e}")
            return False
    
    async def bulk_create(self, entities: List[T]) -> List[str]:
        """Create multiple entities in a single transaction."""
        if not entities:
            return []
        
        try:
            entity_ids = []
            
            async with db_manager.transaction() as conn:
                for entity in entities:
                    entity_dict = self._entity_to_dict(entity)
                    
                    # Remove ID field if it's None
                    if self.id_field in entity_dict and entity_dict[self.id_field] is None:
                        del entity_dict[self.id_field]
                    
                    fields = list(entity_dict.keys())
                    values = list(entity_dict.values())
                    placeholders = [f"${i+1}" for i in range(len(values))]
                    
                    query = f"""
                        INSERT INTO {self.table_name} ({', '.join(fields)})
                        VALUES ({', '.join(placeholders)})
                        RETURNING {self.id_field}
                    """
                    
                    entity_id = await conn.fetchval(query, *values)
                    entity_ids.append(str(entity_id))
            
            logger.debug(f"Bulk created {len(entities)} entities in {self.table_name}")
            return entity_ids
            
        except Exception as e:
            logger.error(f"Failed to bulk create entities in {self.table_name}: {e}")
            raise
    
    async def bulk_update(self, updates: List[tuple[str, Dict[str, Any]]]) -> int:
        """Update multiple entities in a single transaction."""
        if not updates:
            return 0
        
        try:
            updated_count = 0
            
            async with db_manager.transaction() as conn:
                for entity_id, update_data in updates:
                    if not update_data:
                        continue
                    
                    # Add updated_at timestamp
                    update_data = update_data.copy()
                    update_data['updated_at'] = datetime.now()
                    
                    fields = list(update_data.keys())
                    values = list(update_data.values())
                    set_clause = ", ".join([f"{field} = ${i+1}" for i, field in enumerate(fields)])
                    
                    query = f"""
                        UPDATE {self.table_name} 
                        SET {set_clause}
                        WHERE {self.id_field} = ${len(values) + 1}
                    """
                    
                    result = await conn.execute(query, *values, entity_id)
                    if "UPDATE 1" in result:
                        updated_count += 1
            
            logger.debug(f"Bulk updated {updated_count} entities in {self.table_name}")
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to bulk update entities in {self.table_name}: {e}")
            raise
    
    async def bulk_delete(self, entity_ids: List[str]) -> int:
        """Delete multiple entities in a single transaction."""
        if not entity_ids:
            return 0
        
        try:
            placeholders = ",".join([f"${i+1}" for i in range(len(entity_ids))])
            query = f"DELETE FROM {self.table_name} WHERE {self.id_field} IN ({placeholders})"
            
            result = await db_manager.execute(query, *entity_ids)
            
            # Extract number of deleted rows from result
            deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
            
            logger.debug(f"Bulk deleted {deleted_count} entities from {self.table_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to bulk delete entities from {self.table_name}: {e}")
            raise