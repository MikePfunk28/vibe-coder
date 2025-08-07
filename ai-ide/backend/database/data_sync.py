"""
Data synchronization between extension and backend services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from .connection import db_manager
from .caching import cache_manager

logger = logging.getLogger(__name__)

class SyncEventType(Enum):
    """Types of synchronization events."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BULK_UPDATE = "bulk_update"

@dataclass
class SyncEvent:
    """Represents a data synchronization event."""
    id: str
    event_type: SyncEventType
    table_name: str
    entity_id: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime
    source: str  # 'extension' or 'backend'
    processed: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'table_name': self.table_name,
            'entity_id': self.entity_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'processed': self.processed,
            'retry_count': self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncEvent':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            event_type=SyncEventType(data['event_type']),
            table_name=data['table_name'],
            entity_id=data.get('entity_id'),
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            processed=data.get('processed', False),
            retry_count=data.get('retry_count', 0)
        )

class DataSyncManager:
    """Manages data synchronization between extension and backend."""
    
    def __init__(self):
        self.sync_handlers: Dict[str, Callable] = {}
        self.sync_queue_key = "sync:events"
        self.sync_status_key = "sync:status"
        self.max_retry_count = 3
        self.sync_interval = 5  # seconds
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_sync_handler(self, table_name: str, handler: Callable[[SyncEvent], bool]) -> None:
        """Register a handler for synchronizing specific table data."""
        self.sync_handlers[table_name] = handler
        logger.info(f"Registered sync handler for table: {table_name}")
    
    async def queue_sync_event(self, event_type: SyncEventType, table_name: str, 
                              entity_id: Optional[str], data: Dict[str, Any], 
                              source: str = "backend") -> str:
        """Queue a synchronization event."""
        try:
            event = SyncEvent(
                id=str(uuid.uuid4()),
                event_type=event_type,
                table_name=table_name,
                entity_id=entity_id,
                data=data,
                timestamp=datetime.now(),
                source=source
            )
            
            # Add to Redis queue
            await cache_manager.set('sync', f"event:{event.id}", event.to_dict(), ttl=3600*24)  # 24 hours
            
            # Add to processing queue
            queue_length = await cache_manager._redis.lpush(self.sync_queue_key, event.id)
            
            logger.debug(f"Queued sync event {event.id} for {table_name} (queue length: {queue_length})")
            return event.id
            
        except Exception as e:
            logger.error(f"Failed to queue sync event: {e}")
            raise
    
    async def process_sync_events(self) -> int:
        """Process pending synchronization events."""
        processed_count = 0
        
        try:
            while True:
                # Get next event from queue (blocking with timeout)
                event_id = await cache_manager._redis.brpop(self.sync_queue_key, timeout=1)
                if not event_id:
                    break
                
                event_id = event_id[1].decode() if isinstance(event_id[1], bytes) else event_id[1]
                
                # Get event data
                event_data = await cache_manager.get('sync', f"event:{event_id}")
                if not event_data:
                    logger.warning(f"Sync event {event_id} not found in cache")
                    continue
                
                event = SyncEvent.from_dict(event_data)
                
                # Process the event
                success = await self._process_single_event(event)
                
                if success:
                    # Mark as processed
                    event.processed = True
                    await cache_manager.set('sync', f"event:{event.id}", event.to_dict(), ttl=3600)
                    processed_count += 1
                else:
                    # Retry logic
                    event.retry_count += 1
                    if event.retry_count < self.max_retry_count:
                        # Re-queue for retry
                        await cache_manager._redis.lpush(self.sync_queue_key, event.id)
                        await cache_manager.set('sync', f"event:{event.id}", event.to_dict(), ttl=3600*24)
                        logger.warning(f"Re-queued sync event {event.id} for retry ({event.retry_count}/{self.max_retry_count})")
                    else:
                        logger.error(f"Sync event {event.id} failed after {self.max_retry_count} retries")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Failed to process sync events: {e}")
            return processed_count
    
    async def _process_single_event(self, event: SyncEvent) -> bool:
        """Process a single synchronization event."""
        try:
            handler = self.sync_handlers.get(event.table_name)
            if not handler:
                logger.warning(f"No sync handler registered for table: {event.table_name}")
                return True  # Consider it processed to avoid infinite retries
            
            # Call the handler
            success = await handler(event)
            
            if success:
                logger.debug(f"Successfully processed sync event {event.id}")
                
                # Update sync status
                await self._update_sync_status(event.table_name, event.timestamp)
                
                # Invalidate related caches
                await self._invalidate_related_caches(event)
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to process sync event {event.id}: {e}")
            return False
    
    async def _update_sync_status(self, table_name: str, timestamp: datetime) -> None:
        """Update synchronization status for a table."""
        try:
            status_key = f"{self.sync_status_key}:{table_name}"
            status = {
                'last_sync': timestamp.isoformat(),
                'table_name': table_name
            }
            await cache_manager.set('sync', status_key, status, ttl=3600*24*7)  # 7 days
            
        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
    
    async def _invalidate_related_caches(self, event: SyncEvent) -> None:
        """Invalidate caches related to the sync event."""
        try:
            # Invalidate based on table and event type
            if event.table_name == "code_embeddings":
                # Clear embedding caches
                if event.entity_id:
                    # Try to find and clear specific embedding cache
                    embedding = await db_manager.fetchrow(
                        "SELECT file_path, content_hash FROM code_embeddings WHERE id = $1",
                        event.entity_id
                    )
                    if embedding:
                        await cache_manager.delete('embedding', f"{embedding['file_path']}:{embedding['content_hash']}")
                
                # Clear search result caches that might be affected
                await cache_manager.clear_cache_type('search')
            
            elif event.table_name == "agent_interactions":
                # Clear agent-related caches
                if event.data.get('session_id'):
                    await cache_manager.delete('agent', f"session_stats:{event.data['session_id']}")
            
            elif event.table_name == "reasoning_traces":
                # Clear reasoning trace caches
                if event.data.get('interaction_id'):
                    await cache_manager.delete('reasoning', event.data['interaction_id'])
            
            elif event.table_name == "web_search_cache":
                # Clear web search caches
                if event.data.get('query_text') and event.data.get('search_engine'):
                    await cache_manager.delete('web_search', 
                        f"{event.data['search_engine']}:{cache_manager._hash_key(event.data['query_text'])}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate related caches: {e}")
    
    async def get_sync_status(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get synchronization status."""
        try:
            if table_name:
                status_key = f"{self.sync_status_key}:{table_name}"
                status = await cache_manager.get('sync', status_key)
                return status or {}
            
            # Get status for all tables
            all_status = {}
            for table in self.sync_handlers.keys():
                status_key = f"{self.sync_status_key}:{table}"
                status = await cache_manager.get('sync', status_key)
                if status:
                    all_status[table] = status
            
            # Add queue statistics
            queue_length = await cache_manager._redis.llen(self.sync_queue_key)
            all_status['queue_length'] = queue_length
            
            return all_status
            
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {}
    
    async def start_sync_worker(self) -> None:
        """Start the background synchronization worker."""
        if self._running:
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_worker_loop())
        logger.info("Started data synchronization worker")
    
    async def stop_sync_worker(self) -> None:
        """Stop the background synchronization worker."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        logger.info("Stopped data synchronization worker")
    
    async def _sync_worker_loop(self) -> None:
        """Background worker loop for processing sync events."""
        while self._running:
            try:
                processed = await self.process_sync_events()
                if processed > 0:
                    logger.debug(f"Processed {processed} sync events")
                
                # Wait before next iteration
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync worker loop: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def force_sync_table(self, table_name: str) -> bool:
        """Force synchronization of all data for a specific table."""
        try:
            handler = self.sync_handlers.get(table_name)
            if not handler:
                logger.error(f"No sync handler for table: {table_name}")
                return False
            
            # Create a bulk sync event
            event = SyncEvent(
                id=str(uuid.uuid4()),
                event_type=SyncEventType.BULK_UPDATE,
                table_name=table_name,
                entity_id=None,
                data={'force_sync': True},
                timestamp=datetime.now(),
                source='system'
            )
            
            success = await self._process_single_event(event)
            
            if success:
                logger.info(f"Successfully force-synced table: {table_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to force sync table {table_name}: {e}")
            return False
    
    async def clear_sync_queue(self) -> int:
        """Clear all pending sync events."""
        try:
            # Get all event IDs from queue
            event_ids = await cache_manager._redis.lrange(self.sync_queue_key, 0, -1)
            
            # Clear the queue
            cleared_count = await cache_manager._redis.delete(self.sync_queue_key)
            
            # Clean up event data
            for event_id in event_ids:
                event_id = event_id.decode() if isinstance(event_id, bytes) else event_id
                await cache_manager.delete('sync', f"event:{event_id}")
            
            logger.info(f"Cleared {len(event_ids)} sync events from queue")
            return len(event_ids)
            
        except Exception as e:
            logger.error(f"Failed to clear sync queue: {e}")
            return 0

# Global data sync manager instance
data_sync_manager = DataSyncManager()

# Convenience functions
async def queue_sync_event(event_type: SyncEventType, table_name: str, 
                          entity_id: Optional[str], data: Dict[str, Any], 
                          source: str = "backend") -> str:
    """Queue a synchronization event."""
    return await data_sync_manager.queue_sync_event(event_type, table_name, entity_id, data, source)

async def start_sync_worker():
    """Start the synchronization worker."""
    await data_sync_manager.start_sync_worker()

async def stop_sync_worker():
    """Stop the synchronization worker."""
    await data_sync_manager.stop_sync_worker()

def register_sync_handler(table_name: str, handler: Callable[[SyncEvent], bool]):
    """Register a sync handler for a table."""
    data_sync_manager.register_sync_handler(table_name, handler)