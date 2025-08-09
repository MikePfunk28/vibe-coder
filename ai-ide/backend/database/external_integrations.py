"""
External datastore integrations for persistent memory.

Provides lightweight connectors for MongoDB, Airtable, and Pinecone so the
AI IDE can persist embeddings, conversation history, and other artifacts in
free-tier services. Each integration gracefully degrades if optional
dependencies are missing, enabling the project to run without these services
installed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Optional imports for third-party services
try:  # pragma: no cover - dependency may not be installed
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore

try:  # pragma: no cover - dependency may not be installed
    from pyairtable import Table  # type: ignore
except Exception:  # pragma: no cover
    Table = None  # type: ignore

try:  # pragma: no cover - dependency may not be installed
    import pinecone  # type: ignore
except Exception:  # pragma: no cover
    pinecone = None  # type: ignore


class MongoDBStore:
    """Minimal wrapper around MongoDB for storing embeddings and metadata."""

    def __init__(self, uri: str, db_name: str, collection: str) -> None:
        if MongoClient is None:  # pragma: no cover - handled at runtime
            raise ImportError("pymongo is required for MongoDBStore")
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection]

    def store_embedding(self, key: str, embedding: List[float]) -> None:
        """Store or update an embedding vector."""
        self.collection.update_one(
            {"_id": key}, {"$set": {"embedding": embedding}}, upsert=True
        )

    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Retrieve an embedding vector by key."""
        doc = self.collection.find_one({"_id": key})
        return doc.get("embedding") if doc else None


class AirtableStore:
    """Simple Airtable table wrapper for storing structured records."""

    def __init__(self, api_key: str, base_id: str, table_name: str) -> None:
        if Table is None:  # pragma: no cover - handled at runtime
            raise ImportError("pyairtable is required for AirtableStore")
        self.table = Table(api_key, base_id, table_name)

    def store_record(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a record in Airtable."""
        return self.table.create(fields)

    def fetch_records(self, formula: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch records using an optional formula."""
        return self.table.all(formula=formula)


class PineconeStore:
    """Wrapper around Pinecone vector database for embedding search."""

    def __init__(self, api_key: str, index_name: str) -> None:
        if pinecone is None:  # pragma: no cover - handled at runtime
            raise ImportError("pinecone-client is required for PineconeStore")
        pinecone.init(api_key=api_key)
        self.index = pinecone.Index(index_name)

    def upsert(self, items: Dict[str, List[float]]) -> None:
        """Upsert a batch of vectors into the index."""
        vectors = [(key, vec) for key, vec in items.items()]
        self.index.upsert(vectors=vectors)

    def query(self, vector: List[float], top_k: int = 5) -> Any:
        """Query the index for similar vectors."""
        return self.index.query(vector=vector, top_k=top_k)
