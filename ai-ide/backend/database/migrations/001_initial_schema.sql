-- Migration: Initial AI IDE Database Schema
-- Version: 001

-- UP
-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

-- Code embeddings and semantic search
CREATE TABLE IF NOT EXISTS code_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(file_path, content_hash)
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS code_embeddings_embedding_idx ON code_embeddings 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Agent interactions and conversations
CREATE TABLE IF NOT EXISTS agent_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    context JSONB,
    performance_metrics JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    duration_ms INTEGER
);

-- Reasoning traces for debugging and analysis
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID REFERENCES agent_interactions(id),
    step_number INTEGER NOT NULL,
    reasoning_type VARCHAR(50) NOT NULL,
    thought_process TEXT,
    intermediate_results JSONB,
    confidence_score FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Web search cache for performance optimization
CREATE TABLE IF NOT EXISTS web_search_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    search_engine VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    relevance_scores JSONB,
    cached_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    hit_count INTEGER DEFAULT 0
);

-- User interactions and feedback
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    session_id UUID NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    context JSONB,
    ai_response JSONB,
    user_feedback INTEGER CHECK (user_feedback >= 1 AND user_feedback <= 5),
    explicit_feedback TEXT,
    implicit_signals JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    benchmark_context JSONB,
    measurement_timestamp TIMESTAMP DEFAULT NOW()
);

-- Self-improvement history
CREATE TABLE IF NOT EXISTS model_improvements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    improvement_type VARCHAR(50) NOT NULL,
    description TEXT,
    code_changes JSONB,
    performance_before JSONB,
    performance_after JSONB,
    validation_results JSONB,
    applied_at TIMESTAMP DEFAULT NOW(),
    rollback_available BOOLEAN DEFAULT TRUE
);

-- Knowledge graph for RAG relationships
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    entity_name VARCHAR(200) NOT NULL,
    properties JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(entity_type, entity_name)
);

CREATE TABLE IF NOT EXISTS knowledge_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID REFERENCES knowledge_entities(id),
    target_entity_id UUID REFERENCES knowledge_entities(id),
    relationship_type VARCHAR(50) NOT NULL,
    properties JSONB,
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context windows and memory management
CREATE TABLE IF NOT EXISTS context_windows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    window_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    relevance_score FLOAT,
    priority INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- RAG document storage
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(50) NOT NULL,
    source_url TEXT,
    title VARCHAR(500),
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536),
    indexed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(content_hash)
);

-- RAG document chunks for better retrieval
CREATE TABLE IF NOT EXISTS rag_document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES rag_documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_interactions_session ON agent_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_timestamp ON agent_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_interaction ON reasoning_traces(interaction_id);
CREATE INDEX IF NOT EXISTS idx_web_search_cache_expires ON web_search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_interactions_session ON user_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_model ON performance_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_knowledge_entities_type ON knowledge_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_source ON knowledge_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_target ON knowledge_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_context_windows_session ON context_windows(session_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_source ON rag_documents(source_type);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_document_chunks(document_id);

-- Vector similarity indexes
CREATE INDEX IF NOT EXISTS knowledge_entities_embedding_idx ON knowledge_entities 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS context_windows_embedding_idx ON context_windows 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS rag_documents_embedding_idx ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- DOWN
-- Drop all tables and extensions in reverse order
DROP INDEX IF EXISTS rag_chunks_embedding_idx;
DROP INDEX IF EXISTS rag_documents_embedding_idx;
DROP INDEX IF EXISTS context_windows_embedding_idx;
DROP INDEX IF EXISTS knowledge_entities_embedding_idx;
DROP INDEX IF EXISTS idx_rag_chunks_document;
DROP INDEX IF EXISTS idx_rag_documents_source;
DROP INDEX IF EXISTS idx_context_windows_session;
DROP INDEX IF EXISTS idx_knowledge_relationships_target;
DROP INDEX IF EXISTS idx_knowledge_relationships_source;
DROP INDEX IF EXISTS idx_knowledge_entities_type;
DROP INDEX IF EXISTS idx_performance_metrics_model;
DROP INDEX IF EXISTS idx_user_interactions_timestamp;
DROP INDEX IF EXISTS idx_user_interactions_session;
DROP INDEX IF EXISTS idx_web_search_cache_expires;
DROP INDEX IF EXISTS idx_reasoning_traces_interaction;
DROP INDEX IF EXISTS idx_agent_interactions_timestamp;
DROP INDEX IF EXISTS idx_agent_interactions_session;
DROP INDEX IF EXISTS code_embeddings_embedding_idx;

DROP TABLE IF EXISTS rag_document_chunks;
DROP TABLE IF EXISTS rag_documents;
DROP TABLE IF EXISTS context_windows;
DROP TABLE IF EXISTS knowledge_relationships;
DROP TABLE IF EXISTS knowledge_entities;
DROP TABLE IF EXISTS model_improvements;
DROP TABLE IF EXISTS performance_metrics;
DROP TABLE IF EXISTS user_interactions;
DROP TABLE IF EXISTS web_search_cache;
DROP TABLE IF EXISTS reasoning_traces;
DROP TABLE IF EXISTS agent_interactions;
DROP TABLE IF EXISTS code_embeddings;
DROP TABLE IF EXISTS schema_migrations;

DROP EXTENSION IF EXISTS "uuid-ossp";
DROP EXTENSION IF EXISTS vector;