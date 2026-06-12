-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table (512-dim CLIP vectors)
CREATE TABLE embeddings (
    id      SERIAL PRIMARY KEY,
    vector  vector(512) NOT NULL
);

-- Labels table
CREATE TABLE labels (
    id           SERIAL PRIMARY KEY,
    description  TEXT,
    label        TEXT NOT NULL,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE SET NULL
);

-- Images table
CREATE TABLE images (
    id           SERIAL PRIMARY KEY,
    filepath     TEXT NOT NULL,
    label_id     INTEGER REFERENCES labels(id) ON DELETE SET NULL,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE SET NULL
);

-- Index for fast approximate nearest-neighbor search on image embeddings
CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 100);