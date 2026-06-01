-- Drop everything
DROP TABLE IF EXISTS images CASCADE;
DROP TABLE IF EXISTS labels CASCADE;
DROP TABLE IF EXISTS embeddings CASCADE;

-- Recreate with 768-dim vectors
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id      SERIAL PRIMARY KEY,
    vector  vector(768) NOT NULL
);

CREATE TABLE labels (
    id           SERIAL PRIMARY KEY,
    description  TEXT,
    label        TEXT NOT NULL,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE SET NULL
);

CREATE TABLE images (
    id           SERIAL PRIMARY KEY,
    filepath     TEXT NOT NULL,
    label_id     INTEGER REFERENCES labels(id) ON DELETE SET NULL,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE SET NULL
);

CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 100);
