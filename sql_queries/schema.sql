-- ============================================================
-- Patent Analysis Platform — Supabase Schema  (v2)
-- Compatible with Supabase SQL Editor (PostgreSQL 15+)
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. Patent Documents — metadata table
-- ============================================================
CREATE TABLE IF NOT EXISTS patent_documents (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    patent_number    TEXT        NOT NULL UNIQUE,
    title            TEXT        NOT NULL,
    assignee         TEXT,
    jurisdiction     TEXT        NOT NULL DEFAULT 'US',
    publication_date DATE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3. Patent Chunks — content + embeddings table
--
--    NOTE: fts_tokens is a plain TSVECTOR column populated by
--    a trigger (not a GENERATED column). GENERATED columns are
--    rejected by PostgREST when a row is inserted via the
--    Supabase client, causing silent insert failures.
-- ============================================================
CREATE TABLE IF NOT EXISTS patent_chunks (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    patent_id    UUID        NOT NULL REFERENCES patent_documents(id) ON DELETE CASCADE,
    section_type TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    fts_tokens   TSVECTOR,                    -- populated by trigger below
    embedding    VECTOR(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 4. Trigger to auto-populate fts_tokens on INSERT / UPDATE
-- ============================================================
CREATE OR REPLACE FUNCTION patent_chunks_fts_update()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.fts_tokens := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_patent_chunks_fts ON patent_chunks;
CREATE TRIGGER trg_patent_chunks_fts
    BEFORE INSERT OR UPDATE OF content ON patent_chunks
    FOR EACH ROW EXECUTE FUNCTION patent_chunks_fts_update();

-- ============================================================
-- 5. Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_patent_chunks_fts
    ON patent_chunks USING GIN (fts_tokens);

CREATE INDEX IF NOT EXISTS idx_patent_chunks_embedding_hnsw
    ON patent_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_patent_documents_jurisdiction
    ON patent_documents (jurisdiction);

-- ============================================================
-- 6. Hybrid RRF search function
-- ============================================================
CREATE OR REPLACE FUNCTION match_patent_hybrid(
    query_embedding   VECTOR(384),
    query_text        TEXT,
    filter_jurisdiction TEXT    DEFAULT NULL,
    match_count       INT      DEFAULT 3,
    rrf_k             INT      DEFAULT 60
)
RETURNS TABLE (
    chunk_id      UUID,
    patent_id     UUID,
    patent_number TEXT,
    title         TEXT,
    jurisdiction  TEXT,
    section_type  TEXT,
    content       TEXT,
    fts_rank      FLOAT8,
    vector_rank   FLOAT8,
    rrf_score     FLOAT8
)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    WITH
    fts_ranked AS (
        SELECT
            pc.id                                                          AS chunk_id,
            pc.patent_id,
            pd.patent_number,
            pd.title,
            pd.jurisdiction,
            pc.section_type,
            pc.content,
            ts_rank_cd(pc.fts_tokens, plainto_tsquery('english', query_text))::FLOAT8 AS rank_score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(pc.fts_tokens, plainto_tsquery('english', query_text)) DESC
            ) AS row_num
        FROM patent_chunks pc
        JOIN patent_documents pd ON pd.id = pc.patent_id
        WHERE (filter_jurisdiction IS NULL OR pd.jurisdiction = filter_jurisdiction)
          AND pc.fts_tokens IS NOT NULL
          AND pc.fts_tokens @@ plainto_tsquery('english', query_text)
        ORDER BY rank_score DESC
        LIMIT match_count * 5
    ),
    vec_ranked AS (
        SELECT
            pc.id                                    AS chunk_id,
            pc.patent_id,
            pd.patent_number,
            pd.title,
            pd.jurisdiction,
            pc.section_type,
            pc.content,
            (1 - (pc.embedding <=> query_embedding))::FLOAT8 AS rank_score,
            ROW_NUMBER() OVER (
                ORDER BY pc.embedding <=> query_embedding ASC
            ) AS row_num
        FROM patent_chunks pc
        JOIN patent_documents pd ON pd.id = pc.patent_id
        WHERE (filter_jurisdiction IS NULL OR pd.jurisdiction = filter_jurisdiction)
        ORDER BY pc.embedding <=> query_embedding ASC
        LIMIT match_count * 5
    ),
    fused AS (
        SELECT
            COALESCE(f.chunk_id,      v.chunk_id)      AS chunk_id,
            COALESCE(f.patent_id,     v.patent_id)     AS patent_id,
            COALESCE(f.patent_number, v.patent_number) AS patent_number,
            COALESCE(f.title,         v.title)         AS title,
            COALESCE(f.jurisdiction,  v.jurisdiction)  AS jurisdiction,
            COALESCE(f.section_type,  v.section_type)  AS section_type,
            COALESCE(f.content,       v.content)       AS content,
            COALESCE(f.rank_score, 0.0)                AS fts_rank,
            COALESCE(v.rank_score, 0.0)                AS vector_rank,
            (
                COALESCE(1.0 / (rrf_k + f.row_num), 0.0) +
                COALESCE(1.0 / (rrf_k + v.row_num), 0.0)
            )::FLOAT8                                  AS rrf_score
        FROM fts_ranked f
        FULL OUTER JOIN vec_ranked v ON f.chunk_id = v.chunk_id
    )
    SELECT
        fused.chunk_id, fused.patent_id, fused.patent_number, fused.title,
        fused.jurisdiction, fused.section_type, fused.content,
        fused.fts_rank, fused.vector_rank, fused.rrf_score
    FROM fused
    ORDER BY fused.rrf_score DESC
    LIMIT match_count;
END;
$$;