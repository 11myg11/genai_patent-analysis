-- ============================================================
-- Migration 003: Patent figure images table
-- Run in Supabase SQL Editor AFTER schema.sql has been applied.
-- After running, disable RLS on patent_images for the service
-- role (same as patent_chunks) so the Python client can insert.
-- ============================================================

CREATE TABLE IF NOT EXISTS patent_images (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    patent_id   UUID        NOT NULL REFERENCES patent_documents(id) ON DELETE CASCADE,
    page_number INT         NOT NULL,
    width       INT,
    height      INT,
    image_data  BYTEA       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_patent_images_patent_id ON patent_images (patent_id);

-- Verify:
-- SELECT count(*) FROM patent_images;
