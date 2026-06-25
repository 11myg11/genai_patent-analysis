-- ============================================================
-- 005_management_summaries.sql
-- Stores generated one-page Management Summary PDFs (Risk + Design
-- Improvements + Innovation condensed into a single report for
-- management review).
--
-- Mirrors the patent_images pattern: the rendered PDF is stored
-- directly as a BYTEA blob rather than in a separate object storage
-- bucket, since these are small (single-page) files.
--
-- Run this once in the Supabase SQL editor, then disable RLS on this
-- table for the service role (same as patent_images / patent_chunks)
-- so the Python client can insert.
-- ============================================================

CREATE TABLE IF NOT EXISTS management_summaries (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    product_id   TEXT        NOT NULL DEFAULT '',
    domain       TEXT        NOT NULL DEFAULT '',
    pdf_data     BYTEA       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_management_summaries_created_at
    ON management_summaries (created_at DESC);

-- Verify:
-- SELECT id, created_at, product_id, domain FROM management_summaries ORDER BY created_at DESC;
