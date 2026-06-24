-- ============================================================
-- 004_innovation_analyses.sql
-- Stores saved Phase 4 innovation analyses.
--
-- Each row is a complete snapshot of one innovation pipeline run:
--   - Input parameters (domain, scope, jurisdiction, focus_prompt)
--   - patent_ids: UUIDs of all patent_documents included in the corpus.
--     Stored as uuid[] with a GIN index so we can quickly find which
--     analyses reference a given patent. No FK constraint on purpose —
--     deleting a patent should not invalidate a historical analysis.
--   - result: full InnovationResponse payload as JSONB (clusters, gaps,
--     innovations, trend_data). Stored denormalised to keep queries simple.
--
-- Run this once in the Supabase SQL editor.
-- ============================================================

CREATE TABLE IF NOT EXISTS innovation_analyses (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    domain       TEXT        NOT NULL DEFAULT '',
    scope        TEXT        NOT NULL DEFAULT 'full',
    jurisdiction TEXT        NOT NULL DEFAULT 'ALL',
    focus_prompt TEXT        NOT NULL DEFAULT '',
    patent_count INT         NOT NULL DEFAULT 0,
    patent_ids   UUID[]      NOT NULL DEFAULT '{}',
    result       JSONB       NOT NULL
);

-- GIN index enables fast "which analyses include patent X?" lookups
CREATE INDEX IF NOT EXISTS idx_innovation_analyses_patent_ids
    ON innovation_analyses USING GIN (patent_ids);

CREATE INDEX IF NOT EXISTS idx_innovation_analyses_created_at
    ON innovation_analyses (created_at DESC);
