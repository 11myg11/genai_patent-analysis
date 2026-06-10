-- ============================================================
-- Migration v2: Fix fts_tokens generated column → trigger
-- Run this in Supabase SQL Editor if you already ran schema.sql
-- ============================================================

-- Step 1: Drop the generated column (this drops existing fts data, OK)
ALTER TABLE patent_chunks DROP COLUMN IF EXISTS fts_tokens;

-- Step 2: Re-add as a plain nullable tsvector column
ALTER TABLE patent_chunks ADD COLUMN fts_tokens TSVECTOR;

-- Step 3: Back-fill existing rows
UPDATE patent_chunks SET fts_tokens = to_tsvector('english', COALESCE(content, ''));

-- Step 4: Create the trigger function
CREATE OR REPLACE FUNCTION patent_chunks_fts_update()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.fts_tokens := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$;

-- Step 5: Attach trigger
DROP TRIGGER IF EXISTS trg_patent_chunks_fts ON patent_chunks;
CREATE TRIGGER trg_patent_chunks_fts
    BEFORE INSERT OR UPDATE OF content ON patent_chunks
    FOR EACH ROW EXECUTE FUNCTION patent_chunks_fts_update();

-- Step 6: Recreate GIN index
DROP INDEX IF EXISTS idx_patent_chunks_fts;
CREATE INDEX idx_patent_chunks_fts ON patent_chunks USING GIN (fts_tokens);

-- Done. Test with:
-- SELECT count(*) FROM patent_chunks WHERE fts_tokens IS NOT NULL;
