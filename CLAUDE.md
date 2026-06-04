# CLAUDE.md — Patent Analysis Platform

## Project Context

AI-powered patent analysis tool for **Fuyao Europe** (automotive glazing, Heilbronn).
Domain: laminated automotive glass (windshields, HUD zones, PVB interlayers).
Stack: FastAPI + Supabase (pgvector) + Google Gemini + BAAI/bge-small-en-v1.5 embeddings.

Four phases per assignment:
1. Patent ingestion + structured storage
2. Risk identification (design vs. patent claims)
3. Design-around suggestions (with manufacturing audit)
4. Innovation gap analysis across patent portfolios

---

## Target Architecture (refactor goal)

Current state: two monolithic files (`main.py` 1313 lines, `ingest_patents.py` 609 lines) with heavy duplication.

Target structure — **move to this before adding new features**:

```
app/
  main.py           ← FastAPI app factory + lifespan only
  config.py         ← All constants + pydantic-settings (replaces scattered os.environ.get)
  state.py          ← AppState dataclass
  models.py         ← All Pydantic request/response models
  routes/
    ui.py           ← Page routes: /upload /summaries /compare /playground
    api.py          ← API routes: /api/v1/...
  services/
    llm.py          ← Gemini client, retry + fallback chain, in-memory cache
    ingest.py       ← Shared ingestion pipeline (API endpoint + CLI both import this)
    embedder.py     ← SentenceTransformer wrapper
    retrieval.py    ← Hybrid RRF search, _build_context_block
  utils/
    pdf.py          ← PyMuPDF extraction + PaddleOCR fallback
    metadata.py     ← _regex_extract_metadata (currently duplicated in both files)
    translation.py  ← _detect_language + _translate_to_english (also duplicated)
scripts/
  ingest_patents.py ← CLI thin wrapper; imports from app/services/ingest.py
migrations/
  001_schema.sql
  002_migration_v2.sql
templates/          ← stays in root (Jinja2 default)
uploads/            ← stays in root, gitignored
tests/              ← empty, ready
```

---

## Key Design Decisions

- **Embedding model**: `BAAI/bge-small-en-v1.5` (384 dims). Do not change without re-embedding all chunks and updating the `vector(384)` schema.
- **Hybrid search**: Reciprocal Rank Fusion (RRF) of pgvector cosine similarity + full-text search (`tsvector`). The SQL function `match_patent_hybrid` lives in Supabase.
- **pgvector format**: Embeddings must be sent as a string `"[0.123,0.456,...]"` — PostgREST cannot auto-cast Python lists.
- **`fts_tokens` is a trigger column, not GENERATED** — `GENERATED` columns cause silent insert failures via PostgREST. See `migrations/002_migration_v2.sql`.
- **Two-agent pattern**: `_call_agent_generator` produces risk assessment + design-arounds; `_call_agent_auditor` validates against hard glass manufacturing constraints. Keep them separate.
- **Glass domain constants** (PVB thickness, HUD zone, wedge angle) belong in `config.py`, not scattered inline.
- **Gemini fallback chain**: `gemini-2.0-flash-lite` → `gemini-2.0-flash` → `gemini-1.5-flash-001/002`. In-memory cache keyed by SHA-256 of prompt. Per-minute 429s trigger one retry with suggested delay; daily exhaustion skips to next model immediately.

---

## Known Problems to Fix

1. **Duplication**: Language detection, regex metadata extraction, PDF chunking, translation, and DB inserts are copy-pasted between `main.py` and `ingest_patents.py`. Extract to `app/utils/` and `app/services/ingest.py`.
2. **Imports inside functions**: `fitz`, `cv2`, `langdetect`, `deep_translator`, `re`, `hashlib` imported inside function bodies. Move to module top-level.
3. **Synchronous Supabase in async handlers**: All `_state.supabase.*` calls block the event loop. Either use `asyncio.to_thread()` or the async supabase client.
4. **Debug endpoint in production**: `GET /api/v1/debug-insert` writes and deletes rows in prod DB. Remove or gate behind `DEBUG=true` env var.
5. **No config abstraction**: Env vars read via scattered `os.environ.get(...)` calls. Use `pydantic-settings` `BaseSettings` in `config.py`.
6. **In-memory Gemini cache** (`_GEMINI_CACHE`) evaporates on restart and doesn't work across workers. For now acceptable; if scaling, replace with Redis.
7. **Requirements pinned to old versions**: `fastapi==0.110.0`, `uvicorn==0.28.0`, `supabase==2.4.0`. Run `pip-compile` to update.
8. **No `.env.example`**: New team members don't know which keys to set. Create one.
9. **No tests**: Zero test coverage. At minimum add integration tests for the ingestion pipeline and unit tests for `_regex_extract_metadata`.

---

## Environment Variables Required

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.0-flash-lite   # optional, this is default
APP_HOST=127.0.0.1                   # optional
APP_PORT=8000                        # optional
```

---

## Running the App

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # after refactor
# or currently:
python main.py
```

CLI ingestion:
```bash
python ingest_patents.py --pdf path/to/patent.pdf
```

---

## Code Rules

- No comments that explain what the code does — only why (non-obvious constraints, workarounds).
- No backwards-compat shims. Delete dead code.
- No premature abstraction. Solve the actual problem.
- All LLM prompts must request `ONLY minified JSON` — do not change this, the parser depends on it.
- Never commit `.env`. Use `.env.example` for documentation.
- Supabase RLS must be **disabled** on `patent_chunks` for the service role to insert embeddings.
