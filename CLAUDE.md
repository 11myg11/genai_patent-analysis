# CLAUDE.md — Patent Analysis Platform

## Project Context

AI-powered patent analysis tool for **Fuyao Europe** (automotive glazing, Heilbronn).
Domain: laminated automotive glass (windshields, HUD zones, PVB interlayers).
Stack: FastAPI + Supabase (pgvector) + OpenRouter (OpenAI-compatible API) + BAAI/bge-small-en-v1.5 embeddings.

Four phases per assignment:
1. Patent ingestion + structured storage
2. Risk identification (design vs. patent claims)
3. Design-around suggestions (with manufacturing audit)
4. Innovation gap analysis across patent portfolios

---

## Target Architecture

Current state: two monolithic files (`main.py` 1312 lines, `ingest_patents.py` 608 lines) with heavy duplication.

Target structure:

```
app/
  config.py         ← All constants + pydantic-settings BaseSettings (replaces scattered os.environ.get)
  state.py          ← AppState dataclass
  models.py         ← All Pydantic request/response models
  routes/
    ui.py           ← Page routes: / /upload /summaries /compare /playground
    api.py          ← API routes: /api/v1/...
  services/
    llm.py          ← OpenRouter client, retry, in-memory cache ← LLM swap point
    ingest.py       ← Shared ingestion pipeline (web endpoint + CLI both import this)
    retrieval.py    ← Hybrid RRF search, _build_context_block, two agent functions
  utils/
    pdf.py          ← PyMuPDF extraction + PaddleOCR fallback + chunking
    metadata.py     ← _regex_extract_metadata (currently duplicated in both files)
    translation.py  ← _detect_language + _translate_to_english + _translate_chunks
main.py             ← app factory + lifespan only (~30 lines)
scripts/
  ingest_patents.py ← CLI arg parsing only; calls app/services/ingest.py
migrations/
  001_schema.sql
  002_migration_v2.sql
templates/          ← stays in root (Jinja2 default)
uploads/            ← stays in root, gitignored
```

### Refactor phases

**Phase 1 — Foundation (no behavior change)**
1. `app/config.py` — constants + env vars
2. `app/models.py` — 5 Pydantic models from main.py
3. `app/state.py` — AppState dataclass
4. `app/utils/metadata.py` — `_regex_extract_metadata` (main.py version is more complete)
5. `app/utils/translation.py` — `_detect_language` + `_translate_to_english` + `_translate_chunks` (ingest_patents.py versions are more complete)
6. `app/utils/pdf.py` — `_extract_page_text` + `_ocr_page` + `_split_into_chunks` + `determine_section_type` (ingest_patents.py versions)

**Phase 2 — Services**

7. `app/services/llm.py` — OpenRouter client + retry + in-memory SHA-256 cache
8. `app/services/ingest.py` — single ingestion pipeline merging `/api/v1/ingest` body and `ingest_patent()`
9. `app/services/retrieval.py` — `_fetch_hybrid_matches`, `_build_context_block`, `_call_agent_generator`, `_call_agent_auditor`

**Phase 3 — Wire up**

10. `app/routes/ui.py` — 5 page routes
11. `app/routes/api.py` — all API routes; remove `debug-insert` or gate behind `DEBUG` env var
12. Thin `main.py` — app factory + lifespan only
13. `scripts/ingest_patents.py` — CLI thin wrapper

---

## Key Design Decisions

- **Embedding model**: `BAAI/bge-small-en-v1.5` (384 dims). Do not change without re-embedding all chunks and updating the `vector(384)` schema.
- **Hybrid search**: Reciprocal Rank Fusion (RRF) of pgvector cosine similarity + full-text search (`tsvector`). The SQL function `match_patent_hybrid` lives in Supabase.
- **pgvector format**: Embeddings must be sent as a string `"[0.123,0.456,...]"` — PostgREST cannot auto-cast Python lists.
- **`fts_tokens` is a trigger column, not GENERATED** — `GENERATED` columns cause silent insert failures via PostgREST. See `migrations/002_migration_v2.sql`.
- **Two-agent pattern**: `_call_agent_generator` produces risk assessment + design-arounds; `_call_agent_auditor` validates against hard glass manufacturing constraints. Keep them separate.
- **Glass domain constants** (PVB thickness, HUD zone, wedge angle) belong in `config.py`, not scattered inline.
- **LLM client**: OpenRouter via OpenAI-compatible API (`openai` SDK, base URL `https://openrouter.ai/api/v1`). In-memory cache keyed by SHA-256 of prompt. Per-minute 429s trigger one retry with backoff; all retry/fallback logic lives exclusively in `app/services/llm.py`.
- **No separate embedder module**: `_embed()` is 4 lines — keep it inside `services/ingest.py`.

---

## What is duplicated (and where the better version lives)

| Concern | Keep | Delete |
|---|---|---|
| Language detection | `ingest_patents.py` → `utils/translation.py` | inline in `main.py` |
| Translation | `ingest_patents.py` → `utils/translation.py` | inline in `main.py` |
| Regex metadata | `main.py` `_regex_extract_metadata` → `utils/metadata.py` | `ingest_patents.py` `_regex_fallback_metadata` |
| PDF extraction + OCR | `ingest_patents.py` → `utils/pdf.py` | inline in `main.py` |
| Chunk splitting | `ingest_patents.py` → `utils/pdf.py` | `_text_to_chunks` in `main.py` |
| Ingestion pipeline | merge into `services/ingest.py` | both current versions |

---

## Known Problems to Fix

1. **Duplication**: language detection, metadata extraction, PDF chunking, translation, and DB inserts are copy-pasted between the two files. See table above.
2. **Imports inside functions**: `fitz`, `cv2`, `langdetect`, `deep_translator`, `re`, `hashlib` imported inside function bodies. Move to module top-level.
3. **Synchronous Supabase in async handlers**: all `_state.supabase.*` calls block the event loop. Wrap with `asyncio.to_thread()`.
4. **Debug endpoint in production**: `GET /api/v1/debug-insert` writes and deletes rows in prod DB. Remove or gate behind `DEBUG=true`.
5. **No config abstraction**: env vars read via scattered `os.environ.get(...)`. Use `pydantic-settings BaseSettings` in `config.py`.
6. **No `.env.example`**: create one.

---

## Environment Variables Required

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-3.5-haiku   # optional
APP_HOST=127.0.0.1                             # optional
APP_PORT=8000                                  # optional
```

---

## Running the App

```bash
pip install -r requirements.txt
python main.py
# after refactor:
uvicorn app.main:app --reload
```

CLI ingestion:
```bash
python scripts/ingest_patents.py --pdf path/to/patent.pdf
# currently:
python ingest_patents.py --pdf path/to/patent.pdf
```

---

## Code Rules

- Include meaningful comments in the beginning of each file. Explain what the file does and why. Short and on point, 
  focused for quick onboarding for beginner developers. Include global variables, functions, API documentation, complex 
  logic.
- No backwards-compat shims. Delete dead code.
- No premature abstraction. Solve the actual problem.
- All LLM prompts must request `ONLY minified JSON` — do not change this, the parser depends on it.
- Never commit `.env`. Use `.env.example` for documentation.
- Supabase RLS must be **disabled** on `patent_chunks` for the service role to insert embeddings.
