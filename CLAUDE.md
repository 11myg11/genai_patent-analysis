# CLAUDE.md — Patent Analysis Platform

## Project Context

AI-powered patent analysis tool for **Fuyao Europe** (automotive glazing, Heilbronn).
Domain: laminated automotive glass (windshields, HUD zones, PVB interlayers).
Stack: FastAPI + Supabase (pgvector) + OpenRouter (OpenAI-compatible API) + BAAI/bge-small-en-v1.5 embeddings.

Four phases per assignment:
1. Patent ingestion + structured storage ✅
2. Risk identification (design vs. patent claims) ✅
3. Design-around suggestions (with manufacturing audit) ✅
4. Innovation gap analysis across patent portfolios ✅

---

## Current Architecture

The codebase has been fully refactored from two monolithic files into the following modular structure:

```
app/
  config.py         ← All constants + pydantic-settings BaseSettings
  state.py          ← AppState dataclass (embed_model, supabase)
  models.py         ← All Pydantic request/response models
  routes/
    ui.py           ← Page routes: / /upload /patent-library /risk /design-suggestions /innovation /summaries
    api.py          ← API routes: /health /api/v1/...
  services/
    llm.py          ← OpenRouter client, retry, in-memory SHA-256 cache
    ingest.py       ← Shared ingestion pipeline (web endpoint + CLI both import this)
    retrieval.py    ← Hybrid RRF search, risk pipeline, designer + auditor agents (Phases 2–3)
    innovation.py   ← Phase 4: corpus analysis, clustering, gap detection, innovation vectors
  utils/
    pdf.py          ← PyMuPDF extraction + OCR fallback + chunking
    metadata.py     ← _regex_extract_metadata
    translation.py  ← _detect_language + _translate_to_english + _translate_chunks
main.py             ← App factory + lifespan only (~96 lines)
scripts/
  ingest_patents.py ← CLI arg parsing only; calls app/services/ingest.py
migrations/
  001_schema.sql
  002_migration_v2.sql
  003_patent_images.sql
templates/          ← Jinja2 templates (stays in root)
uploads/            ← Temp uploads, gitignored
```

---

## Phase 4 — Innovation Opportunities ✅

### Goal
Analyse the full patent corpus to surface technology clusters, whitespace gaps, and actionable
innovation directions. Assignment: "Review multiple patents to identify common patterns and gaps.
Suggest potential directions for new ideas or improvements."

### What was built
- `app/services/innovation.py` — full pipeline service (corpus fetch, trend aggregation, analyst agent, innovator agent)
- Models in `app/models.py` — `InnovationRequest`, `InnovationResponse`, `TechnologyCluster`, `PatentGap`, `InnovationVector`, `TrendPoint`, `InnovationSaveRequest`, `SavedInnovationSummary`, `SavedInnovationDetail`
- `POST /api/v1/innovation` — streams SSE pipeline progress (steps: corpus, trends, analyst, innovator)
- `POST /api/v1/innovation/save` — persist a completed analysis to `innovation_analyses` table
- `GET /api/v1/innovation/saved` — list all saved analyses (summaries only, no heavy JSONB)
- `GET /api/v1/innovation/saved/{id}` — retrieve one full saved analysis
- `DELETE /api/v1/innovation/saved/{id}` — delete a saved analysis
- `GET /innovation` page route → `templates/innovation.html` (Alpine.js UI)
  - Stats bar: Patents Analysed / Technology Clusters / Gap Areas / Innovation Vectors
  - Three-column results panel: Technology Clusters | Patent Gaps | Innovation Vectors
  - Publication trend bar chart (inline SVG)
  - Save / Load / Delete saved analyses panel
  - Last result and form restored from sessionStorage on page load

### Pipeline (4 steps)

**Step 1 — `fetch_corpus_overview`** — embed domain (if provided), hybrid-search to rank up to
`MAX_CORPUS_PATENTS = 30` patents by relevance (falls back to most-recent order). Fetches
representative chunks per patent: `MAX_CHUNKS_PER_PATENT = 2`, `MAX_CLAIM_CHARS = 400`.
Scope parameter controls section types: `"full"` / `"claims"` / `"description"`.

**Step 2 — `extract_trend_data`** — pure DB aggregation, no LLM. Groups `publication_date`
by year into `[{year, count}]` sorted ascending.

**Step 3 — `call_agent_analyst`** (LLM call 1) — clusters patents into 3–6 technology groups,
identifies 3–6 whitespace gap areas (each with `opportunity_level: HIGH|MEDIUM|LOW`).

**Step 4 — `call_agent_innovator`** (LLM call 2) — generates 3–5 innovation vectors grounded
in the identified gaps. Each vector carries `feasibility`, `novelty` (HIGH/MEDIUM/LOW),
`gap_rationale`, and `addresses_clusters`.

### Token budget
- 30 patents × 2 chunks × 400 chars ≈ 24 000 chars ≈ 6 000 tokens of context per call
- Leaves ample headroom in a 32k context window for prompt + output

---

## Key Design Decisions

- **Embedding model**: `BAAI/bge-small-en-v1.5` (384 dims). Do not change without re-embedding all chunks and updating the `vector(384)` schema.
- **Hybrid search**: Reciprocal Rank Fusion (RRF) of pgvector cosine similarity + full-text search (`tsvector`). The SQL function `match_patent_hybrid` lives in Supabase.
- **pgvector format**: Embeddings must be sent as a string `"[0.123,0.456,...]"` — PostgREST cannot auto-cast Python lists.
- **`fts_tokens` is a trigger column, not GENERATED** — `GENERATED` columns cause silent insert failures via PostgREST. See `migrations/002_migration_v2.sql`.
- **Two-agent pattern**: Phases 2–3 use generator + auditor agents. Phase 4 uses analyst + innovator agents — same pattern, different domain.
- **Glass domain constants** (PVB thickness, HUD zone, wedge angle) belong in `config.py`, not scattered inline.
- **LLM client**: OpenRouter via OpenAI-compatible API (`openai` SDK, base URL `https://openrouter.ai/api/v1`). In-memory cache keyed by SHA-256 of prompt. Per-minute 429s trigger one retry with backoff; all retry/fallback logic lives exclusively in `app/services/llm.py`.
- **No separate embedder module**: `_embed()` is 4 lines — keep it inside `services/ingest.py`.
- **Supabase calls in sync context**: All `state.supabase.*` calls must be wrapped in `asyncio.to_thread()` inside async route handlers.

---

## Known Issues

1. **Imports inside function body**: `import tempfile, os as _os` and `from app.utils.pdf import extract_page_text` are imported inside `extract_metadata_endpoint` in `api.py`. Minor style issue — not urgent.
2. **No `.env.example`**: Missing; should be created for onboarding.

---

## Environment Variables Required

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-3.5-haiku   # optional
APP_HOST=127.0.0.1                             # optional
APP_PORT=8000                                  # optional
DEBUG=false                                    # optional — gates debug endpoints
```

---

## Running the App

```bash
pip install -r requirements.txt
python main.py
# or:
uvicorn main:app --reload
```

CLI ingestion:
```bash
# Single file
python scripts/ingest_patents.py --pdf path/to/patent.pdf

# Directory batch
python scripts/ingest_patents.py --dir path/to/folder/

# Directory batch, skip already-ingested patents
python scripts/ingest_patents.py --dir path/to/folder/ --skip-existing
```

---

## Code Rules

- Include meaningful comments at the beginning of each file: what it does and why, global variables, functions, API docs, complex logic. Short and on point for quick onboarding.
- No backwards-compat shims. Delete dead code.
- No premature abstraction. Solve the actual problem.
- All LLM prompts must request `ONLY minified JSON` — do not change this, the parser depends on it.
- Never commit `.env`. Use `.env.example` for documentation.
- Supabase RLS must be **disabled** on `patent_chunks` for the service role to insert embeddings.
