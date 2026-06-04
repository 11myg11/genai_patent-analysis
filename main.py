"""
main.py
────────
Patent Analysis Platform — FastAPI Backend + UI Server

Pages:
  GET  /            → redirect to /upload
  GET  /upload      → Patent ingestion UI
  GET  /summaries   → Patent summaries browser
  GET  /compare     → Dual-patent comparison
  GET  /playground  → Design-around AI playground

API endpoints:
  POST /api/v1/evaluate-design
  GET  /api/v1/patents              → list all patents
  GET  /api/v1/patents/{id}         → single patent detail + chunks
  POST /api/v1/compare              → compare two patents
  GET  /api/v1/patents/{id}/summary → LLM-generated summary
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

# ─── Load .env ───────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL       = "BAAI/bge-small-en-v1.5"
LLM_MAX_TOKENS        = 1024
TOP_K_CHUNKS          = 3
PVB_MIN_MM            = 0.38
PVB_MAX_MM            = 0.76
GLASS_TOTAL_MIN       = 3.1
GLASS_TOTAL_MAX       = 6.0
HUD_ZONE_CONDUCTIVE_BAN = True

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class DesignEvaluationRequest(BaseModel):
    product_id:              str = Field(...)
    component_scope:         str = Field(...)
    proposed_specifications: str = Field(...)
    jurisdiction:            str = Field(default="US")


class ChunkReference(BaseModel):
    patent_number: str
    title:         str
    section_type:  str
    content:       str
    rrf_score:     float


class DesignAroundProposal(BaseModel):
    id:          str
    description: str
    rationale:   str
    audited:     bool
    audit_notes: Optional[str] = None


class DesignEvaluationResponse(BaseModel):
    product_id:        str
    risk_status:       str
    infringement_map:  List[Dict[str, str]]
    design_arounds:    List[DesignAroundProposal]
    matched_chunks:    List[ChunkReference]
    token_budget_used: int


class CompareRequest(BaseModel):
    patent_id_a: str
    patent_id_b: str
    jurisdiction: str = "US"


# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    embed_model:      SentenceTransformer
    supabase:         Client
    gemini_client:    genai.Client
    gemini_model_name: str

_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Startup: loading embedding model %s…", EMBEDDING_MODEL)
    try:
        _state.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model loaded.")
    except Exception as exc:
        log.critical("Failed to load SentenceTransformer: %s", exc)
        raise

    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    supabase_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not supabase_url or not supabase_key:
        log.critical("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        raise RuntimeError("Missing Supabase credentials.")
    try:
        _state.supabase = create_client(supabase_url, supabase_key)
        log.info("Supabase client initialised.")
    except Exception as exc:
        log.critical("Supabase connection failed: %s", exc)
        raise

    google_api_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
    gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
    if not google_api_key:
        log.critical("GOOGLE_API_KEY must be set in .env")
        raise RuntimeError("Missing Google API key.")

    # Short aliases like "gemini-1.5-flash" are NOT valid for the v1beta endpoint.
    # The google-genai SDK requires versioned names.
    _VALID_MODELS = {
        "gemini-2.0-flash-lite", "gemini-2.0-flash",
        "gemini-1.5-flash-001",  "gemini-1.5-flash-002",
        "gemini-1.5-pro-001",    "gemini-1.5-pro-002",
    }
    if gemini_model_name not in _VALID_MODELS:
        log.warning(
            "GEMINI_MODEL='%s' may not be supported. "
            "Recommended: gemini-2.0-flash-lite. Valid: %s",
            gemini_model_name, ", ".join(sorted(_VALID_MODELS))
        )

    try:
        _state.gemini_client     = genai.Client(api_key=google_api_key)
        _state.gemini_model_name = gemini_model_name
        log.info("Google Gemini client ready (model=%s).", gemini_model_name)
    except Exception as exc:
        log.critical("Gemini client initialisation failed: %s", exc)
        raise

    yield
    log.info("Shutdown complete.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Patent Analysis Platform", version="1.0.0", lifespan=lifespan)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Utility Functions ────────────────────────────────────────────────────────

def _embed(text: str) -> List[float]:
    vec = _state.embed_model.encode(
        [text], normalize_embeddings=True, show_progress_bar=False
    )
    return vec[0].tolist()


# Fallback model chain — tried in order when the primary model hits quota.
# Each model has its own independent daily/minute quota on the free tier.
_FALLBACK_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
]

# In-memory cache for Gemini responses — keyed by prompt hash.
# Avoids re-spending quota on identical requests within the same server session.
import hashlib as _hashlib
_GEMINI_CACHE: Dict[str, str] = {}

def _gemini_call_with_retry(prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """
    Call Gemini with:
      - In-memory response caching (identical prompts reuse the cached result)
      - Automatic retry on per-minute 429s (waits the suggested delay, max 65s)
      - Model fallback chain when daily quota is exhausted
      - Immediate skip on daily quota (limit:0) — no point waiting
    """
    import time
    import re as _re_r

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = _hashlib.sha256(prompt.encode()).hexdigest()
    if cache_key in _GEMINI_CACHE:
        log.info("Gemini cache hit (saved a quota request).")
        return _GEMINI_CACHE[cache_key]

    def _is_daily_exhausted(err_str: str) -> bool:
        """Daily quota shows limit:0 or PerDay in the quota metric name."""
        return "limit: 0" in err_str or "PerDay" in err_str or "PerDayPer" in err_str

    def _extract_wait(err_str: str) -> float:
        m = _re_r.search(r"retryDelay[^:]*:\s*[^\d]*(\d+)s", err_str, _re_r.IGNORECASE)
        return min(float(m.group(1)) if m else 3.0, 65.0)

    def _try_model(model_name: str) -> str:
        response = _state.gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()

    models_to_try = [_state.gemini_model_name] + [
        m for m in _FALLBACK_MODELS if m != _state.gemini_model_name
    ]

    last_error = None
    for model in models_to_try:
        for attempt in range(2):
            try:
                result = _try_model(model)
                _GEMINI_CACHE[cache_key] = result   # cache successful response
                if model != _state.gemini_model_name:
                    log.info("Gemini fallback succeeded on model=%s", model)
                return result

            except Exception as exc:
                err_str = str(exc)
                is_quota     = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_not_found = "404" in err_str or "NOT_FOUND" in err_str
                is_daily     = _is_daily_exhausted(err_str)

                if is_not_found:
                    log.warning("Model %s not available on this key — skipping.", model)
                    break

                if is_quota:
                    last_error = exc
                    if is_daily:
                        # Daily cap — no point waiting, move to next model immediately
                        log.warning("Daily quota exhausted on %s — trying next model.", model)
                        break
                    if attempt == 0:
                        # Per-minute cap — wait then retry once
                        wait = _extract_wait(err_str)
                        log.warning("Per-minute quota on %s — waiting %.0fs…", model, wait)
                        time.sleep(wait)
                        continue
                    else:
                        log.warning("Still rate-limited on %s — trying next model.", model)
                        break

                # Non-quota error
                log.error("Gemini error on %s: %s", model, exc)
                raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    raise HTTPException(
        status_code=429,
        detail=(
            "QUOTA_EXHAUSTED: All Gemini free-tier models have hit their daily limit. "
            "Options: (1) Wait until midnight Pacific Time for quota reset, "
            "(2) Create a new Google Cloud project at console.cloud.google.com and "
            "generate a new API key at aistudio.google.com/app/apikey — "
            "each project has its own independent free quota, "
            "(3) Enable billing on your Google Cloud project for higher limits."
        ),
    )


def _gemini_text(prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """Call Gemini with retry/fallback and return raw text."""
    return _gemini_call_with_retry(prompt, max_tokens)


def _gemini_json(prompt: str) -> Dict[str, Any]:
    """Call Gemini with retry/fallback and parse JSON response."""
    raw = ""
    try:
        raw = _gemini_text(prompt)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Gemini JSON parse error: %s | raw=%s", exc, raw[:300])
        raise HTTPException(status_code=500, detail=f"LLM output parse failure: {exc}")
    except HTTPException:
        raise  # propagate rate-limit and other HTTP errors as-is
    except Exception as exc:
        log.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM inference error: {exc}")


def _fetch_hybrid_matches(embedding, query_text, jurisdiction) -> List[Dict]:
    try:
        resp = _state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     embedding,
            "query_text":          query_text,
            "filter_jurisdiction": jurisdiction,
            "match_count":         TOP_K_CHUNKS,
        }).execute()
        return resp.data or []
    except Exception as exc:
        log.error("Supabase RPC failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Database retrieval error: {exc}")


def _build_context_block(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[REF{i}|{c['patent_number']}|{c['section_type']}|RRF:{c['rrf_score']:.4f}]\n"
            f"{c['content']}"
        )
    return "\n---\n".join(parts)


def _call_agent_generator(context_block, proposed_specs, component_scope) -> Dict:
    prompt = f"""You are a senior IP Engineer specialised in structural patent claim analysis.
Respond ONLY with minified JSON. No markdown. No preamble. No explanation outside JSON.
Use acronyms: PVB=polyvinyl_butyral_interlayer, Tt=total_thickness_mm, HUD=heads_up_display_zone.

Output schema (strict):
{{"risk_status":"HIGH|MEDIUM|LOW|CLEAR","infringement_map":[{{"claim_ref":"...","element":"...","overlap":"..."}}],"design_arounds":[{{"id":"DA1","description":"...","rationale":"..."}},{{"id":"DA2","description":"...","rationale":"..."}}]}}

COMPONENT_SCOPE: {component_scope}
PROPOSED_SPECS: {proposed_specs}
PATENT_CONTEXT:
{context_block}

Analyse proposed specs against each patent claim. Classify risk_status. Map overlapping elements. Propose exactly 2 design-arounds."""
    return _gemini_json(prompt)


def _call_agent_auditor(generator_output, component_scope) -> Dict:
    da_block = json.dumps(generator_output.get("design_arounds", []))
    prompt = f"""You are a Fuyao Glass Manufacturing Auditor with expertise in automotive glazing standards.
Respond ONLY with minified JSON. No markdown. No explanation outside JSON.

HARD CONSTRAINTS (violation = rewrite required):
1. Tt between {GLASS_TOTAL_MIN} and {GLASS_TOTAL_MAX} mm.
2. PVB between {PVB_MIN_MM} and {PVB_MAX_MM} mm.
3. HUD zone: zero conductive materials.
4. Wedge angle ≤ 0.1 mrad.

Output schema:
{{"audited_design_arounds":[{{"id":"...","description":"...","rationale":"...","passed_audit":true,"audit_notes":"..."}}]}}

COMPONENT_SCOPE: {component_scope}
DESIGN_AROUNDS_TO_AUDIT: {da_block}

For each design-around: check constraints, rewrite if violated, confirm if passed."""

    audit_result = _gemini_json(prompt)
    audited_map  = {a["id"]: a for a in audit_result.get("audited_design_arounds", [])}
    merged_das   = []
    for da in generator_output.get("design_arounds", []):
        da_id = da.get("id", "")
        if da_id in audited_map:
            a = audited_map[da_id]
            merged_das.append(DesignAroundProposal(
                id=da_id, description=a.get("description", da.get("description", "")),
                rationale=a.get("rationale", da.get("rationale", "")),
                audited=True, audit_notes=a.get("audit_notes"),
            ))
        else:
            merged_das.append(DesignAroundProposal(
                id=da_id, description=da.get("description", ""),
                rationale=da.get("rationale", ""),
                audited=False, audit_notes="Audit result not returned.",
            ))
    generator_output["design_arounds_merged"] = merged_das
    return generator_output


# ═══════════════════════════════════════════════════════════════════════════════
# UI ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/upload")


@app.get("/upload", include_in_schema=False)
async def page_upload(request: Request):
    try:
        resp = _state.supabase.table("patent_documents") \
            .select("id,patent_number,title,assignee,jurisdiction,publication_date,created_at") \
            .order("created_at", desc=True).limit(20).execute()
        recent = resp.data or []
    except Exception:
        recent = []
    return templates.TemplateResponse(
        request=request, name="upload.html", context={"recent": recent}
    )


@app.get("/summaries", include_in_schema=False)
async def page_summaries(request: Request):
    try:
        resp = _state.supabase.table("patent_documents") \
            .select("id,patent_number,title,assignee,jurisdiction,publication_date") \
            .order("created_at", desc=True).execute()
        patents = resp.data or []
    except Exception:
        patents = []
    return templates.TemplateResponse(
        request=request, name="summaries.html", context={"patents": patents}
    )


@app.get("/compare", include_in_schema=False)
async def page_compare(request: Request):
    try:
        resp = _state.supabase.table("patent_documents") \
            .select("id,patent_number,title,assignee,jurisdiction").execute()
        patents = resp.data or []
    except Exception:
        patents = []
    return templates.TemplateResponse(
        request=request, name="compare.html", context={"patents": patents}
    )


@app.get("/playground", include_in_schema=False)
async def page_playground(request: Request):
    return templates.TemplateResponse(
        request=request, name="playground.html", context={}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", include_in_schema=False)
async def health():
    return JSONResponse({"status": "ok"})

@app.get("/api/v1/debug-insert", include_in_schema=False)
async def debug_insert():
    """
    Inserts one synthetic patent + one chunk with a zero-vector, then deletes both.
    Open http://127.0.0.1:8000/api/v1/debug-insert in browser to diagnose chunk failures.
    """
    patent_id = None
    try:
        doc_resp = _state.supabase.table("patent_documents").insert({
            "patent_number": "__DEBUG_TEST__",
            "title":         "Debug test patent",
            "jurisdiction":  "US",
        }).execute()
        if not doc_resp.data:
            return {"step": "patent_documents insert", "error": "returned empty data"}
        patent_id = doc_resp.data[0]["id"]

        zero_vec = "[" + ",".join(["0.00000000"] * 384) + "]"
        chunk_resp = _state.supabase.table("patent_chunks").insert({
            "patent_id":    patent_id,
            "section_type": "description",
            "content":      "Debug test content.",
            "embedding":    zero_vec,
        }).execute()
        chunk_inserted = len(chunk_resp.data) if chunk_resp.data else 0
        return {
            "result":         "OK" if chunk_inserted > 0 else "FAILED",
            "chunk_inserted": chunk_inserted,
            "raw_resp":       chunk_resp.data,
            "diagnosis": (
                "Vector insert works correctly." if chunk_inserted > 0 else
                "0 rows returned. Fix checklist — run each in Supabase SQL Editor: "
                "1) SELECT column_name, generation_expression FROM information_schema.columns "
                "WHERE table_name=\'patent_chunks\' AND column_name=\'fts_tokens\' — "
                "if generation_expression is not null, run migration_v2.sql. "
                "2) SELECT * FROM pg_policies WHERE tablename=\'patent_chunks\' — "
                "if any RLS policies exist, disable RLS: ALTER TABLE patent_chunks DISABLE ROW LEVEL SECURITY."
            ),
        }
    except Exception as exc:
        return {"step": "exception", "error": str(exc), "repr": repr(exc)}
    finally:
        if patent_id:
            try:
                _state.supabase.table("patent_chunks").delete().eq("patent_id", patent_id).execute()
                _state.supabase.table("patent_documents").delete().eq("id", patent_id).execute()
            except Exception:
                pass


# ── Shared regex metadata extractor (no LLM needed) ─────────────────────────
def _regex_extract_metadata(text: str, filename_hint: str = "") -> Dict[str, str]:
    """
    Extract patent metadata purely via regex — no network calls, no quota.
    Handles USPTO, EPO, CNIPA, JPO, KIPO header formats.
    """
    import re as _re2
    from datetime import datetime as _dt

    meta: Dict[str, str] = {
        "patent_number": "", "title": "", "assignee": "",
        "jurisdiction": "", "publication_date": "",
    }

    # ── Patent number ─────────────────────────────────────────────────────────
    pn_patterns = [
        r'\b(US\s*\d{6,8}\s*[A-Z]\d?)\b',
        r'\b(EP\s*\d{6,7}\s*[A-Z]\d?)\b',
        r'\b(CN\s*\d{8,12}\s*[A-Z]?)\b',
        r'\b(JP\s*\d{7,13}\s*[A-Z]?)\b',
        r'\b(KR\s*\d{7,12}\s*[A-Z]\d?)\b',
        r'\b(WO\s*\d{4}/?\d{4,7}\s*[A-Z]?\d?)\b',
        r'\b(DE\s*\d{9,12}\s*[A-Z]\d?)\b',
        r'\b(FR\s*\d{7,10}\s*[A-Z]?\d?)\b',
        r'\b(GB\s*\d{7}\s*[A-Z]?)\b',
    ]
    for pat in pn_patterns:
        m = _re2.search(pat, text, _re2.IGNORECASE)
        if m:
            meta["patent_number"] = _re2.sub(r'\s+', '', m.group(1)).upper()
            meta["jurisdiction"]   = meta["patent_number"][:2].upper()
            break

    # Fallback: parse patent number from filename
    if not meta["patent_number"] and filename_hint:
        fn = filename_hint.replace(".pdf", "").replace(".PDF", "").strip()
        pn_m = _re2.match(r'^([A-Z]{2}[\d]+[A-Z]?\d?)', fn, _re2.IGNORECASE)
        if pn_m:
            meta["patent_number"] = pn_m.group(1).upper()
            meta["jurisdiction"]   = meta["patent_number"][:2].upper()

    # ── Publication date ──────────────────────────────────────────────────────
    # ISO: 2024-08-15
    m = _re2.search(r'(?:Date of Patent|Publication Date|Pub\.?\s*Date)[:\s]+(\d{4}-\d{2}-\d{2})', text, _re2.IGNORECASE)
    if m:
        meta["publication_date"] = m.group(1)
    if not meta["publication_date"]:
        # US long-form: Aug. 15, 2024
        m = _re2.search(r'(?:Date of Patent|Publication Date)[:\s]+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})', text, _re2.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            for fmt in ('%B %d, %Y', '%b. %d, %Y', '%B %d %Y', '%b %d, %Y'):
                try:
                    meta["publication_date"] = _dt.strptime(raw, fmt).strftime('%Y-%m-%d')
                    break
                except ValueError:
                    pass
    if not meta["publication_date"]:
        # Compact: 20240815
        m = _re2.search(r'\b(2\d{3}[01]\d[0-3]\d)\b', text)
        if m:
            try:
                meta["publication_date"] = _dt.strptime(m.group(1), '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError:
                pass
    if not meta["publication_date"]:
        # European: 15.08.2024
        m = _re2.search(r'\b(\d{2}\.\d{2}\.20\d{2})\b', text)
        if m:
            try:
                meta["publication_date"] = _dt.strptime(m.group(1), '%d.%m.%Y').strftime('%Y-%m-%d')
            except ValueError:
                pass

    # ── Assignee ─────────────────────────────────────────────────────────────
    # INID code (73) on USPTO/EPO documents
    m = _re2.search(r'\(73\)\s*(?:Assignee|Applicant)[:\s]+([^\n\r(]{3,80})', text, _re2.IGNORECASE)
    if m:
        meta["assignee"] = m.group(1).strip().rstrip(',.')
    if not meta["assignee"]:
        m = _re2.search(r'(?:Assignee|Applicant|Anmelder|Titulaire|Patentinhaber)[:\s]+([A-Z][^\n\r(]{3,80}?)(?=\s*[\n\r(]|\s*,\s*[A-Z]{2}\b)', text, _re2.IGNORECASE)
        if m:
            meta["assignee"] = m.group(1).strip().rstrip(',.')
    if not meta["assignee"]:
        m = _re2.search(r'(?:ASSIGNEE|APPLICANT)[:\s]+([A-Z][^\n\r]{3,80})', text)
        if m:
            meta["assignee"] = m.group(1).strip().rstrip(',.')

    # ── Title ─────────────────────────────────────────────────────────────────
    # INID code (54) — present on virtually all modern patents
    m = _re2.search(r'\(54\)\s*([A-Z][^\n\r]{10,150})', text)
    if m:
        val = m.group(1).strip().rstrip('.')
        if not any(w in val.upper() for w in ['UNITED STATES', 'PATENT OFFICE', 'APPLICATION']):
            meta["title"] = val.title() if val.isupper() else val
    if not meta["title"]:
        m = _re2.search(r'(?:TITLE OF INVENTION|Title of Invention|Invention Title)[:\s]+([^\n\r]{10,150})', text, _re2.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.')
            meta["title"] = val.title() if val.isupper() else val
    if not meta["title"]:
        # All-caps line that looks like a title (15–120 chars, not a header)
        for line in text.splitlines():
            line = line.strip()
            if (15 <= len(line) <= 120 and line.isupper()
                    and not any(w in line for w in
                        ['UNITED STATES', 'PATENT', 'OFFICE', 'APPLICATION',
                         'PUBLICATION', 'INTERNATIONAL', 'WORLD'])):
                meta["title"] = line.title()
                break

    # ── Jurisdiction fallback ─────────────────────────────────────────────────
    if not meta["jurisdiction"]:
        _JX = {"CN":"CN","JP":"JP","KR":"KR","DE":"DE","FR":"FR",
               "GB":"GB","EP":"EP","WO":"WO","US":"US","AT":"AT","CH":"CH"}
        prefix = (filename_hint or "")[:2].upper()
        meta["jurisdiction"] = _JX.get(prefix, "US")

    log.info("Regex metadata: number=%s title=%s assignee=%s jx=%s date=%s",
             meta["patent_number"],
             (meta["title"][:40] + "...") if len(meta.get("title","")) > 40 else meta.get("title",""),
             (meta["assignee"][:40] + "...") if len(meta.get("assignee","")) > 40 else meta.get("assignee",""),
             meta["jurisdiction"], meta["publication_date"])
    return meta

# ── Extract metadata from uploaded PDF (called before ingest for auto-fill) ──
@app.post("/api/v1/extract-metadata")
async def extract_metadata(file: UploadFile = File(...), filename_hint: str = Form("")):
    """
    Extract patent metadata from the first 3 pages of a PDF.
    Strategy: regex parsing first (instant, no quota), then Gemini enrichment
    only if quota is available and regex left fields blank.
    """
    import fitz as _fitz

    contents = await file.read()
    tmp_path = UPLOAD_DIR / f"meta_tmp_{file.filename}"
    try:
        tmp_path.write_bytes(contents)
        doc = _fitz.open(str(tmp_path))
        header_text = ""
        for i in range(min(4, len(doc))):
            header_text += doc[i].get_text("text").strip() + "\n\n"
        doc.close()
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"PDF read failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    _name = (filename_hint or file.filename or "").strip()

    # ── Language detection ────────────────────────────────────────────────────
    _PFXMAP = {"CN":"zh","JP":"ja","KR":"ko","DE":"de","AT":"de",
               "CH":"de","FR":"fr","BE":"fr","ES":"es","NL":"nl",
               "RU":"ru","PT":"pt","IT":"it"}
    _prefix = _name[:2].upper()
    src_lang = _PFXMAP.get(_prefix, "")
    if not src_lang:
        try:
            from langdetect import detect
            src_lang = detect(header_text[:2000]) if header_text.strip() else "en"
        except Exception:
            src_lang = "en"

    # ── Translate header to English if needed ─────────────────────────────────
    _GT_MAP = {"zh": "zh-CN", "zh-cn": "zh-CN", "zh-tw": "zh-TW"}
    header_en = header_text
    if not src_lang.startswith("en") and header_text.strip():
        gt_src = _GT_MAP.get(src_lang.lower(), src_lang.split("-")[0])
        try:
            from deep_translator import GoogleTranslator
            header_en = GoogleTranslator(source=gt_src, target="en").translate(
                header_text[:4000]) or header_text
            log.info("Header translated [%s→en]: %d chars", gt_src, len(header_en))
        except Exception as exc:
            log.warning("Header translation failed (%s): %s", gt_src, exc)

    # ── Step 1: Regex extraction (primary — always runs, no quota) ────────────
    meta = _regex_extract_metadata(header_en, _name)

    # ── Step 2: Gemini enrichment (secondary — only fills still-blank fields) ──
    # Only call Gemini if regex left important fields empty, to conserve quota.
    needs_gemini = not meta["patent_number"] or not meta["title"] or not meta["assignee"]
    if needs_gemini:
        google_api_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
        gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
        if google_api_key:
            try:
                missing = [k for k in ("patent_number","title","assignee","publication_date")
                           if not meta.get(k)]
                prompt = (
                    f"Extract these missing patent metadata fields: {missing}.\n"
                    "Respond ONLY with minified JSON matching this schema exactly:\n"
                    '{"patent_number":"","title":"","assignee":"","jurisdiction":"","publication_date":""}\n'
                    "Return empty string for any field you cannot find.\n\n"
                    f"TEXT:\n{header_en[:2500]}"
                )
                response = _state.gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
                )
                raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                gemini_meta = json.loads(raw)
                # Only fill fields that regex left empty
                for k, v in gemini_meta.items():
                    if k in meta and not meta[k] and isinstance(v, str) and v.strip():
                        meta[k] = v.strip()
                        log.info("Gemini filled missing field [%s]: %s", k, v.strip())
            except Exception as exc:
                log.warning("Gemini enrichment skipped: %s", exc)

    return {**meta, "detected_language": src_lang, "translated": not src_lang.startswith("en")}


# ── Browser-initiated PDF ingestion ──────────────────────────────────────────
@app.post("/api/v1/ingest")
async def ingest_from_browser(
    file:          UploadFile = File(...),
    patent_number: str        = Form(""),
    title:         str        = Form(""),
    assignee:      str        = Form(""),
    jurisdiction:  str        = Form(""),
    pub_date:      str        = Form(""),
):
    """
    Full browser ingestion pipeline:
    1. Save PDF, extract text, detect language
    2. Translate non-English content to English
    3. Auto-extract metadata via Gemini; user-supplied form values take priority
    4. Embed chunks and insert into Supabase (pgvector-safe string format)
    """
    import re as _re
    import fitz as _fitz

    # ── Save temp file ────────────────────────────────────────────────────────
    safe_stem = (file.filename or "patent").replace("/", "_").replace("\\", "_")
    tmp_path  = UPLOAD_DIR / f"ingest_{safe_stem}"
    doc       = None
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File save failed: {exc}")

    # ── Open PDF once; keep open until we are done with all pages ────────────
    try:
        doc = _fitz.open(str(tmp_path))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"PDF open failed: {exc}")

    # ── Extract header text (first 3 pages) for language + metadata ───────────
    header_text = ""
    for i in range(min(3, len(doc))):
        try:
            header_text += doc[i].get_text("text").strip() + "\n\n"
        except Exception:
            pass

    # ── Language detection ────────────────────────────────────────────────────
    # Priority 1: patent number prefix (most reliable — CN→zh, JP→ja, KR→ko etc.)
    # Priority 2: langdetect on text (fallback for US/EP/WO/GB which publish in English)
    _PREFIX_LANG = {
        "CN": "zh", "JP": "ja", "KR": "ko",
        "DE": "de", "AT": "de", "CH": "de",
        "FR": "fr", "BE": "fr", "ES": "es",
        "NL": "nl", "RU": "ru", "PT": "pt", "IT": "it",
    }
    _pnum_hint = (patent_number.strip() or safe_stem or "")[:2].upper()
    src_lang = _PREFIX_LANG.get(_pnum_hint, "")
    if src_lang:
        log.info("Language from patent number prefix [%s]: %s", _pnum_hint, src_lang)
    else:
        try:
            from langdetect import detect
            src_lang = detect(header_text[:2000]) if header_text.strip() else "en"
            log.info("Language from langdetect: %s", src_lang)
        except Exception as exc:
            log.warning("Language detection failed: %s — assuming English.", exc)
            src_lang = "en"

    is_english = src_lang.startswith("en")
    log.info("Final language: %s | needs_translation: %s", src_lang, not is_english)

    # ── Translate header to English for Gemini (if needed) ───────────────────
    _GT_HEADER_MAP = {"zh": "zh-CN", "zh-cn": "zh-CN", "zh-tw": "zh-TW"}
    header_en = header_text
    if not is_english and header_text.strip():
        gt_src_header = _GT_HEADER_MAP.get(src_lang.lower(), src_lang.split("-")[0])
        try:
            from deep_translator import GoogleTranslator
            header_en = (
                GoogleTranslator(source=gt_src_header, target="en")
                .translate(header_text[:4000])
                or header_text
            )
            log.info("Header translated from [%s] to English (%d chars).",
                     gt_src_header, len(header_en))
        except Exception as exc:
            log.warning("Header translation failed: %s — using raw text for metadata.", exc)

    # ── Auto-extract metadata: regex first, Gemini only for unfilled fields ────
    # Regex is instant and quota-free — always the primary source.
    auto_meta = _regex_extract_metadata(header_en, safe_stem)

    # Only call Gemini if regex missed important fields, to conserve quota.
    if not auto_meta["patent_number"] or not auto_meta["title"]:
        google_api_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
        gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
        if google_api_key:
            try:
                missing = [k for k in ("patent_number","title","assignee","publication_date")
                           if not auto_meta.get(k)]
                prompt = (
                    f"Extract these missing patent fields: {missing}.\n"
                    "Respond ONLY with minified JSON:\n"
                    '{"patent_number":"","title":"","assignee":"","jurisdiction":"","publication_date":""}\n'
                    f"TEXT:\n{header_en[:2500]}"
                )
                resp_g = _state.gemini_client.models.generate_content(
                    model=gemini_model_name, contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
                )
                raw = resp_g.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                for k, v in parsed.items():
                    if k in auto_meta and not auto_meta[k] and isinstance(v, str) and v.strip():
                        auto_meta[k] = v.strip()
                log.info("Gemini enriched: %s", {k:v for k,v in auto_meta.items() if v})
            except Exception as exc:
                log.warning("Gemini enrichment skipped (quota/error): %s", exc)

    # Merge: user form values win over Gemini where they are non-empty
    final_number = patent_number.strip() or auto_meta["patent_number"] or safe_stem
    final_title  = title.strip()         or auto_meta["title"]          or "Untitled Patent"
    final_assign = assignee.strip()      or auto_meta["assignee"]       or ""
    final_jx     = jurisdiction.strip()  or auto_meta["jurisdiction"]   or "US"
    final_date   = pub_date.strip()      or auto_meta["publication_date"] or None

    log.info("Final metadata → number=%s title=%s assignee=%s jx=%s date=%s",
             final_number, final_title, final_assign, final_jx, final_date)

    # ── Upsert patent_documents ───────────────────────────────────────────────
    try:
        db_resp   = _state.supabase.table("patent_documents").upsert(
            {
                "patent_number":    final_number,
                "title":            final_title,
                "assignee":         final_assign,
                "jurisdiction":     final_jx,
                "publication_date": final_date,
            },
            on_conflict="patent_number",
        ).execute()
        patent_id = db_resp.data[0]["id"]
        log.info("Upserted patent_documents → id=%s", patent_id)
    except Exception as exc:
        doc.close()
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"DB metadata insert failed: {exc}")

    # ── Extract full text from all pages ──────────────────────────────────────
    CLAIM_INDEP = _re.compile(r"^\s*1\.\s", _re.IGNORECASE)
    CLAIM_DEP   = _re.compile(
        r"(\bclaim\s+\d+\b.*\bwherein\b|\bwherein\b|"
        r"the\s+(?:method|apparatus|system|device)\s+of\s+claim\s+\d+)",
        _re.IGNORECASE,
    )

    def _stype(t: str) -> str:
        s = t.strip()
        if CLAIM_INDEP.match(s):  return "claim_independent"
        if CLAIM_DEP.search(s):   return "claim_dependent"
        return "description"

    # ── Extract text: native first, PaddleOCR fallback for scanned pages ───────
    def _ocr_page_to_text(page) -> str:
        """Render page to image and run PaddleOCR. Returns plain text."""
        try:
            import cv2
            import numpy as _np2
            from paddleocr import PaddleOCR
            mat = _fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
            img_arr = _np2.frombuffer(pix.tobytes("png"), dtype=_np2.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            ocr = PaddleOCR(use_angle_cls=True, lang="ch" if src_lang == "zh" else "en",
                            show_log=False)
            result = ocr.ocr(img, cls=True)
            lines = []
            if result:
                for block in result:
                    if block:
                        for line in block:
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                tc = line[1]
                                if isinstance(tc, (list, tuple)) and tc:
                                    lines.append(str(tc[0]))
            return " ".join(lines)
        except Exception as ocr_exc:
            log.warning("OCR failed for page: %s", ocr_exc)
            return ""

    def _extract_text(page) -> str:
        native = page.get_text("text").strip()
        if len(native) >= 50:
            return native
        log.info("  Page %d: only %d native chars — trying OCR fallback",
                 page.number + 1, len(native))
        ocr_text = _ocr_page_to_text(page)
        return ocr_text if ocr_text else native

    def _text_to_chunks(text: str) -> List[Dict[str, str]]:
        chunks = []
        buf = ""
        for para in _re.split(r"\n{2,}", text):
            para = para.strip()
            if not para:
                continue
            if buf and len(para) < 80:
                buf += " " + para
            else:
                if buf:
                    chunks.append({"section_type": _stype(buf), "content": buf})
                buf = para
        if buf:
            chunks.append({"section_type": _stype(buf), "content": buf})
        return chunks

    all_chunks: List[Dict[str, str]] = []
    try:
        total_pages = len(doc)
        for page_num in range(total_pages):
            try:
                text = _extract_text(doc[page_num])
                if text:
                    all_chunks.extend(_text_to_chunks(text))
            except Exception as exc:
                log.warning("Page %d error (skipping): %s", page_num + 1, exc)
    finally:
        doc.close()

    all_chunks = [c for c in all_chunks if len(c["content"]) >= 10]
    log.info("Extracted %d raw chunks from %d pages", len(all_chunks), total_pages)

    if not all_chunks:
        tmp_path.unlink(missing_ok=True)
        return {"patent_id": patent_id, "chunks_inserted": 0,
                "patent_number": final_number, "language": src_lang,
                "warning": "No text content extracted from PDF."}

    # ── Translate chunks to English ───────────────────────────────────────────
    # deep-translator uses Google Translate language codes which differ from
    # ISO 639-1 in several cases. Map our internal codes to GT codes.
    _GT_CODE_MAP = {
        "zh": "zh-CN",   # Chinese Simplified (CN patents)
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "jw": "jw",      # Javanese stays as-is
    }
    if not is_english:
        gt_src = _GT_CODE_MAP.get(src_lang.lower(), src_lang.split("-")[0])
        log.info("Translating %d chunks from [%s] (GT code: %s) → English…",
                 len(all_chunks), src_lang, gt_src)
        translated_chunks: List[Dict[str, str]] = []
        for chunk in all_chunks:
            try:
                from deep_translator import GoogleTranslator
                eng = GoogleTranslator(source=gt_src, target="en").translate(
                    chunk["content"][:4500]
                )
                translated_chunks.append({
                    "section_type": chunk["section_type"],
                    "content":      eng if eng else chunk["content"],
                })
            except Exception as exc:
                log.warning("Chunk translation failed (%s): %s — keeping original.", gt_src, exc)
                translated_chunks.append(chunk)
        all_chunks = translated_chunks
        log.info("Translation complete.")

    # ── Generate embeddings ───────────────────────────────────────────────────
    log.info("Generating embeddings for %d chunks…", len(all_chunks))
    try:
        texts      = [c["content"] for c in all_chunks]
        embeddings = _state.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    # ── Build records — serialize embedding as pgvector string "[x,y,z,...]" ──
    # The Supabase REST API / postgREST requires pgvector values as a JSON
    # string in the format "[0.123,0.456,...]", NOT a plain Python list.
    records: List[Dict[str, Any]] = []
    for chunk, emb in zip(all_chunks, embeddings):
        emb_list   = emb.tolist()
        emb_string = "[" + ",".join(f"{v:.8f}" for v in emb_list) + "]"
        records.append({
            "patent_id":    patent_id,
            "section_type": chunk["section_type"],
            "content":      chunk["content"],
            "embedding":    emb_string,        # ← pgvector-safe string format
        })

    # ── Bulk insert in batches ────────────────────────────────────────────────
    # Insert one record first to surface any schema/type errors clearly,
    # then continue with the rest in batches.
    BATCH = 32
    total = 0
    log.info("Inserting %d chunks into patent_chunks…", len(records))

    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        try:
            resp = _state.supabase.table("patent_chunks").insert(batch).execute()
            inserted = len(resp.data) if resp.data else 0
            total += inserted
            log.info("  Batch offset=%d inserted=%d running_total=%d raw_resp_data_len=%d",
                     i, inserted, total, len(resp.data) if resp.data else 0)
            # If Supabase returned 0 rows on what should be an insert, log the full response
            if inserted == 0:
                log.error("  Zero rows returned from Supabase insert at offset %d. "
                          "resp.data=%r  Check: (1) migration_v2.sql was run, "
                          "(2) RLS is disabled for patent_chunks, "
                          "(3) embedding column type is vector(384).", i, resp.data)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            log.error("Batch insert exception at offset %d: %r", i, exc)
            raise HTTPException(status_code=502,
                                detail=f"Chunk insert failed at offset {i}: {exc}")

    tmp_path.unlink(missing_ok=True)
    log.info("Ingest complete: patent=%s chunks=%d lang=%s translated=%s",
             final_number, total, src_lang, not is_english)

    return {
        "patent_id":       patent_id,
        "chunks_inserted": total,
        "patent_number":   final_number,
        "language":        src_lang,
        "translated":      not is_english,
        "metadata": {
            "title":            final_title,
            "assignee":         final_assign,
            "jurisdiction":     final_jx,
            "publication_date": final_date,
        },
    }


# ── Patent list ───────────────────────────────────────────────────────────────
@app.get("/api/v1/patents")
async def list_patents(jurisdiction: Optional[str] = None, assignee: Optional[str] = None):
    try:
        q = _state.supabase.table("patent_documents") \
            .select("id,patent_number,title,assignee,jurisdiction,publication_date,created_at") \
            .order("created_at", desc=True)
        if jurisdiction:
            q = q.eq("jurisdiction", jurisdiction)
        if assignee:
            q = q.ilike("assignee", f"%{assignee}%")
        resp = q.execute()
        return resp.data or []
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Single patent detail ──────────────────────────────────────────────────────
@app.get("/api/v1/patents/{patent_id}")
async def get_patent(patent_id: str):
    try:
        doc_resp = _state.supabase.table("patent_documents") \
            .select("*").eq("id", patent_id).single().execute()
        chunks_resp = _state.supabase.table("patent_chunks") \
            .select("id,section_type,content") \
            .eq("patent_id", patent_id) \
            .order("id").execute()
        return {
            "document": doc_resp.data,
            "chunks":   chunks_resp.data or [],
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── LLM-generated one-page summary ───────────────────────────────────────────
@app.get("/api/v1/patents/{patent_id}/summary")
async def get_patent_summary(patent_id: str):
    try:
        doc_resp = _state.supabase.table("patent_documents") \
            .select("*").eq("id", patent_id).single().execute()
        chunks_resp = _state.supabase.table("patent_chunks") \
            .select("section_type,content") \
            .eq("patent_id", patent_id) \
            .in_("section_type", ["claim_independent", "claim_dependent", "description"]) \
            .limit(20).execute()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    doc    = doc_resp.data
    chunks = chunks_resp.data or []
    context = "\n\n".join(
        f"[{c['section_type'].upper()}] {c['content']}" for c in chunks
    )

    prompt = f"""You are a patent analyst. Given the following patent content, produce a JSON summary.
Respond ONLY with minified JSON. No markdown. No explanation outside JSON.

Output schema:
{{"abstract":"2-3 sentence technical abstract","key_claims":["claim1","claim2","claim3"],"technology_domain":"...","novelty_statement":"...","commercial_relevance":"..."}}

PATENT: {doc.get('patent_number')} — {doc.get('title')}
ASSIGNEE: {doc.get('assignee')}
CONTENT:
{context[:3000]}"""

    try:
        summary = _gemini_json(prompt)
    except Exception:
        summary = {
            "abstract": "Summary generation failed.",
            "key_claims": [],
            "technology_domain": "",
            "novelty_statement": "",
            "commercial_relevance": "",
        }

    return {"document": doc, "summary": summary, "chunk_count": len(chunks)}


# ── Compare two patents ───────────────────────────────────────────────────────
@app.post("/api/v1/compare")
async def compare_patents(req: CompareRequest):

    # ── Fetch patent metadata ─────────────────────────────────────────────────
    try:
        a_resp = _state.supabase.table("patent_documents")             .select("*").eq("id", req.patent_id_a).single().execute()
        b_resp = _state.supabase.table("patent_documents")             .select("*").eq("id", req.patent_id_b).single().execute()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Patent lookup failed: {exc}")

    # ── Fetch chunks — claims first, fall back to all sections if empty ───────
    def _get_chunks(patent_id: str) -> List[str]:
        rows = (_state.supabase.table("patent_chunks")
                .select("section_type,content")
                .eq("patent_id", patent_id)
                .in_("section_type", ["claim_independent", "claim_dependent"])
                .limit(20).execute().data or [])
        if rows:
            return [r["content"] for r in rows]
        rows = (_state.supabase.table("patent_chunks")
                .select("content")
                .eq("patent_id", patent_id)
                .limit(20).execute().data or [])
        return [r["content"] for r in rows]

    try:
        a_contents = _get_chunks(req.patent_id_a)
        b_contents = _get_chunks(req.patent_id_b)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chunk fetch failed: {exc}")

    a_text = "\n\n".join(a_contents)
    b_text = "\n\n".join(b_contents)

    log.info("compare | A=%s chunks=%d B=%s chunks=%d",
             a_resp.data.get("patent_number"), len(a_contents),
             b_resp.data.get("patent_number"), len(b_contents))

    # ── Similarity score ──────────────────────────────────────────────────────
    similarity_score = 0.0
    if a_text.strip() and b_text.strip():
        try:
            import numpy as np
            ea = _state.embed_model.encode([a_text[:2000]], normalize_embeddings=True)[0]
            eb = _state.embed_model.encode([b_text[:2000]], normalize_embeddings=True)[0]
            similarity_score = float(np.dot(ea, eb))
        except Exception as exc:
            log.warning("Similarity embedding failed: %s", exc)

    # ── Build prompt ──────────────────────────────────────────────────────────
    a_doc = a_resp.data
    b_doc = b_resp.data

    a_block = (
        f"{a_doc.get('patent_number','?')} — {a_doc.get('title','?')}\n"
        f"Assignee: {a_doc.get('assignee','?')}\n\n{a_text[:2000]}"
        if a_text else
        f"{a_doc.get('patent_number','?')} — {a_doc.get('title','?')} (no chunk text indexed)"
    )
    b_block = (
        f"{b_doc.get('patent_number','?')} — {b_doc.get('title','?')}\n"
        f"Assignee: {b_doc.get('assignee','?')}\n\n{b_text[:2000]}"
        if b_text else
        f"{b_doc.get('patent_number','?')} — {b_doc.get('title','?')} (no chunk text indexed)"
    )

    schema = (
        '{"overlap_summary":"string",'
        '"overlapping_claims":[{"claim_a":"string","claim_b":"string",'
        '"overlap_type":"exact|semantic|functional","risk_level":"HIGH|MEDIUM|LOW"}],'
        '"differentiating_features":["string"],'
        '"overall_risk":"HIGH|MEDIUM|LOW|CLEAR",'
        '"recommendation":"string"}'
    )
    prompt = (
        "You are a senior patent IP analyst. Compare the two patents below.\n"
        "Respond ONLY with minified JSON matching this schema — no markdown, no preamble:\n"
        f"{schema}\n\n"
        f"PATENT A:\n{a_block}\n\n"
        f"PATENT B:\n{b_block}\n\n"
        "Identify structural and semantic overlaps between the claims. "
        "Assign a risk level to each overlap. List differentiating features. "
        "Give overall_risk and a one-sentence recommendation."
    )

    # ── Call Gemini — propagate real errors instead of hiding them ────────────
    try:
        analysis = _gemini_json(prompt)
        for key, default in [
            ("overlap_summary",       ""),
            ("overlapping_claims",    []),
            ("differentiating_features", []),
            ("overall_risk",          "UNKNOWN"),
            ("recommendation",        ""),
        ]:
            if key not in analysis:
                analysis[key] = default
        log.info("compare OK | risk=%s overlaps=%d",
                 analysis.get("overall_risk"), len(analysis.get("overlapping_claims", [])))
    except HTTPException:
        raise   # let rate-limit / auth errors surface to the client
    except Exception as exc:
        log.error("compare analysis error: %r", exc)
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    return {
        "patent_a":         a_doc,
        "patent_b":         b_doc,
        "similarity_score": round(similarity_score * 100, 1),
        "analysis":         analysis,
    }

# ── Evaluate design (existing endpoint) ──────────────────────────────────────
@app.post("/api/v1/evaluate-design", response_model=DesignEvaluationResponse)
async def evaluate_design(request: DesignEvaluationRequest):
    log.info("evaluate-design | product_id=%s jurisdiction=%s",
             request.product_id, request.jurisdiction)

    try:
        query_embedding = _embed(request.proposed_specifications)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    raw_chunks = _fetch_hybrid_matches(
        query_embedding, request.proposed_specifications, request.jurisdiction
    )

    if not raw_chunks:
        return DesignEvaluationResponse(
            product_id="", risk_status="CLEAR",
            infringement_map=[], design_arounds=[],
            matched_chunks=[], token_budget_used=0,
        )

    chunk_refs    = [ChunkReference(**{k: c.get(k, "") for k in ChunkReference.model_fields},
                                    rrf_score=float(c.get("rrf_score", 0.0)))
                     for c in raw_chunks]
    context_block = _build_context_block(raw_chunks)
    gen_out       = _call_agent_generator(context_block, request.proposed_specifications,
                                          request.component_scope)
    aud_out       = _call_agent_auditor(gen_out, request.component_scope)
    merged_das    = aud_out.get("design_arounds_merged", [])
    token_est     = (len(context_block) + len(request.proposed_specifications) +
                     len(json.dumps(gen_out))) // 4

    return DesignEvaluationResponse(
        product_id=request.product_id,
        risk_status=aud_out.get("risk_status", "UNKNOWN"),
        infringement_map=aud_out.get("infringement_map", []),
        design_arounds=merged_das,
        matched_chunks=chunk_refs,
        token_budget_used=token_est,
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
