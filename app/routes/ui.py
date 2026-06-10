"""
app/routes/ui.py — Server-side rendered page routes (Jinja2 templates).

Routes:
  GET /                   → redirects to /upload
  GET /upload             → patent upload form
  GET /summaries          → browsable list of all patents
  GET /risk               → Phase 2 — IP risk analysis
  GET /design-suggestions → Phase 3 — design suggestions built on risk output
  GET /innovation         → Phase 4 — innovation opportunities or improvement ideas for gaps or common patterns 
"""
import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.state import state

router = APIRouter()
log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/upload")


@router.get("/upload", include_in_schema=False)
async def page_upload(request: Request):
    try:
        recent = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,assignee,jurisdiction,publication_date,created_at")
                .order("created_at", desc=True).limit(20).execute().data or []
            )
        )
    except Exception:
        recent = []
    return templates.TemplateResponse(request=request, name="upload.html", context={"recent": recent})


@router.get("/summaries", include_in_schema=False)
async def page_summaries(request: Request):
    try:
        patents = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,assignee,jurisdiction,publication_date")
                .order("created_at", desc=True).execute().data or []
            )
        )
    except Exception:
        patents = []
    return templates.TemplateResponse(request=request, name="summaries.html", context={"patents": patents})


@router.get("/risk", include_in_schema=False)
async def page_risk(request: Request):
    """Phase 2 — IP risk identification for a proposed product design."""
    try:
        patents = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,jurisdiction")
                .order("created_at", desc=True).execute().data or []
            )
        )
    except Exception:
        patents = []
    jurisdictions = sorted(set(p["jurisdiction"] for p in patents if p.get("jurisdiction")))
    return templates.TemplateResponse(
        request=request, name="risk.html",
        context={"patents": patents, "jurisdictions": jurisdictions}
    )


@router.get("/design-suggestions", include_in_schema=False)
async def page_design_suggestions(request: Request):
    """Phase 3 — Design suggestions built on top of risk analysis."""
    try:
        patents = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("jurisdiction")
                .execute().data or []
            )
        )
    except Exception:
        patents = []
    jurisdictions = sorted(set(p["jurisdiction"] for p in patents if p.get("jurisdiction")))
    return templates.TemplateResponse(
        request=request, name="design-suggestions.html",
        context={"jurisdictions": jurisdictions}
    )


@router.get("/innovation", include_in_schema=False)
async def page_innovation(request: Request):
    """Phase 4 — Innovation opportunities: gap analysis and new IP directions."""
    try:
        patents = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("jurisdiction")
                .execute().data or []
            )
        )
    except Exception:
        patents = []
    jurisdictions = sorted(set(p["jurisdiction"] for p in patents if p.get("jurisdiction")))
    return templates.TemplateResponse(
        request=request, name="innovation.html",
        context={"jurisdictions": jurisdictions}
    )