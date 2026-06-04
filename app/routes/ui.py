"""
app/routes/ui.py — Server-side rendered page routes (Jinja2 templates).

Serves the five HTML pages of the UI. Each route fetches minimal data from
Supabase to populate the page, then delegates all rendering to a Jinja2 template
in the /templates directory.

Routes:
  GET /             → redirects to /upload
  GET /upload       → patent upload form; shows last 20 ingested patents
  GET /summaries    → browsable list of all patents with LLM summary links
  GET /compare      → side-by-side patent comparison page
  GET /playground   → free-form design-around exploration UI

All Supabase calls are wrapped with asyncio.to_thread() because the supabase-py
client is synchronous and would otherwise block the async event loop.
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


@router.get("/compare", include_in_schema=False)
async def page_compare(request: Request):
    try:
        patents = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,assignee,jurisdiction").execute().data or []
            )
        )
    except Exception:
        patents = []
    return templates.TemplateResponse(request=request, name="compare.html", context={"patents": patents})


@router.get("/playground", include_in_schema=False)
async def page_playground(request: Request):
    return templates.TemplateResponse(request=request, name="playground.html", context={})
