"""
main.py — FastAPI application factory and startup/shutdown lifecycle.

Entry point for the Patent Analysis Platform. Initialises shared resources
at startup and registers all routers. Keep this file thin — business logic
belongs in app/services/ and app/routes/.

Startup sequence (lifespan):
  1. Validate required env vars (fails fast with a clear message if missing)
  2. Load BAAI/bge-small-en-v1.5 embedding model into app.state.state.embed_model
  3. Create Supabase client → app.state.state.supabase
  4. Log the active OpenRouter model

Routers registered:
  app/routes/ui.py  — HTML page routes (/, /upload, /summaries, /compare, /playground)
  app/routes/api.py — JSON API routes (/health, /api/v1/*)

Run:
  python main.py                        (with auto-reload, dev)
  uvicorn main:app --host 0.0.0.0       (production)
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from supabase import create_client

from app.config import EMBEDDING_MODEL, settings
from app.state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env or .env.txt")
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in .env or .env.txt")

    log.info("Startup: loading embedding model %s…", EMBEDDING_MODEL)
    state.embed_model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model loaded.")

    state.supabase = create_client(settings.supabase_url, settings.supabase_anon_key)
    log.info("Supabase client initialised.")

    log.info("LLM: OpenRouter model=%s", settings.openrouter_model)

    yield
    log.info("Shutdown complete.")


app = FastAPI(title="Patent Analysis Platform", version="1.0.0", lifespan=lifespan)

from app.routes import api, ui  # noqa: E402
app.include_router(ui.router)
app.include_router(api.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)
