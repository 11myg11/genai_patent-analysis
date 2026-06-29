"""
app/state.py — Shared application state initialised at startup.

`state` is a module-level singleton populated by the lifespan in main.py.
Import it wherever you need the embedding model or Supabase client:

  from app.state import state
  state.embed_model.encode(...)
  state.supabase.table("patent_documents").select("*").execute()

Fields:
  embed_model  — Loaded SentenceTransformer (jinaai/jina-embeddings-v3, trust_remote_code=True)
  supabase     — Supabase client (sync; wrap with asyncio.to_thread in async routes)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppState:
    embed_model: Optional[object] = None
    supabase: Optional[object] = None


state = AppState()
