"""
app/config.py — Central configuration for the Patent Analysis Platform.

All constants and environment variables live here. Import `settings` for env vars
and the uppercase constants (EMBEDDING_MODEL, PVB_MIN_MM, etc.) for domain values.
New env vars must be added to the Settings class — never use os.environ.get() elsewhere.

Settings (loaded from .env via pydantic-settings):
  SUPABASE_URL / SUPABASE_ANON_KEY  — Supabase project credentials
  OPENROUTER_API_KEY                — OpenRouter API key (openai-compatible)
  OPENROUTER_MODEL                  — Model to use; "openrouter/auto" lets OpenRouter pick
  APP_HOST / APP_PORT               — Uvicorn bind address (default 127.0.0.1:8000)
  DEBUG                             — Enable debug endpoints (default False)

Key constants:
  EMBEDDING_MODEL    — BAAI/bge-small-en-v1.5 (384-dim); do not change without re-embedding
  PVB_MIN/MAX_MM     — PVB interlayer thickness hard limits (Fuyao manufacturing constraint)
  GLASS_TOTAL_MIN/MAX— Total glass stack thickness limits in mm
  BATCH_SIZE         — Supabase insert batch size and embedding encode batch size
  UPLOAD_DIR         — Temp directory for PDF uploads (auto-created)
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_anon_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/auto"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    debug: bool = False

    model_config = SettingsConfigDict(env_file=str(_ROOT / ".env"), extra="ignore")


settings = Settings()

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MAX_TOKENS = 1024
TOP_K_CHUNKS = 3

# PDF processing
BATCH_SIZE = 32
PAGE_DPI = 150
MIN_NATIVE_CHARS = 50
TRANSLATE_CHUNK_LIMIT = 4500
META_EXTRACT_PAGES = 3

# Glass domain hard limits (Fuyao manufacturing constraints)
PVB_MIN_MM = 0.38
PVB_MAX_MM = 0.76
GLASS_TOTAL_MIN = 3.1
GLASS_TOTAL_MAX = 6.0
HUD_ZONE_CONDUCTIVE_BAN = True

BASE_DIR = _ROOT
UPLOAD_DIR = _ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
