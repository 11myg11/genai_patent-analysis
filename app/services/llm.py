"""
app/services/llm.py — OpenRouter LLM client with caching and retry.

This is the single point for all LLM calls in the application. Swap the model or
provider here only — nothing else needs to change.

OpenRouter is OpenAI-API-compatible. We use the official openai SDK pointed at
https://openrouter.ai/api/v1. The model "openrouter/auto" lets OpenRouter pick
the best available model automatically.

Global variables:
  _CACHE  — In-memory dict keyed by SHA-256 of the prompt. Avoids re-spending quota
            on identical prompts within the same server session. Cleared on restart.
  _client — Lazy-initialised OpenAI client singleton (created on first call).

Functions:
  llm_text(prompt, max_tokens) -> str
    Sends a prompt and returns the raw text response. Retries once on 429 (rate
    limit) after a 5-second wait, then raises HTTP 429.

  llm_json(prompt) -> dict
    Calls llm_text and parses the result as JSON. Strips markdown code fences
    (```json ... ```) that some models add despite being told not to. Raises
    HTTP 500 on parse failure. All prompts sent to this function must instruct
    the model to respond with ONLY minified JSON — the parser has no fallback.
"""
import hashlib
import json
import logging
import time
from typing import Any, Dict

from fastapi import HTTPException
from openai import OpenAI, RateLimitError

from app.config import settings, LLM_MAX_TOKENS

log = logging.getLogger(__name__)

# In-memory prompt cache — keyed by SHA-256 of prompt text.
# Avoids re-spending quota on identical requests within the same server session.
_CACHE: Dict[str, str] = {}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def llm_text(prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in _CACHE:
        log.info("LLM cache hit.")
        return _CACHE[key]

    client = _get_client()
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                raise ValueError("LLM returned empty response (None). The model may have timed out or exceeded token limit.")
            result = raw_content.strip()
            _CACHE[key] = result
            return result
        except RateLimitError as exc:
            if attempt == 0:
                log.warning("Rate limited — waiting 5s before retry…")
                time.sleep(5)
                continue
            raise HTTPException(status_code=429, detail=f"LLM rate limit: {exc}") from exc
        except Exception as exc:
            log.error("LLM error: %s", exc)
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    raise HTTPException(status_code=429, detail="LLM rate limit exceeded after retry.")


def llm_json(prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> Dict[str, Any]:
    raw = ""
    try:
        raw = llm_text(prompt, max_tokens=max_tokens)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("LLM JSON parse error: %s | raw=%s", exc, raw[:300])
        raise HTTPException(status_code=500, detail=f"LLM output parse failure: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM inference error: {exc}") from exc