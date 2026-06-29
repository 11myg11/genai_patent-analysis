"""
app/utils/translation.py — Language detection and English translation for patent text.

Used during ingestion to normalise non-English patents into English before embedding,
so all vectors live in the same semantic space regardless of origin language.

Global variables:
  _PREFIX_LANG   — Maps patent-office 2-letter prefix to ISO 639-1 language code.
                   e.g. "CN" → "zh", "JP" → "ja". More reliable than text detection.
  _GT_CODE_MAP   — Maps ISO codes to Google Translate codes where they differ.
                   e.g. "zh" → "zh-CN" (deep-translator requires this exact format).
  TRANSLATE_CHUNK_LIMIT — Max chars per Google Translate request (free-tier limit).

Functions:
  detect_language(text, patent_number_or_filename) -> str
    Returns an ISO 639-1 language code. Checks the patent-number prefix first;
    falls back to langdetect on the first 2000 chars of text.

  translate_to_english(text, source_lang) -> str
    Translates arbitrary-length text to English using deep-translator (Google
    Translate, free, no API key). Splits at newline boundaries to stay within
    the per-request char limit. Returns original text if translation fails.

  translate_chunks(chunks, source_lang) -> list[dict]
    Translates a list of {"section_type": ..., "content": ...} dicts in-place.
    No-ops for English documents.
"""
import logging
from typing import Dict, List

log = logging.getLogger(__name__)

# Patent office prefix → ISO 639-1 language code
# Prefix is the most reliable signal — CN/JP/KR etc. are always published in their national language
_PREFIX_LANG: Dict[str, str] = {
    "CN": "zh", "JP": "ja", "KR": "ko",
    "DE": "de", "AT": "de", "CH": "de",
    "FR": "fr", "BE": "fr", "ES": "es",
    "NL": "nl", "RU": "ru", "PT": "pt", "IT": "it",
}

# deep-translator uses Google Translate codes which differ from ISO 639-1 in some cases
_GT_CODE_MAP: Dict[str, str] = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
}

TRANSLATE_CHUNK_LIMIT = 4500


def detect_language(text: str, patent_number_or_filename: str = "") -> str:
    prefix = (patent_number_or_filename.strip()[:2] or "").upper()
    if prefix in _PREFIX_LANG:
        lang = _PREFIX_LANG[prefix]
        log.info("Language from patent prefix [%s]: %s", prefix, lang)
        return lang
    try:
        from langdetect import detect
        sample = text[:2000].strip()
        if not sample:
            return "en"
        lang = detect(sample)
        log.info("Language from langdetect: %s", lang)
        return lang
    except Exception as exc:
        log.warning("Language detection failed (%s) — assuming English.", exc)
        return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    if not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator

        gt_src = _GT_CODE_MAP.get(source_lang.lower(), source_lang.split("-")[0])
        translator = GoogleTranslator(source=gt_src, target="en")

        # Split at newline boundaries to stay within the free-tier per-request char limit
        chunks: List[str] = []
        remaining = text
        while len(remaining) > TRANSLATE_CHUNK_LIMIT:
            split_at = remaining.rfind("\n", 0, TRANSLATE_CHUNK_LIMIT)
            if split_at == -1:
                split_at = TRANSLATE_CHUNK_LIMIT
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()
        chunks.append(remaining)

        parts: List[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                result = translator.translate(chunk)
                parts.append(result if result else chunk)
            except Exception as exc:
                log.warning("Chunk translation failed: %s — keeping original.", exc)
                parts.append(chunk)

        return "\n".join(parts)

    except ImportError:
        log.error("deep-translator not installed.")
        return text
    except Exception as exc:
        log.warning("Translation failed (%s) — keeping original text.", exc)
        return text


def translate_chunks(
    chunks: List[Dict[str, str]],
    source_lang: str,
) -> List[Dict[str, str]]:
    if source_lang.startswith("en"):
        log.info("Document is English — no translation needed.")
        return chunks

    log.info("Translating %d chunks from [%s] → English…", len(chunks), source_lang)
    translated: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks):
        eng = translate_to_english(chunk["content"], source_lang)
        translated.append({"section_type": chunk["section_type"], "content": eng})
        if (i + 1) % 20 == 0:
            log.info("  Translated %d / %d chunks", i + 1, len(chunks))

    log.info("Translation complete.")
    return translated
