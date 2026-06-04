"""
ingest_patents.py
─────────────────
Patent PDF ingestion pipeline with:
  1. Auto-metadata extraction  — Gemini reads the first 2 pages and extracts
                                  patent number, title, assignee, jurisdiction,
                                  and publication date automatically.
  2. Language detection        — langdetect identifies the document language.
  3. Auto-translation          — deep-translator (Google Translate, free, no key)
                                  translates every non-English chunk to English
                                  before embedding, so all vectors live in the
                                  same English semantic space.
  4. Text extraction           — PyMuPDF native first, PaddleOCR fallback for
                                  scanned pages.
  5. Embedding + Supabase      — BAAI/bge-small-en-v1.5, bulk upsert.

CLI usage (all args are now OPTIONAL — metadata is auto-extracted from the PDF):
  py ingest_patents.py --pdf "path\\to\\patent.pdf"

  # Override any auto-extracted field if needed:
  py ingest_patents.py --pdf "path\\to\\patent.pdf" --patent-number EP1234567A1
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz                              # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL       = "BAAI/bge-small-en-v1.5"
BATCH_SIZE            = 32
PAGE_DPI              = 150
MIN_NATIVE_CHARS      = 50
TRANSLATE_CHUNK_LIMIT = 4500   # Google Translate free tier max chars per call
META_EXTRACT_PAGES    = 3      # number of pages fed to Gemini for metadata

# Section classification regexes
CLAIM_INDEP_RE = re.compile(r"^\s*1\.\s", re.IGNORECASE)
CLAIM_DEP_RE   = re.compile(
    r"(\bclaim\s+\d+\b.*\bwherein\b|\bwherein\b|"
    r"the\s+(?:method|apparatus|system|device)\s+of\s+claim\s+\d+)",
    re.IGNORECASE,
)
CLAIM_NUM_RE = re.compile(r"^\s*\d{1,3}\.\s")


# ═══════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════

# Prefix → language mapping (patent office code is the most reliable signal)
_PREFIX_LANG: Dict[str, str] = {
    "CN": "zh", "JP": "ja", "KR": "ko",
    "DE": "de", "AT": "de", "CH": "de",
    "FR": "fr", "BE": "fr", "ES": "es",
    "NL": "nl", "RU": "ru", "PT": "pt", "IT": "it",
}

def _detect_language(text: str, patent_number: str = "") -> str:
    """
    Detect language with two-stage fallback:
    1. Patent number prefix (CN→zh, JP→ja, KR→ko, DE→de …) — most reliable.
    2. langdetect on body text — fallback for US/EP/WO/GB patents which are
       published in English regardless of applicant origin.
    """
    prefix = (patent_number.strip()[:2] or "").upper()
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


# ═══════════════════════════════════════════════════════════════════
# TRANSLATION
# ═══════════════════════════════════════════════════════════════════

def _translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text to English using Google Translate (free, no API key).
    Splits text into chunks ≤ TRANSLATE_CHUNK_LIMIT chars to stay within
    the free tier per-request limit.
    Returns original text unchanged if translation fails.
    """
    if not text.strip():
        return text

    try:
        from deep_translator import GoogleTranslator

        # Normalise language code: langdetect returns 'zh-cn', translator wants 'zh-CN' etc.
        src = source_lang.split("-")[0]   # 'zh-cn' → 'zh'

        # deep-translator uses Google Translate codes; zh≠zh-CN
        _GT_MAP = {"zh": "zh-CN", "zh-cn": "zh-CN", "zh-tw": "zh-TW"}
        gt_src = _GT_MAP.get(source_lang.lower(), src)
        translator = GoogleTranslator(source=gt_src, target="en")

        # Split into safe chunks on sentence boundaries where possible
        chunks: List[str] = []
        remaining = text
        while len(remaining) > TRANSLATE_CHUNK_LIMIT:
            # Try to break at the last newline within the limit
            split_at = remaining.rfind("\n", 0, TRANSLATE_CHUNK_LIMIT)
            if split_at == -1:
                split_at = TRANSLATE_CHUNK_LIMIT
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()
        chunks.append(remaining)

        translated_parts: List[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                result = translator.translate(chunk)
                translated_parts.append(result if result else chunk)
            except Exception as chunk_exc:
                log.warning("Chunk translation failed: %s — keeping original.", chunk_exc)
                translated_parts.append(chunk)

        return "\n".join(translated_parts)

    except ImportError:
        log.error("deep-translator not installed. Run: pip install deep-translator")
        return text
    except Exception as exc:
        log.warning("Translation failed (%s) — keeping original text.", exc)
        return text


def _translate_chunks(
    chunks: List[Dict[str, str]],
    source_lang: str,
) -> List[Dict[str, str]]:
    """
    Translate all chunks in-place if source language is not English.
    Keeps original_content field for traceability.
    """
    if source_lang.startswith("en"):
        log.info("Document is English — no translation needed.")
        return chunks

    log.info("Translating %d chunks from [%s] → English…", len(chunks), source_lang)
    translated: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks):
        orig = chunk["content"]
        eng  = _translate_to_english(orig, source_lang)
        translated.append({
            "section_type":     chunk["section_type"],
            "content":          eng,
            "original_content": orig,   # stored for audit trail
        })
        if (i + 1) % 20 == 0:
            log.info("  Translated %d / %d chunks", i + 1, len(chunks))

    log.info("Translation complete.")
    return translated


# ═══════════════════════════════════════════════════════════════════
# AUTO-METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _extract_metadata_with_gemini(
    first_pages_text: str,
    google_api_key:   str,
    gemini_model:     str = "gemini-2.0-flash",
) -> Dict[str, str]:
    """
    Feed the first few pages of the patent to Gemini and extract:
      patent_number, title, assignee, jurisdiction, publication_date.

    Returns a dict; any field that can't be found is returned as "".
    """
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=google_api_key)

        prompt = f"""You are a patent document parser. Extract metadata from the following patent text.
Respond ONLY with minified JSON — no markdown, no explanation outside JSON.

Output schema (all fields are strings):
{{"patent_number":"","title":"","assignee":"","jurisdiction":"US|EP|CN|JP|KR|DE|FR|GB|other","publication_date":"YYYY-MM-DD or empty string"}}

Rules:
- patent_number: the official patent/application number (e.g. US10678081B2, EP3456789A1, CN112345678A).
- title: the invention title in English (translate if necessary).
- assignee: the patent owner / applicant company name.
- jurisdiction: the 2-letter country/office code. Infer from patent_number prefix if not stated.
- publication_date: ISO 8601 date string, or empty string if not found.

PATENT TEXT (first {META_EXTRACT_PAGES} pages):
{first_pages_text[:4000]}"""

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
        )
        raw = response.text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        meta = json.loads(raw)
        log.info("Auto-extracted metadata: %s", meta)
        return meta

    except Exception as exc:
        log.warning("Gemini metadata extraction failed: %s — using empty metadata.", exc)
        return {"patent_number": "", "title": "", "assignee": "",
                "jurisdiction": "US", "publication_date": ""}


def _regex_fallback_metadata(text: str) -> Dict[str, str]:
    """
    Regex-based fallback metadata extractor for when Gemini is unavailable.
    Handles common USPTO, EPO, and CNIPA header formats.
    """
    meta: Dict[str, str] = {
        "patent_number": "", "title": "", "assignee": "",
        "jurisdiction": "US", "publication_date": "",
    }

    # Patent number patterns: US/EP/CN/JP/KR/DE/WO + digits + optional suffix
    pn_match = re.search(
        r"\b(US|EP|CN|JP|KR|DE|WO|FR|GB)\s*(\d[\d,\s]{4,10}[A-Z0-9]*)\b",
        text, re.IGNORECASE
    )
    if pn_match:
        meta["patent_number"] = (pn_match.group(1).upper() +
                                  pn_match.group(2).replace(" ", "").replace(",", ""))
        meta["jurisdiction"]  = pn_match.group(1).upper()

    # Date: look for (NN) Date of Patent lines (USPTO style)
    date_match = re.search(
        r"(?:Date of Patent|Publication Date|Pub\. Date)[:\s]+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        text
    )
    if date_match:
        raw_date = date_match.group(1).strip()
        # Try ISO format first
        if re.match(r"\d{4}-\d{2}-\d{2}", raw_date):
            meta["publication_date"] = raw_date
        else:
            try:
                from datetime import datetime
                for fmt in ("%B %d, %Y", "%b. %d, %Y", "%B %d %Y", "%b %d, %Y"):
                    try:
                        meta["publication_date"] = datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

    # Assignee: look for (73) Assignee: pattern
    assignee_match = re.search(r"(?:Assignee|Applicant)[:\s]+([A-Z][^\n,]{3,60})", text)
    if assignee_match:
        meta["assignee"] = assignee_match.group(1).strip()

    # Title: look for first all-caps or title-case multi-word line after patent number
    title_match = re.search(
        r"(?:TITLE OF INVENTION|Title)[:\s]*\n?\s*([A-Z][^\n]{10,120})", text
    )
    if title_match:
        meta["title"] = title_match.group(1).strip().title()

    return meta


# ═══════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════

_ocr_engine = None

def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        log.info("Initialising PaddleOCR engine…")
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:
            log.error("PaddleOCR init failed: %s", exc)
            raise
    return _ocr_engine


def _ocr_page(page: fitz.Page) -> str:
    import cv2
    mat     = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix     = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    result  = _get_ocr_engine().ocr(img, cls=True)
    lines: List[str] = []
    if result:
        for block in result:
            if block:
                for line in block:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        tc = line[1]
                        if isinstance(tc, (list, tuple)) and len(tc) >= 1:
                            lines.append(str(tc[0]))
    return " ".join(lines)


def _extract_page_text(page: fitz.Page) -> str:
    native = page.get_text("text").strip()
    if len(native) >= MIN_NATIVE_CHARS:
        return native
    log.info("  Page %d: sparse text (%d chars) → OCR fallback", page.number + 1, len(native))
    try:
        return _ocr_page(page)
    except Exception as exc:
        log.warning("  OCR failed: %s", exc)
        return native


def determine_section_type(text: str) -> str:
    s = text.strip()
    if CLAIM_INDEP_RE.match(s):  return "claim_independent"
    if CLAIM_DEP_RE.search(s):   return "claim_dependent"
    if CLAIM_NUM_RE.match(s):    return "claim_dependent"
    return "description"


def _split_into_chunks(full_text: str) -> List[Dict[str, str]]:
    paragraphs = re.split(r"\n{2,}", full_text)
    chunks: List[Dict[str, str]] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if buffer and len(para) < 80:
            buffer += " " + para
        else:
            if buffer:
                chunks.append({"section_type": determine_section_type(buffer), "content": buffer})
            buffer = para
    if buffer:
        chunks.append({"section_type": determine_section_type(buffer), "content": buffer})
    return [c for c in chunks if len(c["content"]) >= 10]


# ═══════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════════

def ingest_patent(
    pdf_path:         Path,
    supabase:         Client,
    embed_model:      SentenceTransformer,
    google_api_key:   str,
    gemini_model:     str    = "gemini-2.0-flash",
    # Optional overrides — if provided, skip auto-extraction for that field
    patent_number:    str    = "",
    title:            str    = "",
    assignee:         str    = "",
    jurisdiction:     str    = "",
    publication_date: str    = "",
) -> Dict[str, Any]:
    """
    Full ingestion pipeline. Returns a summary dict.
    """
    log.info("Opening PDF: %s", pdf_path)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.error("Failed to open PDF: %s", exc)
        raise

    # ── Step 1: Extract first N pages for metadata + language detection ───────
    log.info("Extracting header text for metadata + language detection…")
    header_pages_text = ""
    for i in range(min(META_EXTRACT_PAGES, len(doc))):
        header_pages_text += _extract_page_text(doc[i]) + "\n\n"

    # ── Step 2: Detect language ───────────────────────────────────────────────
    # Pass patent_number so the prefix (CN/JP/KR…) is checked first
    source_lang = _detect_language(header_pages_text, patent_number=patent_number)
    is_english  = source_lang.startswith("en")
    log.info("Source language: %s | Needs translation: %s", source_lang, not is_english)

    # ── Step 3: Auto-extract metadata ─────────────────────────────────────────
    log.info("Auto-extracting metadata via Gemini…")
    # Translate header text to English first if needed, so Gemini can read it
    header_for_meta = (
        header_pages_text if is_english
        else _translate_to_english(header_pages_text, source_lang)
    )

    auto_meta = _extract_metadata_with_gemini(header_for_meta, google_api_key, gemini_model)

    # CLI overrides take priority over auto-extracted values
    final_patent_number  = patent_number  or auto_meta.get("patent_number",  "") or pdf_path.stem
    final_title          = title          or auto_meta.get("title",          "") or "Untitled Patent"
    final_assignee       = assignee       or auto_meta.get("assignee",       "")
    final_jurisdiction   = jurisdiction   or auto_meta.get("jurisdiction",   "US")
    final_pub_date       = publication_date or auto_meta.get("publication_date", "") or None

    log.info("Metadata → number=%s | title=%s | assignee=%s | jx=%s | date=%s",
             final_patent_number, final_title, final_assignee,
             final_jurisdiction, final_pub_date)

    # ── Step 4: Upsert patent_documents ──────────────────────────────────────
    patent_meta = {
        "patent_number":    final_patent_number,
        "title":            final_title,
        "assignee":         final_assignee,
        "jurisdiction":     final_jurisdiction,
        "publication_date": final_pub_date,
    }
    try:
        resp = (
            supabase.table("patent_documents")
            .upsert(patent_meta, on_conflict="patent_number")
            .execute()
        )
        patent_id: str = resp.data[0]["id"]
        log.info("Upserted patent_documents → id=%s", patent_id)
    except Exception as exc:
        log.error("Supabase upsert failed: %s", exc)
        doc.close()
        raise

    # ── Step 5: Extract full text from all pages ──────────────────────────────
    all_chunks: List[Dict[str, str]] = []
    for page_num in range(len(doc)):
        log.info("Processing page %d / %d", page_num + 1, len(doc))
        try:
            page_text = _extract_page_text(doc[page_num])
            chunks    = _split_into_chunks(page_text)
            log.info("  → %d chunks", len(chunks))
            all_chunks.extend(chunks)
        except Exception as exc:
            log.warning("Page %d error (skipping): %s", page_num + 1, exc)

    doc.close()
    log.info("Total raw chunks: %d", len(all_chunks))

    if not all_chunks:
        log.warning("No content extracted — aborting.")
        return {"patent_id": patent_id, "chunks_inserted": 0,
                "language": source_lang, "metadata": patent_meta}

    # ── Step 6: Translate all chunks to English ───────────────────────────────
    all_chunks = _translate_chunks(all_chunks, source_lang)

    # ── Step 7: Embed ─────────────────────────────────────────────────────────
    log.info("Generating embeddings…")
    texts = [c["content"] for c in all_chunks]
    try:
        embeddings: np.ndarray = embed_model.encode(
            texts, batch_size=BATCH_SIZE,
            show_progress_bar=True, normalize_embeddings=True,
        )
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        raise

    # ── Step 8: Build + insert records ───────────────────────────────────────
    # pgvector via Supabase REST requires embeddings as a string "[x,y,z,...]"
    # NOT a plain Python list — the PostgREST layer cannot auto-cast lists.
    records: List[Dict[str, Any]] = [
        {
            "patent_id":    patent_id,
            "section_type": chunk["section_type"],
            "content":      chunk["content"],
            "embedding":    "[" + ",".join(f"{v:.8f}" for v in emb.tolist()) + "]",
        }
        for chunk, emb in zip(all_chunks, embeddings)
    ]

    log.info("Uploading %d chunks to Supabase…", len(records))
    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        try:
            supabase.table("patent_chunks").insert(batch).execute()
            total += len(batch)
            log.info("  Inserted %d / %d", total, len(records))
        except Exception as exc:
            log.error("Batch insert failed at index %d: %s", i, exc)
            raise

    log.info("Done. %d chunks stored for %s.", total, final_patent_number)
    return {
        "patent_id":       patent_id,
        "patent_number":   final_patent_number,
        "chunks_inserted": total,
        "language":        source_lang,
        "translated":      not is_english,
        "metadata":        patent_meta,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a patent PDF — metadata is auto-extracted, all args optional.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully automatic (recommended):
  py ingest_patents.py --pdf "CN112345678A.pdf"

  # Override specific fields:
  py ingest_patents.py --pdf "patent.pdf" --jurisdiction EP --assignee "Pilkington"
""",
    )
    p.add_argument("--pdf",            required=True,  help="Path to patent PDF.")
    p.add_argument("--patent-number",  default="",     help="Override patent number.")
    p.add_argument("--title",          default="",     help="Override title.")
    p.add_argument("--assignee",       default="",     help="Override assignee.")
    p.add_argument("--jurisdiction",   default="",     help="Override jurisdiction (US/EP/CN…).")
    p.add_argument("--pub-date",       default="",     help="Override publication date YYYY-MM-DD.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    supabase_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    google_key   = os.environ.get("GOOGLE_API_KEY", "").strip()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()

    if not supabase_url or not supabase_key:
        log.error("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        sys.exit(1)
    if not google_key:
        log.error("GOOGLE_API_KEY must be set in .env for metadata extraction")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        log.error("SentenceTransformer load failed: %s", exc)
        sys.exit(1)

    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        log.info("Supabase connected.")
    except Exception as exc:
        log.error("Supabase connection failed: %s", exc)
        sys.exit(1)

    result = ingest_patent(
        pdf_path         = pdf_path,
        supabase         = supabase,
        embed_model      = embed_model,
        google_api_key   = google_key,
        gemini_model     = gemini_model,
        patent_number    = args.patent_number,
        title            = args.title,
        assignee         = args.assignee,
        jurisdiction     = args.jurisdiction,
        publication_date = args.pub_date,
    )
    log.info("Ingestion summary: %s", json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
