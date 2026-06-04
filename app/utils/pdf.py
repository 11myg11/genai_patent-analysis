"""
app/utils/pdf.py — PDF text extraction, OCR fallback, and paragraph chunking.

Handles the full pipeline from a fitz.Page object to a list of labelled text chunks
ready for embedding. Used exclusively by app/services/ingest.py.

Global variables:
  CLAIM_INDEP_RE / CLAIM_DEP_RE / CLAIM_NUM_RE — Pre-compiled regexes for classifying
    a paragraph as an independent claim, dependent claim, or description section.
  _ocr_engines — Dict of lazy-loaded PaddleOCR singletons keyed by language ("en"/"ch").
    OCR is expensive to initialise; singletons avoid reloading per page.

Functions:
  extract_page_text(page, src_lang) -> str
    Extracts text from a fitz.Page. Uses native PyMuPDF text first; falls back to
    PaddleOCR if the native result is shorter than MIN_NATIVE_CHARS (50 chars),
    which indicates a scanned/image-only page.

  determine_section_type(text) -> str
    Classifies a paragraph as "claim_independent", "claim_dependent", or "description"
    based on regex patterns. Used to label chunks stored in Supabase.

  split_into_chunks(full_text) -> list[dict]
    Splits a page's full text on double newlines into labelled paragraphs.
    Short paragraphs (< 80 chars) are merged into the previous one to avoid
    fragmenting numbered list items. Chunks shorter than 10 chars are dropped.
"""
import logging
import re
from typing import Dict, List

import fitz
import numpy as np

from app.config import PAGE_DPI, MIN_NATIVE_CHARS

log = logging.getLogger(__name__)

CLAIM_INDEP_RE = re.compile(r"^\s*1\.\s", re.IGNORECASE)
CLAIM_DEP_RE = re.compile(
    r"(\bclaim\s+\d+\b.*\bwherein\b|\bwherein\b|"
    r"the\s+(?:method|apparatus|system|device)\s+of\s+claim\s+\d+)",
    re.IGNORECASE,
)
CLAIM_NUM_RE = re.compile(r"^\s*\d{1,3}\.\s")

# OCR engine singletons keyed by language — lazy-initialised on first use
_ocr_engines: Dict[str, object] = {}


def _get_ocr_engine(lang: str = "en"):
    if lang not in _ocr_engines:
        log.info("Initialising PaddleOCR engine (lang=%s)…", lang)
        from paddleocr import PaddleOCR
        _ocr_engines[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_engines[lang]


def _ocr_page(page: fitz.Page, src_lang: str = "en") -> str:
    import cv2
    ocr_lang = "ch" if src_lang == "zh" else "en"
    mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    result = _get_ocr_engine(ocr_lang).ocr(img, cls=True)
    lines: List[str] = []
    if result:
        for block in result:
            if block:
                for line in block:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        tc = line[1]
                        if isinstance(tc, (list, tuple)) and tc:
                            lines.append(str(tc[0]))
    return " ".join(lines)


def extract_page_text(page: fitz.Page, src_lang: str = "en") -> str:
    native = page.get_text("text").strip()
    if len(native) >= MIN_NATIVE_CHARS:
        return native
    log.info("Page %d: sparse text (%d chars) — OCR fallback", page.number + 1, len(native))
    try:
        return _ocr_page(page, src_lang)
    except Exception as exc:
        log.warning("OCR failed: %s", exc)
        return native


def determine_section_type(text: str) -> str:
    s = text.strip()
    if CLAIM_INDEP_RE.match(s):
        return "claim_independent"
    if CLAIM_DEP_RE.search(s):
        return "claim_dependent"
    if CLAIM_NUM_RE.match(s):
        return "claim_dependent"
    return "description"


def split_into_chunks(full_text: str) -> List[Dict[str, str]]:
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
