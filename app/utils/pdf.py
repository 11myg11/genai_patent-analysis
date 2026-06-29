"""
app/utils/pdf.py — PDF text extraction, OCR fallback, paragraph chunking, and figure extraction.

Handles the full pipeline from a fitz.Page object to labelled text chunks and figure images
ready for storage. Used exclusively by app/services/ingest.py.

OCR backend: pytesseract (wraps system Tesseract). Install with:
  brew install tesseract && pip install pytesseract
For non-English patents install the matching language pack, e.g.:
  brew install tesseract-lang   (all languages)

Global variables:
  CLAIM_INDEP_RE / CLAIM_DEP_RE / CLAIM_NUM_RE — Pre-compiled regexes for classifying
    a paragraph as an independent claim, dependent claim, or description section.
  _text_splitter — LangChain RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)

Functions:
  extract_page_text(page, src_lang) -> str
    Extracts text from a fitz.Page. Uses native PyMuPDF text first; falls back to
    Tesseract OCR if the native result is shorter than MIN_NATIVE_CHARS (50 chars),
    which indicates a scanned/image-only page.

  determine_section_type(text) -> str
    Classifies a paragraph as "claim_independent", "claim_dependent", or "description"
    based on regex patterns. Used to label chunks stored in Supabase.

  split_into_chunks(full_text) -> list[dict]
    Splits full document text using RecursiveCharacterTextSplitter (500 chars, 50 overlap),
    then labels each chunk via determine_section_type. Chunks shorter than 10 chars are dropped.

  extract_figure_pages(doc) -> list[dict]
    Iterates all pages in the document and renders qualifying pages as PNG images.
    A page qualifies as a figure page if it has embedded images OR its text content
    is below FIGURE_TEXT_THRESHOLD (200 chars) — the latter catches vector drawings
    which page.get_images() cannot detect. Returns dicts with page_number, width,
    height, and image_data (PNG bytes).
"""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import PAGE_DPI, MIN_NATIVE_CHARS, FIGURE_DPI, FIGURE_TEXT_THRESHOLD

log = logging.getLogger(__name__)

# Resolve tessdata once at import time so every _ocr_page call can pass it explicitly.
# PyMuPDF 1.23.x raises "TESSDATA_PREFIX not set" without this; 1.24+ auto-discovers it.
def _find_tessdata() -> Optional[str]:
    # Honour explicit env var first
    if os.environ.get("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    tess_bin = shutil.which("tesseract")
    if tess_bin:
        candidate = Path(tess_bin).parent.parent / "share" / "tessdata"
        if candidate.is_dir():
            return str(candidate)
    return None

_TESSDATA: Optional[str] = _find_tessdata()

CLAIM_INDEP_RE = re.compile(r"^\s*1\.\s", re.IGNORECASE)
CLAIM_DEP_RE = re.compile(
    r"(\bclaim\s+\d+\b.*\bwherein\b|\bwherein\b|"
    r"the\s+(?:method|apparatus|system|device)\s+of\s+claim\s+\d+)",
    re.IGNORECASE,
)
CLAIM_NUM_RE = re.compile(r"^\s*\d{1,3}\.\s")

# Used by split_into_chunks to detect a numbered claim boundary regardless of
# surrounding whitespace — e.g. "7. The laminate comprising…"
CLAIM_BOUNDARY_RE = re.compile(r"(?=^\s*\d{1,3}\.\s)", re.MULTILINE)

# Matches numbered claims that reference another claim → dependent
# e.g. "7. The laminate of claim 1, wherein..." or "3. The method of claim 2..."
CLAIM_DEP_REF_RE = re.compile(
    r"^\s*\d{1,3}\.\s.*\bof\s+claim\s+\d+\b",
    re.IGNORECASE,
)

_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Map ISO 639-1 codes → Tesseract language codes (only non-obvious ones needed)
_TESS_LANG: Dict[str, str] = {
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "ru": "rus",
    "nl": "nld",
    "pt": "por",
    "it": "ita",
}


def _ocr_page(page: fitz.Page, src_lang: str = "en") -> str:
    """Run Tesseract OCR via PyMuPDF's built-in get_textpage_ocr.

    Passes tessdata path explicitly — required by PyMuPDF <=1.23.x which raises
    'TESSDATA_PREFIX not set' without it. Harmless on newer versions.
    """
    tess_lang = _TESS_LANG.get(src_lang.split("-")[0], "eng")
    tp = page.get_textpage_ocr(language=tess_lang, dpi=PAGE_DPI, full=True, tessdata=_TESSDATA)
    return page.get_text(textpage=tp).strip()


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
    """
    Classify a chunk as claim_independent, claim_dependent, or description.

    Priority order:
    1. If the chunk references another claim ("of claim N") → dependent
    2. If it contains "wherein" → dependent
    3. If it starts with a number + period (any claim number) and is NOT
       a back-reference → independent claim
    4. Everything else → description
    """
    s = text.strip()

    # Rule 1 — numbered claim that explicitly references another claim → dependent
    if CLAIM_DEP_REF_RE.match(s):
        return "claim_dependent"

    # Rule 2 — contains "wherein" (classic dependent claim language) → dependent
    if CLAIM_DEP_RE.search(s):
        return "claim_dependent"

    # Rule 3 — starts with "N. " and no back-reference → independent claim
    if CLAIM_NUM_RE.match(s):
        return "claim_independent"

    # Rule 4 — everything else → description
    return "description"


def split_into_chunks(full_text: str) -> List[Dict[str, str]]:
    """
    Split document text into labelled chunks using LangChain's
    RecursiveCharacterTextSplitter (500 chars, 50 overlap), then classify
    each chunk with determine_section_type so the risk pipeline can distinguish
    independent claims from dependent claims and description text.
    """
    raw_chunks = _text_splitter.split_text(full_text)
    result: List[Dict[str, str]] = []
    for text in raw_chunks:
        text = text.strip()
        if len(text) < 10:
            continue
        result.append({
            "section_type": determine_section_type(text),
            "content":      text,
        })
    return result


def extract_figure_pages(doc: fitz.Document) -> List[Dict]:
    """Render qualifying pages as PNG and return their raw bytes.

    A page qualifies if it has embedded images (raster figures) OR if its text
    content is below FIGURE_TEXT_THRESHOLD, which catches pages that consist
    entirely of vector drawings — those are invisible to page.get_images().
    """
    figures: List[Dict] = []
    mat = fitz.Matrix(FIGURE_DPI / 72, FIGURE_DPI / 72)

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            has_images = len(page.get_images(full=True)) > 0
            # Byte length (not code-point length) so CJK characters (3 bytes each)
            # are weighted correctly — a 70-char Chinese paragraph is ~210 bytes,
            # safely above the 200-byte threshold that filters out figure labels.
            text_len = len(page.get_text("text").strip().encode("utf-8"))
            is_figure_page = has_images or text_len < FIGURE_TEXT_THRESHOLD
            if not is_figure_page:
                continue

            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            image_bytes = pix.tobytes("png")
            figures.append({
                "page_number": page_num + 1,  # 1-based for display
                "width":       pix.width,
                "height":      pix.height,
                "image_data":  image_bytes,
            })
            log.debug("Figure page %d: %dx%d %d bytes", page_num + 1, pix.width, pix.height, len(image_bytes))
        except Exception as exc:
            log.warning("Figure extraction failed on page %d: %s", page_num + 1, exc)

    return figures
