"""
app/services/patent_sheet.py — Per-patent one-page Analysis Sheet PDF.

A different PDF from the multi-patent Management Summary in
management_summary.py: this one is *per patent* and follows the standard
Fuyao-style analysis layout —

  Header:  bibliographic block (Patent No., Patentee, Application Date,
           Legal Status, Countries) — styled to match management_summary.py
           (accent divider, same fonts, same margins)
  Body:    Technical Issue, Technical Solution, Patent Assessment, Risk Analysis

Fed by the LLM output from GET /api/v1/patents/{id}/summary (the same call
that drives the Patent Library "Generate Summary" card) — no schema changes,
no re-ingestion. Missing fields render as "—" rather than going blank.

Functions:
  build_patent_sheet_pdf(doc, summary) -> bytes  — renders one-page PDF
"""
import logging
from typing import Any, Dict

import fitz

log = logging.getLogger(__name__)

# Brand palette — IDENTICAL values + variable names to management_summary.py,
# so both PDFs render visually consistent if the brand colours ever shift.
_ACCENT = (0 / 255, 96 / 255, 169 / 255)
_TEXT   = (0x0F / 255, 0x17 / 255, 0x2A / 255)
_TEXT3  = (0x64 / 255, 0x74 / 255, 0x8B / 255)
_BORDER = (0.85, 0.87, 0.9)

_PAGE_W, _PAGE_H = fitz.paper_size("a4")
_MARGIN = 40

# Match management_summary.py's section sizing exactly: title box >=19pt at
# fontsize 10.5 (insert_textbox draws NOTHING if even slightly short, see the
# warning in management_summary.py — that's what silently dropped the section
# headings in the previous version of this file).
_SECTION_TITLE_H = 20
_BODY_FONTSIZE   = 11
_BODY_LINEHEIGHT = 1.4


def _wrapped_line_count(text: str, fontsize: float, width: float, fontname: str = "helv") -> int:
    """Mirrors insert_textbox's greedy word-wrap so each body box can be sized
    to its actual content. See same-named helper in management_summary.py."""
    total = 0
    for paragraph in text.split("\n"):
        if not paragraph:
            total += 1
            continue
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fontsize) <= width:
                line = trial
            else:
                total += 1
                line = word
        total += 1
    return max(total, 1)


def _fallback(value: Any) -> str:
    """Missing/empty fields render as 'n/a'. We deliberately avoid the em-dash
    (U+2014) here even though it reads better — PyMuPDF's built-in 'helv' font
    is base-14 Helvetica, which has no glyph for U+2014 and renders it as a
    missing-glyph "?" box instead. Same issue management_summary.py worked
    around for the U+2022 bullet. Whitespace-only counts as missing too — the
    LLM occasionally returns ' ' for fields it can't find."""
    if value is None:
        return "n/a"
    s = str(value).strip()
    return s if s else "n/a"


def _draw_header(page: fitz.Page, doc: Dict[str, Any], summary: Dict[str, Any]) -> float:
    """Top-of-page bibliographic block — 5 mandated fields rendered as a clean
    label/value list (no boxed background, to stay visually consistent with the
    Management Summary header), followed by the accent divider line that
    management_summary.py uses between header and body."""

    fields = [
        ("Patent No.",       _fallback(doc.get("patent_number"))),
        ("Patentee",         _fallback(doc.get("assignee"))),
        ("Application Date", _fallback(summary.get("application_date"))),
        ("Legal Status",     _fallback(summary.get("legal_status"))),
        # Countries: prefer LLM-extracted designated states; fall back to the
        # single-country jurisdiction column if LLM returned nothing.
        ("Countries",        _fallback(summary.get("countries") or doc.get("jurisdiction"))),
    ]

    label_w   = 110
    value_w   = _PAGE_W - 2 * _MARGIN - label_w
    # Row height is now PER ROW, sized to the wrapped value text. The previous
    # fixed row_h=20 clipped long values (e.g. "GLOBALFOUNDRIES INC., Grand
    # Cayman" → "...Grand"). Single-line min stays 20pt to clear
    # insert_textbox's silent "box too short -> draws nothing" threshold for
    # 10.5pt bold; multi-line rows expand from there.
    y = _MARGIN

    for label, value in fields:
        n_lines = _wrapped_line_count(value, 10.5, value_w, fontname="hebo")
        row_h = max(20.0, n_lines * 10.5 * 1.35 + 4)
        # Label in accent (matches the section-heading colour in
        # management_summary.py for a consistent visual language).
        page.insert_textbox(
            fitz.Rect(_MARGIN, y, _MARGIN + label_w, y + row_h),
            label.upper(), fontsize=9.5, fontname="helv", color=_ACCENT,
        )
        page.insert_textbox(
            fitz.Rect(_MARGIN + label_w, y, _PAGE_W - _MARGIN, y + row_h),
            value, fontsize=10.5, fontname="hebo", color=_TEXT, lineheight=1.35,
        )
        y += row_h

    # Title beneath the bibliography (smaller, grey) — gives the reader context
    # without competing with the headline. Same colour role as the meta line in
    # management_summary.py.
    title = _fallback(doc.get("title"))
    title_w = _PAGE_W - 2 * _MARGIN
    title_lines = _wrapped_line_count(title, 10.5, title_w)
    title_h = title_lines * 10.5 * 1.3 + 4
    page.insert_textbox(
        fitz.Rect(_MARGIN, y + 4, _PAGE_W - _MARGIN, y + 4 + title_h),
        title, fontsize=10.5, fontname="heit", color=_TEXT3, lineheight=1.3,
    )
    y += 4 + title_h + 10

    # Accent divider — same width/colour/y-offset pattern as in
    # management_summary.py's header→body separator.
    page.draw_line(
        (_MARGIN, y), (_PAGE_W - _MARGIN, y), color=_ACCENT, width=1.4,
    )
    return y + 14


def _section(page: fitz.Page, y: float, title: str, body: str) -> float:
    """One labelled body block — pixel-identical layout to
    management_summary.py._section (same title fontsize, font, accent colour,
    border line, +6 buffer on the body height) so the two PDFs render the same
    section style. Returns the y of the next section."""
    page.insert_textbox(
        fitz.Rect(_MARGIN, y, _PAGE_W - _MARGIN, y + _SECTION_TITLE_H),
        title.upper(), fontsize=10.5, fontname="helv", color=_ACCENT,
    )
    page.draw_line(
        (_MARGIN, y + _SECTION_TITLE_H + 1), (_PAGE_W - _MARGIN, y + _SECTION_TITLE_H + 1),
        color=_BORDER, width=0.6,
    )
    body_y = y + _SECTION_TITLE_H + 6
    width = _PAGE_W - 2 * _MARGIN
    body = _fallback(body)
    n_lines = _wrapped_line_count(body, _BODY_FONTSIZE, width)
    # +6 buffer matches management_summary.py — empirically the smallest margin
    # that still clears insert_textbox's silent "box too short -> draws
    # nothing" mode across 1–8 line bodies.
    height = n_lines * _BODY_FONTSIZE * _BODY_LINEHEIGHT + 6
    page.insert_textbox(
        fitz.Rect(_MARGIN, body_y, _PAGE_W - _MARGIN, body_y + height),
        body, fontsize=_BODY_FONTSIZE, fontname="helv", color=_TEXT,
        lineheight=_BODY_LINEHEIGHT,
    )
    return body_y + height + 14


def build_patent_sheet_pdf(doc: Dict[str, Any], summary: Dict[str, Any]) -> bytes:
    """Render the per-patent Analysis Sheet as raw PDF bytes.

    `doc` is one row of patent_documents (with patent_number, title, assignee,
    jurisdiction, publication_date). `summary` is the JSON dict from the
    analyst LLM call in api.py:_generate_patent_summary. Both are forgiving:
    any missing field renders as "—" rather than throwing."""
    doc = doc or {}
    summary = summary or {}

    pdf = fitz.open()
    page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)

    y = _draw_header(page, doc, summary)

    y = _section(page, y, "Technical Issue",    summary.get("technical_issue"))
    y = _section(page, y, "Technical Solution", summary.get("technical_solution"))
    y = _section(page, y, "Patent Assessment",  summary.get("patent_assessment"))
    _section(page, y,     "Risk Analysis",      summary.get("risk_analysis"))

    return pdf.tobytes()
