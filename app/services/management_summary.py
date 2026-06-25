"""
app/services/management_summary.py — One-page Management Summary PDF.

Condenses a Risk Analysis + Design Improvements + Innovation run into a single
printable page for management review: the original idea, the single highest
risk found, the single best design-around, and the highest-opportunity
innovation gaps. Any of the three source results may be missing (e.g. the user
only ran Risk Analysis and Innovation) — the affected section then renders a
short "not available" note instead of failing.

Built with PyMuPDF (already a dependency for PDF ingestion) rather than adding
a new templating/rendering library — a fixed one-page layout with a handful of
text blocks doesn't need HTML/CSS rendering, and `insert_textbox` clips text
that doesn't fit instead of overflowing, which is what keeps this at exactly
one page by construction (no second page is ever created).

Functions:
  build_summary_pdf(...) -> bytes   — renders the one-pager, returns raw PDF bytes
  save_summary(...)      -> dict    — inserts a row (metadata + pdf_data bytea)
  list_summaries()        -> list    — newest-first, metadata only (no pdf_data)
  get_summary(id)         -> dict|None — one row including pdf_data
  delete_summary(id)      -> None
"""
import datetime
import logging
from typing import Any, Dict, List, Optional

import fitz

from app.state import state

log = logging.getLogger(__name__)

# Brand colours (see --accent / --text / --text3 in templates/base.html),
# expressed as PyMuPDF's 0-1 float RGB tuples rather than CSS hex strings.
_ACCENT = (0 / 255, 96 / 255, 169 / 255)
_TEXT   = (0x0F / 255, 0x17 / 255, 0x2A / 255)
_TEXT3  = (0x64 / 255, 0x74 / 255, 0x8B / 255)
_BORDER = (0.85, 0.87, 0.9)

_PAGE_W, _PAGE_H = fitz.paper_size("a4")
_MARGIN = 40


_SECTION_TITLE_H = 17  # min. ~16 needed for a single fontsize-9 line — insert_textbox
                        # draws nothing at all (not even a clipped line) if the box is
                        # even slightly too short for one full line, so these heights
                        # are deliberately generous rather than tightly fitted.


def _section(page: fitz.Page, y: float, title: str, body: str, height: float) -> float:
    """Draws one section (accent-coloured heading + body text clipped to `height`)
    and returns the y-coordinate immediately below it, for the next section."""
    page.insert_textbox(
        fitz.Rect(_MARGIN, y, _PAGE_W - _MARGIN, y + _SECTION_TITLE_H),
        title.upper(), fontsize=9, fontname="helv", color=_ACCENT,
    )
    page.draw_line(
        (_MARGIN, y + _SECTION_TITLE_H + 1), (_PAGE_W - _MARGIN, y + _SECTION_TITLE_H + 1),
        color=_BORDER, width=0.6,
    )
    body_y = y + _SECTION_TITLE_H + 6
    page.insert_textbox(
        fitz.Rect(_MARGIN, body_y, _PAGE_W - _MARGIN, body_y + height),
        body, fontsize=9.5, fontname="helv", color=_TEXT, lineheight=1.4,
    )
    return body_y + height + 14


def _pick_top_risk(risk_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    assessments = (risk_result or {}).get("patent_assessments") or []
    if not assessments:
        return None
    return max(assessments, key=lambda a: a.get("risk_score", 0))


def _pick_best_design(design_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    suggestions = (design_result or {}).get("suggestions") or []
    if not suggestions:
        return None
    # CLEAR ranks above LOW (the only two labels a surviving proposal can have);
    # ties keep the designer agent's own ordering (first one wins).
    rank = {"CLEAR": 0, "LOW": 1}
    return min(suggestions, key=lambda s: rank.get(s.get("risk_score", "LOW"), 1))


def _pick_top_gaps(innovation_result: Optional[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    gaps = (innovation_result or {}).get("gaps") or []
    return [g for g in gaps if g.get("opportunity_level") == "HIGH"][:limit]


def build_summary_pdf(
    product_id:        str,
    component_scope:   str,
    domain:             str,
    risk_result:        Optional[Dict[str, Any]],
    design_result:      Optional[Dict[str, Any]],
    innovation_result:  Optional[Dict[str, Any]],
) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=_PAGE_W, height=_PAGE_H)

    # Header
    page.insert_textbox(
        fitz.Rect(_MARGIN, _MARGIN, _PAGE_W - _MARGIN, _MARGIN + 32),
        "Management Summary", fontsize=18, fontname="hebo", color=_TEXT,
    )
    meta = f"{product_id or 'Unnamed product'}" + (f"  ·  {domain}" if domain else "")
    meta += "  ·  " + datetime.date.today().strftime("%d %b %Y")
    page.insert_textbox(
        fitz.Rect(_MARGIN, _MARGIN + 34, _PAGE_W - _MARGIN, _MARGIN + 50),
        meta, fontsize=9.5, fontname="helv", color=_TEXT3,
    )
    page.draw_line(
        (_MARGIN, _MARGIN + 56), (_PAGE_W - _MARGIN, _MARGIN + 56), color=_ACCENT, width=1.4,
    )

    y = _MARGIN + 70

    # 1 — Original idea
    idea_body = component_scope.strip() if component_scope.strip() else "No design specification on file."
    y = _section(page, y, "Original Idea", idea_body[:420], height=44)

    # 2 — Highest risk
    top_risk = _pick_top_risk(risk_result)
    if top_risk:
        risk_body = (
            f"{top_risk.get('patent_number', '?')}  ·  risk score {top_risk.get('risk_score', 0)}/100\n"
            f"{(top_risk.get('overlap_explanation') or '').strip()[:380]}"
        )
    else:
        risk_body = "No risk analysis on file."
    y = _section(page, y, "Highest Risk Identified", risk_body, height=58)

    # 3 — Best design improvement
    best_design = _pick_best_design(design_result)
    if best_design:
        design_body = (
            f"Risk after revision: {best_design.get('risk_score', '?')}\n"
            f"{(best_design.get('description') or '').strip()[:380]}"
        )
    else:
        design_body = "No design improvement on file."
    y = _section(page, y, "Recommended Design Improvement", design_body, height=58)

    # 4 — Innovation opportunities
    top_gaps = _pick_top_gaps(innovation_result)
    if top_gaps:
        # "helv" (base-14 Helvetica) has no glyph for U+2022 (•) — renders as a
        # missing-glyph box/"?" instead. A plain hyphen avoids that entirely.
        gaps_body = "\n".join(
            f"- {g.get('area', '?')} - {(g.get('description') or '').strip()[:160]}"
            for g in top_gaps
        )
    else:
        gaps_body = "No high-opportunity innovation gaps on file."
    _section(page, y, "Innovation Opportunities (High Priority)", gaps_body, height=70)

    return doc.tobytes()


# ---------------------------------------------------------------------------
# Persistence — save / list / get / delete (mirrors innovation_analyses)
# ---------------------------------------------------------------------------

def save_summary(product_id: str, domain: str, pdf_bytes: bytes) -> Dict[str, Any]:
    """Insert a generated summary. Returns the new row (id + created_at)."""
    row = (
        state.supabase.table("management_summaries")
        .insert({
            "product_id": product_id,
            "domain":     domain,
            "pdf_data":   "\\x" + pdf_bytes.hex(),
        })
        .execute()
        .data[0]
    )
    log.info("save_summary: saved id=%s product_id=%r", row.get("id"), product_id)
    return row


def list_summaries() -> List[Dict[str, Any]]:
    """Return all saved summaries newest-first, without the PDF bytes."""
    return (
        state.supabase.table("management_summaries")
        .select("id,created_at,product_id,domain")
        .order("created_at", desc=True)
        .execute()
        .data or []
    )


def get_summary(summary_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one summary including its PDF bytes. Returns None if not found."""
    rows = (
        state.supabase.table("management_summaries")
        .select("*")
        .eq("id", summary_id)
        .execute()
        .data or []
    )
    return rows[0] if rows else None


def delete_summary(summary_id: str) -> None:
    state.supabase.table("management_summaries").delete().eq("id", summary_id).execute()
    log.info("delete_summary: deleted id=%s", summary_id)
