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

from app.services.retrieval import _score_to_label
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


_SECTION_TITLE_H = 20  # min. ~19 needed for a single fontsize-10.5 line — insert_textbox
                        # draws nothing at all (not even a clipped line) if the box is
                        # even slightly too short for one full line, so these heights
                        # are deliberately generous rather than tightly fitted.
_BODY_FONTSIZE = 11
_BODY_LINEHEIGHT = 1.4


def _wrapped_line_count(text: str, fontsize: float, width: float, fontname: str = "helv") -> int:
    """Estimates how many lines `text` will wrap into inside `width` points,
    mirroring insert_textbox's own greedy word-wrap (measured via
    fitz.get_text_length) since PyMuPDF doesn't expose the wrapped line count
    directly. Used to size each section's body box to its actual content
    instead of a fixed worst-case height — the latter left a large, varying
    gap before the next heading whenever the real text was shorter."""
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


def _section(page: fitz.Page, y: float, title: str, body: str) -> float:
    """Draws one section (accent-coloured heading + body text sized to fit
    `body` exactly) and returns the y-coordinate immediately below it, for
    the next section."""
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
    n_lines = _wrapped_line_count(body, _BODY_FONTSIZE, width)
    # +6 buffer: empirically the smallest margin that still clears
    # insert_textbox's silent "box too short -> draws nothing" failure
    # across 1-6 line bodies (see module docstring / _SECTION_TITLE_H above).
    height = n_lines * _BODY_FONTSIZE * _BODY_LINEHEIGHT + 6
    page.insert_textbox(
        fitz.Rect(_MARGIN, body_y, _PAGE_W - _MARGIN, body_y + height),
        body, fontsize=_BODY_FONTSIZE, fontname="helv", color=_TEXT, lineheight=_BODY_LINEHEIGHT,
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


def _first_sentence(text: str, max_chars: int = 105) -> str:
    """Returns the first sentence of `text` (up to and including its period).
    If that sentence doesn't fit within `max_chars`, cuts at the last comma
    within budget instead of an arbitrary word — descriptions here are
    typically "<main clause>, <trailing participial clause>." (e.g. "Use X
    instead of Y, avoiding Z entirely."), so cutting at the comma keeps the
    complete main clause rather than landing mid-clause on a dangling word
    like "claimed" or "the". Only falls back to a plain word-boundary cut if
    no comma is found either. Always ends on a real period — never "...".
    max_chars=105 keeps the full headline (this plus its fixed prefix) inside
    3 lines at the headline's fontsize — see build_summary_pdf."""
    text = text.strip()
    cut = text.find(". ")
    sentence = text[: cut + 1] if cut != -1 else text
    if len(sentence) <= max_chars:
        return sentence

    window = sentence[:max_chars]
    comma = window.rfind(",")
    if comma > max_chars * 0.4:
        return window[:comma].rstrip() + "."
    return window.rsplit(" ", 1)[0].rstrip(",;:. ") + "."


def _build_headline(
    top_risk:    Optional[Dict[str, Any]],
    best_design: Optional[Dict[str, Any]],
) -> str:
    """
    Plain-language, at-a-glance summary of the whole page for the big bold
    headline — deliberately free of patent numbers/product codes (not useful
    to a non-technical reader at a glance) and always ends on a real sentence
    so nothing reads as cut off.
    """
    if not top_risk:
        return "No significant patent risk was identified for this design."

    risk_label = _score_to_label(top_risk.get("risk_score", 0))
    if not best_design:
        return f"This design carries {risk_label} patent risk, with no validated design-around on file yet."

    fix = _first_sentence((best_design.get("description") or "").strip())
    return f"This design carries {risk_label} patent risk. Recommended fix: {fix}"


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

    top_risk    = _pick_top_risk(risk_result)
    best_design = _pick_best_design(design_result)

    # Header — the actual one-sentence management summary as the big bold
    # headline (this is the point of the whole page), with "Fuyao Patent OS"
    # demoted to a small grey meta line underneath alongside product/domain/
    # date. The headline's height is computed from its actual wrapped line
    # count (via _wrapped_line_count) rather than a fixed guess — a fixed
    # height previously went silently blank in production whenever the real
    # data produced a longer headline than the test cases it was sized for
    # (insert_textbox draws nothing at all if the box is even slightly too
    # short — see _SECTION_TITLE_H above). _first_sentence's max_chars=105
    # keeps the headline at 3 lines in the overwhelming majority of cases,
    # but this still degrades gracefully (grows the box) rather than going
    # blank if some edge case ever produces a 4th line.
    headline_w = _PAGE_W - 2 * _MARGIN
    headline = _build_headline(top_risk, best_design)
    headline_lines = _wrapped_line_count(headline, 17, headline_w, fontname="hebo")
    headline_h = headline_lines * 17 * 1.3 + 8
    page.insert_textbox(
        fitz.Rect(_MARGIN, _MARGIN, _PAGE_W - _MARGIN, _MARGIN + headline_h),
        headline, fontsize=17, fontname="hebo", color=_TEXT, lineheight=1.3,
    )

    # product_id/domain are free-text form fields with no length cap upstream,
    # so they're capped here too — long values get a 2-line allowance below,
    # but nothing should be able to grow this past that.
    meta = f"Fuyao Patent OS · {(product_id or 'Unnamed product').strip()[:40]}"
    if domain:
        meta += f" · {domain.strip()[:40]}"
    meta += " · " + datetime.date.today().strftime("%d %b %Y")
    meta_y = _MARGIN + headline_h + 4
    meta_lines = _wrapped_line_count(meta, 11, headline_w)
    meta_h = meta_lines * 11 * 1.3 + 6
    page.insert_textbox(
        fitz.Rect(_MARGIN, meta_y, _PAGE_W - _MARGIN, meta_y + meta_h),
        meta, fontsize=11, fontname="helv", color=_TEXT3, lineheight=1.3,
    )
    divider_y = meta_y + meta_h + 10
    page.draw_line(
        (_MARGIN, divider_y), (_PAGE_W - _MARGIN, divider_y), color=_ACCENT, width=1.4,
    )

    y = divider_y + 14

    # 1 — Original idea
    idea_body = component_scope.strip() if component_scope.strip() else "No design specification on file."
    y = _section(page, y, "Original Idea", idea_body[:420])

    # 2 — Highest risk
    if top_risk:
        risk_body = (
            f"{top_risk.get('patent_number', '?')} · risk score {top_risk.get('risk_score', 0)}/100\n"
            f"{(top_risk.get('overlap_explanation') or '').strip()[:380]}"
        )
    else:
        risk_body = "No risk analysis on file."
    y = _section(page, y, "Highest Risk Identified", risk_body)

    # 3 — Best design improvement
    if best_design:
        design_body = (
            f"Risk after revision: {best_design.get('risk_score', '?')}\n"
            f"{(best_design.get('description') or '').strip()[:380]}"
        )
    else:
        design_body = "No design improvement on file."
    y = _section(page, y, "Recommended Design Improvement", design_body)

    # 4 — Innovation opportunities
    top_gaps = _pick_top_gaps(innovation_result)
    if top_gaps:
        # "helv" (base-14 Helvetica) has no glyph for U+2022 (•) — renders as a
        # missing-glyph box/"?" instead. A plain hyphen-bullet avoids that; the
        # area/description separator is a colon rather than a second hyphen so
        # the line doesn't read as "- X - Y".
        gaps_body = "\n".join(
            f"- {g.get('area', '?')}: {(g.get('description') or '').strip()[:160]}"
            for g in top_gaps
        )
    else:
        gaps_body = "No high-opportunity innovation gaps on file."
    _section(page, y, "Innovation Opportunities", gaps_body)

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
            "summary":    "",
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
