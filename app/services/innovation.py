"""
app/services/innovation.py — Phase 4: Innovation opportunity analysis across the patent corpus.

Pipeline (run_innovation_pipeline):
  Step 1 — fetch_corpus_overview: query up to MAX_CORPUS_PATENTS patents + representative chunks,
            ranked by semantic relevance to the domain (or most recent if no domain given).
  Step 2 — extract_trend_data: group publication_date by year (pure DB, no LLM).
  Step 3 — call_agent_analyst: cluster patents into technology groups, identify whitespace gaps.
  Step 4 — call_agent_innovator: generate actionable innovation vectors from the gaps.

Persistence functions (save / list / get / delete):
  save_analysis         — insert a completed analysis into innovation_analyses table
  list_saved_analyses   — fetch all saved analyses as summaries (no heavy JSONB in list)
  get_saved_analysis    — fetch one full saved analysis by id
  delete_saved_analysis — delete one saved analysis by id

Key constants:
  MAX_CORPUS_PATENTS    — cap on how many patents enter the pipeline (token budget)
  MAX_CHUNKS_PER_PATENT — representative chunks fetched per patent for the analyst prompt
  MAX_CLAIM_CHARS       — each chunk is truncated to this length before being sent to the LLM

Scope parameter controls which section_types are fetched per patent:
  "full"        → claim_independent + claim_dependent + description
  "claims"      → claim_independent + claim_dependent
  "description" → description only
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.config import GLASS_TOTAL_MIN, GLASS_TOTAL_MAX, PVB_MIN_MM, PVB_MAX_MM
from app.services.llm import llm_json
from app.state import state

log = logging.getLogger(__name__)

MAX_CORPUS_PATENTS    = 30
MAX_CHUNKS_PER_PATENT = 2
MAX_CLAIM_CHARS       = 400

SCOPE_SECTION_TYPES: Dict[str, List[str]] = {
    "full":        ["claim_independent", "claim_dependent", "description"],
    "claims":      ["claim_independent", "claim_dependent"],
    "description": ["description"],
}


# ---------------------------------------------------------------------------
# Step 1 — Corpus preparation
# ---------------------------------------------------------------------------

def fetch_corpus_overview(
    jurisdiction: str,
    scope: str,
    domain_embedding: Optional[List[float]],
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Fetch patent metadata and representative claim/description chunks.

    If domain_embedding is provided, hybrid search ranks patents by semantic
    relevance to the domain first; otherwise falls back to most recent patents.
    Caps at MAX_CORPUS_PATENTS to keep the analyst prompt within token budget.
    """
    filter_jx     = None if (not jurisdiction or jurisdiction.upper() == "ALL") else jurisdiction
    section_types = SCOPE_SECTION_TYPES.get(scope, SCOPE_SECTION_TYPES["full"])

    if domain_embedding and domain.strip():
        resp   = state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     domain_embedding,
            "query_text":          domain,
            "filter_jurisdiction": filter_jx,
            "match_count":         MAX_CORPUS_PATENTS * 3,
        }).execute()
        chunks = resp.data or []

        # Deduplicate patent IDs in relevance order
        seen_ids: List[str] = []
        for c in chunks:
            pid = c.get("patent_id")
            if pid and pid not in seen_ids:
                seen_ids.append(pid)
            if len(seen_ids) >= MAX_CORPUS_PATENTS:
                break

        if seen_ids:
            rows    = (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,assignee,jurisdiction,publication_date")
                .in_("id", seen_ids)
                .execute()
                .data or []
            )
            id_map  = {d["id"]: d for d in rows}
            patent_docs = [id_map[pid] for pid in seen_ids if pid in id_map]
        else:
            patent_docs = []
    else:
        q = (
            state.supabase.table("patent_documents")
            .select("id,patent_number,title,assignee,jurisdiction,publication_date")
        )
        if filter_jx:
            q = q.eq("jurisdiction", filter_jx)
        patent_docs = q.order("created_at", desc=True).limit(MAX_CORPUS_PATENTS).execute().data or []

    overviews: List[Dict[str, Any]] = []
    for doc in patent_docs:
        chunk_rows = (
            state.supabase.table("patent_chunks")
            .select("section_type,content")
            .eq("patent_id", doc["id"])
            .in_("section_type", section_types)
            .limit(MAX_CHUNKS_PER_PATENT)
            .execute()
            .data or []
        )
        # Prioritise independent claims first by sorting before joining
        ordered = sorted(chunk_rows, key=lambda c: (
            0 if c.get("section_type") == "claim_independent" else
            1 if c.get("section_type") == "claim_dependent" else 2
        ))
        claim_summary = " | ".join(c["content"][:MAX_CLAIM_CHARS] for c in ordered)
        overviews.append({
            "id":               doc.get("id", ""),
            "patent_number":    doc.get("patent_number", ""),
            "title":            doc.get("title", ""),
            "assignee":         doc.get("assignee", ""),
            "publication_date": doc.get("publication_date", ""),
            "claim_summary":    claim_summary or "(no chunks indexed)",
        })

    log.info("Step 1 — corpus overview: %d patents (scope=%s jx=%s)", len(overviews), scope, filter_jx or "ALL")
    return overviews


# ---------------------------------------------------------------------------
# Step 2 — Trend data (no LLM)
# ---------------------------------------------------------------------------

def extract_trend_data(jurisdiction: str) -> List[Dict[str, int]]:
    """
    Group patent publication dates by year for the trend bar chart.
    Rows with missing or malformed dates are silently skipped.
    """
    filter_jx = None if (not jurisdiction or jurisdiction.upper() == "ALL") else jurisdiction
    q         = state.supabase.table("patent_documents").select("publication_date")
    if filter_jx:
        q = q.eq("jurisdiction", filter_jx)
    rows = q.execute().data or []

    year_counts: Dict[int, int] = {}
    for row in rows:
        date_str = (row.get("publication_date") or "").strip()
        m = re.match(r"(\d{4})", date_str)
        if m:
            year = int(m.group(1))
            year_counts[year] = year_counts.get(year, 0) + 1

    trend = [{"year": y, "count": c} for y, c in sorted(year_counts.items())]
    log.info("Step 2 — trend data: %d years with data", len(trend))
    return trend


# ---------------------------------------------------------------------------
# Step 3 — Analyst agent: clusters + gaps
# ---------------------------------------------------------------------------

def call_agent_analyst(
    patent_overviews: List[Dict[str, Any]],
    domain: str,
    focus_prompt: str,
) -> Dict[str, Any]:
    """
    LLM call 1 — Groups patents into technology clusters and identifies patent whitespace gaps.

    The LLM receives a compact JSON block of patent summaries and returns structured
    clusters (recurring themes) and gaps (unprotected adjacent areas).
    patent_count per cluster is computed server-side from the returned patent_numbers list.
    """
    corpus_block = json.dumps([
        {
            "patent_number": p["patent_number"],
            "title":         p["title"],
            "claim_summary": p["claim_summary"],
        }
        for p in patent_overviews
    ])

    context_lines = "\n".join(filter(None, [
        f"Domain: {domain}"      if domain.strip()       else "",
        f"Focus: {focus_prompt}" if focus_prompt.strip() else "",
    ]))

    prompt = (
        "You are a patent landscape analyst specialising in automotive glass technology.\n"
        "Respond ONLY with minified JSON. No markdown. No preamble.\n\n"
        'Output schema: {"clusters":[{"name":"string","summary":"1-2 sentence description",'
        '"patent_numbers":["EP1234"]}],"gaps":[{"area":"string","description":"1-2 sentence '
        'description of the gap","opportunity_level":"HIGH|MEDIUM|LOW","related_patents":["EP1234"]}]}\n\n'
        "Rules:\n"
        "- Identify 3-6 technology clusters grouping patents by recurring technical themes.\n"
        "- Identify 3-6 whitespace gaps: technical areas adjacent to the clusters NOT covered by any listed patent.\n"
        "- opportunity_level HIGH = clear commercial value + low existing IP density.\n"
        "- related_patents = patent numbers closest to the boundary of the gap.\n"
        "- Base analysis on claim language and technical substance, NOT just titles.\n\n"
        + (context_lines + "\n\n" if context_lines else "")
        + f"PATENT CORPUS:\n{corpus_block}"
    )

    result = llm_json(prompt)
    result.setdefault("clusters", [])
    result.setdefault("gaps", [])

    for cluster in result["clusters"]:
        cluster["patent_count"] = len(cluster.get("patent_numbers", []))

    log.info(
        "Step 3 — analyst: %d clusters, %d gaps",
        len(result["clusters"]), len(result["gaps"]),
    )
    return result


# ---------------------------------------------------------------------------
# Step 4 — Innovator agent: innovation vectors
# ---------------------------------------------------------------------------

def call_agent_innovator(
    clusters: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    domain: str,
    focus_prompt: str,
) -> List[Dict[str, Any]]:
    """
    LLM call 2 — Generates actionable innovation vectors grounded in the identified gaps.

    Each vector is scored for feasibility (achievability within Fuyao manufacturing)
    and novelty (distance from existing prior art).
    """
    clusters_block = json.dumps([
        {"name": c["name"], "summary": c["summary"]} for c in clusters
    ])
    gaps_block = json.dumps([
        {"area": g["area"], "description": g["description"], "opportunity_level": g["opportunity_level"]}
        for g in gaps
    ])

    context_lines = "\n".join(filter(None, [
        f"Domain: {domain}"      if domain.strip()       else "",
        f"Focus: {focus_prompt}" if focus_prompt.strip() else "",
    ]))

    prompt = (
        "You are an IP innovation strategist specialising in automotive glass.\n"
        "Respond ONLY with minified JSON. No markdown. No preamble.\n\n"
        'Output schema: {"innovations":[{"title":"short descriptive title","description":"2-3 sentences",'
        '"feasibility":"HIGH|MEDIUM|LOW","novelty":"HIGH|MEDIUM|LOW",'
        '"gap_rationale":"1 sentence: why this gap currently exists","addresses_clusters":["Cluster name"]}]}\n\n'
        "Rules:\n"
        "- Generate 3-5 innovation vectors, each grounded in one or more identified gaps.\n"
        "- feasibility HIGH = achievable with current Fuyao manufacturing; MEDIUM = requires R&D; LOW = long-term.\n"
        "- novelty HIGH = no clear prior art; MEDIUM = adjacent prior art exists; LOW = incremental.\n"
        f"- Fuyao constraints: glass stack {GLASS_TOTAL_MIN}-{GLASS_TOTAL_MAX}mm, "
        f"PVB {PVB_MIN_MM}-{PVB_MAX_MM}mm, no conductive layers in HUD zone, wedge ≤0.1 mrad.\n"
        "- addresses_clusters: list cluster names from the provided clusters exactly.\n\n"
        + (context_lines + "\n\n" if context_lines else "")
        + f"TECHNOLOGY CLUSTERS:\n{clusters_block}\n\nIDENTIFIED GAPS:\n{gaps_block}"
    )

    result     = llm_json(prompt)
    innovations = result.get("innovations", [])

    log.info("Step 4 — innovator: %d innovation vectors generated", len(innovations))
    return innovations


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_innovation_pipeline(
    domain: str,
    scope: str,
    jurisdiction: str,
    focus_prompt: str,
    domain_embedding: Optional[List[float]],
) -> Dict[str, Any]:
    """
    Full innovation analysis pipeline. Steps 1-4 in sequence.
    Returns a dict matching the InnovationResponse schema.
    """
    overviews    = fetch_corpus_overview(jurisdiction, scope, domain_embedding, domain)
    patent_count = len(overviews)
    patent_ids   = [o["id"] for o in overviews if o.get("id")]

    if not overviews:
        log.warning("run_innovation_pipeline: no patents in corpus — returning empty response")
        return {
            "domain":       domain or "General patent corpus",
            "patent_count": 0,
            "patent_ids":   [],
            "clusters":     [],
            "gaps":         [],
            "innovations":  [],
            "trend_data":   [],
        }

    trend_data      = extract_trend_data(jurisdiction)
    analyst_result  = call_agent_analyst(overviews, domain, focus_prompt)
    clusters        = analyst_result.get("clusters", [])
    gaps            = analyst_result.get("gaps", [])
    innovations     = call_agent_innovator(clusters, gaps, domain, focus_prompt)

    log.info(
        "run_innovation_pipeline complete: patents=%d clusters=%d gaps=%d innovations=%d",
        patent_count, len(clusters), len(gaps), len(innovations),
    )
    return {
        "domain":       domain or "General patent corpus",
        "patent_count": patent_count,
        "patent_ids":   patent_ids,
        "clusters":     clusters,
        "gaps":         gaps,
        "innovations":  innovations,
        "trend_data":   trend_data,
    }


# ---------------------------------------------------------------------------
# Persistence — save / list / get / delete
# ---------------------------------------------------------------------------

def save_analysis(
    domain:       str,
    scope:        str,
    jurisdiction: str,
    focus_prompt: str,
    patent_count: int,
    patent_ids:   List[str],
    result:       Dict[str, Any],
) -> Dict[str, Any]:
    """Insert a completed innovation analysis. Returns the new row (id + created_at)."""
    row = (
        state.supabase.table("innovation_analyses")
        .insert({
            "domain":       domain,
            "scope":        scope,
            "jurisdiction": jurisdiction,
            "focus_prompt": focus_prompt,
            "patent_count": patent_count,
            "patent_ids":   patent_ids,
            "result":       result,
        })
        .execute()
        .data[0]
    )
    log.info("save_analysis: saved id=%s domain=%r", row.get("id"), domain)
    return row


def list_saved_analyses() -> List[Dict[str, Any]]:
    """
    Return all saved analyses ordered newest-first.
    Fetches result JSONB only to compute cluster/gap/innovation counts;
    the heavy payload is not forwarded to the caller.
    patent_ids is included so the frontend can build per-patent analysis counts.
    """
    rows = (
        state.supabase.table("innovation_analyses")
        .select("id,created_at,domain,scope,jurisdiction,patent_count,patent_ids,result")
        .order("created_at", desc=True)
        .execute()
        .data or []
    )
    summaries = []
    for row in rows:
        result = row.get("result") or {}
        summaries.append({
            "id":               row["id"],
            "created_at":       row["created_at"],
            "domain":           row.get("domain", ""),
            "scope":            row.get("scope", "full"),
            "jurisdiction":     row.get("jurisdiction", "ALL"),
            "patent_count":     row.get("patent_count", 0),
            "patent_ids":       row.get("patent_ids") or [],
            "cluster_count":    len(result.get("clusters", [])),
            "gap_count":        len(result.get("gaps", [])),
            "innovation_count": len(result.get("innovations", [])),
        })
    return summaries


def get_saved_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one full saved analysis by id. Returns None if not found."""
    rows = (
        state.supabase.table("innovation_analyses")
        .select("*")
        .eq("id", analysis_id)
        .execute()
        .data or []
    )
    return rows[0] if rows else None


def delete_saved_analysis(analysis_id: str) -> None:
    """Delete a saved analysis by id."""
    state.supabase.table("innovation_analyses").delete().eq("id", analysis_id).execute()
    log.info("delete_saved_analysis: deleted id=%s", analysis_id)
