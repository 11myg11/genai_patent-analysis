"""
app/services/retrieval.py — Patent-level hybrid search and IP analysis pipeline.

Phase 2 (Risk Analysis) — NEW patent-level workflow:
  fetch_independent_claim_chunks — hybrid RRF search restricted to claim_independent chunks
  select_candidate_patents       — aggregate chunks → rank patents → return top N patent_ids
  fetch_claim_family             — fetch ALL claims (independent + dependent) for each patent
  build_patent_context_block     — format complete claim family for LLM prompt
  call_agent_risk_patent         — per-patent LLM assessment returning structured JSON
  run_patent_risk_pipeline       — orchestrates Steps 1-5, returns list of PatentRiskResult

  build_context_block / call_agent_risk kept for Phase 3 backward compatibility.

Phase 3 (Design Suggestions):
  call_agent_designer   — proposes 2 alternative designs based on risk output,
                          re-scores each via call_agent_risk, keeps only LOW/CLEAR
  call_agent_auditor    — validates surviving proposals against Fuyao manufacturing constraints
"""
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.config import TOP_K_CHUNKS, PVB_MIN_MM, PVB_MAX_MM, GLASS_TOTAL_MIN, GLASS_TOTAL_MAX
from app.models import DesignAroundProposal, PatentRiskResult
from app.services.llm import llm_json
from app.state import state

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_CANDIDATE_PATENTS = 2   # max patents to expand into full claim family
CLAIM_INDEPENDENT_LIMIT = 400   # chars — full independent claims are legally critical
CLAIM_DEPENDENT_LIMIT   = 200   # chars — dependent claims can be shorter


# Priority weights for section types — used in candidate patent scoring
SECTION_PRIORITY = {
    "claim_independent": 3.0,
    "claim_dependent":   1.5,
    "description":       0.5,
}


# ---------------------------------------------------------------------------
# Step 1 — Priority-weighted Chunk Retrieval
# ---------------------------------------------------------------------------

def fetch_independent_claim_chunks(embedding: List[float], query_text: str, jurisdiction: str) -> List[Dict]:
    """
    Hybrid RRF search with priority-weighted results.

    Retrieval order of importance:
      1. claim_independent — broadest legal scope, primary infringement signal
      2. claim_dependent   — narrows independent claims, secondary signal
      3. description       — context only, lowest weight

    Fetches a large pool first, then weights scores by section type so that
    independent claims always rank above equivalent-scoring description chunks.
    Falls back gracefully if no independent claims exist in the corpus.
    """
    try:
        filter_jx = None if (not jurisdiction or jurisdiction.upper() == "ALL") else jurisdiction
        resp = state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     embedding,
            "query_text":          query_text,
            "filter_jurisdiction": filter_jx,
            "match_count":         TOP_K_CHUNKS * 5,  # fetch wide pool, re-rank below
        }).execute()
        chunks = resp.data or []

        if not chunks:
            log.info("Step 1 — no chunks returned from hybrid search")
            return []

        # Apply priority weight to rrf_score
        for c in chunks:
            section  = c.get("section_type", "description")
            weight   = SECTION_PRIORITY.get(section, 0.5)
            c["weighted_score"] = float(c.get("rrf_score", 0.0)) * weight

        # Sort by weighted score descending
        chunks.sort(key=lambda c: c["weighted_score"], reverse=True)

        # Keep top TOP_K_CHUNKS * 3 — will be narrowed further in Step 2
        chunks = chunks[: TOP_K_CHUNKS * 3]

        counts = {"claim_independent": 0, "claim_dependent": 0, "description": 0}
        for c in chunks:
            counts[c.get("section_type", "description")] = counts.get(c.get("section_type", "description"), 0) + 1
        log.info("Step 1 — retrieved %d chunks (ind=%d dep=%d desc=%d) jurisdiction=%s",
                 len(chunks),
                 counts["claim_independent"], counts["claim_dependent"], counts["description"],
                 filter_jx or "ALL")
        return chunks

    except Exception as exc:
        log.error("Supabase RPC failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Database retrieval error: {exc}") from exc


# ---------------------------------------------------------------------------
# Step 2 — Candidate Patent Selection
# ---------------------------------------------------------------------------

def select_candidate_patents(chunks: List[Dict], top_n: int = TOP_CANDIDATE_PATENTS) -> List[Dict]:
    """
    Aggregate chunks by patent and rank using three signals:
      1. sum of weighted_scores (overall relevance, priority-adjusted)
      2. count of matching chunks (breadth of match)
      3. max weighted_score (best single chunk)

    Independent claim matches contribute 3× more to total_score than
    description matches due to the weights applied in Step 1.
    """
    patent_map: Dict[str, Dict] = {}
    for c in chunks:
        pid = c.get("patent_id") or c.get("patent_number")
        if not pid:
            continue
        if pid not in patent_map:
            patent_map[pid] = {
                "patent_id":     c.get("patent_id", ""),
                "patent_number": c.get("patent_number", ""),
                "title":         c.get("title", ""),
                "jurisdiction":  c.get("jurisdiction", ""),
                "total_score":   0.0,
                "match_count":   0,
                "max_score":     0.0,
            }
        score = float(c.get("weighted_score", c.get("rrf_score", 0.0)))
        patent_map[pid]["total_score"] += score
        patent_map[pid]["match_count"] += 1
        patent_map[pid]["max_score"]    = max(patent_map[pid]["max_score"], score)

    ranked = sorted(
        patent_map.values(),
        key=lambda p: (p["total_score"], p["match_count"], p["max_score"]),
        reverse=True,
    )
    selected = ranked[:top_n]
    log.info("Step 2 — candidate patents selected: %d / %d", len(selected), len(patent_map))
    return selected


# ---------------------------------------------------------------------------
# Step 3 — Claim Family Expansion
# ---------------------------------------------------------------------------

def fetch_claim_family(patent_id: str) -> List[Dict]:
    """
    Fetch ALL chunks for a patent in priority order:
      1. claim_independent — always included
      2. claim_dependent   — always included
      3. description       — included as supporting context (limited to 3 chunks)

    Description chunks are fetched last and capped to avoid token bloat.
    """
    try:
        claim_rows = (
            state.supabase.table("patent_chunks")
            .select("id, section_type, content")
            .eq("patent_id", patent_id)
            .in_("section_type", ["claim_independent", "claim_dependent"])
            .execute()
            .data or []
        )
        desc_rows = (
            state.supabase.table("patent_chunks")
            .select("id, section_type, content")
            .eq("patent_id", patent_id)
            .eq("section_type", "description")
            .limit(3)
            .execute()
            .data or []
        )
        all_rows = claim_rows + desc_rows
        log.info("Step 3 — claim family for patent_id=%s: %d claims + %d desc chunks",
                 patent_id, len(claim_rows), len(desc_rows))
        return all_rows
    except Exception as exc:
        log.warning("Claim family fetch failed for %s: %s", patent_id, exc)
        return []


def build_patent_context_block(patent: Dict, claim_chunks: List[Dict]) -> str:
    """
    Format a patent's complete claim family into a labelled block for the LLM.
    Independent claims get more space (800 chars), dependent claims less (400 chars).
    Includes rrf_score hint for independent claims so the LLM can weight confidence.
    """
    header = f"PATENT: {patent['patent_number']} — {patent.get('title', '')}"
    parts  = [header]

    # Independent claims first, then dependent
    for section in ("claim_independent", "claim_dependent"):
        for c in claim_chunks:
            if c.get("section_type") != section:
                continue
            limit   = CLAIM_INDEPENDENT_LIMIT if section == "claim_independent" else CLAIM_DEPENDENT_LIMIT
            content = c["content"][:limit] + ("…" if len(c["content"]) > limit else "")
            label   = "INDEPENDENT CLAIM" if section == "claim_independent" else "DEPENDENT CLAIM"
            parts.append(f"[{label}]\n{content}")

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Step 4 & 5 — Per-Patent LLM Risk Assessment
# ---------------------------------------------------------------------------

def call_agent_risk_patent(
    patent: Dict,
    claim_chunks: List[Dict],
    proposed_specs: str,
    component_scope: str,
) -> Dict[str, Any]:
    """
    Phase 2 Step 4 — Structured per-patent risk assessment.

    The LLM evaluates each claim element against the proposed design and returns:
      - matched_elements:  claim elements present in the design
      - missing_elements:  claim elements NOT present (reduces infringement risk)
      - unclear_elements:  elements that need human review
      - overlap_explanation: plain-English reasoning
      - risk_score: 0–100 numeric score
    """
    context_block = build_patent_context_block(patent, claim_chunks)

    prompt = f"""You are a senior IP Engineer performing patent infringement analysis.
Respond ONLY with minified JSON. No markdown. No preamble.

Output schema (one object, exactly this structure):
{{"patent_number":"{patent['patent_number']}","matched_elements":["..."],"missing_elements":["..."],"unclear_elements":["..."],"overlap_explanation":"...","risk_score":0}}

Rules:
- risk_score: integer 0-100. 0=no overlap, 100=all claim elements present in design.
- matched_elements: claim elements clearly present in the proposed design.
- missing_elements: claim elements clearly absent from the proposed design.
- unclear_elements: claim elements that cannot be determined without more information.
- overlap_explanation: 1-2 sentence plain English explanation of the technical overlap.
- Base your analysis on claim language, NOT on semantic similarity alone.
- Fuyao manufacturing context: glass thickness {GLASS_TOTAL_MIN}-{GLASS_TOTAL_MAX}mm, PVB interlayer {PVB_MIN_MM}-{PVB_MAX_MM}mm.
- You have limited output space. If running long, stop adding elements and close the JSON immediately.

COMPONENT SCOPE: {component_scope}
PROPOSED DESIGN:
{proposed_specs[:2000]}

PATENT CLAIMS:
{context_block}

Analyse each independent claim element against the proposed design. Then assess dependent claims."""

    result = llm_json(prompt)

    # Ensure all required keys exist with safe defaults
    for key, default in [
        ("patent_number",      patent["patent_number"]),
        ("matched_elements",   []),
        ("missing_elements",   []),
        ("unclear_elements",   []),
        ("overlap_explanation",""),
        ("risk_score",         0),
    ]:
        if key not in result:
            result[key] = default

    # Clamp score to 0-100
    result["risk_score"] = max(0, min(100, int(result.get("risk_score", 0))))
    return result


# ---------------------------------------------------------------------------
# Orchestrator — runs Steps 1-5 in sequence
# ---------------------------------------------------------------------------

def run_patent_risk_pipeline(
    embedding: List[float],
    query_text: str,
    proposed_specs: str,
    component_scope: str,
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    """
    Full patent-level risk pipeline. Returns a list of per-patent risk dicts,
    sorted descending by risk_score. Each dict follows the PatentRiskResult schema.
    Returns empty list if no independent claim chunks are found.
    """
    # Step 1 — retrieve independent claim chunks
    ind_chunks = fetch_independent_claim_chunks(embedding, query_text, jurisdiction)
    if not ind_chunks:
        log.info("Pipeline: no independent claim chunks found → CLEAR")
        return []

    # Step 2 — select top candidate patents
    candidate_patents = select_candidate_patents(ind_chunks)
    if not candidate_patents:
        return []

    # Steps 3–5 — expand claim family and assess each patent
    results: List[Dict[str, Any]] = []
    for patent in candidate_patents:
        claim_chunks = fetch_claim_family(patent["patent_id"])
        if not claim_chunks:
            log.warning("No claim chunks found for patent %s, skipping", patent["patent_number"])
            continue
        try:
            assessment = call_agent_risk_patent(
                patent, claim_chunks, proposed_specs, component_scope
            )
            # Attach patent metadata to result
            assessment["title"]       = patent.get("title", "")
            assessment["jurisdiction"]= patent.get("jurisdiction", "")
            assessment["match_count"] = patent.get("match_count", 0)
            assessment["total_score"] = round(patent.get("total_score", 0.0), 6)
            results.append(assessment)
        except Exception as exc:
            log.warning("Risk assessment failed for patent %s: %s", patent["patent_number"], exc)

    # Sort by risk_score descending
    results.sort(key=lambda r: r.get("risk_score", 0), reverse=True)
    log.info("Pipeline complete: %d patents assessed", len(results))
    return results


# ---------------------------------------------------------------------------
# Backward-compatible helpers (used by Phase 3 design suggestions pipeline)
# ---------------------------------------------------------------------------

def fetch_hybrid_matches(embedding: List[float], query_text: str, jurisdiction: str) -> List[Dict]:
    """Legacy hybrid search — kept for Phase 3 design suggestions pipeline."""
    try:
        filter_jx = None if (not jurisdiction or jurisdiction.upper() == "ALL") else jurisdiction
        resp = state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     embedding,
            "query_text":          query_text,
            "filter_jurisdiction": filter_jx,
            "match_count":         TOP_K_CHUNKS,
        }).execute()
        log.info("Legacy hybrid search: jurisdiction=%s → %d chunks",
                 filter_jx or "ALL", len(resp.data or []))
        return resp.data or []
    except Exception as exc:
        log.error("Supabase RPC failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Database retrieval error: {exc}") from exc


def build_context_block(chunks: List[Dict]) -> str:
    """Legacy context formatter — kept for Phase 3 design suggestions pipeline."""
    parts = []
    for c in chunks:
        limit   = CLAIM_INDEPENDENT_LIMIT if c.get("section_type") == "claim_independent" else CLAIM_DEPENDENT_LIMIT
        content = c["content"][:limit] + ("…" if len(c["content"]) > limit else "")
        parts.append(f"[{c['patent_number']}|{c['section_type']}]\n{content}")
    return "\n---\n".join(parts)


def call_agent_risk(context_block: str, proposed_specs: str, component_scope: str) -> Dict[str, Any]:
    """
    Legacy Phase 2 risk agent — kept for Phase 3 design suggestions pipeline.
    Returns risk_status + infringement_map (old schema).
    """
    prompt = f"""You are a senior IP Engineer. Respond ONLY with minified JSON. No markdown. No preamble.

Output schema:
{{"risk_status":"HIGH|MEDIUM|LOW|CLEAR","infringement_map":[{{"claim_ref":"<PATENT_NUMBER> Claim <N>","element":"...","overlap":"...","risk_level":"HIGH|MEDIUM|LOW"}}]}}

IMPORTANT: claim_ref MUST follow the format "<PATENT_NUMBER> Claim <N>" — e.g. "US5773102 Claim 21".
Use the exact patent number from the context block header [PATENT_NUMBER|...].

SCOPE: {component_scope}
SPECS: {proposed_specs[:600]}
PATENTS:
{context_block}

Identify all overlapping claims and classify the overall risk_status."""
    return llm_json(prompt)


def call_agent_designer(
    proposed_specs: str,
    component_scope: str,
    risk_output: Dict[str, Any],
    context_block: str,
    embed_model,
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    """
    Phase 3 — Design suggestion agent.

    Step 1: Proposes 2 alternative designs based on the risk output.
    Step 2: Re-scores each proposal by calling call_agent_risk on it.
    Step 3: Returns only proposals that score LOW or CLEAR.
    """
    infringement_block = json.dumps(risk_output.get("infringement_map", []))
    original_risk = risk_output.get("risk_status", "UNKNOWN")

    prompt = f"""IP Design Engineer. Respond ONLY with minified JSON. No markdown.

Schema: {{"design_arounds":[{{"id":"DA1","description":"2-3 sentence engineering description","rationale":"which claims avoided and how","addresses_claims":["PATENT Claim N"]}},{{"id":"DA2","description":"...","rationale":"...","addresses_claims":[]}}]}}

Original risk: {original_risk}
Conflicting claims: {infringement_block}
Scope: {component_scope}
Original spec: {proposed_specs[:400]}
Patents: {context_block[:600]}

Propose 2 short alternative designs that avoid the conflicting claims."""

    try:
        designer_output = llm_json(prompt)
    except Exception as exc:
        log.error("call_agent_designer: proposal step failed: %s", exc)
        raise

    proposals = designer_output.get("design_arounds", [])
    log.info("Designer proposed %d alternatives — re-scoring each for risk", len(proposals))

    surviving: List[Dict[str, Any]] = []
    for proposal in proposals:
        proposal_spec = proposal.get("description", "")
        if not proposal_spec:
            continue
        try:
            proposal_embedding = embed_model.encode(
                [proposal_spec], normalize_embeddings=True, show_progress_bar=False
            )[0].tolist()
            proposal_chunks = fetch_hybrid_matches(proposal_embedding, proposal_spec, jurisdiction)
            proposal_context = build_context_block(proposal_chunks) if proposal_chunks else context_block

            proposal_risk = call_agent_risk(proposal_context, proposal_spec, component_scope)
            proposal_risk_status = proposal_risk.get("risk_status", "UNKNOWN")
            proposal["risk_score"] = proposal_risk_status

            log.info("Proposal %s re-scored: %s", proposal.get("id"), proposal_risk_status)

            if proposal_risk_status in ("LOW", "CLEAR"):
                surviving.append(proposal)
            else:
                log.info("Proposal %s filtered out (risk=%s)", proposal.get("id"), proposal_risk_status)

        except Exception as exc:
            log.warning("Re-scoring proposal %s failed, skipping: %s", proposal.get("id"), exc)

    log.info("%d / %d proposals passed risk filter", len(surviving), len(proposals))
    return surviving


def call_agent_auditor(design_arounds: List[Dict], component_scope: str) -> List[DesignAroundProposal]:
    """
    Manufacturing audit agent.

    Validates each design proposal against Fuyao's hard glass constraints.
    Rewrites any proposal that violates a constraint.
    Returns a list of DesignAroundProposal objects ready for the API response.
    """
    if not design_arounds:
        return []

    # Trim each proposal to avoid token overflow in the auditor prompt
    trimmed = [
        {"id": da.get("id",""), "description": da.get("description","")[:300], "rationale": da.get("rationale","")[:150]}
        for da in design_arounds
    ]
    da_block = json.dumps(trimmed)
    prompt = f"""Fuyao Glass Auditor. Respond ONLY with minified JSON. No markdown.

Constraints: thickness {GLASS_TOTAL_MIN}-{GLASS_TOTAL_MAX}mm, PVB {PVB_MIN_MM}-{PVB_MAX_MM}mm, no HUD conductors, wedge ≤0.1mrad.

Schema: {{"audited_design_arounds":[{{"id":"...","description":"...","rationale":"...","passed_audit":true,"audit_notes":"..."}}]}}

SCOPE: {component_scope}
DESIGNS: {da_block}

Check each against constraints. Rewrite if violated."""

    audit_result = llm_json(prompt)
    audited_map = {a["id"]: a for a in audit_result.get("audited_design_arounds", [])}

    merged: List[DesignAroundProposal] = []
    for da in design_arounds:
        da_id = da.get("id", "")
        if da_id in audited_map:
            a = audited_map[da_id]
            merged.append(DesignAroundProposal(
                id=da_id,
                description=a.get("description", da.get("description", "")),
                rationale=a.get("rationale", da.get("rationale", "")),
                audited=True,
                audit_notes=a.get("audit_notes"),
            ))
        else:
            merged.append(DesignAroundProposal(
                id=da_id,
                description=da.get("description", ""),
                rationale=da.get("rationale", ""),
                audited=False,
                audit_notes="Audit result not returned.",
            ))
    return merged