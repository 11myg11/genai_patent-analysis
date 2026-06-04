"""
app/services/retrieval.py — Hybrid patent search and two-agent IP analysis.

Implements the core analysis workflow used by POST /api/v1/evaluate-design:
  1. fetch_hybrid_matches — retrieves the top-K most relevant patent chunks via
     Supabase RPC (RRF of pgvector cosine similarity + full-text tsvector search)
  2. build_context_block  — formats chunks into a labelled block for LLM prompts
  3. call_agent_generator — IP Engineer agent: assesses risk and proposes design-arounds
  4. call_agent_auditor   — Manufacturing Auditor agent: validates design-arounds against
     Fuyao's hard glass constraints (PVB thickness, HUD zone, wedge angle)

The two-agent split is intentional: the generator optimises for IP clearance,
the auditor enforces physical manufacturing limits. Merging them degrades both.

Functions:
  fetch_hybrid_matches(embedding, query_text, jurisdiction) -> list[dict]
    Calls the Supabase SQL function `match_patent_hybrid`. Returns up to TOP_K_CHUNKS
    results with keys: patent_number, title, section_type, content, rrf_score.

  build_context_block(chunks) -> str
    Formats chunks as [REF{n}|patent_number|section_type|RRF:score] blocks
    separated by "---". This format is referenced in the generator prompt.

  call_agent_generator(context_block, proposed_specs, component_scope) -> dict
    Returns: risk_status, infringement_map, design_arounds.

  call_agent_auditor(generator_output, component_scope) -> dict
    Adds "design_arounds_merged" key to generator_output with audited proposals.
"""
import json
import logging
from typing import Any, Dict, List

from fastapi import HTTPException

from app.config import TOP_K_CHUNKS, PVB_MIN_MM, PVB_MAX_MM, GLASS_TOTAL_MIN, GLASS_TOTAL_MAX
from app.models import DesignAroundProposal
from app.services.llm import llm_json
from app.state import state

log = logging.getLogger(__name__)


def fetch_hybrid_matches(embedding: List[float], query_text: str, jurisdiction: str) -> List[Dict]:
    try:
        resp = state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     embedding,
            "query_text":          query_text,
            "filter_jurisdiction": jurisdiction,
            "match_count":         TOP_K_CHUNKS,
        }).execute()
        return resp.data or []
    except Exception as exc:
        log.error("Supabase RPC failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Database retrieval error: {exc}") from exc


def build_context_block(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[REF{i}|{c['patent_number']}|{c['section_type']}|RRF:{c['rrf_score']:.4f}]\n"
            f"{c['content']}"
        )
    return "\n---\n".join(parts)


def call_agent_generator(context_block: str, proposed_specs: str, component_scope: str) -> Dict[str, Any]:
    prompt = f"""You are a senior IP Engineer specialised in structural patent claim analysis.
Respond ONLY with minified JSON. No markdown. No preamble. No explanation outside JSON.
Use acronyms: PVB=polyvinyl_butyral_interlayer, Tt=total_thickness_mm, HUD=heads_up_display_zone.

Output schema (strict):
{{"risk_status":"HIGH|MEDIUM|LOW|CLEAR","infringement_map":[{{"claim_ref":"...","element":"...","overlap":"..."}}],"design_arounds":[{{"id":"DA1","description":"...","rationale":"..."}},{{"id":"DA2","description":"...","rationale":"..."}}]}}

COMPONENT_SCOPE: {component_scope}
PROPOSED_SPECS: {proposed_specs}
PATENT_CONTEXT:
{context_block}

Analyse proposed specs against each patent claim. Classify risk_status. Map overlapping elements. Propose exactly 2 design-arounds."""
    return llm_json(prompt)


def call_agent_auditor(generator_output: Dict[str, Any], component_scope: str) -> Dict[str, Any]:
    da_block = json.dumps(generator_output.get("design_arounds", []))
    prompt = f"""You are a Fuyao Glass Manufacturing Auditor with expertise in automotive glazing standards.
Respond ONLY with minified JSON. No markdown. No explanation outside JSON.

HARD CONSTRAINTS (violation = rewrite required):
1. Tt between {GLASS_TOTAL_MIN} and {GLASS_TOTAL_MAX} mm.
2. PVB between {PVB_MIN_MM} and {PVB_MAX_MM} mm.
3. HUD zone: zero conductive materials.
4. Wedge angle ≤ 0.1 mrad.

Output schema:
{{"audited_design_arounds":[{{"id":"...","description":"...","rationale":"...","passed_audit":true,"audit_notes":"..."}}]}}

COMPONENT_SCOPE: {component_scope}
DESIGN_AROUNDS_TO_AUDIT: {da_block}

For each design-around: check constraints, rewrite if violated, confirm if passed."""

    audit_result = llm_json(prompt)
    audited_map = {a["id"]: a for a in audit_result.get("audited_design_arounds", [])}
    merged: List[DesignAroundProposal] = []
    for da in generator_output.get("design_arounds", []):
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
    generator_output["design_arounds_merged"] = merged
    return generator_output
