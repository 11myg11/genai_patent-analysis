"""
app/models.py — Pydantic request/response models for all API endpoints.

Phase 2 — Risk Analysis:
  RiskAnalysisRequest   — input for POST /api/v1/risk-analysis
  RiskAnalysisResponse  — output: risk_status + infringement_map + matched chunks

Phase 3 — Design Suggestions:
  DesignSuggestionRequest  — input for POST /api/v1/design-suggestions
  DesignSuggestionResponse — output: original risk + audited proposals (LOW/CLEAR only)
  DesignAroundProposal     — a single audited design proposal (shared by both phases)

Other:
  CompareRequest        — input for POST /api/v1/compare
  ChunkReference        — a single matched patent chunk
  PatentUpdateRequest   — input for PATCH /api/v1/patents/{id}
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class ChunkReference(BaseModel):
    patent_number: str
    title:         str
    section_type:  str
    content:       str
    rrf_score:     float


class DesignAroundProposal(BaseModel):
    id:               str
    description:      str
    rationale:        str
    audited:          bool
    audit_notes:      Optional[str] = None
    risk_score:       Optional[str] = None   # LOW or CLEAR — set after re-scoring


# ── Phase 2: Risk Analysis ─────────────────────────────────────────────────────

class PatentRiskResult(BaseModel):
    """Per-patent structured risk assessment — new Step 5 output schema."""
    patent_number:       str
    title:               str                = ""
    jurisdiction:        str                = ""
    risk_score:          int                = Field(0, ge=0, le=100)
    matched_elements:    List[str]          = []
    missing_elements:    List[str]          = []
    unclear_elements:    List[str]          = []
    overlap_explanation: str               = ""
    match_count:         int                = 0   # how many independent claim chunks matched
    total_score:         float              = 0.0  # aggregated RRF score


class RiskAnalysisRequest(BaseModel):
    product_id:              str = Field(...)
    component_scope:         str = Field(...)
    proposed_specifications: str = Field(...)
    jurisdiction:            str = Field(default="ALL")


class RiskAnalysisResponse(BaseModel):
    product_id:        str
    risk_status:       str                  # derived: HIGH/MEDIUM/LOW/CLEAR from top patent score
    patent_assessments: List[PatentRiskResult]  # one entry per candidate patent
    token_budget_used: int


# ── Phase 3: Design Suggestions ───────────────────────────────────────────────

class DesignSuggestionRequest(BaseModel):
    product_id:              str = Field(...)
    component_scope:         str = Field(...)
    proposed_specifications: str = Field(...)
    jurisdiction:            str = Field(default="ALL")


class DesignSuggestionResponse(BaseModel):
    product_id:           str
    original_risk_status: str              # risk of the original spec
    suggestions:          List[DesignAroundProposal]  # LOW/CLEAR only, audited
    proposals_generated:  int              # total proposals before filtering
    proposals_passed:     int              # how many survived the LOW/CLEAR filter


# ── Other ─────────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    patent_id_a:  str
    patent_id_b:  str
    jurisdiction: str = "ALL"


class PatentUpdateRequest(BaseModel):
    patent_number:    Optional[str] = None
    title:            Optional[str] = None
    assignee:         Optional[str] = None
    jurisdiction:     Optional[str] = None
    publication_date: Optional[str] = None