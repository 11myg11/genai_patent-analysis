"""
app/models.py — Pydantic request/response models for all API endpoints.

These models validate incoming JSON, document the API contract, and are used by
FastAPI to generate the OpenAPI schema. Import the model you need directly.

Models:
  DesignEvaluationRequest  — Input for POST /api/v1/evaluate-design
  DesignEvaluationResponse — Output for POST /api/v1/evaluate-design
  ChunkReference           — A single matched patent chunk (part of the response above)
  DesignAroundProposal     — A generated + audited design-around suggestion
  CompareRequest           — Input for POST /api/v1/compare
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DesignEvaluationRequest(BaseModel):
    product_id:              str = Field(...)
    component_scope:         str = Field(...)
    proposed_specifications: str = Field(...)
    jurisdiction:            str = Field(default="US")


class ChunkReference(BaseModel):
    patent_number: str
    title:         str
    section_type:  str
    content:       str
    rrf_score:     float


class DesignAroundProposal(BaseModel):
    id:          str
    description: str
    rationale:   str
    audited:     bool
    audit_notes: Optional[str] = None


class DesignEvaluationResponse(BaseModel):
    product_id:        str
    risk_status:       str
    infringement_map:  List[Dict[str, str]]
    design_arounds:    List[DesignAroundProposal]
    matched_chunks:    List[ChunkReference]
    token_budget_used: int


class CompareRequest(BaseModel):
    patent_id_a:  str
    patent_id_b:  str
    jurisdiction: str = "US"
