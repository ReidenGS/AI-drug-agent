"""Step 13 — scientific_evidence_table (EvidenceAgent output)."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


class EvidenceRecord(BaseModel):
    evidence_id: str
    candidate_id: str
    therapeutic_area: Optional[str] = None
    disease_context: Optional[str] = None
    target: Optional[str] = None
    mechanism: Optional[str] = None
    evidence_type: Optional[str] = None
    key_finding: str = ""
    source: str = ""
    confidence_score: float = 0.0
    # ── Systematic review hardening (additive) ──────────────────────────────
    # Filled when the evidence record is derived from a deduplicated literature
    # hit (title / DOI / year / theme). Raw abstract / full payload stay in
    # `tool_outputs/step_13/{tool_call_id}.json`; only compact references are
    # carried here.
    title: Optional[str] = None
    doi: Optional[str] = None
    link: Optional[str] = None
    year: Optional[int] = None
    theme: Optional[str] = None
    query_role: Optional[str] = None
    query_term: Optional[str] = None
    relevance_score: Optional[float] = None
    sources: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class ScientificEvidenceTable(BaseModel):
    run_id: str
    step_id: str = "step_13_evidence"
    created_at: str
    review_status: Literal["ok", "partial", "failed"] = "partial"
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
