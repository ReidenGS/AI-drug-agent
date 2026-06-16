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


class ScientificEvidenceTable(BaseModel):
    run_id: str
    step_id: str = "step_13_evidence"
    created_at: str
    review_status: Literal["ok", "partial", "failed"] = "partial"
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
