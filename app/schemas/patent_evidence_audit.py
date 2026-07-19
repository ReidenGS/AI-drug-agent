"""Typed compact audit shared by both Patent-Evidence output artifacts."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .patent_evidence_contract import PatentEvidenceInputRole
from .patent_evidence_request import SafeIdentifier, SafeSourcePath


class PatentEvidenceLaneAssessmentAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_lane: Literal["evidence", "patent"]
    status: Literal["planned", "missing_inputs", "not_applicable"]
    reason: str


class PatentEvidenceRejectionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    reason: str


class PatentEvidencePlanningAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_cache_layout_version: str = ""
    llm_call_count: int = Field(default=0, ge=0)
    catalog_visible_count: int = Field(default=0, ge=0)
    eligible_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    executed_count: int = Field(default=0, ge=0)
    lane_assessments: list[PatentEvidenceLaneAssessmentAudit] = Field(
        default_factory=list
    )
    rejections: list[PatentEvidenceRejectionAudit] = Field(default_factory=list)


PatentEvidenceResolverFailureCode = Literal[
    "source_artifact_unavailable",
    "source_path_invalid",
    "source_path_not_list",
    "source_path_index_missing",
    "source_value_missing",
]


class PatentEvidenceResolverAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: SafeIdentifier
    source_artifact: SafeIdentifier
    source_path: SafeSourcePath
    role: PatentEvidenceInputRole
    resolved: bool
    unresolved_reason: Optional[PatentEvidenceResolverFailureCode] = None


__all__ = [
    "PatentEvidenceLaneAssessmentAudit",
    "PatentEvidencePlanningAudit",
    "PatentEvidenceRejectionAudit",
    "PatentEvidenceResolverAuditEntry",
]
