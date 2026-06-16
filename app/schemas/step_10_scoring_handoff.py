"""Step 10 — external AEE / scoring handoff package.

Step 10 prepares a normalized package that the external Yufei AEE scoring
module will consume. The module is **external** to this repo; we do NOT
fabricate scores here. The handoff package only carries:
- per-candidate normalized summary (no raw MCP payload)
- pointers to upstream Step 5/6/7/8/9 artifact ids
- explicit `awaiting_external_scoring` status when delivery hasn't happened

When the external system completes, the scoring result is expected to land at
`inputs/external_scoring_result.json` under the run directory — see
`docs` in `app/services/scoring_validation_service.py` for the expected shape.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


HandoffStatus = Literal[
    "awaiting_external_scoring",
    "delivered",
    "partial",
    "failed",
]


class StructureConfidenceSummary(BaseModel):
    structure_input_id: Optional[str] = None
    run_status: Optional[str] = None
    confidence_count: int = 0
    partial_run_flag: bool = False


class CompoundScreeningSummary(BaseModel):
    compound_id: Optional[str] = None
    source_library: Optional[str] = None
    source_database_version: Optional[str] = None
    source_tool_name: Optional[str] = None
    source_runtime_status: Optional[str] = None


class SourceArtifactRefs(BaseModel):
    """Stable upstream refs by artifact id. Raw MCP outputs stay under
    `tool_outputs/...` and are NOT carried in this package."""

    candidate_context_table_id: Optional[str] = None
    structured_liability_summary_id: Optional[str] = None
    prepared_structure_input_package_id: Optional[str] = None
    structure_prediction_and_interface_results_id: Optional[str] = None
    structure_variant_and_compound_screening_id: Optional[str] = None


class CandidateSummary(BaseModel):
    candidate_id: str
    candidate_label: Optional[str] = None
    candidate_type: Optional[str] = None
    developability_label: Optional[str] = None
    recommended_action: Optional[str] = None
    structure_confidence: list[StructureConfidenceSummary] = Field(default_factory=list)
    compound_screening: list[CompoundScreeningSummary] = Field(default_factory=list)
    source_artifact_refs: SourceArtifactRefs = Field(default_factory=SourceArtifactRefs)


class ScoringHandoff(BaseModel):
    run_id: str
    step_id: str = "step_10_scoring_handoff"
    created_at: str
    handoff_status: HandoffStatus = "awaiting_external_scoring"
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_summaries: list[CandidateSummary] = Field(default_factory=list)
    payload_storage_path: str = ""
    external_module: str = "yufei_aee"
    expected_result_storage_path: str = "inputs/external_scoring_result.json"
    missing_inputs: list[str] = Field(default_factory=list)
    notes: str = ""
