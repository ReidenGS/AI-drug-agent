"""Step 5 — candidate_context_table (CandidateContextAgent output)."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


class Identifier(BaseModel):
    id_type: str
    id_value: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class Material(BaseModel):
    material_id: str
    material_type: str  # e.g. target_antigen_name, antibody_heavy_chain_seq, pdb_id, payload_smiles, linker_smiles, dar
    value: str
    value_format: Optional[str] = None
    extraction_status: Literal["extracted", "pending", "failed"] = "extracted"
    validation_status: Literal["valid", "invalid", "unknown"] = "unknown"


class ADCLinks(BaseModel):
    target_material_ids: list[str] = Field(default_factory=list)
    antibody_material_ids: list[str] = Field(default_factory=list)
    payload_material_ids: list[str] = Field(default_factory=list)
    linker_material_ids: list[str] = Field(default_factory=list)
    dar_material_ids: list[str] = Field(default_factory=list)


class CandidateRecord(BaseModel):
    candidate_id: str
    candidate_label: str
    candidate_type: Literal[
        "target_antigen", "antibody", "adc_construct", "compound_component", "unknown"
    ]
    source_records: list[str] = Field(default_factory=list)
    identifiers: list[Identifier] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list)
    adc_links: ADCLinks = Field(default_factory=ADCLinks)
    candidate_status: Literal["ready_for_step6", "partially_ready_for_step6"] = "partially_ready_for_step6"
    candidate_notes: Optional[str] = None


class CandidateContextTable(BaseModel):
    run_id: str
    step_id: str = "step_05_candidate_context"
    created_at: str
    context_build_status: Literal["ok", "partial", "failed"] = "partial"
    candidate_records: list[CandidateRecord] = Field(default_factory=list)
    missing_context_flags: list[str] = Field(default_factory=list)
    # Runtime extension: the canonical Step 5 schema does not require these,
    # but the Step 5 runtime records every MCP enrichment call here so raw
    # tool outputs stay outside `candidate_records[]` and can still be
    # audited (per IO Schema §Step 5/Step 6 tool_output_ref convention).
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
