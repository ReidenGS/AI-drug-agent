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
    # Step 5 batch-6 additions (additive, optional):
    # `role` carries the ADC role we believe this material plays
    # (target, antibody, payload, linker, linker_payload, structure, …)
    # so downstream agents don't have to re-classify; `role_status`
    # records whether that role was user-provided (explicit) or
    # inferred from an alias decomposition.
    role: Optional[str] = None
    role_status: Literal["explicit", "inferred", "unknown"] = "unknown"
    # Compact, raw-safe descriptor of the material's content (additive,
    # optional). Used when `value` is a storage ref rather than the raw
    # content itself (e.g. an explicit ESM `prompt_sequence` material whose
    # raw masked prompt is held in storage, never written into this artifact):
    # downstream projection layers read length / hash / format flags from here
    # without touching the raw bytes. Only holds non-sensitive fingerprints
    # (length, sha256 prefix, mask flags) — never the raw sequence/prompt.
    content_descriptor: Optional[dict] = None


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
    # Step 5 batch-6 additions (additive, optional). Step 5 is a material /
    # context organization step; it must distinguish reference ADCs the user
    # cited (e.g. T-DM1, T-DXd) from novel candidates the user actually
    # wants the pipeline to generate. None of these fields are required by
    # legacy callers — defaults preserve prior behavior.
    candidate_role: Literal[
        "reference_benchmark",
        "comparator",
        "partial_context",
        "user_provided_candidate",
        "material_only",
        "unknown",
    ] = "unknown"
    is_generated_candidate: bool = False
    context_status: Literal[
        "complete_reference", "partial", "material_pool", "unknown"
    ] = "unknown"
    data_gaps: list[str] = Field(default_factory=list)
    missing_material_roles: list[str] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)


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
    # Step 5 batch-6 addition (additive): downstream-step search hints,
    # prioritized linker-payload / payload / linker / compound / target /
    # complete ADC. Antibody is included ONLY when the user explicitly
    # provided one. Each entry: {"entity": str, "role": str,
    # "explicit_or_inferred": "explicit"|"inferred", "source": str}.
    downstream_query_hints: list[dict] = Field(default_factory=list)
    # Step 5 LLM-assisted selection audit (additive). Keyed by
    # ``candidate_id``; each value holds compact eligible_tools /
    # selected_tools / skipped_eligible_tools / tool_selection_source /
    # llm_call_status / llm_dropped_out_of_scope strings. Never holds
    # raw LLM responses, raw tool payloads, full prompts, or API keys.
    enrichment_selection_audit: dict = Field(default_factory=dict)
