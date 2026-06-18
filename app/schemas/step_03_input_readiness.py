"""Step 3 — input_readiness_status + missing_input_checklist."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class SourceRefs(BaseModel):
    raw_request_record_id: str
    structured_query_id: str


class BasicADCInputPresence(BaseModel):
    target_or_antigen_present: bool = False
    antibody_candidate_present: bool = False
    payload_present: bool = False
    linker_present: bool = False
    structure_or_sequence_present: bool = False
    constraints_present: bool = False
    # Step 3 batch 3 additions — finer signals required by the production
    # plan (Step1_4_Orchestration_Component_Plan_v0.1.md §Step 3).
    adc_task_intent_present: bool = False
    structure_input_present: bool = False
    sequence_input_present: bool = False
    candidate_file_present: bool = False
    # Evidence: which artifact/field surfaced the signal.
    target_evidence: Optional[str] = None
    antibody_evidence: Optional[str] = None
    payload_evidence: Optional[str] = None
    linker_evidence: Optional[str] = None
    structure_or_sequence_evidence: Optional[str] = None
    constraints_evidence: Optional[str] = None
    adc_task_intent_evidence: Optional[str] = None
    structure_input_evidence: Optional[str] = None
    sequence_input_evidence: Optional[str] = None
    candidate_file_evidence: Optional[str] = None


FileRole = Literal[
    "pdb_or_cif_structure",
    "fasta_sequence",
    "csv_or_table",
    "json_metadata",
    "image",
    "unknown",
]


class UploadedFileCheck(BaseModel):
    file_id: str
    exists: bool
    checksum_ok: bool
    format_ok: bool
    inferred_role: FileRole = "unknown"
    storage_path_present: bool = False
    content_type_present: bool = False
    size_bytes_present: bool = False
    notes: Optional[str] = None


GapCategory = Literal[
    "target",
    "antibody",
    "payload_or_linker",
    "structure_or_sequence",
    "constraints",
    "task_intent",
    "raw_user_query",
    "uploaded_file",
    "other",
]


class MissingInputItem(BaseModel):
    field: str
    severity: Literal["blocking", "warning", "optional"]
    message: str
    category: GapCategory = "other"
    evidence_field: Optional[str] = None


class InputReadinessStatus(BaseModel):
    run_id: str
    step_id: str = "step_03_input_readiness"
    checked_at: str
    source_refs: SourceRefs
    input_readiness_status: Literal["ready", "needs_user_input", "blocked"]
    readiness_summary: str = ""
    basic_adc_input_presence: BasicADCInputPresence = Field(default_factory=BasicADCInputPresence)
    uploaded_file_checks: list[UploadedFileCheck] = Field(default_factory=list)
    missing_input_checklist: list[MissingInputItem] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
