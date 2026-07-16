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
    # Blocking means routing is forbidden. Recoverable says whether a
    # concrete clarification can return the request to Step 2/3.
    recoverable: bool = True


class ClarificationRequest(BaseModel):
    """A user-facing question derived from a required-slot gap.

    This is the minimal backend skeleton of the Step 3 clarification loop:
    Step 3 turns gaps (Step 2 `missing_slots` and its own deterministic
    checks) into structured, stable, machine-readable questions. There is
    NO UI and NO multi-turn graph here — `resolved` always starts False and
    nothing in Step 3 resolves it yet.

    `request_id` is deterministic (slot identity + a short content hash) so
    re-running Step 3 on the same input yields the same id, which lets a
    store dedupe answers without random UUID churn.
    """

    request_id: str
    slot_name: str
    slot_category: str
    severity: Literal["blocking", "warning", "optional"]
    question: str
    reason: str = ""
    source: str = "step2_missing_slots"
    evidence_field: Optional[str] = None
    resolved: bool = False


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
    # Additive (Step 3 clarification-loop skeleton). Defaults to [] so old
    # artifacts that predate the field still validate.
    clarification_requests: list[ClarificationRequest] = Field(default_factory=list)
    # User-facing follow-up message. Passed through from Step 2's
    # `structured_query.response` when readiness is not `ready`; falls back
    # to a deterministic join of clarification questions when Step 2 left it
    # empty. Step 3 NEVER calls an LLM to produce this. None when ready.
    response: Optional[str] = None
