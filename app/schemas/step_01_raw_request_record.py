"""Step 1 — raw_request_record.

Per ADC_Pipeline_IO_Schema_v0.1.md §Step 1: produced by IntakeService when a
user submits an ADC run request via UI/API/notebook/script.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from app.utils.ids import SessionId


class UserProvidedContext(BaseModel):
    target_or_antigen_text: Optional[str] = None
    candidate_text: Optional[str] = None
    payload_linker_text: Optional[str] = None
    constraints_text: Optional[str] = None
    notes: Optional[str] = None
    # ── Clarification-loop carry-over (additive; all default empty so first-
    # turn intake and old artifacts are unchanged). These are populated only
    # on a clarification revision turn so the Step 2 LLM can remember the
    # previous intent and combine the user's short follow-up answer with it.
    # They are loose containers (the program passes them through; the LLM
    # re-parses). They must never carry prompts, keys, or extracted sequences.
    previous_task_intent: Optional[dict] = None
    previous_missing_slots: list[dict] = Field(default_factory=list)
    previous_clarification_requests: list[dict] = Field(default_factory=list)
    clarification_answers: list[dict] = Field(default_factory=list)
    # Previous turn's LLM canonical_query, carried so the Step 2 LLM can
    # update it with the new answers instead of re-deriving from scratch.
    previous_canonical_query: Optional[str] = None


class UploadedFile(BaseModel):
    file_id: str
    original_filename: str
    storage_path: str
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    related_candidate_id: Optional[str] = None
    related_candidate_ids: list[str] = Field(default_factory=list)
    role: Optional[str] = None
    chain_role: Optional[str] = None
    chain_id: Optional[str] = None
    chain_roles: dict[str, str] = Field(default_factory=dict)


class RawRequestRecord(BaseModel):
    run_id: str
    # Optional only for reading historical artifacts. New intake always sets it.
    session_id: Optional[SessionId] = None
    run_artifact_registry_id: str
    step_id: Literal["step_01_intake"] = "step_01_intake"
    created_at: str
    entry_source: Literal["ui", "api", "notebook", "script"] = "api"
    submitted_by: Optional[str] = None
    raw_user_query: str
    user_provided_context: UserProvidedContext = Field(default_factory=UserProvidedContext)
    uploaded_files: list[UploadedFile] = Field(default_factory=list)
    intake_status: Literal["received", "queued", "rejected"] = "received"
