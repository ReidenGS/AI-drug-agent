"""Step 1 — raw_request_record.

Per ADC_Pipeline_IO_Schema_v0.1.md §Step 1: produced by IntakeService when a
user submits an ADC run request via UI/API/notebook/script.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class UserProvidedContext(BaseModel):
    target_or_antigen_text: Optional[str] = None
    candidate_text: Optional[str] = None
    payload_linker_text: Optional[str] = None
    constraints_text: Optional[str] = None
    notes: Optional[str] = None


class UploadedFile(BaseModel):
    file_id: str
    original_filename: str
    storage_path: str
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class RawRequestRecord(BaseModel):
    run_id: str
    run_artifact_registry_id: str
    step_id: Literal["step_01_intake"] = "step_01_intake"
    created_at: str
    entry_source: Literal["ui", "api", "notebook", "script"] = "api"
    submitted_by: Optional[str] = None
    raw_user_query: str
    user_provided_context: UserProvidedContext = Field(default_factory=UserProvidedContext)
    uploaded_files: list[UploadedFile] = Field(default_factory=list)
    intake_status: Literal["received", "queued", "rejected"] = "received"
