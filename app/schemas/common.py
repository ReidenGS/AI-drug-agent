"""Cross-step common objects: registry refs, tool-call records, artifact refs."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class RegistryRef(BaseModel):
    """Reference to the run artifact registry — used by Step 2+ inputs."""

    run_artifact_registry_id: str
    snapshot_version: Optional[int] = None


class ArtifactRef(BaseModel):
    artifact_id: str
    artifact_type: str
    storage_path: Optional[str] = None
    sha256: Optional[str] = None


ToolCallRunStatus = Literal[
    "success",
    "failed",
    "skipped",
    "dependency_unavailable",
    "partial",
    "pending",
    "not_run",
]


class ToolCallRecord(BaseModel):
    """Canonical tool-call record (ADC_Pipeline_IO_Schema_v0.1.md).

    Raw tool outputs MUST live outside this record — referenced via
    `tool_output_artifact_id` (registered artifact handle) or `tool_output_ref`
    (free-form storage ref, e.g. S3 URI). Never embed raw payloads here.
    """

    tool_call_id: str
    tool_name: str
    agent_name: Optional[str] = None
    step_id: Optional[str] = None
    run_status: ToolCallRunStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    idempotency_key: Optional[str] = None
    tool_input_summary: Optional[dict[str, Any]] = None
    tool_output_artifact_id: Optional[str] = None
    tool_output_ref: Optional[str] = None
    error_message: Optional[str] = None
