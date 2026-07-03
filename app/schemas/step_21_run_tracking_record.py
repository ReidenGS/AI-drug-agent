"""Step 21 — run tracking / memory update scaffold."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrackedArtifacts(BaseModel):
    run_artifact_registry_id: str
    artifact_registry_snapshot_ref: str


class StorageRecord(BaseModel):
    storage_record_id: str
    record_type: Literal[
        "run_metadata",
        "artifact_registry_snapshot",
        "tool_call_index",
        "log_bundle",
        "final_output_package",
        "memory_entry",
        "other",
    ]
    storage_ref: str
    storage_type: Literal[
        "database_record",
        "s3_path",
        "local_run_storage",
        "vector_db",
        "other",
    ] = "local_run_storage"
    write_status: Literal["success", "failed", "skipped"] = "success"


class MemoryUpdateRecord(BaseModel):
    memory_record_id: str
    memory_type: Literal[
        "run_summary",
        "candidate_result_summary",
        "user_facing_summary",
        "decision_summary",
        "reusable_context",
        "other",
    ] = "run_summary"
    source_artifact_refs: list[str] = Field(default_factory=list)
    memory_storage_ref: Optional[str] = None
    embedding_ref: Optional[str] = None
    write_status: Literal["success", "failed", "skipped"] = "skipped"
    failure_reason: Optional[str] = None


class LogRef(BaseModel):
    log_ref_id: str
    log_type: Literal[
        "run_log",
        "tool_call_log",
        "error_log",
        "warning_log",
        "user_action_log",
        "system_event_log",
        "storage_log",
        "other",
    ] = "run_log"
    storage_ref: str
    write_status: Literal["success", "failed", "skipped"] = "skipped"


class TrackingWarning(BaseModel):
    warning_type: Literal[
        "missing_artifact_ref",
        "memory_write_failed",
        "log_write_failed",
        "storage_write_failed",
        "registry_snapshot_failed",
        "other",
    ]
    message: str
    related_artifact_ref: Optional[str] = None


class RunTrackingMemoryUpdateRecord(BaseModel):
    run_id: str
    step_id: str = "step_21"
    created_at: str
    tracking_status: Literal[
        "completed",
        "completed_with_warnings",
        "partial",
        "failed",
    ] = "completed_with_warnings"
    tracked_run_status: Literal[
        "completed",
        "completed_with_warnings",
        "stopped",
        "failed",
        "partial",
    ] = "completed_with_warnings"
    tracked_artifacts: TrackedArtifacts
    storage_records: list[StorageRecord] = Field(default_factory=list)
    memory_update_records: list[MemoryUpdateRecord] = Field(default_factory=list)
    log_refs: list[LogRef] = Field(default_factory=list)
    tracking_warnings: list[TrackingWarning] = Field(default_factory=list)
    tracking_notes: Optional[str] = None
