"""Generic pipeline snapshot manifest.

A snapshot is a portable, reusable export of a run's completed-prefix
artifacts (through an arbitrary ``through_step``), so a new run can be
hydrated from it and continued from the next step WITHOUT re-running the
prior LLM / tool-heavy steps. This is a development / test acceleration
layer and reusable-artifact mechanism — it changes no production step
behavior.

The manifest is deliberately step-agnostic: it records file paths, ids,
hashes, and sizes discovered from the artifact registry + run storage. It
never embeds raw artifact content, raw tool payloads, prompts, LLM
responses, keys, or biological sequences.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ArtifactSnapshotEntry(BaseModel):
    artifact_name: str
    artifact_type: str = "artifact"  # artifact | registry | workflow_state | tool_output
    artifact_id: Optional[str] = None
    source_storage_key: str
    # Run-relative key (no prefix/runs/<run_id>) — drives restore on hydrate.
    run_relative_path: str
    # Where the bytes live inside the snapshot directory.
    snapshot_relative_path: str
    content_type: str = "application/json"
    sha256: str = ""
    size_bytes: int = 0


class UploadedFileSnapshotEntry(BaseModel):
    file_id: str
    original_filename: Optional[str] = None
    inferred_role: Optional[str] = None
    content_type: Optional[str] = None
    source_storage_ref: str
    run_relative_path: str
    snapshot_relative_path: str
    sha256: str = ""
    size_bytes: int = 0


class PipelineSnapshotManifest(BaseModel):
    snapshot_id: str
    snapshot_version: str = "1"
    created_at: str
    source_run_id: str
    hydrated_from_snapshot_id: Optional[str] = None
    through_step: str
    completed_steps: list[str] = Field(default_factory=list)
    active_artifacts: dict = Field(default_factory=dict)
    artifact_files: list[ArtifactSnapshotEntry] = Field(default_factory=list)
    uploaded_files: list[UploadedFileSnapshotEntry] = Field(default_factory=list)
    tool_output_files: list[ArtifactSnapshotEntry] = Field(default_factory=list)
    workflow_state_summary: dict = Field(default_factory=dict)
    registry_summary: dict = Field(default_factory=dict)
    notes: Optional[dict] = None
