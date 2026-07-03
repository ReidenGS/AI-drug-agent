"""Small helpers shared by Step 15-21 scaffold services."""

from __future__ import annotations

from typing import Any

from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage


def read_json_if_exists(storage: Storage, run_id: str, key: str) -> dict[str, Any] | None:
    storage_key = storage.run_key(run_id, key)
    if not storage.exists(storage_key):
        return None
    data = storage.read_json(storage_key)
    return data if isinstance(data, dict) else None


def active_artifact_refs(registry: ArtifactRegistryService, run_id: str) -> dict[str, str]:
    active = registry.get(run_id).active_artifacts.model_dump()
    return {k: v for k, v in active.items() if v}


def missing_active_refs(
    registry: ArtifactRegistryService,
    run_id: str,
    required: dict[str, str],
) -> list[tuple[str, str]]:
    active = registry.get(run_id).active_artifacts
    missing: list[tuple[str, str]] = []
    for field_name, artifact_label in required.items():
        if not getattr(active, field_name, None):
            missing.append((field_name, artifact_label))
    return missing


def safe_workflow_mark(workflow_state, run_id: str, step_key: str, status: str) -> None:
    try:
        workflow_state.mark(run_id, step_key, status)
    except FileNotFoundError:
        return
