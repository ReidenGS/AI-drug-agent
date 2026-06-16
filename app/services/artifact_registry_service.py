"""Maintain run_artifact_registry: active artifacts + snapshots."""

from __future__ import annotations

from ..schemas.registry import RunArtifactRegistry, ActiveArtifacts
from ..utils.ids import new_registry_id
from ..utils.time import now_iso
from .storage_service import Storage

_REGISTRY_KEY = "registry/current.json"
_SNAPSHOT_KEY = "registry/snapshot_{version:04d}.json"


class ArtifactRegistryService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def init_registry(self, run_id: str) -> RunArtifactRegistry:
        now = now_iso()
        reg = RunArtifactRegistry(
            run_id=run_id,
            run_artifact_registry_id=new_registry_id(),
            version=1,
            created_at=now,
            updated_at=now,
            active_artifacts=ActiveArtifacts(),
        )
        self._save(run_id, reg)
        return reg

    def get(self, run_id: str) -> RunArtifactRegistry:
        return RunArtifactRegistry.model_validate(
            self.storage.read_json(self.storage.run_key(run_id, _REGISTRY_KEY))
        )

    def update_active(self, run_id: str, **updates: str) -> RunArtifactRegistry:
        reg = self.get(run_id)
        # snapshot first
        self.storage.write_json(
            self.storage.run_key(run_id, _SNAPSHOT_KEY.format(version=reg.version)),
            reg.model_dump(),
        )
        active = reg.active_artifacts.model_dump()
        active.update(updates)
        reg = reg.model_copy(
            update={
                "active_artifacts": ActiveArtifacts(**active),
                "version": reg.version + 1,
                "updated_at": now_iso(),
            }
        )
        self._save(run_id, reg)
        return reg

    def _save(self, run_id: str, reg: RunArtifactRegistry) -> None:
        self.storage.write_json(self.storage.run_key(run_id, _REGISTRY_KEY), reg.model_dump())
