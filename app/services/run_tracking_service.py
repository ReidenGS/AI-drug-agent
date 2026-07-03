"""Step 21 scaffold service — write compact run tracking metadata."""

from __future__ import annotations

from ..schemas.step_21_run_tracking_record import (
    LogRef,
    MemoryUpdateRecord,
    RunTrackingMemoryUpdateRecord,
    StorageRecord,
    TrackedArtifacts,
    TrackingWarning,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import active_artifact_refs, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "run_tracking_memory_update_record.json"


class RunTrackingService:
    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def record_tracking(self, run_id: str) -> RunTrackingMemoryUpdateRecord:
        reg = self.registry.get(run_id)
        refs = active_artifact_refs(self.registry, run_id)
        snapshot_id = new_artifact_id("registry_snapshot")
        snapshot_key = self.storage.run_key(run_id, f"registry/{snapshot_id}.json")
        self.storage.write_json(
            snapshot_key,
            {
                "registry_snapshot_id": snapshot_id,
                "snapshot_reason": "step21_final_tracking",
                "snapshot_created_at": now_iso(),
                **reg.model_dump(),
            },
        )
        warnings: list[TrackingWarning] = []
        if not reg.active_artifacts.final_output_package_record_id:
            warnings.append(
                TrackingWarning(
                    warning_type="missing_artifact_ref",
                    message="final_output_package_record_id is missing from registry.",
                    related_artifact_ref="final_output_package_record",
                )
            )
        storage_records = [
            StorageRecord(
                storage_record_id=new_artifact_id("storage_record"),
                record_type="artifact_registry_snapshot",
                storage_ref=snapshot_key,
                write_status="success",
            )
        ]
        if reg.active_artifacts.final_output_package_record_id:
            storage_records.append(
                StorageRecord(
                    storage_record_id=new_artifact_id("storage_record"),
                    record_type="final_output_package",
                    storage_ref=self.storage.run_key(run_id, "final_output_package_record.json"),
                    write_status="success",
                )
            )

        artifact = RunTrackingMemoryUpdateRecord(
            run_id=run_id,
            created_at=now_iso(),
            tracking_status="completed_with_warnings",
            tracked_run_status="completed_with_warnings" if warnings else "completed",
            tracked_artifacts=TrackedArtifacts(
                run_artifact_registry_id=reg.run_artifact_registry_id,
                artifact_registry_snapshot_ref=snapshot_key,
            ),
            storage_records=storage_records,
            memory_update_records=[
                MemoryUpdateRecord(
                    memory_record_id=new_artifact_id("memory_record"),
                    source_artifact_refs=list(refs.values()),
                    write_status="skipped",
                    failure_reason="external memory service intentionally deferred in scaffold",
                )
            ],
            log_refs=[
                LogRef(
                    log_ref_id=new_artifact_id("log_ref"),
                    storage_ref=self.storage.run_key(run_id, "logs/"),
                    write_status="skipped",
                )
            ],
            tracking_warnings=warnings,
            tracking_notes=(
                "Step 21 scaffold records traceability only. It does not call external memory, "
                "embedding, or log aggregation services."
            ),
        )
        artifact_id = new_artifact_id("run_tracking_memory_update_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        safe_workflow_mark(self.workflow_state, run_id, "step_21", "completed")
        return artifact
