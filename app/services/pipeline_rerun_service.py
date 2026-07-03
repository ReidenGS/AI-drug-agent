"""Step 19 scaffold service — record rerun intent without executing reruns."""

from __future__ import annotations

from ..schemas.step_19_pipeline_rerun_result_record import (
    PipelineRerunResultRecord,
    RerunTaskResult,
    RerunWarning,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "pipeline_rerun_result_record.json"


class PipelineRerunService:
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

    def record_rerun_status(self, run_id: str) -> PipelineRerunResultRecord:
        active = self.registry.get(run_id).active_artifacts
        redesign = read_json_if_exists(self.storage, run_id, "redesign_optimization_task_record.json") or {}
        tasks = redesign.get("redesign_tasks") or []
        results = [
            RerunTaskResult(
                redesign_task_id=task.get("redesign_task_id", "missing"),
                source_candidate_id=task.get("candidate_id"),
                rerun_status="skipped",
                rerun_start_step=task.get("suggested_rerun_start_step") or "other",
                rerun_end_step="other",
                skip_or_failure_reason="no_rerun_required",
            )
            for task in tasks
        ]
        warnings = [
            RerunWarning(
                warning_type="partial_rerun",
                related_redesign_task_id=task.get("redesign_task_id"),
                related_candidate_id=task.get("candidate_id"),
                message="Pipeline rerun is intentionally deferred in scaffold mode.",
            )
            for task in tasks
            if task.get("requires_pipeline_rerun")
        ]
        artifact = PipelineRerunResultRecord(
            run_id=run_id,
            created_at=now_iso(),
            rerun_status="partial" if results else "skipped",
            rerun_iteration_id=None,
            source_redesign_task_record_id=(
                active.redesign_optimization_task_record_id or "missing"
            ),
            rerun_task_results=results,
            rerun_warnings=warnings,
            rerun_notes=(
                "Step 19 scaffold records rerun intent only. No pipeline step was re-executed "
                "and no registry pointer was promoted."
            ),
        )
        artifact_id = new_artifact_id("pipeline_rerun_result_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, pipeline_rerun_result_record_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_19", "completed")
        return artifact
