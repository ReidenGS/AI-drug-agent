"""Step 19 API — pipeline rerun status scaffold."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.pipeline_rerun_service import PipelineRerunService
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/19", tags=["step-19-pipeline-rerun"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_19(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_19_pipeline_rerun")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 19 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_19_pipeline_rerun",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    svc = PipelineRerunService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.record_rerun_status(run_id).model_dump()
