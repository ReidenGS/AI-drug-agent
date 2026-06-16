"""Step 11 API — scoring_validation."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.scoring_validation_service import ScoringValidationService
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/11", tags=["step-11-scoring-validation"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_11(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_11_scoring_validation")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 11 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_11_scoring_validation",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    svc = ScoringValidationService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.validate(run_id).model_dump()
