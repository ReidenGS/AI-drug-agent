"""Step 17 API — human review decision record scaffold."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.human_review_service import HumanReviewService
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/17", tags=["step-17-human-review"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_17(run_id: str, review_payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_17_human_review")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 17 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_17_human_review",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    svc = HumanReviewService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.record(run_id, review_payload=review_payload).model_dump()
