"""Step 10 API — scoring_handoff_package.

External Yufei AEE scoring is **out of scope** for this endpoint. We only
prepare the handoff package; the result file is expected to land at
`{run_id}/inputs/external_scoring_result.json` (see Step 11 docstring).
"""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.scoring_handoff_service import ScoringHandoffService
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/10", tags=["step-10-scoring-handoff"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_10(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_10_scoring_handoff")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 10 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_10_scoring_handoff",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    svc = ScoringHandoffService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.prepare(run_id).model_dump()
