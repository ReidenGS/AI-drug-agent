"""Step 15 API — ip_risk_integrated_shortlist scaffold."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.ip_risk_integration_service import IPRiskIntegrationService
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/15", tags=["step-15-ip-risk"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_15(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_15_ip_risk_integration")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 15 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_15_ip_risk_integration",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    svc = IPRiskIntegrationService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.integrate(run_id).model_dump()
