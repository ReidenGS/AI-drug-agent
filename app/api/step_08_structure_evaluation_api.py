"""Step 8 API — structure_prediction_and_interface_results."""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.structure_and_design_agent import StructureAndDesignAgent
from ..deps import (
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/8", tags=["step-08-structure-evaluation"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_08(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_08_structure_evaluation")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 8 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_08_structure_evaluation",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    agent = StructureAndDesignAgent(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
    )
    return agent.run_step_8(run_id).model_dump()
