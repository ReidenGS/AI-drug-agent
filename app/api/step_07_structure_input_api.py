"""Step 7 API — prepared_structure_input_package.

Uses the shared inventory-scoped MCP client and honours the Step 4
`execution_decision` gate exactly like Step 5/6.
"""

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

router = APIRouter(prefix="/runs/{run_id}/steps/7", tags=["step-07-structure-input"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_07(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_07_structure_input")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 7 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_07_structure_input",
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
    return agent.run_step_7(run_id).model_dump()
