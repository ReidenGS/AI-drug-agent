"""Step 6 API — structured_liability_summary (DevelopabilityAgent).

Same Step 4 plan-gate as Step 5: plan_status `wait_for_input` / `blocked`
short-circuits the agent and returns 409 with structured detail.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.developability_agent import DevelopabilityAgent
from ..deps import (
    get_llm_provider,
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/6", tags=["step-06-developability"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_06(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_06_developability")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 6 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_06_developability",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    agent = DevelopabilityAgent(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
        llm=get_llm_provider(),
    )
    return agent.run(run_id).model_dump()
