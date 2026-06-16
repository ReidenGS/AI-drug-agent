"""Step 14 API — patent_prior_art_table (PatentIPAgent)."""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.patent_ip_agent import PatentIPAgent
from ..deps import (
    get_llm_provider,
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/14", tags=["step-14-patent-ip"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_14(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_14_patent_ip")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 14 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_14_patent_ip",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    agent = PatentIPAgent(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
        llm=get_llm_provider(),
    )
    return agent.run(run_id).model_dump()
