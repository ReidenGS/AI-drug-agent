"""Step 5 API — candidate_context_table (CandidateContextAgent).

Uses the shared inventory-scoped MCP client from `app.deps.get_mcp_client`.
The Step 4 `run_step_plan` is consulted as a hard gate before the agent runs:
- plan_status `wait_for_input` / `blocked` → 409 with structured detail,
  the agent is never invoked and no Step 5 artifact is written.
- planned_status `skip` / `blocked` for Step 5 → same.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.candidate_context_agent import CandidateContextAgent
from ..deps import (
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/5", tags=["step-05-candidate-context"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_05(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_05_candidate_context")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 5 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_05_candidate_context",
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            },
        )
    agent = CandidateContextAgent(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
    )
    return agent.run(run_id).model_dump()
