"""Step 9 API — compound_screening_artifact.

MVP only emits the compound-screening lane. The protein-design lane
(`protein_design_artifact`, RFdiffusion / ProteinMPNN / ESM) is deferred:
this agent must NOT freely emit RFdiffusion `contigs_dsl` — that path
belongs in services/structure_service.py per architecture v0.1.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.structure_and_design_agent import StructureAndDesignAgent
from ..deps import (
    get_llm_provider,
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.workflow_setup_service import execution_decision
from ..utils.errors import WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/9", tags=["step-09-structure-design"])


def _load_plan(storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    return storage.read_json(key) if storage.exists(key) else None


@router.post("/execute")
def execute_step_09(run_id: str) -> dict:
    storage = get_storage()
    decision = execution_decision(_load_plan(storage, run_id), "step_09_structure_design")
    if not decision.allow:
        raise WorkflowStateError(
            "Step 9 cannot execute under current Step 4 plan",
            detail={
                "step_id": "step_09_structure_design",
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
        llm=get_llm_provider(),
    )
    return agent.run_step_9(run_id).model_dump()
