"""Step 4 API — run_step_plan (deterministic)."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.workflow_setup_service import WorkflowSetupService

router = APIRouter(prefix="/runs/{run_id}/steps/4", tags=["step-04-workflow-setup"])


@router.post("/execute")
def execute_step_04(run_id: str) -> dict:
    svc = WorkflowSetupService(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.plan(run_id).model_dump()
