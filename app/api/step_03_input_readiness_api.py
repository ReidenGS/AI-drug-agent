"""Step 3 API — input_readiness_status (deterministic)."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.input_readiness_service import InputReadinessService

router = APIRouter(prefix="/runs/{run_id}/steps/3", tags=["step-03-input-readiness"])


@router.post("/execute")
def execute_step_03(run_id: str) -> dict:
    svc = InputReadinessService(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.check(run_id).model_dump()
