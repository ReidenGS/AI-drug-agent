"""Run-level endpoints: create a run and inspect its state."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.intake_service import IntakeService
from ..utils.errors import NotFoundError

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    raw_user_query: str
    entry_source: str = "api"
    submitted_by: Optional[str] = None
    user_provided_context: Optional[dict] = None
    uploaded_files: Optional[list[dict]] = None


@router.post("")
def create_run(req: CreateRunRequest) -> dict:
    storage = get_storage()
    intake = IntakeService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    record = intake.submit(**req.model_dump(exclude_none=True))
    return {"run_id": record.run_id, "raw_request_record": record.model_dump()}


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    storage = get_storage()
    state_key = storage.run_key(run_id, "state/workflow_state.json")
    if not storage.exists(state_key):
        raise NotFoundError(f"run {run_id} not found")
    return {
        "run_id": run_id,
        "workflow_state": storage.read_json(state_key),
        "registry": get_registry_service().get(run_id).model_dump(),
    }
