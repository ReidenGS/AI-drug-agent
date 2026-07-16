"""Run-level endpoints: create a run and inspect its state."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.intake_service import IntakeService
from ..services.raw_request_authority import (
    RawRequestAuthorityError,
    load_active_raw_request,
)
from ..utils.errors import NotFoundError
from ..utils.ids import SessionId

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    raw_user_query: str
    session_id: Optional[str] = None
    entry_source: str = "api"
    submitted_by: Optional[str] = None
    user_provided_context: Optional[dict] = None
    uploaded_files: Optional[list[dict]] = None


@router.post("")
def create_run(req: CreateRunRequest) -> dict:
    payload = req.model_dump(exclude_none=True)
    if req.session_id is not None:
        try:
            payload["session_id"] = TypeAdapter(SessionId).validate_python(
                req.session_id, strict=True
            )
        except ValidationError:
            raise HTTPException(
                status_code=422, detail="session_id_invalid"
            ) from None
    storage = get_storage()
    intake = IntakeService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    record = intake.submit(**payload)
    return {
        "run_id": record.run_id,
        "session_id": record.session_id,
        "raw_request_record": record.model_dump(),
    }


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    storage = get_storage()
    state_key = storage.run_key(run_id, "state/workflow_state.json")
    if not storage.exists(state_key):
        raise NotFoundError(f"run {run_id} not found")
    try:
        raw = load_active_raw_request(
            run_id=run_id,
            registry=get_registry_service(),
            storage=storage,
        )
    except RawRequestAuthorityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {
        "run_id": run_id,
        "session_id": raw.session_id,
        "workflow_state": storage.read_json(state_key),
        "registry": get_registry_service().get(run_id).model_dump(),
    }
