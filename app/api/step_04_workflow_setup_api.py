"""Production Step 4 Orchestrator API over durable HTTP A2A execution."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from ..a2a.orchestrator_application_service import (
    OrchestratorApplicationServiceError,
    unavailable_orchestrator_response,
)
from ..schemas.orchestrator_api import OrchestratorStep4Response
from ..schemas.orchestrator_execution_state import RunId

router = APIRouter(prefix="/runs/{run_id}/steps/4", tags=["step-04-orchestrator"])


@router.post("/execute", response_model=OrchestratorStep4Response)
async def execute_step_04(
    run_id: str, request: Request
) -> OrchestratorStep4Response | JSONResponse:
    """Start a fresh durable run, or idempotently resume its checkpoint."""
    checked_run_id = _validated_run_id(run_id)
    if checked_run_id is None:
        return _error_response(
            None, "orchestrator_request_invalid", status_code=422
        )
    run_id = checked_run_id
    service = _service(request)
    if service is None:
        return _error_response(
            run_id,
            _unavailable_code(request),
            status_code=503,
        )
    try:
        return await service.execute(run_id)
    except OrchestratorApplicationServiceError as exc:
        return _service_error_response(run_id, exc)


@router.post("/resume", response_model=OrchestratorStep4Response)
async def resume_step_04(
    run_id: str, request: Request
) -> OrchestratorStep4Response | JSONResponse:
    """Explicitly resume/reconcile one existing durable checkpoint."""
    checked_run_id = _validated_run_id(run_id)
    if checked_run_id is None:
        return _error_response(
            None, "orchestrator_request_invalid", status_code=422
        )
    run_id = checked_run_id
    service = _service(request)
    if service is None:
        return _error_response(
            run_id,
            _unavailable_code(request),
            status_code=503,
        )
    try:
        return await service.resume(run_id)
    except OrchestratorApplicationServiceError as exc:
        return _service_error_response(run_id, exc)


@router.get("/status", response_model=OrchestratorStep4Response)
async def status_step_04(
    run_id: str, request: Request
) -> OrchestratorStep4Response | JSONResponse:
    """Read durable compact orchestration state without network dispatch."""
    checked_run_id = _validated_run_id(run_id)
    if checked_run_id is None:
        return _error_response(
            None, "orchestrator_request_invalid", status_code=422
        )
    run_id = checked_run_id
    service = _service(request)
    if service is None:
        return _error_response(
            run_id,
            _unavailable_code(request),
            status_code=503,
        )
    try:
        return await service.status(run_id)
    except OrchestratorApplicationServiceError as exc:
        return _service_error_response(run_id, exc)


def _service(request: Request) -> Any | None:
    return getattr(request.app.state, "orchestrator_service", None)


def _validated_run_id(value: str) -> str | None:
    try:
        return TypeAdapter(RunId).validate_python(value, strict=True)
    except ValidationError:
        return None


def _unavailable_code(request: Request) -> str:
    code = getattr(
        request.app.state,
        "orchestrator_unavailable_code",
        "orchestrator_checkpoint_runtime_unavailable",
    )
    return code or "orchestrator_checkpoint_runtime_unavailable"


def _service_error_response(
    run_id: str, exc: OrchestratorApplicationServiceError
) -> JSONResponse:
    code = str(exc)
    status = 404 if code == "orchestrator_checkpoint_not_found" else 503
    return _error_response(run_id, code, status_code=status)


def _error_response(
    run_id: str | None, code: str, *, status_code: int
) -> JSONResponse:
    response = unavailable_orchestrator_response(run_id, code)
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json"),
    )


__all__ = ["router"]
