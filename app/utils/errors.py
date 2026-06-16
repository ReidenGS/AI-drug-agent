"""Unified error types and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ADCError(Exception):
    status_code: int = 500
    code: str = "adc_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(ADCError):
    status_code = 404
    code = "not_found"


class SchemaValidationError(ADCError):
    status_code = 422
    code = "schema_validation_error"


class WorkflowStateError(ADCError):
    status_code = 409
    code = "workflow_state_error"


class ToolNotAllowedError(ADCError):
    status_code = 403
    code = "tool_not_allowed"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ADCError)
    async def _handle(_: Request, exc: ADCError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "detail": exc.detail},
        )
