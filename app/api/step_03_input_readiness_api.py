"""Step 3 API — input_readiness_status (deterministic)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..a2a.orchestrator_readiness import OrchestratorReadinessError
from ..agents.supervisor_agent import SupervisorAgent
from ..deps import (
    get_llm_provider,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..schemas.step_03_input_readiness import ClarificationRequest
from ..services.clarification_service import (
    ClarificationConflictError,
    ClarificationRequestError,
    ClarificationService,
)
from ..services.input_readiness_service import InputReadinessService
from ..services.raw_request_authority import (
    RawRequestAuthorityError,
    load_active_raw_request,
)
from ..utils.ids import SessionId

router = APIRouter(prefix="/runs/{run_id}/steps/3", tags=["step-03-input-readiness"])


class ClarificationAnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    answered_at: str = Field(min_length=1)


class ClarificationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[ClarificationAnswerInput] = Field(min_length=1)


class ClarificationRoundResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    session_id: SessionId
    clarification_revision_id: str
    input_readiness_status: Literal["ready", "needs_user_input", "blocked"]
    response: str | None = None
    clarification_requests: list[ClarificationRequest]


@router.post("/execute")
def execute_step_03(run_id: str) -> dict:
    svc = InputReadinessService(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    return svc.check(run_id).model_dump()


@router.post(
    "/clarifications", response_model=ClarificationRoundResponse
)
def submit_step_03_clarifications(
    run_id: str, submission: ClarificationSubmission
) -> ClarificationRoundResponse:
    """Persist one clarification revision and reparse Step 2/3 in this run."""

    storage = get_storage()
    registry = get_registry_service()
    workflow_state = get_workflow_state_service()
    try:
        source = load_active_raw_request(
            run_id=run_id, registry=registry, storage=storage
        )
        result = ClarificationService(
            storage, registry, workflow_state
        ).submit_and_reparse(
            run_id,
            [answer.model_dump() for answer in submission.answers],
            SupervisorAgent(llm=get_llm_provider()),
        )
        readiness = result.input_readiness_status
        if readiness is None:
            raise RuntimeError("clarification_readiness_missing")
        current_raw = load_active_raw_request(
            run_id=run_id,
            registry=registry,
            storage=storage,
        )
        if current_raw.session_id != source.session_id:
            raise RuntimeError("clarification_session_mismatch")
    except (
        RawRequestAuthorityError,
        OrchestratorReadinessError,
        ClarificationConflictError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ClarificationRequestError:
        raise HTTPException(
            status_code=422, detail="clarification_request_invalid"
        ) from None
    except Exception:
        raise HTTPException(
            status_code=503, detail="clarification_reparse_failed"
        ) from None
    assert source.session_id is not None
    return ClarificationRoundResponse(
        run_id=run_id,
        session_id=source.session_id,
        clarification_revision_id=result.state.revision_id,
        input_readiness_status=readiness.input_readiness_status,
        response=readiness.response,
        clarification_requests=readiness.clarification_requests,
    )
