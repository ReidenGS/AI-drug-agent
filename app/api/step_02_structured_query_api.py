"""Step 2 API — structured_query.

Delegates to `StructuredQueryService`, which composes `SupervisorAgent` with
the configured `LLMProvider` (Mock by default; Gemini when configured).
Note: this file does NOT import google-genai or python-a2a directly —
that wiring lives in `app/llm/` and `app/a2a/` respectively.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..agents.supervisor_agent import SupervisorAgent
from ..deps import (
    get_llm_provider,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from ..services.structured_query_service import StructuredQueryService

router = APIRouter(prefix="/runs/{run_id}/steps/2", tags=["step-02-structured-query"])


@router.post("/execute")
def execute_step_02(run_id: str) -> dict:
    supervisor = SupervisorAgent(llm=get_llm_provider())
    svc = StructuredQueryService(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        supervisor=supervisor,
    )
    return svc.parse(run_id).model_dump()
