"""Step 3 clarification-loop schemas.

Minimal backend artifacts for the multi-turn clarification loop. There is NO
LangGraph memory / checkpointer here — clarification state is a normal
artifact persisted via the existing LocalStorage + ArtifactRegistryService
pattern, exactly like every other step output.

`ClarificationAnswer` is the user's short follow-up to a single
`ClarificationRequest`. `ClarificationState` records a clarification turn:
which requests were answered (resolved) vs still open (unresolved), the
source artifacts it was derived from, and the next revision run created so
Step 2 can re-parse with the previous intent + the new answers.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ClarificationAnswer(BaseModel):
    request_id: str
    answer_text: str
    answered_at: str
    source: str = "user"
    target_slot_name: Optional[str] = None
    target_slot_category: Optional[str] = None


class ClarificationState(BaseModel):
    run_id: str
    step_id: str = "step_03_clarification_state"
    source_input_readiness_status_id: str
    source_structured_query_id: str
    source_raw_request_record_id: str
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)
    resolved_request_ids: list[str] = Field(default_factory=list)
    unresolved_request_ids: list[str] = Field(default_factory=list)
    # The revision run created from these answers (so the loop can re-run
    # Step 2/3 without overwriting the original run's artifacts).
    next_run_id: Optional[str] = None
    next_raw_request_record_id: Optional[str] = None
    created_at: str = ""
