"""Step 3 clarification-loop schemas.

Minimal backend artifacts for the multi-turn clarification loop. There is NO
LangGraph memory / checkpointer here — clarification state is a normal
artifact persisted via the existing LocalStorage + ArtifactRegistryService
pattern, exactly like every other step output.

`ClarificationAnswer` is the user's short follow-up to a single
`ClarificationRequest`. `ClarificationState` records a same-run clarification
revision, its source authority, replay fingerprint, phase status, and compact
Step 2/3 output identities.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer_text: str
    answered_at: str
    source: str = "user"
    target_slot_name: Optional[str] = None
    target_slot_category: Optional[str] = None


class ClarificationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    revision_id: str = Field(pattern=r"^clarification_state_[0-9a-f]{12}$")
    revision_number: int = Field(ge=1)
    step_id: str = "step_03_clarification_state"
    source_input_readiness_status_id: str
    source_structured_query_id: str
    source_raw_request_record_id: str
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)
    resolved_request_ids: list[str] = Field(default_factory=list)
    unresolved_request_ids: list[str] = Field(default_factory=list)
    submission_fingerprint: str = Field(
        pattern=r"^clarification_submission_[0-9a-f]{64}$"
    )
    revision_status: Literal[
        "submitted", "step2_completed", "completed", "reparse_failed"
    ] = "submitted"
    output_structured_query_id: Optional[str] = None
    output_input_readiness_status_id: Optional[str] = None
    failure_code: Optional[
        Literal["clarification_step2_failed", "clarification_step3_failed"]
    ] = None
    created_at: str = ""
    updated_at: str = ""
