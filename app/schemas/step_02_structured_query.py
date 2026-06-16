"""Step 2 — structured_query (SupervisorAgent output)."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SourceRawRequestRef(BaseModel):
    raw_request_record_id: str


class TaskIntent(BaseModel):
    task_type: str
    task_type_confidence: float = 0.0
    modality: str = "ADC"
    modality_confidence: float = 0.0
    user_goal_summary: str = ""


class MentionedEntities(BaseModel):
    target_or_antigen_text: Optional[str] = None
    disease_or_indication_text: Optional[str] = None
    antibody_candidate_text: Optional[str] = None
    payload_text: Optional[str] = None
    linker_text: Optional[str] = None


class StructuredQuery(BaseModel):
    run_id: str
    step_id: str = "step_02_structured_query"
    parsed_at: str
    source_raw_request_ref: SourceRawRequestRef
    task_intent: TaskIntent
    mentioned_entities: MentionedEntities = Field(default_factory=MentionedEntities)
    referenced_inputs: list[dict] = Field(default_factory=list)
    requested_outputs: list[str] = Field(default_factory=list)
    user_constraints: list[dict] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
