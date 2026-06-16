"""Step 11 — deterministic scoring table validation.

The validator does not produce scores; it validates a scoring result file the
external module dropped at `inputs/external_scoring_result.json`. Raw rows
stay in that file; the normalized artifact carries only:
- per-row issue list (severity + field + message)
- the set of validated candidate ids that survived
- a storage ref back to the raw input
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


ValidationStatus = Literal[
    "awaiting_external_input",
    "completed",
    "completed_with_warnings",
    "failed",
]


class ValidationIssue(BaseModel):
    candidate_id: Optional[str] = None
    severity: Literal["warning", "error"]
    field: Optional[str] = None
    message: str


class ScoringValidation(BaseModel):
    run_id: str
    step_id: str = "step_11_scoring_validation"
    created_at: str
    validation_status: ValidationStatus = "awaiting_external_input"
    external_scoring_input_ref: Optional[str] = None
    scoring_table_storage_path: str = ""
    validated_candidate_ids: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    row_count: int = 0
    notes: Optional[str] = None
