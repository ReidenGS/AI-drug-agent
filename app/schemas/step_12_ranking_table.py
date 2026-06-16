"""Step 12 — deterministic ranking_table.

If Step 11 returned `awaiting_external_input`, Step 12 emits a table with
`ranking_status="awaiting_external_scoring"` and an empty `ranked_candidates`
list. We never invent scores.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


RankingStatus = Literal[
    "awaiting_external_scoring",
    "completed",
    "failed",
    "skipped",
]


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    final_rank_score: float
    notes: Optional[str] = None


class RankingTable(BaseModel):
    run_id: str
    step_id: str = "step_12_ranking"
    created_at: str
    ranking_status: RankingStatus = "awaiting_external_scoring"
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    shortlist_size: int = 0
    storage_path: str = ""
    source_scoring_validation_id: Optional[str] = None
    notes: Optional[str] = None
