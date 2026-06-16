"""Job envelope for SQS-driven step execution."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class StepJob(BaseModel):
    run_id: str
    step_id: str
    payload_ref: Optional[str] = None     # S3 key to large payload; None for inline
    inline_payload: Optional[dict] = None
    idempotency_key: Optional[str] = None
