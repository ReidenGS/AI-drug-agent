"""Execution policies for the ADC LangGraph."""

from __future__ import annotations

from enum import Enum


class ExecutionPolicy(str, Enum):
    DEFAULT = "default"             # serial 1→14
    COST_SAVING = "cost_saving"     # Step 6 first, then 7/8 only for acceptable/review candidates
    PARALLEL = "parallel"           # 6 and 7/8 concurrent

    # Step 13 and 14 always run concurrently when reached.
