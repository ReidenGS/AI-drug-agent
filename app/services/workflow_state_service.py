"""Track Step 1-14 execution state per run.

Persists a small state document per run; LangGraph node functions update it
between transitions. SQS job tracking lives here too.
"""

from __future__ import annotations

from typing import Literal

from ..utils.time import now_iso
from .storage_service import AtomicJsonStorage, Storage

_STATE_KEY = "state/workflow_state.json"

StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


class WorkflowStateService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def init_run(self, run_id: str) -> dict:
        state = {
            "run_id": run_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "steps": {f"step_{i:02d}": "pending" for i in range(1, 15)},
            "current_step": "step_01",
        }
        self._save(run_id, state)
        return state

    def get(self, run_id: str) -> dict:
        return self.storage.read_json(self.storage.run_key(run_id, _STATE_KEY))

    def mark(self, run_id: str, step_key: str, status: StepStatus) -> dict:
        key = self.storage.run_key(run_id, _STATE_KEY)

        def mutate(state: dict) -> dict:
            state["steps"][step_key] = status
            state["updated_at"] = now_iso()
            if status == "running":
                state["current_step"] = step_key
            return state

        if isinstance(self.storage, AtomicJsonStorage):
            return self.storage.atomic_update_json(key, mutate)
        # S3Storage has no conditional-write/CAS contract yet; retain the
        # existing single-writer semantics rather than claiming local locking.
        state = mutate(self.storage.read_json(key))
        self.storage.write_json(key, state)
        return state

    def _save(self, run_id: str, state: dict) -> None:
        self.storage.write_json(self.storage.run_key(run_id, _STATE_KEY), state)
