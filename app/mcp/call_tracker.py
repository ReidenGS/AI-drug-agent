"""Record every MCP tool call to DDB `TOOL_CALL#{step_id}#{agent}#{tool_call_id}`."""

from __future__ import annotations

from typing import Any
from ..schemas.common import ToolCallRecord
from ..utils.ids import new_tool_call_id
from ..utils.time import now_iso


class MCPCallTracker:
    def __init__(self, tool_call_repo: Any | None = None) -> None:
        self.tool_call_repo = tool_call_repo

    def open(self, *, run_id: str, step_id: str, agent_name: str, tool_name: str, idempotency_key: str | None = None) -> ToolCallRecord:
        rec = ToolCallRecord(
            tool_call_id=new_tool_call_id(),
            tool_name=tool_name,
            agent_name=agent_name,
            step_id=step_id,
            run_status="pending",
            started_at=now_iso(),
            idempotency_key=idempotency_key,
        )
        if self.tool_call_repo is not None:
            self.tool_call_repo.put(run_id, rec)
        return rec

    def close(self, run_id: str, rec: ToolCallRecord, *, status: str, output_artifact_id: str | None = None, error: str | None = None) -> ToolCallRecord:
        rec = rec.model_copy(update={
            "run_status": status,  # type: ignore[arg-type]
            "finished_at": now_iso(),
            "tool_output_artifact_id": output_artifact_id,
            "error_message": error,
        })
        if self.tool_call_repo is not None:
            self.tool_call_repo.put(run_id, rec)
        return rec
