"""TOOL_CALL#{step_id}#{agent_name}#{tool_call_id}."""

from __future__ import annotations

from ..schemas.common import ToolCallRecord
from .dynamodb_client import DynamoDBClient


class ToolCallRepo:
    def __init__(self, ddb: DynamoDBClient) -> None:
        self.ddb = ddb

    def put(self, run_id: str, rec: ToolCallRecord) -> None:
        self.ddb.put({
            "PK": f"RUN#{run_id}",
            "SK": f"TOOL_CALL#{rec.step_id}#{rec.agent_name}#{rec.tool_call_id}",
            **rec.model_dump(),
        })
