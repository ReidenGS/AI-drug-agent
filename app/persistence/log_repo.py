"""LOG#{timestamp}#{log_id}."""

from __future__ import annotations

from .dynamodb_client import DynamoDBClient
from ..utils.time import now_iso


class LogRepo:
    def __init__(self, ddb: DynamoDBClient) -> None:
        self.ddb = ddb

    def append(self, run_id: str, log_id: str, payload: dict) -> None:
        self.ddb.put({
            "PK": f"RUN#{run_id}",
            "SK": f"LOG#{now_iso()}#{log_id}",
            **payload,
        })
