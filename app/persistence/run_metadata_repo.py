"""RUN_METADATA / STEP#{step_id} / AGENT_STATE# reads & writes."""

from __future__ import annotations

from .dynamodb_client import DynamoDBClient


class RunMetadataRepo:
    def __init__(self, ddb: DynamoDBClient) -> None:
        self.ddb = ddb

    def put_run(self, run_id: str, meta: dict) -> None:
        self.ddb.put({"PK": f"RUN#{run_id}", "SK": "RUN_METADATA", **meta})

    def put_step(self, run_id: str, step_id: str, item: dict) -> None:
        self.ddb.put({"PK": f"RUN#{run_id}", "SK": f"STEP#{step_id}", **item})
