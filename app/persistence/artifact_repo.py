"""ARTIFACT#{artifact_type}#{artifact_id}."""

from __future__ import annotations

from .dynamodb_client import DynamoDBClient


class ArtifactRepo:
    def __init__(self, ddb: DynamoDBClient) -> None:
        self.ddb = ddb

    def put(self, run_id: str, artifact_type: str, artifact_id: str, item: dict) -> None:
        self.ddb.put({
            "PK": f"RUN#{run_id}",
            "SK": f"ARTIFACT#{artifact_type}#{artifact_id}",
            **item,
        })
