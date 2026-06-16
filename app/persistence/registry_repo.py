"""REGISTRY#CURRENT and REGISTRY_SNAPSHOT#{version}."""

from __future__ import annotations

from .dynamodb_client import DynamoDBClient


class RegistryRepo:
    def __init__(self, ddb: DynamoDBClient) -> None:
        self.ddb = ddb

    def put_current(self, run_id: str, registry: dict) -> None:
        self.ddb.put({"PK": f"RUN#{run_id}", "SK": "REGISTRY#CURRENT", **registry})

    def put_snapshot(self, run_id: str, version: int, registry: dict) -> None:
        self.ddb.put({"PK": f"RUN#{run_id}", "SK": f"REGISTRY_SNAPSHOT#{version:04d}", **registry})
