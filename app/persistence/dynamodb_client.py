"""DynamoDB single-table client.

Key model (architecture v0.1):
    PK = RUN#{run_id}
    SK = one of:
        RUN_METADATA
        REGISTRY#CURRENT
        REGISTRY_SNAPSHOT#{version}
        STEP#{step_id}
        ARTIFACT#{artifact_type}#{artifact_id}
        TOOL_CALL#{step_id}#{agent_name}#{tool_call_id}
        AGENT_STATE#{agent_name}#{step_id}
        A2A_TASK#{source_agent}#{target_agent}#{task_id}
        MCP_TOOL#{tool_name}
        LOG#{timestamp}#{log_id}
"""

from __future__ import annotations

from typing import Any


class DynamoDBClient:
    def __init__(self, table: str, region: str) -> None:
        self.table_name = table
        self.region = region
        self._table: Any | None = None

    @property
    def table(self) -> Any:
        if self._table is None:
            import boto3

            self._table = boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)
        return self._table

    def put(self, item: dict) -> None:
        self.table.put_item(Item=item)

    def get(self, pk: str, sk: str) -> dict | None:
        resp = self.table.get_item(Key={"PK": pk, "SK": sk})
        return resp.get("Item")

    def query_pk(self, pk: str, sk_prefix: str | None = None) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        key = Key("PK").eq(pk)
        if sk_prefix:
            key &= Key("SK").begins_with(sk_prefix)
        return self.table.query(KeyConditionExpression=key).get("Items", [])
