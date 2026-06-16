"""S3 storage adapter (boto3). Skeleton only — wire up in non-dev deployments."""

from __future__ import annotations

import json
from typing import Any


class S3Storage:
    def __init__(self, bucket: str, prefix: str, region: str) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # lazy import so dev mode does not need AWS creds

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def write_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def read_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def write_json(self, key: str, payload: dict) -> str:
        return self.write_bytes(
            key, json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        )

    def read_json(self, key: str) -> dict:
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_prefix(self, key_prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                out.append(obj["Key"])
        return out

    def run_key(self, run_id: str, *parts: str) -> str:
        return "/".join([self.prefix, "runs", run_id, *parts])
