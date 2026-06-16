"""Abstract storage interface.

Path template (per ADC_S3_Structure.md):
    adc_pilot/runs/{run_id}/{inputs|tool_outputs|evidence|patent_review|logs}/...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    prefix: str

    def write_bytes(self, key: str, data: bytes) -> str: ...
    def read_bytes(self, key: str) -> bytes: ...
    def write_json(self, key: str, payload: dict) -> str: ...
    def read_json(self, key: str) -> dict: ...
    def exists(self, key: str) -> bool: ...
    def list_prefix(self, key_prefix: str) -> list[str]: ...
    def delete(self, key: str) -> bool:
        """Delete a stored key. Returns True if the key existed.

        Must be idempotent: deleting a missing key is not an error.
        """
        ...

    def run_key(self, run_id: str, *parts: str) -> str:
        return "/".join([self.prefix, "runs", run_id, *parts])
