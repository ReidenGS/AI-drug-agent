"""ID generators for run/artifact/tool-call/registry."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone


def _today_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def new_run_id() -> str:
    """Mint a unique run id.

    Format: ``run_YYYYMMDD_<8 hex chars>``. The date prefix matches the
    canonical `ADC_metadata_template.json` convention (`run_YYYYMMDD_*`) so
    humans can still tell when a run was created, but the random suffix
    eliminates collisions between concurrent intakes (smoke scripts, multi-
    user requests, parallel CI workers). Storage layout / registry / API
    consumers only require the value to be `str` and start with ``run_``.
    """
    return f"run_{_today_compact()}_{secrets.token_hex(4)}"


def new_tool_call_id() -> str:
    return f"tc_{secrets.token_hex(8)}"


def new_artifact_id(artifact_type: str) -> str:
    return f"{artifact_type}_{secrets.token_hex(6)}"


def new_registry_id() -> str:
    return f"reg_{secrets.token_hex(6)}"


def new_file_id() -> str:
    return f"file_{secrets.token_hex(6)}"
