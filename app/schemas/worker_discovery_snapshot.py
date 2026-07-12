"""Worker discovery snapshot (Turn D).

Additive, LLM-safe persisted record of one Orchestrator run's worker discovery
result. It stores ONLY compact routing metadata:

- per-worker :class:`WorkerStatusSummary` (availability + failure reason + compact
  discovery_error code; never an endpoint URL, raw exception, or response body),
- the compact LLM-safe card catalog,
- available / unavailable agent id lists.

It never stores full ``python_a2a.AgentCard`` objects, endpoints, auth, HTTP
response bodies, or any artifact body. The full AgentCard cache lives only in
process memory (``WorkerDiscoveryRunCache``) and is used by the deterministic
validator / dispatcher, never persisted and never placed in an LLM prompt.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..a2a.contracts import WorkerStatusSummary

_FORBID = ConfigDict(extra="forbid")

DiscoveryStatus = Literal["all_available", "partially_available", "unavailable"]


class WorkerDiscoverySnapshot(BaseModel):
    """Persisted, compact result of one run's worker discovery."""

    model_config = _FORBID

    run_id: str
    created_at: str
    discovery_status: DiscoveryStatus
    worker_statuses: list[WorkerStatusSummary] = Field(default_factory=list)
    # Compact, LLM-safe catalog (fixed order: Step5, Step6, Structure). Each entry
    # is the compact card view augmented with availability — never a URL/auth/body.
    compact_card_catalog: list[dict[str, Any]] = Field(default_factory=list)
    available_agent_ids: list[str] = Field(default_factory=list)
    unavailable_agent_ids: list[str] = Field(default_factory=list)
    cache_frozen: bool = True


__all__ = ["WorkerDiscoverySnapshot", "DiscoveryStatus"]
