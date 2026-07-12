"""Orchestrator worker discovery + deterministic target validation (Turn D).

This is the Step 4 Orchestrator's run-start worker discovery. It fetches each
configured worker's AgentCard over REAL HTTP A2A, validates the card + health
against a fixed deployment identity contract, freezes a per-run full AgentCard
cache, builds an LLM-safe compact catalog, persists a compact discovery snapshot,
and offers deterministic ``resolve_dispatch_target`` validation.

It does NOT (Turn D boundary):

- construct or send ``python_a2a.Task`` / ``WorkerExecutionRequest`` (no dispatch),
- run any LLM, MCP tool, or LangGraph,
- import or call any worker agent business method,
- build a local AgentCard to substitute for HTTP discovery,
- fall back to an in-process direct call on any failure.

Discovery flow (per worker, see orchestrator_routing_design.md
"Run-start AgentCard Discovery"):

    Settings endpoint
    -> python_a2a.A2AClient(endpoint).get_agent_card()   (real HTTP)
    -> validate adc_agent_contract + expected identity/capabilities/url
    -> GET endpoint/health (real HTTP) + validate health identity
    -> available -> freeze into full AgentCard cache
    -> build compact LLM-safe catalog entry
    -> persist compact snapshot
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests
from python_a2a import A2AClient, AgentCard

from .agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    AdcAgentContract,
    AgentContractError,
    build_compact_card_for_agent,
    parse_adc_agent_contract,
)
from .contracts import WorkerStatusSummary
from .worker_server import effective_url_port

# Compact, stable discovery_error codes have a hard upper bound so a raw
# exception / endpoint / response body can never leak through this field.
MAX_DISCOVERY_ERROR_LEN = 64

_DISPATCH_MODE = "python_a2a"


# ─────────────────────────────────────────────────────────────────────────────
# Public exceptions
# ─────────────────────────────────────────────────────────────────────────────
class WorkerUnavailableError(RuntimeError):
    """Raised by resolve_dispatch_target when the target worker is not available."""


class DispatchTargetValidationError(ValueError):
    """Raised by resolve_dispatch_target for an unknown/inconsistent target."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal classification signals (compact codes only — never raw detail)
# ─────────────────────────────────────────────────────────────────────────────
class _CardInvalid(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _HealthError(Exception):
    def __init__(self, *, agent_failure_reason: str, code: str) -> None:
        super().__init__(code)
        self.agent_failure_reason = agent_failure_reason
        self.code = code


# ─────────────────────────────────────────────────────────────────────────────
# Config + result types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExpectedWorkerEndpoint:
    """Fixed deployment identity contract for one worker.

    The Orchestrator NEVER trusts a remote card's self-declared agent_id; it must
    equal ``agent_id`` here, and its capability ids must equal ``capability_ids``.
    """

    agent_id: str
    capability_ids: tuple[str, ...]
    endpoint_url: str


@dataclass
class DiscoveredWorker:
    """Full (in-memory only) discovery record for one worker."""

    expected: ExpectedWorkerEndpoint
    availability: str
    agent_failure_reason: str
    discovery_error: Optional[str]
    worker_status: WorkerStatusSummary
    compact: dict[str, Any]
    # Present only when available; the full card / contract / dispatch url stay
    # in memory and are never persisted or put in an LLM prompt.
    agent_card: Optional[AgentCard] = None
    contract: Optional[AdcAgentContract] = None
    dispatch_url: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.availability == "available"


@dataclass
class DispatchTarget:
    """Deterministic, validated in-memory dispatch target (NOT sent this turn)."""

    agent_id: str
    capability_id: str
    dispatch_url: str
    dispatch_mode: str


@dataclass
class WorkerDiscoveryRunCache:
    """Canonical per-run full cache (content-frozen: built once, never
    re-discovered). The service keeps the single canonical instance internal and
    only ever hands out deep copies, so external mutation cannot change routing.
    """

    run_id: str
    workers: dict[str, DiscoveredWorker]
    ordered_agent_ids: list[str]
    snapshot: "WorkerDiscoverySnapshot"
    cache_frozen: bool = True


def default_expected_workers(settings: Any) -> list[ExpectedWorkerEndpoint]:
    """Fixed three-worker identity contract, in canonical order Step5/6/Structure."""
    return [
        ExpectedWorkerEndpoint(
            agent_id=AGENT_ID_STEP5,
            capability_ids=(CAP_STEP5_CANDIDATE_CONTEXT,),
            endpoint_url=settings.step5_worker_url,
        ),
        ExpectedWorkerEndpoint(
            agent_id=AGENT_ID_STEP6,
            capability_ids=(CAP_STEP6_DEVELOPABILITY,),
            endpoint_url=settings.step6_worker_url,
        ),
        ExpectedWorkerEndpoint(
            agent_id=AGENT_ID_STRUCTURE,
            capability_ids=(CAP_STRUCTURE_DESIGN_WORKFLOW,),
            endpoint_url=settings.structure_worker_url,
        ),
    ]


# Import placed after dataclass defs to avoid a forward-ref cycle at module import.
from ..schemas.worker_discovery_snapshot import (  # noqa: E402
    DiscoveryStatus,
    WorkerDiscoverySnapshot,
)
from ..utils.ids import new_artifact_id  # noqa: E402
from ..utils.time import now_iso  # noqa: E402

_SNAPSHOT_STORAGE_KEY = "inputs/worker_discovery_snapshot.json"


class WorkerDiscoveryService:
    """Per-run worker discovery + deterministic dispatch-target validation.

    Constructing the service performs NO network I/O. Only
    :meth:`discover_for_run` reaches the workers, and only once per ``run_id``.
    """

    def __init__(
        self,
        *,
        expected_workers: list[ExpectedWorkerEndpoint],
        storage: Any,
        registry: Any,
        discovery_timeout_seconds: float,
        health_timeout_seconds: float,
        client_factory: Callable[..., A2AClient] = A2AClient,
        http_get: Callable[..., Any] = requests.get,
    ) -> None:
        self._expected = list(expected_workers)
        self._storage = storage
        self._registry = registry
        self._discovery_timeout = float(discovery_timeout_seconds)
        self._health_timeout = float(health_timeout_seconds)
        self._client_factory = client_factory
        self._http_get = http_get

        self._run_caches: dict[str, WorkerDiscoveryRunCache] = {}
        self._meta_lock = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}

    # ── public API ──────────────────────────────────────────────────────────
    #
    # The CANONICAL per-run cache (``self._run_caches``) is the single source of
    # truth for deterministic routing and is NEVER handed out. Every public
    # accessor returns a defensive DEEP COPY, so a caller mutating a returned
    # snapshot / cache / catalog (its nested AgentCard, contract, availability,
    # compact capability lists, …) cannot alter the frozen routing state that
    # ``resolve_dispatch_target`` reads. "Frozen" means the *content* is fixed and
    # never re-discovered — not that the same Python object is returned twice.
    def discover_for_run(self, run_id: str) -> WorkerDiscoverySnapshot:
        """Discover all configured workers for ``run_id`` (once) and return a
        defensive deep copy of the frozen compact snapshot. Re-invocations return
        an equal (deep-copied) snapshot without any new HTTP call."""
        cached = self._run_caches.get(run_id)
        if cached is not None:
            return self._external_snapshot(cached)

        run_lock = self._get_run_lock(run_id)
        with run_lock:
            # Double-checked: another thread for the same run may have built it.
            cached = self._run_caches.get(run_id)
            if cached is not None:
                return self._external_snapshot(cached)
            cache = self._build_run_cache(run_id)
            self._run_caches[run_id] = cache
            return self._external_snapshot(cache)

    def get_full_card_cache(self, run_id: str) -> WorkerDiscoveryRunCache:
        """Return a defensive DEEP COPY of the canonical per-run cache (full
        AgentCards + validated contracts included). Mutating the returned copy
        cannot affect internal routing state."""
        return copy.deepcopy(self._canonical_cache(run_id))

    def get_compact_card_catalog(self, run_id: str) -> list[dict[str, Any]]:
        """Return a deep copy of the compact catalog; nested capability /
        routing entries are independent of the canonical cache."""
        return copy.deepcopy(self._canonical_cache(run_id).snapshot.compact_card_catalog)

    def _canonical_cache(self, run_id: str) -> WorkerDiscoveryRunCache:
        """Internal-only accessor for the canonical (mutable) per-run cache.
        Never expose its return value directly to callers."""
        cache = self._run_caches.get(run_id)
        if cache is None:
            raise DispatchTargetValidationError(
                f"no discovery cache for run_id '{run_id}'; call discover_for_run first"
            )
        return cache

    @staticmethod
    def _external_snapshot(cache: WorkerDiscoveryRunCache) -> WorkerDiscoverySnapshot:
        """Defensive deep copy of the frozen snapshot for external callers."""
        return cache.snapshot.model_copy(deep=True)

    def resolve_dispatch_target(
        self,
        run_id: str,
        *,
        agent_id: str,
        capability_id: str,
        dispatch_mode: str = _DISPATCH_MODE,
    ) -> DispatchTarget:
        """Deterministically validate and return an in-memory dispatch target.

        Reads the internal CANONICAL cache (never a caller-mutable copy). Does NOT
        build or send a Task. Raises :class:`WorkerUnavailableError` or
        :class:`DispatchTargetValidationError` on any inconsistency; never returns
        a worker object and never falls back to a local call.
        """
        cache = self._canonical_cache(run_id)  # canonical, never a defensive copy

        worker = cache.workers.get(agent_id)
        if worker is None:
            raise DispatchTargetValidationError(
                f"agent_id '{agent_id}' is not a configured worker"
            )
        if not worker.is_available:
            raise WorkerUnavailableError(
                f"worker '{agent_id}' is not available "
                f"(availability={worker.availability})"
            )
        if worker.agent_card is None or worker.contract is None or not worker.dispatch_url:
            raise DispatchTargetValidationError(
                f"worker '{agent_id}' has no validated card in the discovery cache"
            )

        card_capability_ids = {c.capability_id for c in worker.contract.capabilities}
        if capability_id not in card_capability_ids:
            raise DispatchTargetValidationError(
                f"capability '{capability_id}' is not served by worker '{agent_id}'"
            )
        if dispatch_mode != _DISPATCH_MODE:
            raise DispatchTargetValidationError(
                f"dispatch_mode '{dispatch_mode}' is not supported (only '{_DISPATCH_MODE}')"
            )
        if worker.contract.dispatch_modes != [_DISPATCH_MODE]:
            raise DispatchTargetValidationError(
                f"worker '{agent_id}' card does not allow dispatch_mode '{_DISPATCH_MODE}'"
            )
        if not worker.contract.routable:
            raise DispatchTargetValidationError(f"worker '{agent_id}' is not routable")
        if worker.contract.status != "active":
            raise DispatchTargetValidationError(
                f"worker '{agent_id}' status is '{worker.contract.status}', not active"
            )

        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=worker.dispatch_url,
            dispatch_mode=_DISPATCH_MODE,
        )

    # ── internal build ──────────────────────────────────────────────────────
    def _get_run_lock(self, run_id: str) -> threading.Lock:
        with self._meta_lock:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.Lock()
                self._run_locks[run_id] = lock
            return lock

    def _build_run_cache(self, run_id: str) -> WorkerDiscoveryRunCache:
        workers: dict[str, DiscoveredWorker] = {}
        ordered_ids: list[str] = []
        for expected in self._expected:
            discovered = self._discover_one(expected)
            workers[expected.agent_id] = discovered
            ordered_ids.append(expected.agent_id)

        available_ids = [aid for aid in ordered_ids if workers[aid].is_available]
        unavailable_ids = [aid for aid in ordered_ids if not workers[aid].is_available]

        if not unavailable_ids:
            status: DiscoveryStatus = "all_available"
        elif available_ids:
            status = "partially_available"
        else:
            status = "unavailable"

        snapshot = WorkerDiscoverySnapshot(
            run_id=run_id,
            created_at=now_iso(),
            discovery_status=status,
            worker_statuses=[workers[aid].worker_status for aid in ordered_ids],
            compact_card_catalog=[workers[aid].compact for aid in ordered_ids],
            available_agent_ids=available_ids,
            unavailable_agent_ids=unavailable_ids,
            cache_frozen=True,
        )
        self._persist_snapshot(run_id, snapshot)
        return WorkerDiscoveryRunCache(
            run_id=run_id,
            workers=workers,
            ordered_agent_ids=ordered_ids,
            snapshot=snapshot,
        )

    def _persist_snapshot(self, run_id: str, snapshot: WorkerDiscoverySnapshot) -> None:
        artifact_id = new_artifact_id("worker_discovery_snapshot")
        self._storage.write_json(
            self._storage.run_key(run_id, _SNAPSHOT_STORAGE_KEY),
            {"artifact_id": artifact_id, **snapshot.model_dump()},
        )
        self._registry.update_active(run_id, worker_discovery_snapshot_id=artifact_id)

    # ── per-worker discovery ────────────────────────────────────────────────
    def _discover_one(self, expected: ExpectedWorkerEndpoint) -> DiscoveredWorker:
        endpoint = expected.endpoint_url
        try:
            card = self._fetch_card(endpoint)
            if _is_unknown_fallback_card(card):
                # A2AClient swallowed a fetch failure and returned its default
                # "Unknown Agent" card. Classify WHY over real HTTP.
                net = self._classify_network_failure(endpoint)
                if net is not None:
                    return self._unavailable(expected, agent_failure_reason=net, code=net)
                # Reachable but no usable card body -> card_invalid via validation.
            contract = self._validate_card(card, expected, endpoint)
            self._check_health(expected, endpoint, contract)
        except _CardInvalid as exc:
            return self._unavailable(expected, agent_failure_reason="card_invalid", code=exc.code)
        except _HealthError as exc:
            return self._unavailable(
                expected, agent_failure_reason=exc.agent_failure_reason, code=exc.code
            )
        except Exception:  # noqa: BLE001 — never abort other workers, never leak detail
            return self._unavailable(
                expected, agent_failure_reason="server_error", code="server_error"
            )

        return self._available(expected, endpoint, card, contract)

    def _fetch_card(self, endpoint: str) -> AgentCard:
        """Obtain the AgentCard via the REQUIRED python_a2a.A2AClient path.

        A2AClient's constructor performs the well-known-URL fetch and, on any
        error, returns a default "Unknown Agent" card (it never raises) — so the
        caller detects that fallback and classifies the network cause separately.
        """
        client = self._client_factory(endpoint, timeout=self._client_timeout())
        return client.get_agent_card()

    def _client_timeout(self) -> int:
        # A2AClient takes an int timeout; keep it >= 1s and no shorter than the
        # configured discovery budget.
        return max(1, math.ceil(self._discovery_timeout))

    def _classify_network_failure(self, endpoint: str) -> Optional[str]:
        """Classify a discovery fetch failure over real HTTP. Returns a compact
        code (discovery_timeout / discovery_connection_failed / server_error) or
        None when the endpoint is actually reachable (bad body -> card_invalid)."""
        url = _join(endpoint, "/.well-known/agent.json")
        try:
            self._http_get(
                url, timeout=self._discovery_timeout, headers={"Accept": "application/json"}
            )
        except requests.Timeout:
            return "discovery_timeout"
        except requests.ConnectionError:
            return "discovery_connection_failed"
        except requests.RequestException:
            return "server_error"
        return None

    def _validate_card(
        self, card: AgentCard, expected: ExpectedWorkerEndpoint, endpoint: str
    ) -> AdcAgentContract:
        try:
            contract = parse_adc_agent_contract(card)
        except AgentContractError:
            raise _CardInvalid("adc_agent_contract_invalid")

        if contract.agent_role != "worker":
            raise _CardInvalid("agent_role_not_worker")
        if contract.agent_id != expected.agent_id:
            raise _CardInvalid("agent_id_mismatch")
        card_caps = {c.capability_id for c in contract.capabilities}
        if card_caps != set(expected.capability_ids):
            raise _CardInvalid("capability_ids_mismatch")
        if not contract.routable:
            raise _CardInvalid("not_routable")
        if contract.status != "active":
            raise _CardInvalid("status_not_active")
        if contract.dispatch_modes != [_DISPATCH_MODE]:
            raise _CardInvalid("dispatch_mode_invalid")

        # A2AClient normalizes the returned card's ``url`` to the endpoint it was
        # constructed with, so the worker's SELF-DECLARED url must be read from the
        # raw well-known discovery body. A worker may not redirect its dispatch URL
        # to another host/port.
        declared_url = self._declared_card_url(endpoint)
        if declared_url is None or effective_url_port(declared_url) is None:
            raise _CardInvalid("card_url_invalid")
        if _normalize_url(declared_url) != _normalize_url(endpoint):
            raise _CardInvalid("card_url_mismatch")
        return contract

    def _declared_card_url(self, endpoint: str) -> Optional[str]:
        """Read the worker's self-declared ``url`` from the raw well-known card
        body (A2AClient discards it by overwriting with the endpoint)."""
        url = _join(endpoint, "/.well-known/agent.json")
        try:
            resp = self._http_get(
                url, timeout=self._discovery_timeout, headers={"Accept": "application/json"}
            )
        except requests.RequestException:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(body, dict) and isinstance(body.get("url"), str):
            return body["url"]
        return None

    def _check_health(
        self, expected: ExpectedWorkerEndpoint, endpoint: str, contract: AdcAgentContract
    ) -> None:
        url = _join(endpoint, "/health")
        try:
            resp = self._http_get(
                url, timeout=self._health_timeout, headers={"Accept": "application/json"}
            )
        except requests.Timeout:
            raise _HealthError(agent_failure_reason="health_timeout", code="health_timeout")
        except requests.ConnectionError:
            raise _HealthError(agent_failure_reason="health_failed", code="health_connection_failed")
        except requests.RequestException:
            raise _HealthError(agent_failure_reason="health_failed", code="health_request_failed")

        if getattr(resp, "status_code", None) != 200:
            raise _HealthError(agent_failure_reason="health_failed", code="health_non_200")
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 — malformed health JSON
            raise _HealthError(agent_failure_reason="health_failed", code="health_bad_json")
        if not isinstance(body, dict):
            raise _HealthError(agent_failure_reason="health_failed", code="health_bad_json")
        if body.get("status") != "ok":
            raise _HealthError(agent_failure_reason="health_failed", code="health_status_not_ok")
        if body.get("agent_id") != expected.agent_id:
            raise _HealthError(
                agent_failure_reason="health_failed", code="health_agent_id_mismatch"
            )
        caps = body.get("capabilities")
        card_caps = {c.capability_id for c in contract.capabilities}
        if not isinstance(caps, list) or {str(c) for c in caps} != card_caps:
            raise _HealthError(
                agent_failure_reason="health_failed", code="health_capabilities_mismatch"
            )

    # ── result assembly ─────────────────────────────────────────────────────
    def _available(
        self,
        expected: ExpectedWorkerEndpoint,
        endpoint: str,
        card: AgentCard,
        contract: AdcAgentContract,
    ) -> DiscoveredWorker:
        compact = {
            **build_compact_card_for_agent(card),
            "availability": "available",
            "agent_failure_reason": "none",
        }
        status = WorkerStatusSummary(
            agent_id=expected.agent_id,
            availability="available",
            agent_failure_reason="none",
            discovery_error=None,
            routable=True,
            status="active",
        )
        return DiscoveredWorker(
            expected=expected,
            availability="available",
            agent_failure_reason="none",
            discovery_error=None,
            worker_status=status,
            compact=compact,
            agent_card=card,
            contract=contract,
            # Dispatch target is always the configured endpoint (validated to
            # equal the worker's declared url), never a worker-supplied redirect.
            dispatch_url=_normalize_url(endpoint),
        )

    def _unavailable(
        self, expected: ExpectedWorkerEndpoint, *, agent_failure_reason: str, code: str
    ) -> DiscoveredWorker:
        discovery_error = _compact_code(code)
        # status is deliberately NOT faked as "active": no trusted card was
        # validated, so the worker is marked disabled/unavailable.
        status = WorkerStatusSummary(
            agent_id=expected.agent_id,
            availability="unavailable",
            agent_failure_reason=agent_failure_reason,  # type: ignore[arg-type]
            discovery_error=discovery_error,
            routable=False,
            status="disabled",
        )
        compact = {
            "agent_id": expected.agent_id,
            "availability": "unavailable",
            "agent_failure_reason": agent_failure_reason,
            "discovery_error": discovery_error,
            "routable": False,
            "status": "disabled",
            "capabilities": [],
        }
        return DiscoveredWorker(
            expected=expected,
            availability="unavailable",
            agent_failure_reason=agent_failure_reason,
            discovery_error=discovery_error,
            worker_status=status,
            compact=compact,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_unknown_fallback_card(card: AgentCard) -> bool:
    """Detect A2AClient's default card returned when the fetch failed."""
    return (
        str(getattr(card, "name", "") or "") == "Unknown Agent"
        and str(getattr(card, "version", "") or "") == "unknown"
    )


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _join(endpoint: str, path: str) -> str:
    return _normalize_url(endpoint) + path


def _compact_code(code: str) -> str:
    """Guarantee a compact, safe discovery_error: bounded length, code-shaped."""
    safe = (code or "unknown_error").strip()
    if len(safe) > MAX_DISCOVERY_ERROR_LEN:
        safe = safe[:MAX_DISCOVERY_ERROR_LEN]
    return safe


__all__ = [
    "ExpectedWorkerEndpoint",
    "DiscoveredWorker",
    "DispatchTarget",
    "WorkerDiscoveryRunCache",
    "WorkerDiscoveryService",
    "WorkerUnavailableError",
    "DispatchTargetValidationError",
    "default_expected_workers",
    "MAX_DISCOVERY_ERROR_LEN",
]
