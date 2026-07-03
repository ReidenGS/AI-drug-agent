"""ToolUniverse runtime adapter (lazy, scope-filtered).

This is the ONLY place in the codebase that imports `tooluniverse`. Live
mode for every migrated MCP wrapper routes through this adapter — there
is no parallel manual httpx path, no settings kill switch, and no
per-tool gate. The `_live` flag itself is the gate.

End-to-end control flow:

1. `LocalMCPClient.call_tool` consults the existing
   `MCP_LIVE_TOOLS` + `MCP_LIVE_TOOL_ALLOWLIST` policy. When the policy
   says "go live", it injects `_live=True` into the wrapper kwargs.
2. The wrapper inspects `_live`:
   - `False` → returns the deterministic mock envelope (unchanged).
   - `True`  → calls `tooluniverse_adapter.call_tool(name, args)` and
     returns whatever envelope this adapter built. No httpx, no
     filesystem fallback.
3. The adapter lazily builds a singleton `ToolUniverse()` instance the
   first time it is needed, loads ONLY the inventory-scoped tool names,
   dispatches the call, and normalizes the response.

Design rules:

- **Lazy.** `ToolUniverse.load_tools()` materializes >2 000 tool specs;
  default pytest must not pay that cost. The singleton is built on first
  `call_tool(...)` and never during module import or settings access.
- **Scope-filtered.** `load_tools(include_tools=…)` restricts the
  registry to the v0.2 inventory tool names only — we never widen MCP
  scope by going through TU.
- **Tests don't load TU.** `_reset_for_tests()` lets monkeypatched tests
  inject a fake universe so the real registry is never touched offline.
- **Envelope.** `call_tool` returns a uniform
  `{status, source, executor, arguments, payload}` dict so callers can
  persist via `tool_output_ref` without caring whether the upstream was
  TU or a manual wrapper. TU's structured-error response becomes
  `status="upstream_error"` with the original `error_message`.
"""

from __future__ import annotations

import os
import signal
import socket
import hashlib
import threading
import time
from contextlib import contextmanager
from typing import Any
from typing import Callable

import requests

from ..services.tool_inventory_service import ToolInventoryService


# ── transient-failure retry policy ─────────────────────────────────────────
#
# Live ToolUniverse calls can hit transient upstream/network failures
# (proxy drops, remote disconnects, timeouts, HTTP 5xx / rate limits). We
# retry ONLY those a small bounded number of times with a short backoff.
# Deterministic failures (validation / schema / missing-input / not-found)
# are NEVER retried. A final failure always keeps `status="upstream_error"`
# — we never mock a success or swallow the error.

_MAX_LIVE_RETRIES = 2  # up to 2 retries → 3 total attempts
_RETRY_BACKOFF_BASE_SECONDS = 0.25

_TRANSIENT_ERROR_TOKENS = (
    "proxyerror",
    "remotedisconnected",
    "remote disconnected",
    "remote end closed",
    "timed out",
    "timeout",
    "connection reset",
    "connectionreset",
    "connection aborted",
    "connection refused",
    "max retries exceeded",
    "connectionerror",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "server error",  # requests' "5xx Server Error"
    "rate limit",
    "too many requests",
    "429",
)


def _is_transient_error(message: str | None) -> bool:
    """Classify an upstream error message as transient (worth retrying).

    Conservative: only known network/upstream-transient signatures match.
    Validation / schema / missing-input / not-found errors do not, so they
    are never retried.
    """
    if not message:
        return False
    lowered = message.lower()
    return any(token in lowered for token in _TRANSIENT_ERROR_TOKENS)


def _compact_error(message: str | None, *, limit: int = 200) -> str:
    """One-line, length-capped error string (never a raw payload/sequence)."""
    text = " ".join((message or "").split())
    return text[:limit]


_BIO_SEQUENCE_CHARS = set("ACDEFGHIKLMNPQRSTVWYBZXUO*")


def _looks_like_amino_acid_sequence(value: str) -> bool:
    """Best-effort heuristic for raw AA sequences.

    Used only for envelope redaction of persisted arguments, not for runtime
    execution.
    """
    cleaned = "".join(ch for ch in value.upper() if ch.isalpha())
    if len(cleaned) < 12:
        return False
    if len(cleaned) > 40_000:
        return True
    return (len(cleaned) / max(1, len(value)) >= 0.70 and set(cleaned) <= _BIO_SEQUENCE_CHARS)


def _looks_like_raw_structural_payload(value: str) -> bool:
    lowered = value.lower()
    if "header" in lowered or "atom" in lowered or "hetatm" in lowered or "data_" in lowered or "loop_" in lowered:
        return True
    if ">" in value and ("\n" in value or "\\n" in value):
        return True
    return False


def _safely_redact_argument_string(value: str, *, key: str | None = None) -> str | dict[str, Any]:
    keep_short_literals = {
        "operation",
        "pdb_id",
        "output_format",
        "tool",
        "tool_name",
        "agent",
        "task",
    }
    if key and key.lower() in keep_short_literals and len(value) <= 200:
        return value
    lowered_key = (key or "").lower()
    if any(secret in lowered_key for secret in ("key", "token", "secret", "password", "api")):
        return {
            "redacted": True,
            "length": len(value),
            "reason": "sensitive credential-like key",
        }
    if _looks_like_raw_structural_payload(value) or _looks_like_amino_acid_sequence(value) or len(value) > 120:
        sha = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return {
            "redacted": True,
            "length": len(value),
            "sha256_prefix": sha,
            "reason": "argument value omitted for compact audit",
        }
    return value


def _compact_argument_value(value: Any, *, key: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _safely_redact_argument_string(value, key=key)
    if isinstance(value, list):
        return [_compact_argument_value(v) for v in value]
    if isinstance(value, tuple):
        return [_compact_argument_value(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key == "arguments" and isinstance(child_value, dict):
                out[child_key] = _compact_argument_value(child_value)
            else:
                out[child_key] = _compact_argument_value(
                    child_value, key=str(child_key)
                )
        return out
    return value


def _sleep_backoff(attempt: int) -> None:
    # `attempt` is 1-based for the retry just completed.
    time.sleep(_RETRY_BACKOFF_BASE_SECONDS * attempt)


def _live_call_timeout_seconds(tool_name: str | None = None) -> float:
    """Return the configured outer timeout for one TU live call."""
    try:
        from ..settings import get_settings

        settings = get_settings()
        if str(tool_name or "").startswith("NvidiaNIM_"):
            return float(settings.nvidia_nim_live_call_timeout or 0)
        return float(settings.tooluniverse_live_call_timeout or 0)
    except Exception:  # noqa: BLE001 — timeout lookup must not break dispatch
        return 0.0


def _inject_session_request_timeout(
    timeout_seconds: float,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Patch `Session.request` to inject a default timeout if caller omitted one."""

    def _patch_request(original: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(session: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
            if "timeout" not in kwargs:
                kwargs["timeout"] = timeout_seconds
            return original(session, method, url, *args, **kwargs)

        return _wrapped

    return _patch_request


@contextmanager
def _bounded_live_call(timeout_seconds: float):
    """Raise TimeoutError if a ToolUniverse call blocks past the outer limit.

    ToolUniverse wraps many upstream clients. Some of those clients use
    `requests` without a timeout, so our normal retry loop never sees an
    exception. On the main thread, a short SIGALRM guard turns that socket
    hang into a normal transient `TimeoutError`. Off the main thread, signal
    alarms are unavailable, so we degrade to no outer timeout rather than
    changing thread semantics.
    """
    if (
        timeout_seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    previous_socket_timeout = socket.getdefaulttimeout()
    previous_request = requests.sessions.Session.request

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"ToolUniverse live call exceeded {timeout_seconds:g}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    socket.setdefaulttimeout(timeout_seconds)
    requests.sessions.Session.request = _inject_session_request_timeout(timeout_seconds)(
        previous_request
    )
    try:
        yield
    finally:
        requests.sessions.Session.request = previous_request
        socket.setdefaulttimeout(previous_socket_timeout)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


# ── env hydration ──────────────────────────────────────────────────────────
#
# ToolUniverse's tools read selected credentials directly from
# `os.environ` (e.g. `GEMINI_API_KEY`, `NVIDIA_API_KEY`). Our codebase loads `.env` via
# `pydantic-settings`, which populates the `Settings` object but does NOT
# write to `os.environ`. Without bridging, TU sees no key and silently
# fails to load the agentic sub-tools (`IntentAnalyzerAgent`,
# `KeywordExtractorAgent`, …) used by `MultiAgentLiteratureSearch` etc.
#
# `_hydrate_env_from_settings()` runs ONCE, before the TU universe is
# built. It copies a small allowlist of keys from `Settings` into
# `os.environ` IF and ONLY IF `os.environ` does not already define them
# (operator-set env wins). It NEVER prints the values.


def _hydrate_env_from_settings() -> None:
    """Bridge selected `Settings` fields into `os.environ` for ToolUniverse."""
    try:
        from ..settings import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 — env hydration must never break tool dispatch
        return

    # (settings attribute, env var name TU's AgenticTool reads)
    bridges: tuple[tuple[str, str], ...] = (
        ("gemini_api_key", "GEMINI_API_KEY"),
        ("gemini_model", "GEMINI_MODEL_ID"),
        ("gemini_model", "TOOLUNIVERSE_LLM_MODEL_DEFAULT"),
        ("nvidia_api_key", "NVIDIA_API_KEY"),
    )
    for attr, env_name in bridges:
        if os.environ.get(env_name):
            continue  # operator-set env always wins
        value = getattr(settings, attr, "") or ""
        if value:
            os.environ[env_name] = value


class ToolUniverseAdapterError(RuntimeError):
    """Raised when the adapter cannot construct a ToolUniverse instance."""


_universe: Any | None = None
_inventory_names: frozenset[str] | None = None


def _resolve_inventory_names() -> frozenset[str]:
    """Load the v0.2 inventory tool name set once and cache it."""
    global _inventory_names
    if _inventory_names is not None:
        return _inventory_names
    from ..settings import get_settings

    try:
        entries = ToolInventoryService(get_settings().tool_inventory_xlsx).load()
        _inventory_names = frozenset(e.tool_name for e in entries if e.tool_name)
    except Exception:  # noqa: BLE001 - inventory load is best-effort here
        _inventory_names = frozenset()
    return _inventory_names


def _get_universe() -> Any:
    """Lazy-build the ToolUniverse instance filtered to inventory names."""
    global _universe
    if _universe is not None:
        return _universe
    try:
        from tooluniverse import ToolUniverse  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ToolUniverseAdapterError(
            "tooluniverse is not installed in this environment; install it "
            "before using _live=True ToolUniverse-backed MCP wrappers."
        ) from exc

    # Bridge `.env` → `os.environ` BEFORE TU starts loading tools so
    # `AgenticTool` sub-tools can see `GEMINI_API_KEY` etc.
    _hydrate_env_from_settings()

    inventory_names = _resolve_inventory_names()
    inst = ToolUniverse()
    include = sorted(inventory_names) or None
    inst.load_tools(quiet=True, include_tools=include)
    _universe = inst
    return _universe


def _reset_for_tests() -> None:
    """Test hook — drop the cached universe + inventory set."""
    global _universe, _inventory_names
    _universe = None
    _inventory_names = None


def list_available_tools() -> list[str]:
    """Return the inventory-scoped tool names that ToolUniverse can serve."""
    universe = _get_universe()
    names = universe.get_available_tools(name_only=True) or []
    return sorted(names)


def has_tool(tool_name: str) -> bool:
    try:
        return tool_name in set(list_available_tools())
    except ToolUniverseAdapterError:
        return False


# ── official tool metadata (description + parameter schema) ───────────────
#
# Progressive tool selection (`app/agents/tool_selection_policy.py`) prefers
# the OFFICIAL ToolUniverse metadata over the hand-written `CAPABILITY_REGISTRY`
# entries — Stage 1 description and Stage 2 parameter schema both come from
# TU when TU recognizes the name. These helpers expose just enough of TU's
# metadata API for the selector while keeping the same hard rules as
# `call_tool`:
#
# - Lazy: never touches the TU registry until first call.
# - Scope-filtered: the underlying `_get_universe()` already loads with
#   `include_tools=<inventory-scoped names>`, so any tool name absent from
#   the inventory returns `None` here too.
# - Test-friendly: monkeypatching `_get_universe` to a `FakeUniverse` works.
# - Safe degradation: when `tooluniverse` is not installed (ImportError) or
#   the lookup fails, helpers return `None` / `[]` so the selector keeps
#   working off the hand-written fallback instead of crashing.
# - Logs nothing: we never log the full spec or parameter blob — only the
#   tool name on a failure path, never the schema or description bodies.


def _safe_universe() -> Any | None:
    """Return the lazy TU singleton or `None` if it can't be built.

    Differs from `_get_universe` by swallowing `ToolUniverseAdapterError`
    so metadata callers can fall back instead of crashing.
    """
    try:
        return _get_universe()
    except ToolUniverseAdapterError:
        return None


def get_tool_specification(tool_name: str) -> dict[str, Any] | None:
    """Return the TU official tool spec dict for one tool, or `None`."""
    if not tool_name:
        return None
    universe = _safe_universe()
    if universe is None:
        return None
    try:
        specs = universe.get_tool_specification_by_names([tool_name]) or []
    except Exception:  # noqa: BLE001 — TU metadata path must never crash callers
        # We deliberately do NOT log spec contents here.
        return None
    for spec in specs:
        if isinstance(spec, dict) and spec.get("name") == tool_name:
            return spec
    return None


def get_tool_specifications(
    tool_names: list[str] | tuple[str, ...] | frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Bulk-fetch official specs for a list of tool names.

    The returned dict only contains entries TU recognized — callers can
    treat missing keys as "use fallback". Inventory scope is already
    enforced by the singleton's `include_tools=` filter; the public API
    contract is "the caller passed in the agent's allowed set" — we do
    NOT widen scope by fetching anything else.
    """
    names = [n for n in (tool_names or []) if n]
    if not names:
        return {}
    universe = _safe_universe()
    if universe is None:
        return {}
    try:
        specs = universe.get_tool_specification_by_names(list(names)) or []
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if isinstance(spec, dict):
            name = spec.get("name")
            if name in names:
                out[name] = spec
    return out


def get_required_parameters(tool_name: str) -> list[str]:
    """Return the TU-declared required parameter names for a tool, or []."""
    if not tool_name:
        return []
    universe = _safe_universe()
    if universe is None:
        return []
    try:
        req = universe.get_required_parameters(tool_name) or []
    except Exception:  # noqa: BLE001
        return []
    return [str(p) for p in req if p]


def call_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Invoke a ToolUniverse tool and normalize the response.

    The returned envelope always carries `status` (one of `ok` / `empty` /
    `upstream_error`), `source=tool_name`, the input `arguments`, and
    `payload` (the raw TU dict) so callers can persist it via
    `tool_output_ref` rather than embed it into normalized artifacts.
    """
    args = dict(arguments or {})
    redacted_args = _compact_argument_value(args)
    envelope_base = {
        "source": tool_name,
        "executor": "tooluniverse",
        "arguments": redacted_args,
    }
    try:
        universe = _get_universe()
    except ToolUniverseAdapterError as exc:
        # Building the universe failed (e.g. tooluniverse not installed) —
        # deterministic, not a transient upstream failure: do not retry.
        return {
            **envelope_base,
            "status": "upstream_error",
            "error_message": str(exc),
            "retry_count": 0,
            "retryable": False,
            "final_error_type": type(exc).__name__,
        }

    # Bounded retry loop for TRANSIENT upstream failures only. `retry_count`
    # counts retries performed (0 on first-try success). Deterministic errors
    # break out immediately and are returned as upstream_error with
    # retryable=False.
    retry_count = 0
    last_error_message: str | None = None
    last_error_type: str | None = None
    last_error_details: Any = None

    while True:
        try:
            with _bounded_live_call(_live_call_timeout_seconds(tool_name)):
                raw = universe.run_one_function(
                    {"name": tool_name, "arguments": args},
                    validate=True,
                    use_cache=False,
                )
            error_message: str | None = None
            error_type: str | None = None
            error_details: Any = None
            if isinstance(raw, dict) and raw.get("status") == "error":
                error_message = raw.get("error") or "tooluniverse_error"
                error_type = "tooluniverse_error"
                error_details = raw.get("error_details")
        except Exception as exc:  # noqa: BLE001 — TU should not raise but be defensive
            raw = None
            error_message = f"{type(exc).__name__}: {exc}"
            error_type = type(exc).__name__
            error_details = None

        if error_message is None:
            break  # success (raw is a usable response)

        last_error_message = error_message
        last_error_type = error_type
        last_error_details = error_details
        transient = _is_transient_error(error_message)
        if transient and retry_count < _MAX_LIVE_RETRIES:
            _sleep_backoff(retry_count + 1)
            retry_count += 1
            continue
        # Exhausted retries OR non-retryable: surface upstream_error honestly.
        return {
            **envelope_base,
            "status": "upstream_error",
            "error_message": _compact_error(last_error_message),
            "error_details": last_error_details,
            "retry_count": retry_count,
            "retryable": transient,
            "final_error_type": last_error_type,
        }

    status = "ok"
    # Treat obviously-empty container responses as `empty` so callers can
    # branch the same way they do for manual wrappers.
    if isinstance(raw, dict):
        for key in (
            "results",
            "items",
            "records",
            "activities",
            "features",
            "patents",
            "data",
        ):
            value = raw.get(key)
            if isinstance(value, list) and len(value) == 0:
                status = "empty"
                break
    elif isinstance(raw, list) and not raw:
        status = "empty"

    envelope = {
        **envelope_base,
        "status": status,
        "payload": raw,
        "retry_count": retry_count,
        "retryable": False,
    }
    if retry_count:
        # Succeeded only after retrying a transient failure — record the
        # last transient error compactly for audit (never a raw payload).
        envelope["recovered_after_transient_error"] = True
        envelope["final_error_type"] = last_error_type
        envelope["final_error_message"] = _compact_error(last_error_message)
    return envelope
