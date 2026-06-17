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
from typing import Any

from ..services.tool_inventory_service import ToolInventoryService


# ── env hydration ──────────────────────────────────────────────────────────
#
# ToolUniverse's `AgenticTool` instances read LLM credentials directly from
# `os.environ` (e.g. `GEMINI_API_KEY`). Our codebase loads `.env` via
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


def call_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Invoke a ToolUniverse tool and normalize the response.

    The returned envelope always carries `status` (one of `ok` / `empty` /
    `upstream_error`), `source=tool_name`, the input `arguments`, and
    `payload` (the raw TU dict) so callers can persist it via
    `tool_output_ref` rather than embed it into normalized artifacts.
    """
    args = dict(arguments or {})
    envelope_base = {
        "source": tool_name,
        "executor": "tooluniverse",
        "arguments": args,
    }
    try:
        universe = _get_universe()
    except ToolUniverseAdapterError as exc:
        return {
            **envelope_base,
            "status": "upstream_error",
            "error_message": str(exc),
        }

    try:
        raw = universe.run_one_function(
            {"name": tool_name, "arguments": args},
            validate=True,
            use_cache=False,
        )
    except Exception as exc:  # noqa: BLE001 — TU should not raise but be defensive
        return {
            **envelope_base,
            "status": "upstream_error",
            "error_message": f"{type(exc).__name__}: {exc}",
        }

    if isinstance(raw, dict) and raw.get("status") == "error":
        return {
            **envelope_base,
            "status": "upstream_error",
            "error_message": raw.get("error") or "tooluniverse_error",
            "error_details": raw.get("error_details"),
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

    return {
        **envelope_base,
        "status": status,
        "payload": raw,
    }
