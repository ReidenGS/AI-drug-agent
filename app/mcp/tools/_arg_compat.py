"""Shared helpers for ToolUniverse official-arg compatibility.

Background: Stage 2 of the LLM-assisted selector now builds tool arguments
from ToolUniverse's official `parameter` schema (Q8 of the audit doc).
That changed the keys the selector sends — e.g. `operation`,
`assay_chembl_id__exact`, `pmids`. Our wrappers had legacy positional/keyword
signatures that pre-date the official schema. This module provides the
small, uniform helpers used by every wrapper that needs to accept both
shapes without inviting a "global parameter translation layer" into the
MCP client.

The contract every wrapper using these helpers follows:

- Official kwarg names listed in the wrapper signature (so introspection
  matches the TU spec) — never just a `**kwargs` sink for required args.
- Legacy kwarg names also stay in the signature so existing call sites
  keep working.
- `_resolve_operation(...)` validates a TU-required `operation` enum (the
  wrapper hard-codes the only legal value) and never silently coerces.
- `_pick(...)` picks between an official and a legacy alias, preferring
  official when both are set, and raising on contradiction.
- `**_extra: object` at the end of each wrapper soaks up unknown optional
  aliases without crashing the call — never forwarded to the adapter
  unless the wrapper explicitly opts in.
"""

from __future__ import annotations

from typing import Any


def resolve_operation(provided: str | None, expected: str) -> str:
    """Validate the official `operation` enum and return it.

    TU multi-op tools (DNATool / RfamTool / SwissADMETool / iPTMnetTool /
    ZINC / Crystal etc.) ship a fixed `operation` enum that the LLM may
    populate when constructing args from the official schema. Each
    wrapper is bound to one of those operations; this helper enforces
    that pairing.
    """
    if provided is None or provided == "":
        return expected
    p = str(provided)
    if p != expected:
        raise ValueError(
            f"operation={p!r} does not match this wrapper's operation "
            f"({expected!r})"
        )
    return expected


def pick(official: Any, legacy: Any, *, name: str) -> Any:
    """Choose between an official-schema kwarg and a legacy alias.

    - Both None / empty → returns None.
    - Only one set → returns it.
    - Both set + equal → returns the value (no fuss).
    - Both set + different → raises so the caller can't pass a
      contradictory pair (we never silently prefer one over the other
      when they disagree).
    """
    if _is_empty(official):
        return None if _is_empty(legacy) else legacy
    if _is_empty(legacy):
        return official
    if official == legacy:
        return official
    raise ValueError(
        f"argument {name!r} received contradictory official "
        f"({official!r}) and legacy ({legacy!r}) values"
    )


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    return False
