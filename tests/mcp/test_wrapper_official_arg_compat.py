"""Regression: every wrapper must accept ToolUniverse-official kwargs.

Stage 2 of the LLM-assisted selector builds tool arguments from the
official ToolUniverse `parameter` schema. If a wrapper's `inspect.signature`
doesn't accept the official arg names — required OR optional — the
`LocalMCPClient.call_tool(...)` call will TypeError ("unexpected
keyword argument") and run as `failed`, even though the LLM did its
job correctly.

This test enforces the contract at audit time, not at runtime:

- Every binding's parameter set must cover every TU-declared REQUIRED
  arg by name (or via `**kwargs`).
- Every binding's parameter set must cover every TU-declared OPTIONAL
  arg by name (or via `**kwargs`) so the LLM can populate them without
  crashing the call.

When a future TU release adds new args to a tool, this test fails first
and surfaces the gap before it can hit a runtime user.
"""

from __future__ import annotations

import inspect

import pytest

from app.mcp import tooluniverse_adapter
from app.mcp.tools._registry import _all_bindings


def _wrapper_accepts(fn, arg_name: str) -> bool:
    sig = inspect.signature(fn)
    if arg_name in sig.parameters:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _spec_args(spec: dict) -> tuple[list[str], list[str]]:
    """Return (required, optional) arg names from a TU spec."""
    param = (spec or {}).get("parameter") or {}
    props = param.get("properties") or {}
    if not isinstance(props, dict):
        return [], []
    declared_required = param.get("required") or []
    if not isinstance(declared_required, list):
        declared_required = []
    required: list[str] = []
    optional: list[str] = []
    for name, p in props.items():
        if not isinstance(p, dict):
            continue
        is_req = name in declared_required or p.get("required") is True
        if is_req:
            required.append(name)
        else:
            optional.append(name)
    return required, optional


def _all_specs():
    names = [n for n, _ in _all_bindings()]
    return tooluniverse_adapter.get_tool_specifications(names)


def test_every_wrapper_accepts_official_required_args():
    """The hard gate — required args MUST land in the wrapper."""
    specs = _all_specs()
    if not specs:
        pytest.skip(
            "ToolUniverse runtime not available — audit needs real TU specs."
        )
    bad: list[tuple[str, str]] = []
    for name, fn in _all_bindings():
        spec = specs.get(name)
        if not spec:
            continue
        required, _opt = _spec_args(spec)
        for arg in required:
            if not _wrapper_accepts(fn, arg):
                bad.append((name, arg))
    assert bad == [], (
        "Wrappers reject TU-required official args; selector will fail with "
        "`unexpected keyword argument` at runtime. Add the named kwarg or a "
        "`**_extra` sink. Offenders: " + ", ".join(f"{n}({a})" for n, a in bad)
    )


def test_every_wrapper_accepts_official_optional_args():
    """Optional args also must not crash the wrapper — `**_extra` is fine."""
    specs = _all_specs()
    if not specs:
        pytest.skip(
            "ToolUniverse runtime not available — audit needs real TU specs."
        )
    bad: list[tuple[str, str]] = []
    for name, fn in _all_bindings():
        spec = specs.get(name)
        if not spec:
            continue
        _req, optional = _spec_args(spec)
        for arg in optional:
            if not _wrapper_accepts(fn, arg):
                bad.append((name, arg))
    assert bad == [], (
        "Wrappers reject TU-optional official args. Even when ignored, they "
        "must not TypeError. Add `**_extra: Any` to the signature. "
        "Offenders: " + ", ".join(f"{n}({a})" for n, a in bad)
    )
