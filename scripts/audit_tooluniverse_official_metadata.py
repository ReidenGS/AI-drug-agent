"""Opt-in live audit: official ToolUniverse metadata coverage.

NOT part of default pytest. Run on demand to verify that every
MCP-registered binding can source its Stage 1 description and Stage 2
parameter schema from the REAL ToolUniverse runtime — not from the
hand-written CAPABILITY_REGISTRY fallback.

What this script does:

1. Enumerates the tool names in `app/mcp/tools/_registry.py` (the set
   the MCP server actually registers).
2. Bulk-fetches official specs via
   `tooluniverse_adapter.get_tool_specifications(<allowed names>)` —
   which loads the real `ToolUniverse` lazily and is scope-filtered to
   the v0.2 inventory by construction. No fake universe, no widening of
   the registered scope, no per-tool invocation.
3. For each name, classifies whether TU served:
   - any spec
   - a non-empty `description`
   - a usable `parameter` block (after the selector's normalizer)
   - a schema free of any `_live` leak
4. Prints a summary of counts plus the per-bucket tool-name lists.
   Description bodies, full schemas, and raw payloads are NEVER printed —
   only tool names and integer counts. API keys are never read here.

Exit code:
    0 — every bucket is empty (all MCP-registered tools are TU-covered).
    1 — at least one bucket has entries; the script surfaces which.

Hard rules:
- No `_live=True` calls. We only touch the metadata side of TU.
- No `.env` parsing beyond what the adapter already does for env hydration
  during TU's lazy load.
- No network probes against tool endpoints. (TU's load loads tool
  registrations; it does not call the underlying APIs.)
- Does NOT alter the MCP-registered scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `app.*` importable when running this script directly.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from app.agents.tool_selection_policy import _normalize_official_schema  # noqa: E402
from app.mcp import tooluniverse_adapter  # noqa: E402
from app.mcp.tools._registry import _all_bindings  # noqa: E402


def _registered_binding_names() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name, _fn in _all_bindings():
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return sorted(ordered)


def _print_bucket(label: str, names: list[str]) -> None:
    print(f"{label}: {len(names)}")
    for n in names:
        print(f"  - {n}")


def main() -> int:
    bindings = _registered_binding_names()
    print(f"registered_bindings: {len(bindings)}")

    # Bulk lookup against the real ToolUniverse — scope-filtered to the
    # inventory by the adapter, never widened here.
    specs = tooluniverse_adapter.get_tool_specifications(bindings)
    print(f"official_specs: {len(specs)}")

    missing_spec: list[str] = []
    missing_description: list[str] = []
    missing_parameter_schema: list[str] = []
    normalized_schema_failures: list[str] = []
    live_leak: list[str] = []

    for name in bindings:
        spec = specs.get(name)
        if not spec:
            missing_spec.append(name)
            continue

        desc = (spec.get("description") or "").strip() if isinstance(spec, dict) else ""
        if not desc:
            missing_description.append(name)

        param = spec.get("parameter") or spec.get("parameters") or {}
        if not isinstance(param, dict) or not param.get("properties"):
            missing_parameter_schema.append(name)
            # If there's no parameter block, no point checking normalizer or _live leak.
            continue

        normalized = _normalize_official_schema(name, spec)
        if normalized is None or not normalized.get("properties"):
            normalized_schema_failures.append(name)
            continue

        if "_live" in normalized.get("properties", {}):
            live_leak.append(name)

    _print_bucket("missing_spec", missing_spec)
    _print_bucket("missing_description", missing_description)
    _print_bucket("missing_parameter_schema", missing_parameter_schema)
    _print_bucket("normalized_schema_failures", normalized_schema_failures)
    _print_bucket("_live_leak", live_leak)

    failed = any(
        [
            missing_spec,
            missing_description,
            missing_parameter_schema,
            normalized_schema_failures,
            live_leak,
        ]
    )
    if failed:
        print("\nRESULT: FAIL — at least one bucket has entries.")
        return 1
    print("\nRESULT: PASS — every MCP-registered tool has a TU-official "
          "description and parameter schema; no _live leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
