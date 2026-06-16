"""Smoke-test the FastMCP transport: build inventory-scoped server, verify
registered tool set is bounded by v0.2 inventory, and confirm an agent's
scope view through `FastMCPClient` doesn't leak Step 6/7/8 tools into Step 5.

This does NOT make any real network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mcp.server import build_default_server, is_fastmcp_available  # noqa: E402


def main() -> int:
    if not is_fastmcp_available():
        print("python-a2a FastMCP not importable; install `python-a2a` to run this smoke test.")
        return 0  # graceful no-op, not a hard failure

    server = build_default_server()
    fm = server.build()
    if fm is None:
        print("FastMCP build returned None; aborting.")
        return 2

    registered = server.registered_tool_names()
    allowed = server.allowed_tool_names()
    extras = set(registered) - allowed
    print(f"FastMCP registered tools: {len(registered)}")
    print(f"v0.2 inventory tool universe: {len(allowed)}")

    assert registered, "no tools registered"
    assert not extras, f"FastMCP registered tools outside v0.2 inventory: {extras}"
    assert len(registered) <= len(allowed), "registered set exceeds inventory size"

    # Scope guard: walk the client and confirm Step 6/7/8 tools never appear in
    # the Step 5 visible set for candidate_context_agent.
    from app.mcp.client import FastMCPClient

    client = FastMCPClient.attach_server(server, inventory=server.inventory)
    visible_step5 = client.list_tools(agent_name="candidate_context_agent", step_id="step_05")
    forbidden = {
        "ProteinsPlus_profile_structure_quality",  # step 6
        "alphafold_get_prediction",                # step 7 (v0.2 inventory)
        "NvidiaNIM_alphafold2_multimer",           # step 8
    }
    leaked = forbidden & set(visible_step5)
    assert not leaked, f"Step 6/7/8 tools leaked into Step 5 scope: {leaked}"

    # Out-of-step call must be skipped, not invoked.
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ProteinsPlus_profile_structure_quality",
        pdb_id="1n8z",
    )
    assert res["run_status"] == "skipped", res

    print(f"Step 5 scoped view: {len(visible_step5)} tools (sample): {visible_step5[:5]}")
    print("OK: FastMCP transport smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
