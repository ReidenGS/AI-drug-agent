"""FastMCPClient must enforce the same agent/step + inventory scope guard as
LocalMCPClient — the underlying FastMCP server may register tools across many
steps, but an individual agent only ever sees its scoped subset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.mcp.server import (
    ToolUniversityInventoryMCPServer,
    is_fastmcp_available,
)
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def server() -> ToolUniversityInventoryMCPServer:
    if not is_fastmcp_available():
        pytest.skip("python-a2a FastMCP not installed")
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    s = ToolUniversityInventoryMCPServer(inventory=ToolInventoryService(xlsx))
    s.build()
    return s


def test_fastmcp_client_list_tools_blocks_other_steps(server):
    from app.mcp.client import FastMCPClient

    client = FastMCPClient.attach_server(server, inventory=server.inventory)
    step5 = set(client.list_tools(agent_name="candidate_context_agent", step_id="step_05"))
    assert "ChEMBL_search_molecules" in step5
    # Step 6/7/8 tools must not appear:
    assert "ProteinsPlus_profile_structure_quality" not in step5
    assert "alphafold_get_prediction" not in step5  # step 7 in v0.2 inventory
    assert "NvidiaNIM_alphafold2_multimer" not in step5


def test_fastmcp_client_call_tool_blocks_out_of_step(server):
    from app.mcp.client import FastMCPClient

    client = FastMCPClient.attach_server(server, inventory=server.inventory)
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ProteinsPlus_profile_structure_quality",
        pdb_id="1n8z",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"


def test_fastmcp_client_blocks_wrong_agent(server):
    from app.mcp.client import FastMCPClient

    client = FastMCPClient.attach_server(server, inventory=server.inventory)
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="MMAE",
    )
    assert res["run_status"] == "skipped"


def test_fastmcp_client_dispatches_in_scope_tool(server):
    """Patch the underlying FastMCP `ToolDefinition.handler` for an in-scope
    tool and confirm `FastMCPClient.call_tool` actually goes through the
    FastMCP transport (returns its `MCPResponse` payload). This proves the
    client uses the MCP protocol rather than the agent's bindings dict."""
    from app.mcp.client import FastMCPClient
    from app.mcp.scope_filter import AGENT_STEP_MAP

    fm = server._fastmcp
    assert "ChEMBL_search_molecules" in fm.tools
    fm.tools["ChEMBL_search_molecules"].handler = lambda **kw: {"echoed_query": kw.get("query")}

    client = FastMCPClient.attach_server(server, inventory=server.inventory)
    assert "step_05" in AGENT_STEP_MAP["candidate_context_agent"]
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="MMAE",
    )
    assert res["run_status"] == "success", res
    assert res["payload"] == {"echoed_query": "MMAE"}


def test_fastmcp_client_requires_python_a2a():
    """When `python-a2a` is missing, instantiation must raise ImportError so
    test suites skip rather than silently fall back to a wider tool surface."""
    if is_fastmcp_available():
        pytest.skip("python-a2a is installed; cannot test the missing-dep branch")
    from app.mcp.client import FastMCPClient

    with pytest.raises(ImportError):
        FastMCPClient(fastmcp=None, remote_client=None)
