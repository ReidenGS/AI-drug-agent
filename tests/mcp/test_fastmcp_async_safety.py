"""FastMCPClient sync/async boundary safety.

The sync `call_tool` must refuse to run inside a running event loop (instead
of trying `run_until_complete` and producing a confusing RuntimeError from
asyncio internals). The async `async_call_tool` must work in both cases.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.mcp.server import ToolUniversityInventoryMCPServer, is_fastmcp_available
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def client_with_in_scope_handler():
    if not is_fastmcp_available():
        pytest.skip("python-a2a FastMCP not installed")
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    server = ToolUniversityInventoryMCPServer(inventory=ToolInventoryService(xlsx))
    server.build()
    server._fastmcp.tools["ChEMBL_search_molecules"].handler = (
        lambda **kw: {"echoed": kw.get("query")}
    )
    from app.mcp.client import FastMCPClient

    return FastMCPClient.attach_server(server, inventory=server.inventory)


def test_sync_call_tool_works_outside_event_loop(client_with_in_scope_handler):
    res = client_with_in_scope_handler.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="MMAE",
    )
    assert res["run_status"] == "success"
    assert res["payload"] == {"echoed": "MMAE"}


def test_sync_call_tool_refuses_inside_running_loop(client_with_in_scope_handler):
    async def _inner():
        client_with_in_scope_handler.call_tool(
            agent_name="candidate_context_agent",
            step_id="step_05",
            tool_name="ChEMBL_search_molecules",
            query="MMAE",
        )

    with pytest.raises(RuntimeError, match="async_call_tool"):
        asyncio.run(_inner())


def test_async_call_tool_works_inside_running_loop(client_with_in_scope_handler):
    async def _inner() -> dict:
        return await client_with_in_scope_handler.async_call_tool(
            agent_name="candidate_context_agent",
            step_id="step_05",
            tool_name="ChEMBL_search_molecules",
            query="MMAE",
        )

    res = asyncio.run(_inner())
    assert res["run_status"] == "success"
    assert res["payload"] == {"echoed": "MMAE"}


def test_async_call_tool_still_enforces_scope(client_with_in_scope_handler):
    async def _inner() -> dict:
        return await client_with_in_scope_handler.async_call_tool(
            agent_name="developability_agent",  # wrong agent for Step 5 tool
            step_id="step_05",
            tool_name="ChEMBL_search_molecules",
            query="MMAE",
        )

    res = asyncio.run(_inner())
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"
