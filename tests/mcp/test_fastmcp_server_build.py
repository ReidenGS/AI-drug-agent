"""FastMCP server build must register only the v0.2 inventory subset."""

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
def inventory() -> ToolInventoryService:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    return ToolInventoryService(xlsx)


@pytest.fixture
def server(inventory: ToolInventoryService) -> ToolUniversityInventoryMCPServer:
    if not is_fastmcp_available():
        pytest.skip("python-a2a FastMCP not installed")
    return ToolUniversityInventoryMCPServer(inventory=inventory)


def test_fastmcp_build_returns_server_with_registered_tools(server):
    fm = server.build()
    assert fm is not None, "FastMCP build returned None even though python-a2a is importable"
    names = server.registered_tool_names()
    assert names, "FastMCP server registered zero tools"


def test_fastmcp_registration_is_subset_of_v02_inventory(server):
    server.build()
    registered = set(server.registered_tool_names())
    allowed = server.allowed_tool_names()
    extras = registered - allowed
    assert not extras, f"FastMCP registered tools outside v0.2 inventory: {extras}"


def test_fastmcp_does_not_register_full_tool_universe(server):
    """`ToolUniverse` claims thousands of tools; v0.2 inventory is ~113. We
    must never grow past the inventory size."""
    server.build()
    registered = set(server.registered_tool_names())
    allowed = server.allowed_tool_names()
    assert len(registered) <= len(allowed)
    # Sanity bound (ToolUniverse full extract is ~2366).
    assert len(registered) < 500, f"unexpectedly large registration: {len(registered)} tools"


def test_fastmcp_server_get_tools_matches_registered_names(server):
    fm = server.build()
    listed = {(t["name"] if isinstance(t, dict) else getattr(t, "name", "")) for t in fm.get_tools()}
    assert listed == set(server.registered_tool_names())
