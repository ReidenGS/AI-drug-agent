"""Assert FastMCP would register a subset of v0.2 inventory only.

Hard constraint (README_FOR_CLAUDE.md): we may NOT register the full ToolUniverse
extract. The tool name set declared in `app/mcp/tools/*` BINDINGS must be ⊆ the
v0.2 inventory name set.

If the xlsx is unavailable (e.g. CI without the docs folder), the test is skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.mcp.tools._registry import _all_bindings
from app.services.tool_inventory_service import ToolInventoryService


@pytest.fixture
def inventory() -> ToolInventoryService:
    xlsx = os.environ.get(
        "TOOL_INVENTORY_XLSX",
        str(Path(__file__).resolve().parents[2].parent / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"),
    )
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    return ToolInventoryService(xlsx)


def test_registered_tools_are_subset_of_v02_inventory(inventory):
    declared = {name for name, _ in _all_bindings()}
    allowed = inventory.names()
    extras = declared - allowed
    assert not extras, f"Tools declared in app/mcp/tools but NOT in v0.2 inventory: {extras}"
