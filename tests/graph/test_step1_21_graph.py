"""Compile coverage for the Step 1-21 scaffold graph builder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.graph.adc_graph import build_step1_21_graph
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = PROJECT_ROOT.parent / "ToolUniversity_inventory_v0.2.xlsx"


def test_build_step1_21_graph_compiles(
    local_storage,
    registry_service,
    workflow_state_service,
):
    xlsx = Path(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX)))
    if not xlsx.exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    graph = build_step1_21_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=ToolInventoryService(str(xlsx))),
    )
    assert graph is not None
