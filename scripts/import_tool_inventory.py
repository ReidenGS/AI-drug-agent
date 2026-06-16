"""Read ToolUniversity_inventory_v0.2.xlsx and (in real deployment) persist
each entry as DynamoDB item MCP_TOOL#{tool_name}.

This file is the SINGLE source of MCP-callable tool names. FastMCP registration
intersects against it at startup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.tool_inventory_service import ToolInventoryService  # noqa: E402


def main() -> None:
    xlsx = os.environ.get(
        "TOOL_INVENTORY_XLSX",
        str(ROOT.parent / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"),
    )
    svc = ToolInventoryService(xlsx)
    entries = svc.load()
    print(f"Loaded {len(entries)} tool inventory entries from {xlsx}")
    print(json.dumps([e.tool_name for e in entries[:20]], indent=2))
    # In real deployment: write each entry to DDB as MCP_TOOL#{tool_name}.


if __name__ == "__main__":
    main()
