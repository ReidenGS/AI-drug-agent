"""ToolUniversityInventoryMCPServer.

Uses `python-a2a` FastMCP. Only tools present in
`ToolUniversity_inventory_v0.2.xlsx` are registered.

HARD CONSTRAINT (README_FOR_CLAUDE.md):
- MUST use python-a2a FastMCP; no handwritten MCP protocol code.
- MUST NOT register the full ToolUniverse extract.

Design notes:
- We do not write any JSON-RPC, message framing, or task envelope code here.
  When `python-a2a` is installed, `build()` constructs the FastMCP server and
  hands tool registration to `tools/_registry.register_all`.
- In dev / test environments where `python-a2a` is absent, `build()` returns
  None and emits a warning. The agent path is exercised via `LocalMCPClient`
  (see `app/mcp/client.py`), which still goes through the same scope filter
  and the same `BINDINGS` table.
"""

from __future__ import annotations

import logging
from typing import Any

from ..services.tool_inventory_service import ToolInventoryService
from .tools._registry import register_all

logger = logging.getLogger(__name__)


def _try_import_fastmcp() -> Any | None:
    """Try the few public locations python-a2a is known to expose FastMCP."""
    candidates = (
        ("python_a2a", "FastMCP"),         # 0.5.x: top-level re-export
        ("python_a2a.mcp", "FastMCP"),
        ("python_a2a.mcp.fastmcp", "FastMCP"),
        ("python_a2a.server.mcp", "FastMCP"),
    )
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            return getattr(module, attr)
        except (ImportError, AttributeError):
            continue
    return None


def is_fastmcp_available() -> bool:
    return _try_import_fastmcp() is not None


class ToolUniversityInventoryMCPServer:
    """Builds the inventory-scoped FastMCP server.

    Only tools present in `ToolUniversity_inventory_v0.2.xlsx` are registered.
    `build()` returns the FastMCP instance (so callers can `.run(...)` it on a
    real transport). `registered_tool_names` lets tests audit that we did not
    silently widen the surface to the full ToolUniverse.
    """

    def __init__(self, inventory: ToolInventoryService) -> None:
        self.inventory = inventory
        self._fastmcp: Any | None = None
        self._registered: list[str] = []

    def build(self) -> Any | None:
        FastMCP = _try_import_fastmcp()
        if FastMCP is None:
            logger.warning(
                "python-a2a FastMCP not importable; skipping MCP server build. "
                "Use LocalMCPClient for in-process agent calls."
            )
            return None
        # We do NOT pass through any arguments that would let us bypass MCP
        # framing — this is just a plain registry whose only job is to bind
        # the v0.2 inventory names to FastMCP's `@tool` decorator.
        self._fastmcp = FastMCP(
            name="ToolUniversityInventory",
            version="0.2.0",
            description="ADC pipeline tool whitelist (v0.2 inventory subset).",
        )
        self._registered = register_all(self._fastmcp, self.inventory)
        logger.info("Registered %d v0.2 inventory tools onto FastMCP", len(self._registered))
        return self._fastmcp

    def registered_tool_names(self) -> list[str]:
        return list(self._registered)

    def allowed_tool_names(self) -> set[str]:
        return self.inventory.names()

    def run(self, *, host: str = "127.0.0.1", port: int = 5050, transport: str = "fastapi") -> None:
        """Start the FastMCP HTTP transport. Blocking call.

        Delegates entirely to `FastMCP.run`; we never frame a request ourselves.
        """
        if self._fastmcp is None:
            self.build()
        if self._fastmcp is None:
            raise RuntimeError(
                "FastMCP unavailable — install `python-a2a` to expose the MCP transport."
            )
        self._fastmcp.run(transport=transport, host=host, port=port)


def build_default_server() -> ToolUniversityInventoryMCPServer:
    """Convenience constructor used by the CLI / smoke script.

    Pulls inventory path from `Settings`. Returns the server (not yet built);
    callers decide whether to `.build()` (in-process) or `.run(...)` (HTTP).
    """
    from ..settings import get_settings

    return ToolUniversityInventoryMCPServer(
        inventory=ToolInventoryService(get_settings().tool_inventory_xlsx)
    )
