"""MCP client abstraction.

Agents depend on the `MCPClient` Protocol, never on a concrete transport.
Two concrete implementations:

- `LocalMCPClient` — in-process router over `app/mcp/tools/*.BINDINGS`.
  Honors `scope_filter` so agents only see their step-allowed tool subset.
  Used by tests and by Step 5 MVP in dev mode.
- `FastMCPClient` — would proxy over `python-a2a` FastMCP. Skeleton — wire in
  later iterations. Importantly, we still do NOT handwrite MCP envelopes.

Calling a tool returns a dict with `run_status` ∈ canonical
`ToolCallRunStatus` plus an optional `payload` and `tool_output_ref`. Raw
payloads are intended to be persisted by the agent under
`tool_outputs/step_XX/{tool_call_id}.json` and referenced (never embedded).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Protocol, runtime_checkable

from .scope_filter import AGENT_STEP_MAP, ScopeRequest, filter_inventory
from .tools._registry import _all_bindings
from ..services.tool_inventory_service import InventoryEntry, ToolInventoryService


@runtime_checkable
class MCPClient(Protocol):
    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]: ...
    def call_tool(
        self, *, agent_name: str, step_id: str, tool_name: str, **kwargs: Any
    ) -> dict: ...


class LocalMCPClient:
    """In-process router. Always passes through `scope_filter`."""

    def __init__(
        self,
        inventory: ToolInventoryService | None = None,
        *,
        bindings: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.inventory = inventory
        self._bindings: dict[str, Callable[..., Any]] = (
            dict(bindings) if bindings is not None else dict(_all_bindings())
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _entries(self) -> list[InventoryEntry]:
        if self.inventory is None:
            return [
                InventoryEntry(
                    tool_name=name,
                    step_id=None,
                    pipeline_stage=None,
                    tool_status="available",
                    runtime_status="ok",
                    category=None,
                    notes=None,
                )
                for name in self._bindings
            ]
        return self.inventory.load()

    def _allow(self, agent_name: str, step_id: str, tool_name: str) -> bool:
        allowed_steps = AGENT_STEP_MAP.get(agent_name, set())
        if step_id not in allowed_steps:
            return False
        scoped = {
            e.tool_name
            for e in filter_inventory(
                self._entries(), ScopeRequest(agent_name=agent_name, step_id=step_id)
            )
        }
        # If inventory provided, restrict by both inventory scope AND step_map.
        # Without inventory (tests), step_map alone is the guard.
        if self.inventory is not None and tool_name not in scoped:
            return False
        return tool_name in self._bindings

    # ── public API ─────────────────────────────────────────────────────────
    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]:
        allowed_steps = AGENT_STEP_MAP.get(agent_name, set())
        if step_id not in allowed_steps:
            return []
        if self.inventory is None:
            return sorted(self._bindings)
        scoped = filter_inventory(
            self._entries(), ScopeRequest(agent_name=agent_name, step_id=step_id)
        )
        return sorted(e.tool_name for e in scoped if e.tool_name in self._bindings)

    def call_tool(
        self, *, agent_name: str, step_id: str, tool_name: str, **kwargs: Any
    ) -> dict:
        if not self._allow(agent_name, step_id, tool_name):
            return {
                "run_status": "skipped",
                "skip_reason": "tool_not_in_agent_scope",
                "tool_name": tool_name,
                "agent_name": agent_name,
                "step_id": step_id,
            }
        fn = self._bindings[tool_name]
        call_kwargs = dict(kwargs)
        # Live-mode injection per `settings.should_use_live`: live ON + a
        # non-empty allowlist limits `_live=True` to listed tools, while live
        # ON + an empty allowlist injects `_live=True` for every scoped tool
        # (production all-live). Callers who pass `_live` themselves keep control.
        if "_live" not in call_kwargs:
            try:
                from ..settings import get_settings

                if get_settings().should_use_live(tool_name):
                    call_kwargs["_live"] = True
            except Exception:  # noqa: BLE001 - settings must never break tool dispatch
                pass
        try:
            payload = fn(**call_kwargs)
            return {
                "run_status": "success",
                "tool_name": tool_name,
                "payload": payload,
                "executor": _classify_executor(payload),
            }
        except NotImplementedError:
            return {
                "run_status": "dependency_unavailable",
                "tool_name": tool_name,
                "reason": "wrapper_not_wired",
                "executor": "deferred",
            }
        except Exception as e:  # noqa: BLE001
            return {
                "run_status": "failed",
                "tool_name": tool_name,
                "error_message": str(e),
                "executor": "error",
            }


class FastMCPClient:
    """MCP transport client backed by `python-a2a` FastMCP.

    The transport itself (request framing, tools/list, tools/call) is owned by
    FastMCP — we never construct an MCP envelope by hand. This class:

    1. Looks up the FastMCP instance (built from the v0.2 inventory).
    2. Applies the same `scope_filter` + inventory whitelist guard as
       `LocalMCPClient` — the underlying server may have a broader registry
       than what an individual agent / step is allowed to see.
    3. Awaits `fastmcp.call_tool` and normalizes the `MCPResponse` to the
       same dict shape `LocalMCPClient` returns.

    Two construction modes:
    - `attach_server(...)`: in-process FastMCP instance (the server you built
       via `ToolUniversityInventoryMCPServer.build()`). Useful for tests and
       Python-level wiring without going over the network.
    - `connect_remote(server_url)`: connects to a running FastMCP HTTP
       transport via `python-a2a` `MCPClient`. Not used by the smoke test;
       provided so production / cross-process callers have one entry point.

    If `python-a2a` is not importable, instantiation raises ImportError so
    tests skip cleanly instead of silently widening the tool surface.
    """

    def __init__(
        self,
        *,
        fastmcp: Any | None = None,
        remote_client: Any | None = None,
        inventory: ToolInventoryService | None = None,
    ) -> None:
        from .server import _try_import_fastmcp

        if _try_import_fastmcp() is None:
            raise ImportError(
                "python-a2a FastMCP is not available; install `python-a2a` to use FastMCPClient."
            )
        if fastmcp is None and remote_client is None:
            raise ValueError(
                "FastMCPClient needs either a `fastmcp` instance or a `remote_client`."
            )
        self._fastmcp = fastmcp
        self._remote = remote_client
        self.inventory = inventory

    # ── factory helpers ─────────────────────────────────────────────────────
    @classmethod
    def attach_server(
        cls, server: Any, *, inventory: ToolInventoryService | None = None
    ) -> "FastMCPClient":
        # If `server` is already a FastMCP, use it as-is. Otherwise, treat it
        # as a `ToolUniversityInventoryMCPServer` and reuse its built instance
        # (build only if one hasn't been built yet — never re-build, since
        # that would throw away any in-place handler overrides).
        if hasattr(server, "build"):
            fm = getattr(server, "_fastmcp", None) or server.build()
            inv = inventory or getattr(server, "inventory", None)
        else:
            fm = server
            inv = inventory
        if fm is None:
            raise RuntimeError("FastMCP instance is None — call server.build() first")
        return cls(fastmcp=fm, inventory=inv)

    @classmethod
    def connect_remote(
        cls, server_url: str, *, inventory: ToolInventoryService | None = None
    ) -> "FastMCPClient":
        from python_a2a import MCPClient as RemoteMCPClient  # type: ignore

        return cls(remote_client=RemoteMCPClient(server_url=server_url), inventory=inventory)

    # ── scope guard ─────────────────────────────────────────────────────────
    def _scope_ok(self, *, agent_name: str, step_id: str, tool_name: str) -> bool:
        allowed_steps = AGENT_STEP_MAP.get(agent_name, set())
        if step_id not in allowed_steps:
            return False
        if self.inventory is not None:
            scoped = {
                e.tool_name
                for e in filter_inventory(
                    self.inventory.load(), ScopeRequest(agent_name=agent_name, step_id=step_id)
                )
            }
            return tool_name in scoped
        # No inventory: still must be in the server's registered set.
        return tool_name in self._server_tool_names()

    def _server_tool_names(self) -> set[str]:
        if self._fastmcp is not None:
            tools = self._fastmcp.get_tools()
            return {t["name"] if isinstance(t, dict) else getattr(t, "name", "") for t in tools}
        # remote path: fall back to inventory; remote tools/list would require
        # an extra round-trip we don't need for the scope guard.
        return self.inventory.names() if self.inventory is not None else set()

    # ── MCPClient protocol ──────────────────────────────────────────────────
    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]:
        allowed_steps = AGENT_STEP_MAP.get(agent_name, set())
        if step_id not in allowed_steps:
            return []
        server = self._server_tool_names()
        if self.inventory is None:
            return sorted(server)
        scoped = filter_inventory(
            self.inventory.load(), ScopeRequest(agent_name=agent_name, step_id=step_id)
        )
        return sorted(e.tool_name for e in scoped if e.tool_name in server)

    def call_tool(
        self, *, agent_name: str, step_id: str, tool_name: str, **kwargs: Any
    ) -> dict:
        """Synchronous tool call.

        Refuses to run inside an already-running event loop — calling
        `asyncio.run` or `loop.run_until_complete` from inside a running loop
        raises `RuntimeError` and is a common source of subtle bugs. Use
        `async_call_tool` from async code instead.
        """
        if self._inside_running_loop():
            raise RuntimeError(
                "FastMCPClient.call_tool() is sync but an event loop is already "
                "running. Use `await async_call_tool(...)` from async code."
            )
        if not self._scope_ok(agent_name=agent_name, step_id=step_id, tool_name=tool_name):
            return _skipped_payload(agent_name, step_id, tool_name)
        try:
            response = self._dispatch_sync(tool_name, kwargs)
        except NotImplementedError:
            return _dep_unavailable_payload(tool_name)
        except Exception as e:  # noqa: BLE001
            return {"run_status": "failed", "tool_name": tool_name, "error_message": str(e)}
        return _normalize_response(response, tool_name)

    async def async_call_tool(
        self, *, agent_name: str, step_id: str, tool_name: str, **kwargs: Any
    ) -> dict:
        """Async tool call — safe to use from any running event loop."""
        if not self._scope_ok(agent_name=agent_name, step_id=step_id, tool_name=tool_name):
            return _skipped_payload(agent_name, step_id, tool_name)
        try:
            response = await self._dispatch_async(tool_name, kwargs)
        except NotImplementedError:
            return _dep_unavailable_payload(tool_name)
        except Exception as e:  # noqa: BLE001
            return {"run_status": "failed", "tool_name": tool_name, "error_message": str(e)}
        return _normalize_response(response, tool_name)

    # ── transport dispatch ──────────────────────────────────────────────────
    @staticmethod
    def _inside_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _dispatch_sync(self, tool_name: str, params: dict[str, Any]) -> Any:
        if self._fastmcp is not None:
            return asyncio.run(self._fastmcp.call_tool(tool_name, params))
        # remote path: python_a2a.MCPClient exposes call_tool_sync.
        return self._remote.call_tool_sync(tool_name, params)

    async def _dispatch_async(self, tool_name: str, params: dict[str, Any]) -> Any:
        if self._fastmcp is not None:
            return await self._fastmcp.call_tool(tool_name, params)
        # remote path: prefer the native async API if available; fall back to
        # offloading the sync helper to a worker thread so we never block the
        # caller's event loop.
        remote_async = getattr(self._remote, "call_tool", None)
        if remote_async is not None and asyncio.iscoroutinefunction(remote_async):
            return await remote_async(tool_name, params)
        return await asyncio.to_thread(self._remote.call_tool_sync, tool_name, params)


def _classify_executor(payload: Any) -> str:
    """Tag the executor that produced the wrapper payload.

    Used for audit/visibility — agents and tests can read
    `result["executor"]` to tell `tooluniverse` (live adapter path) from
    `mock` (deterministic mock envelope) without parsing wrapper-specific
    shapes. Never touches raw payload contents beyond the two recognized
    discriminator fields.
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("executor"), str) and payload.get("executor"):
            return str(payload["executor"])
        if payload.get("status") == "mocked":
            return "mock"
    return "unknown"


def _skipped_payload(agent_name: str, step_id: str, tool_name: str) -> dict:
    return {
        "run_status": "skipped",
        "skip_reason": "tool_not_in_agent_scope",
        "tool_name": tool_name,
        "agent_name": agent_name,
        "step_id": step_id,
    }


def _dep_unavailable_payload(tool_name: str) -> dict:
    return {
        "run_status": "dependency_unavailable",
        "tool_name": tool_name,
        "reason": "wrapper_not_wired",
    }


def _normalize_response(response: Any, tool_name: str) -> dict:
    if getattr(response, "is_error", False):
        return {
            "run_status": "failed",
            "tool_name": tool_name,
            "error_message": _extract_text(response),
        }
    return {
        "run_status": "success",
        "tool_name": tool_name,
        "payload": _extract_payload(response),
    }


def _extract_text(response: Any) -> str:
    parts = getattr(response, "content", None) or []
    return " ".join(
        (p.get("text") if isinstance(p, dict) else str(p)) for p in parts
    ).strip()


def _extract_payload(response: Any) -> Any:
    parts = getattr(response, "content", None) or []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text") or ""
            try:
                return json.loads(text)
            except Exception:  # noqa: BLE001
                return text
    return getattr(response, "to_dict", lambda: {"content": parts})()
