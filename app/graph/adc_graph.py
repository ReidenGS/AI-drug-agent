"""LangGraph wiring for ADC.

Five builders:

- `build_minimal_graph(...)` — Step 1→2→3→4 only. Deterministic services +
  LLM Supervisor; no MCP client required.
- `build_pipeline_graph(...)` — Step 1→…→6.
- `build_step1_9_graph(...)` — Step 1→…→9.
- `build_step1_12_graph(...)` — Step 1→…→12. Adds external Yufei AEE
  handoff (Step 10), validation (Step 11), and deterministic ranking
  (Step 12).
- `build_step1_14_graph(...)` — Step 1→…→14. Adds Step 13 (`EvidenceAgent`)
  and Step 14 (`PatentIPAgent`).

Step 13/14 topology: architecture v0.1 wants Step 13 ∥ Step 14 in parallel
after Step 12. The MVP graph runs them **sequentially** (Step 12 → 13 → 14
→ END) to keep node fan-out simple; both agents are independent and their
artifacts don't read from each other, so a future `add_node` swap to
parallel branches is mechanical.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END

from ..agents.supervisor_agent import SupervisorAgent
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import LocalMCPClient, MCPClient
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.tool_inventory_service import ToolInventoryService
from ..services.workflow_state_service import WorkflowStateService
from . import nodes as N


def build_minimal_graph(
    *,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    llm: LLMProvider | None = None,
) -> Any:
    """Compile the deterministic Step 1→2→3→4 graph.

    LLM defaults to MockLLMProvider so the graph runs without API keys. No
    MCP client needed — Steps 1-4 don't call any tools.
    """
    llm = llm or MockLLMProvider()
    supervisor = SupervisorAgent(llm=llm)

    graph = StateGraph(N.PipelineState)
    graph.add_node("step_01", N.make_node_step_01(storage, registry, workflow_state))
    graph.add_node("step_02", N.make_node_step_02(storage, registry, workflow_state, supervisor))
    graph.add_node("step_03", N.make_node_step_03(storage, registry, workflow_state))
    graph.add_node("step_04", N.make_node_step_04(storage, registry, workflow_state))

    graph.add_edge(START, "step_01")
    graph.add_edge("step_01", "step_02")
    graph.add_edge("step_02", "step_03")
    graph.add_edge("step_03", "step_04")
    graph.add_edge("step_04", END)

    return graph.compile()


def build_pipeline_graph(
    *,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
) -> Any:
    """Compile the Step 1→2→3→4→5→6 pipeline.

    `mcp_client` MUST be inventory-scoped — i.e. constructed with a
    `ToolInventoryService`. We refuse to accept a bare `LocalMCPClient()`
    here because Steps 5 and 6 would otherwise see a wider tool surface than
    v0.2 inventory allows. The check applies to both `LocalMCPClient` and
    `FastMCPClient`.
    """
    _require_inventory_scoped(mcp_client)
    llm = llm or MockLLMProvider()
    supervisor = SupervisorAgent(llm=llm)

    graph = StateGraph(N.PipelineState)
    graph.add_node("step_01", N.make_node_step_01(storage, registry, workflow_state))
    graph.add_node("step_02", N.make_node_step_02(storage, registry, workflow_state, supervisor))
    graph.add_node("step_03", N.make_node_step_03(storage, registry, workflow_state))
    graph.add_node("step_04", N.make_node_step_04(storage, registry, workflow_state))
    graph.add_node("step_05", N.make_node_step_05(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_06", N.make_node_step_06(storage, registry, workflow_state, mcp_client, llm=llm))

    graph.add_edge(START, "step_01")
    graph.add_edge("step_01", "step_02")
    graph.add_edge("step_02", "step_03")
    graph.add_edge("step_03", "step_04")
    graph.add_edge("step_04", "step_05")
    graph.add_edge("step_05", "step_06")
    graph.add_edge("step_06", END)

    return graph.compile()


def build_step1_9_graph(
    *,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
) -> Any:
    """Compile the Step 1→…→9 pipeline.

    Adds Step 7/8/9 (`StructureAndDesignAgent`) on top of the Step 1-6 graph.
    Same `mcp_client` is reused across Step 5/6/7/8/9 — it must be inventory
    scoped (see `_require_inventory_scoped`).
    """
    _require_inventory_scoped(mcp_client)
    llm = llm or MockLLMProvider()
    supervisor = SupervisorAgent(llm=llm)

    graph = StateGraph(N.PipelineState)
    graph.add_node("step_01", N.make_node_step_01(storage, registry, workflow_state))
    graph.add_node("step_02", N.make_node_step_02(storage, registry, workflow_state, supervisor))
    graph.add_node("step_03", N.make_node_step_03(storage, registry, workflow_state))
    graph.add_node("step_04", N.make_node_step_04(storage, registry, workflow_state))
    graph.add_node("step_05", N.make_node_step_05(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_06", N.make_node_step_06(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_07", N.make_node_step_07(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_08", N.make_node_step_08(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_09", N.make_node_step_09(storage, registry, workflow_state, mcp_client, llm=llm))

    graph.add_edge(START, "step_01")
    graph.add_edge("step_01", "step_02")
    graph.add_edge("step_02", "step_03")
    graph.add_edge("step_03", "step_04")
    graph.add_edge("step_04", "step_05")
    graph.add_edge("step_05", "step_06")
    graph.add_edge("step_06", "step_07")
    graph.add_edge("step_07", "step_08")
    graph.add_edge("step_08", "step_09")
    graph.add_edge("step_09", END)

    return graph.compile()


def build_step1_12_graph(
    *,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
) -> Any:
    """Compile the Step 1→…→12 pipeline.

    Step 10-12 are deterministic services that don't touch MCP; the same
    inventory-scoped client is still required by Step 5-9.
    """
    _require_inventory_scoped(mcp_client)
    llm = llm or MockLLMProvider()
    supervisor = SupervisorAgent(llm=llm)

    graph = StateGraph(N.PipelineState)
    graph.add_node("step_01", N.make_node_step_01(storage, registry, workflow_state))
    graph.add_node("step_02", N.make_node_step_02(storage, registry, workflow_state, supervisor))
    graph.add_node("step_03", N.make_node_step_03(storage, registry, workflow_state))
    graph.add_node("step_04", N.make_node_step_04(storage, registry, workflow_state))
    graph.add_node("step_05", N.make_node_step_05(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_06", N.make_node_step_06(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_07", N.make_node_step_07(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_08", N.make_node_step_08(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_09", N.make_node_step_09(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_10", N.make_node_step_10(storage, registry, workflow_state))
    graph.add_node("step_11", N.make_node_step_11(storage, registry, workflow_state))
    graph.add_node("step_12", N.make_node_step_12(storage, registry, workflow_state))

    graph.add_edge(START, "step_01")
    graph.add_edge("step_01", "step_02")
    graph.add_edge("step_02", "step_03")
    graph.add_edge("step_03", "step_04")
    graph.add_edge("step_04", "step_05")
    graph.add_edge("step_05", "step_06")
    graph.add_edge("step_06", "step_07")
    graph.add_edge("step_07", "step_08")
    graph.add_edge("step_08", "step_09")
    graph.add_edge("step_09", "step_10")
    graph.add_edge("step_10", "step_11")
    graph.add_edge("step_11", "step_12")
    graph.add_edge("step_12", END)

    return graph.compile()


def build_step1_14_graph(
    *,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
) -> Any:
    """Compile the Step 1→…→14 pipeline.

    Step 13 (EvidenceAgent) and Step 14 (PatentIPAgent) currently run
    **sequentially** after Step 12. Both are independent and will eventually
    run in parallel; see this module's docstring for the migration note.
    """
    _require_inventory_scoped(mcp_client)
    llm = llm or MockLLMProvider()
    supervisor = SupervisorAgent(llm=llm)

    graph = StateGraph(N.PipelineState)
    graph.add_node("step_01", N.make_node_step_01(storage, registry, workflow_state))
    graph.add_node("step_02", N.make_node_step_02(storage, registry, workflow_state, supervisor))
    graph.add_node("step_03", N.make_node_step_03(storage, registry, workflow_state))
    graph.add_node("step_04", N.make_node_step_04(storage, registry, workflow_state))
    graph.add_node("step_05", N.make_node_step_05(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_06", N.make_node_step_06(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_07", N.make_node_step_07(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_08", N.make_node_step_08(storage, registry, workflow_state, mcp_client))
    graph.add_node("step_09", N.make_node_step_09(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_10", N.make_node_step_10(storage, registry, workflow_state))
    graph.add_node("step_11", N.make_node_step_11(storage, registry, workflow_state))
    graph.add_node("step_12", N.make_node_step_12(storage, registry, workflow_state))
    graph.add_node("step_13", N.make_node_step_13(storage, registry, workflow_state, mcp_client, llm=llm))
    graph.add_node("step_14", N.make_node_step_14(storage, registry, workflow_state, mcp_client, llm=llm))

    graph.add_edge(START, "step_01")
    for src, dst in (
        ("step_01", "step_02"), ("step_02", "step_03"), ("step_03", "step_04"),
        ("step_04", "step_05"), ("step_05", "step_06"), ("step_06", "step_07"),
        ("step_07", "step_08"), ("step_08", "step_09"), ("step_09", "step_10"),
        ("step_10", "step_11"), ("step_11", "step_12"), ("step_12", "step_13"),
        ("step_13", "step_14"),
    ):
        graph.add_edge(src, dst)
    graph.add_edge("step_14", END)
    return graph.compile()


def _require_inventory_scoped(mcp_client: MCPClient) -> None:
    inventory = getattr(mcp_client, "inventory", None)
    if not isinstance(inventory, ToolInventoryService):
        raise ValueError(
            "build_pipeline_graph requires an inventory-scoped MCP client "
            "(LocalMCPClient(inventory=...) or FastMCPClient(..., inventory=...)). "
            "Refusing to expose Steps 5/6 to an unfiltered tool surface."
        )
