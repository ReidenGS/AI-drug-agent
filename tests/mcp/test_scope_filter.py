from __future__ import annotations

from app.mcp.scope_filter import AGENT_STEP_MAP, ScopeRequest, filter_inventory
from app.services.tool_inventory_service import InventoryEntry


def _entry(name: str, step: str | None = None, runtime: str | None = None) -> InventoryEntry:
    return InventoryEntry(
        tool_name=name,
        step_id=step,
        pipeline_stage=None,
        tool_status="available",
        runtime_status=runtime,
        category=None,
        notes=None,
    )


def test_filter_returns_empty_when_agent_not_allowed_for_step():
    entries = [_entry("X", step="step_05")]
    out = filter_inventory(entries, ScopeRequest(agent_name="developability_agent", step_id="step_05"))
    assert out == []


def test_filter_drops_unstable_runtime():
    entries = [_entry("A", step="step_05", runtime="ok"), _entry("B", step="step_05", runtime="broken")]
    out = filter_inventory(
        entries, ScopeRequest(agent_name="candidate_context_agent", step_id="step_05")
    )
    assert [e.tool_name for e in out] == ["A"]


def test_agent_map_covers_all_llm_steps():
    expected = {"step_05", "step_06", "step_07", "step_08", "step_09", "step_13", "step_14"}
    covered = set().union(*AGENT_STEP_MAP.values())
    assert expected.issubset(covered)
