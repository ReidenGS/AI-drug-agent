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


def test_europepmc_visible_to_patent_ip_step_14_override():
    # EuropePMC has a Step 13 inventory row; the (patent_ip_agent, step_14)
    # override routes it to Step 14 too (Enola literature/prior-art evidence).
    entries = [_entry("EuropePMC_search_articles", step="step_13", runtime="ok")]
    out = filter_inventory(
        entries, ScopeRequest(agent_name="patent_ip_agent", step_id="step_14")
    )
    assert [e.tool_name for e in out] == ["EuropePMC_search_articles"]


def test_europepmc_still_visible_to_evidence_step_13_unchanged():
    entries = [_entry("EuropePMC_search_articles", step="step_13", runtime="ok")]
    out = filter_inventory(
        entries, ScopeRequest(agent_name="evidence_agent", step_id="step_13")
    )
    assert [e.tool_name for e in out] == ["EuropePMC_search_articles"]


def test_europepmc_not_leaked_to_unrelated_agent_step():
    # A Step 13 row must not appear for developability/step_06 (no override).
    entries = [_entry("EuropePMC_search_articles", step="step_13", runtime="ok")]
    out = filter_inventory(
        entries, ScopeRequest(agent_name="developability_agent", step_id="step_06")
    )
    assert out == []
