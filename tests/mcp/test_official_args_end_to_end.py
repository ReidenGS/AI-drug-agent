"""End-to-end: LLM-style official kwargs through LocalMCPClient.call_tool.

Reproduces the five real-failure examples reported by ops:

    DNA_calculate_gc_content   {"operation": "calculate_gc_content", "sequence": "ATCG"}
    Rfam_get_family            {"operation": "get_family", "family_id": "RF00001"}
    SwissADME_calculate_adme   {"operation": "calculate_adme", "smiles": "CCO"}
    ChEMBL_get_assay_activities{"assay_chembl_id__exact": "CHEMBL123", "limit": 1, "offset": 0}
    PubTator3_get_annotations  {"pmids": "12345"}

Each used to fail with `unexpected keyword argument` at the wrapper
layer. The wrapper-level fix means LocalMCPClient now routes them to a
clean `executor="tooluniverse"` envelope (adapter-backed) or
`dependency_unavailable` (deferred) — never `failed`.

Also covers:
- A deferred wrapper (`drugbank`) receiving official kwargs still surfaces
  `dependency_unavailable` and does NOT touch the fake universe.
- ZINC stays `intentionally_disabled` even when called with official
  kwargs; fake universe records zero calls; no `live_ready` / `ZINC22`.
"""

from __future__ import annotations

import pytest

from app.mcp.client import LocalMCPClient


def _enable_live(monkeypatch, tools: list[str]) -> None:
    from app.settings import get_settings

    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.setenv("MCP_LIVE_TOOL_ALLOWLIST", ",".join(tools))
    get_settings.cache_clear()


# Each row: (tool_name, agent_name, step_id, official_args, expected_forward)
_LLM_STYLE_CASES = [
    (
        "DNA_calculate_gc_content",
        # Step 6 future / oligo tools aren't in any agent scope yet — for
        # this real-link audit we use the no-inventory `LocalMCPClient`
        # which lets AGENT_STEP_MAP gate by step alone. Developability_agent
        # owns step_06, which is where these would land.
        "developability_agent",
        "step_06",
        {"operation": "calculate_gc_content", "sequence": "ATCG"},
        {"operation": "calculate_gc_content", "sequence": "ATCG"},
    ),
    (
        "Rfam_get_family",
        "developability_agent",
        "step_06",
        {"operation": "get_family", "family_id": "RF00001"},
        {"operation": "get_family", "family_id": "RF00001"},
    ),
    (
        "SwissADME_calculate_adme",
        "developability_agent",
        "step_06",
        {"operation": "calculate_adme", "smiles": "CCO"},
        {"operation": "calculate_adme", "smiles": "CCO"},
    ),
    (
        "ChEMBL_get_assay_activities",
        "developability_agent",
        "step_06",
        {"assay_chembl_id__exact": "CHEMBL123", "limit": 1, "offset": 0},
        {"assay_chembl_id__exact": "CHEMBL123", "limit": 1, "offset": 0},
    ),
    (
        "PubTator3_get_annotations",
        "evidence_agent",
        "step_13",
        {"pmids": "12345"},
        {"pmids": "12345"},
    ),
]


@pytest.mark.parametrize(
    "tool,agent,step,args,forwarded", _LLM_STYLE_CASES
)
def test_official_kwargs_reach_adapter_cleanly(
    monkeypatch, install_universe, tool, agent, step, args, forwarded
):
    fake = install_universe(
        tools={tool: lambda a: {"results": [{"echo": dict(a)}]}}
    )
    _enable_live(monkeypatch, [tool])
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name=agent, step_id=step, tool_name=tool, **args
    )
    assert res["run_status"] == "success", res
    assert res["executor"] == "tooluniverse"
    # The argument set actually forwarded to TU matches what the LLM sent
    # (after wrapper normalization). No `_live` and no synthetic noise.
    assert fake.calls[0]["arguments"] == forwarded
    assert "_live" not in fake.calls[0]["arguments"]


# ── Contradictory official + legacy args raise honestly ───────────────────


def test_official_and_legacy_contradiction_raises():
    """`pick(official, legacy)` rejects mismatched pairs instead of silently
    preferring one — surfaces as `run_status="failed"`."""
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="PubTator3_get_annotations",
        pmid="111",
        pmids="222",
    )
    assert res["run_status"] == "failed"
    assert "contradictory" in res["error_message"].lower()


def test_legacy_only_still_works():
    """Old call sites passing `pmid` unchanged keep mocking cleanly."""
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="PubTator3_get_annotations",
        pmid="12345",
    )
    assert res["run_status"] == "success"
    assert res["payload"]["status"] == "mocked"


# ── Wrapper-hard-coded `operation` rejects a contradictory value ──────────


def test_operation_must_match_wrapper_hardcoded_value():
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="SwissADME_calculate_adme",
        operation="check_druglikeness",  # wrong op for this wrapper
        smiles="CCO",
    )
    assert res["run_status"] == "failed"
    assert "operation" in res["error_message"].lower()


# ── Deferred wrapper still surfaces dependency_unavailable ────────────────


def test_drugbank_with_official_args_stays_deferred(monkeypatch, install_universe):
    fake = install_universe(
        tools={"drugbank_get_drug_references_by_drug_name_or_id": lambda a: {"ok": 1}}
    )
    _enable_live(monkeypatch, ["drugbank_get_drug_references_by_drug_name_or_id"])
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="patent_ip_agent",
        step_id="step_14",
        tool_name="drugbank_get_drug_references_by_drug_name_or_id",
        query="imatinib",
        case_sensitive=False,
        exact_match=False,
        limit=20,
    )
    assert res["run_status"] == "dependency_unavailable"
    assert res["executor"] == "deferred"
    assert fake.calls == []


# ── ZINC stays intentionally_disabled under official kwargs ───────────────


@pytest.mark.parametrize(
    "tool,args",
    [
        ("ZINC_search_compounds", {"operation": "search_compounds", "query": "imatinib"}),
        ("ZINC_get_compound", {"operation": "get_compound", "zinc_id": "ZINC123"}),
        ("ZINC_search_by_smiles", {"operation": "search_by_smiles", "smiles": "CCO"}),
        ("ZINC_get_purchasable", {"operation": "get_purchasable", "tier": "in-stock", "zinc_id": "ZINC123"}),
    ],
)
def test_zinc_official_args_still_intentionally_disabled(
    monkeypatch, install_universe, tool, args
):
    fake = install_universe(tools={tool: lambda a: {"hits": [{"silent": "win"}]}})
    _enable_live(monkeypatch, [tool])
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name=tool,
        **args,
    )
    assert res["run_status"] == "dependency_unavailable"
    assert res["executor"] == "deferred"
    assert fake.calls == []
    # Mock envelope (no _live) must still NOT claim live_ready / ZINC22.
    mock = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name=tool,
        **args,
    )
    # In mock mode without live, run_status==success and we can read the mock body.
    # Re-call without the live env (allowlist hit but inject only fires when env truthy).
    # ZINC raises NotImplementedError on _live=True regardless — we've already proved
    # the deferred path. The mock-mode check is for `live_ready` / `ZINC22` absence.
    monkeypatch.setenv("MCP_LIVE_TOOLS", "false")
    from app.settings import get_settings

    get_settings.cache_clear()
    mock_only = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name=tool,
        **args,
    )
    payload = mock_only["payload"]
    flat = str(payload).lower()
    assert "live_ready" not in flat
    assert "zinc22" not in flat
