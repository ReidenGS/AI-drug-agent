"""Runtime readiness audit — Agent → LocalMCPClient → wrapper → adapter.

This suite goes beyond per-wrapper unit tests: it verifies the full path
that the real agents will exercise in production, for each of the five
agent/step lanes that have at least one migrated wrapper.

For every lane we assert:

1. With `MCP_LIVE_TOOLS=true` and the tool on `MCP_LIVE_TOOL_ALLOWLIST`,
   `LocalMCPClient` injects `_live=True` so the wrapper routes through
   `ToolUniverseAdapter` and the result envelope carries
   `executor="tooluniverse"`.
2. With live ON and an empty allowlist, `_live=True` is injected for every
   scoped tool (production all-live). With live ON and a non-empty allowlist,
   tools not on that allowlist stay on their deterministic mock envelope.
3. A still-deferred tool inside the agent's allowed step surfaces
   `dependency_unavailable` and the fake universe records zero calls —
   no silent fall-through to a fake "success".
4. Cross-step scope refusal: a tool outside the agent's step map returns
   `skipped / tool_not_in_agent_scope`.
5. The fake adapter payload is NOT inlined into the agent's normalized
   artifacts — agents only persist it under `tool_output_ref` /
   `tool_output_artifact_id` (verified at the LocalMCPClient layer via
   the `payload` envelope shape — agents are tested elsewhere).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XLSX = (
    _PROJECT_ROOT.parent / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"
)


@pytest.fixture
def inventory_client():
    """LocalMCPClient backed by the real v0.2 inventory so cross-step
    scope rejection actually fires (the inventory carries each tool's
    canonical step_id; the no-inventory client uses only AGENT_STEP_MAP)."""
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(_DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


# ── live-injection policy ──────────────────────────────────────────────────


def _set_live(monkeypatch, *, allowlist: str | None = None, enable: bool = True) -> None:
    """Configure `MCP_LIVE_TOOLS` + allowlist via env, then bust the cache."""
    from app.settings import get_settings

    monkeypatch.setenv("MCP_LIVE_TOOLS", "true" if enable else "false")
    if allowlist is None:
        monkeypatch.delenv("MCP_LIVE_TOOL_ALLOWLIST", raising=False)
    else:
        monkeypatch.setenv("MCP_LIVE_TOOL_ALLOWLIST", allowlist)
    get_settings.cache_clear()


# ── Step 5: CandidateContextAgent ──────────────────────────────────────────


@pytest.mark.parametrize(
    "tool_name,kwargs,fake_payload",
    [
        ("ChEMBL_search_molecules", {"query": "imatinib"}, {"results": [{"id": "X"}]}),
        ("SAbDab_search_structures", {"query": "HER2"}, {"results": [{"pdb": "1KZK"}]}),
        ("TheraSAbDab_search_by_target", {"target": "EGFR"}, {"results": []}),
    ],
)
def test_step5_candidate_context_live_path(
    monkeypatch, install_universe, tool_name, kwargs, fake_payload
):
    fake = install_universe(tools={tool_name: lambda args: fake_payload})
    _set_live(monkeypatch, allowlist=tool_name)
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name=tool_name,
        **kwargs,
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert res["payload"]["executor"] == "tooluniverse"
    assert len(fake.calls) == 1


def test_step5_empty_allowlist_is_all_live(monkeypatch, install_universe):
    """Production all-live: live ON + empty allowlist injects `_live=True`
    for every scoped tool, so the call routes through ToolUniverse."""
    fake = install_universe(
        tools={"ChEMBL_search_molecules": lambda args: {"results": [{"x": 1}]}}
    )
    _set_live(monkeypatch, allowlist="")  # live ON, allowlist empty => all-live
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="imatinib",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert len(fake.calls) == 1


def test_step5_nonempty_allowlist_miss_keeps_mock(monkeypatch, install_universe):
    """Constrained smoke/debug: with a non-empty allowlist, a tool NOT on it
    stays on the deterministic mock envelope."""
    fake = install_universe(
        tools={"ChEMBL_search_molecules": lambda args: {"results": [{"x": 1}]}}
    )
    _set_live(monkeypatch, allowlist="SAbDab_search_structures")  # other tool only
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="imatinib",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "mock"
    assert res["payload"]["status"] == "mocked"
    assert fake.calls == []


def test_step5_live_disabled_keeps_mock(monkeypatch, install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_molecules": lambda args: {"results": []}}
    )
    _set_live(monkeypatch, allowlist="ChEMBL_search_molecules", enable=False)
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="x",
    )
    assert res["executor"] == "mock"
    assert fake.calls == []


def test_step5_zinc_intentionally_disabled_does_not_touch_universe(
    monkeypatch, install_universe
):
    """ZINC must never reach the adapter, even with live + allowlist set."""
    fake = install_universe(
        tools={"ZINC_search_by_smiles": lambda args: {"ok": True}}
    )
    _set_live(monkeypatch, allowlist="ZINC_search_by_smiles")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ZINC_search_by_smiles",
        smiles="CCO",
    )
    assert res["run_status"] == "dependency_unavailable"
    assert res["executor"] == "deferred"
    assert fake.calls == []


# ── Step 6: DevelopabilityAgent ────────────────────────────────────────────


def test_step6_swissadme_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={
            "SwissADME_calculate_adme": lambda args: {
                "results": [{"smiles": args["smiles"]}]
            }
        }
    )
    _set_live(monkeypatch, allowlist="SwissADME_calculate_adme")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="SwissADME_calculate_adme",
        smiles="CCO",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {
        "operation": "calculate_adme",
        "smiles": "CCO",
    }


def test_step6_chembl_search_activities_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_activities": lambda args: {"activities": [{"id": "A1"}]}}
    )
    _set_live(monkeypatch, allowlist="ChEMBL_search_activities")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="ChEMBL_search_activities",
        target_chembl_id="CHEMBL25",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert len(fake.calls) == 1


def test_step6_admetai_dispatches_through_tooluniverse_when_live(
    monkeypatch, install_universe
):
    """ADMETAI wrappers are now live-wired: under MCP_LIVE_TOOLS with
    the tool in the allowlist, the LocalMCPClient routes the call
    through the ToolUniverseAdapter and surfaces a successful envelope."""
    fake = install_universe(
        tools={"ADMETAI_predict_toxicity": lambda args: {"smiles": args.get("smiles")}}
    )
    _set_live(monkeypatch, allowlist="ADMETAI_predict_toxicity")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="ADMETAI_predict_toxicity",
        smiles="CCO",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert len(fake.calls) == 1
    assert fake.calls[0]["arguments"] == {"smiles": "CCO"}


def test_step6_admetai_surfaces_upstream_error_when_admet_ai_missing(
    monkeypatch, install_universe
):
    """If the runtime ``admet_ai`` package is missing the TU
    ``ADMETAITool`` reports an error. The adapter normalises this to
    an envelope carrying ``status="upstream_error"`` — the wrapper does
    NOT raise (so LocalMCPClient still records ``run_status="success"``
    + ``executor="tooluniverse"``), but the persisted envelope's
    ``status`` is preserved so post-hoc inspection can audit it. This
    matches the path the Step 5 agent already uses to distinguish a
    real upstream error from a real success."""
    fake = install_universe(
        tools={"ADMETAI_predict_toxicity": lambda args: {
            "status": "error",
            "error": "ADMETModel requires 'admet-ai' package",
        }}
    )
    _set_live(monkeypatch, allowlist="ADMETAI_predict_toxicity")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="ADMETAI_predict_toxicity",
        smiles="CCO",
    )
    # LocalMCPClient run_status reflects whether the wrapper raised; it
    # did not. The TU upstream error is preserved INSIDE the envelope.
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    payload = res["payload"]
    assert payload["status"] == "upstream_error"
    assert "admet-ai" in (payload.get("error_message") or "")
    assert len(fake.calls) == 1


def test_step6_developability_cannot_call_step9_tool(
    monkeypatch, install_universe, inventory_client
):
    """AlphaMissense lives at step_09 — developability_agent must not see it.

    Uses the inventory-backed client so the step_id mismatch in the v0.2
    inventory actually rejects the call (the no-inventory client only
    enforces AGENT_STEP_MAP, which step_06 belongs to)."""
    fake = install_universe(
        tools={"AlphaMissense_get_variant_score": lambda args: {"score": 0.5}}
    )
    _set_live(monkeypatch, allowlist="AlphaMissense_get_variant_score")
    res = inventory_client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="AlphaMissense_get_variant_score",
        uniprot_id="P00533",
        variant="V600E",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"
    assert fake.calls == []


# ── Step 9: StructureAndDesignAgent / compound screening ───────────────────


def test_step9_alphamissense_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={
            "AlphaMissense_get_variant_score": lambda args: {
                "score": 0.82,
                "classification": "likely_pathogenic",
            }
        }
    )
    _set_live(monkeypatch, allowlist="AlphaMissense_get_variant_score")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name="AlphaMissense_get_variant_score",
        uniprot_id="P00533",
        variant="L858R",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {
        "uniprot_id": "P00533",
        "variant": "L858R",
    }


# P2 migration: DynaMut2 / ESM_generate / ESM_score / NvidiaNIM_rfdiffusion /
# NvidiaNIM_proteinmpnn are now thin ToolUniverse adapter bindings too (no
# Step 9 active tool remains deferred). Each `(tool_name, kwargs, echo)` proves
# the same Agent -> LocalMCPClient -> wrapper -> adapter path AlphaMissense
# already exercised above — live routes through the fake universe with
# `executor="tooluniverse"`, never a mocked/deferred shortcut.
_STEP9_TU_WIRED_CASES = [
    (
        "DynaMut2_predict_stability",
        {"operation": "predict_stability", "pdb_id": "1N8Z", "chain": "A", "mutation": "V777L"},
    ),
    ("ESM_generate_protein_sequence", {"prompt_sequence": "MKTAYIAK"}),
    ("ESM_score_variant_sae_batch", {"sequence": "MKTAYIAK", "variants": ["V1L"]}),
    ("NvidiaNIM_rfdiffusion", {"contigs": "A:1-10", "input_pdb": "run/structure.pdb"}),
    ("NvidiaNIM_proteinmpnn", {"input_pdb": "run/structure.pdb"}),
]


@pytest.mark.parametrize("tool_name,kwargs", _STEP9_TU_WIRED_CASES)
def test_step9_tu_wired_tools_live_path(monkeypatch, install_universe, tool_name, kwargs):
    fake = install_universe(tools={tool_name: lambda args: {"ok": True, **args}})
    _set_live(monkeypatch, allowlist=tool_name)
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name=tool_name,
        **kwargs,
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == kwargs


@pytest.mark.parametrize("tool_name,kwargs", _STEP9_TU_WIRED_CASES)
def test_step9_tu_wired_tools_live_disabled_stays_dependency_unavailable(
    monkeypatch, install_universe, tool_name, kwargs
):
    """Live OFF: the wrapper's own `_live=False` guard still fires (via
    `LocalMCPClient` catching `NotImplementedError`) — no silent mock
    success, and the fake universe records zero calls."""
    fake = install_universe(tools={tool_name: lambda args: {"ok": True, **args}})
    _set_live(monkeypatch, enable=False)
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name=tool_name,
        **kwargs,
    )
    assert res["run_status"] == "dependency_unavailable"
    assert fake.calls == []


def test_step9_compound_screening_chembl_carve_out_live(
    monkeypatch, install_universe
):
    """`structure_and_design_agent` step_09 has an architecture override
    allowing ChEMBL_search_similarity (otherwise step_05). Live path
    must still route through the adapter cleanly."""
    fake = install_universe(
        tools={"ChEMBL_search_similarity": lambda args: {"results": [{"id": "S1"}]}}
    )
    _set_live(monkeypatch, allowlist="ChEMBL_search_similarity")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="structure_and_design_agent",
        step_id="step_09",
        tool_name="ChEMBL_search_similarity",
        smiles="CCO",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert len(fake.calls) == 1


# ── Step 13: EvidenceAgent ─────────────────────────────────────────────────


def test_step13_europepmc_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"pmid": "1"}]}}
    )
    _set_live(monkeypatch, allowlist="EuropePMC_search_articles")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2 imatinib",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    # EuropePMC wrapper forwards `query` + default `page_size=25` as `limit`.
    assert fake.calls[0]["arguments"]["query"] == "HER2 imatinib"


def test_step13_multi_agent_literature_search_live_path(
    monkeypatch, install_universe
):
    fake = install_universe(
        tools={
            "MultiAgentLiteratureSearch": lambda args: {
                "success": True,
                "total_papers": 0,
                "max_iterations_used": args.get("max_iterations"),
            }
        }
    )
    _set_live(monkeypatch, allowlist="MultiAgentLiteratureSearch")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="MultiAgentLiteratureSearch",
        query="ADC payload class",
        max_iterations=8,  # caller asks for 8, wrapper clamps to 1
        quality_threshold=0.5,
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    # Hard clamp invariant — must hold even from the agent path.
    assert fake.calls[0]["arguments"]["max_iterations"] == 1


def test_step13_tool_not_visible_outside_step13(
    monkeypatch, install_universe, inventory_client
):
    """MultiAgentLiteratureSearch must not be callable from other agents."""
    fake = install_universe(
        tools={"MultiAgentLiteratureSearch": lambda args: {"success": True}}
    )
    _set_live(monkeypatch, allowlist="MultiAgentLiteratureSearch")
    res = inventory_client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="MultiAgentLiteratureSearch",
        query="x",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"
    assert fake.calls == []


# ── Step 14: PatentIPAgent ─────────────────────────────────────────────────


def test_step14_pubchem_patents_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={
            "PubChem_get_associated_patents_by_CID": lambda args: {
                "patents": [{"patent_number": "US123"}]
            }
        }
    )
    _set_live(monkeypatch, allowlist="PubChem_get_associated_patents_by_CID")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="patent_ip_agent",
        step_id="step_14",
        tool_name="PubChem_get_associated_patents_by_CID",
        cid=2244,
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"cid": 2244}


def test_step14_fda_orangebook_live_path(monkeypatch, install_universe):
    fake = install_universe(
        tools={
            "FDA_OrangeBook_get_patent_info": lambda args: {
                "patents": [{"patent_no": "1234567"}]
            }
        }
    )
    _set_live(monkeypatch, allowlist="FDA_OrangeBook_get_patent_info")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="patent_ip_agent",
        step_id="step_14",
        tool_name="FDA_OrangeBook_get_patent_info",
        application_number="N021588",
    )
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    assert len(fake.calls) == 1


def test_step14_drugbank_remains_deferred(monkeypatch, install_universe):
    """DrugBank wrapper is manual_wrapper/key_required and stays deferred."""
    fake = install_universe(
        tools={"drugbank_get_drug_references_by_drug_name_or_id": lambda args: {"ok": True}}
    )
    _set_live(monkeypatch, allowlist="drugbank_get_drug_references_by_drug_name_or_id")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="patent_ip_agent",
        step_id="step_14",
        tool_name="drugbank_get_drug_references_by_drug_name_or_id",
        drug_name_or_id="imatinib",
    )
    assert res["run_status"] == "dependency_unavailable"
    assert res["executor"] == "deferred"
    assert fake.calls == []


# ── raw payload isolation ──────────────────────────────────────────────────


def test_raw_payload_stays_in_envelope_not_in_run_status(
    monkeypatch, install_universe
):
    """The raw TU payload must live inside `result["payload"]` (which the
    agent persists to `tool_output_ref`). It must not leak into the
    top-level `run_status` shape that drives normalized artifacts."""
    raw_secret = {"hidden": "raw-data-blob", "rows": list(range(100))}
    install_universe(
        tools={"ChEMBL_search_molecules": lambda args: {"results": [raw_secret]}}
    )
    _set_live(monkeypatch, allowlist="ChEMBL_search_molecules")
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="x",
    )
    # Raw payload must only surface inside `payload`, not anywhere else.
    assert res["run_status"] == "success"
    assert res["executor"] == "tooluniverse"
    flat_top_level = {k: v for k, v in res.items() if k != "payload"}
    assert "rows" not in str(flat_top_level)
    assert "raw-data-blob" not in str(flat_top_level)
    # Payload IS where it lives, ready for tool_output_ref persistence.
    assert raw_secret in res["payload"]["payload"]["results"]


# ── classify executor: unknown / null payload ──────────────────────────────


def test_executor_unknown_when_wrapper_returns_non_dict(install_universe):
    install_universe()
    bindings = {"NonDictTool": lambda **kw: "plain-string"}
    client = LocalMCPClient(bindings=bindings)
    # The scope guard would reject this (no agent map). Bypass by patching:
    from app.mcp import scope_filter as sf

    sf.AGENT_STEP_MAP.setdefault("__audit__", set()).add("step_99")
    try:
        client._bindings["NonDictTool"] = lambda: "plain-string"
        res = client.call_tool(
            agent_name="__audit__",
            step_id="step_99",
            tool_name="NonDictTool",
        )
        assert res["run_status"] == "success"
        assert res["executor"] == "unknown"
    finally:
        sf.AGENT_STEP_MAP["__audit__"].discard("step_99")
