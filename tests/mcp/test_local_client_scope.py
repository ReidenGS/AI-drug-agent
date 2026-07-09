from __future__ import annotations

from app.mcp.client import LocalMCPClient


def test_local_client_blocks_out_of_scope_tool():
    client = LocalMCPClient()
    # developability_agent is not allowed at step_05; SAbDab is a Step 5 tool.
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_05",
        tool_name="SAbDab_search_structures",
        query="HER2",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"


def test_local_client_allows_in_scope_tool():
    """An in-scope tool with a wired wrapper reaches its binding and
    returns the wrapper's envelope. ``ADMETAI_predict_toxicity`` was
    migrated to a live wrapper that routes through
    ``ToolUniverseAdapter`` when ``_live=True`` and otherwise returns a
    deterministic mock envelope. Here we exercise the mock-mode path
    (no ``_live`` injection) to prove the binding is callable end-to-end
    and that scope routing succeeds."""
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="developability_agent",
        step_id="step_06",
        tool_name="ADMETAI_predict_toxicity",
        smiles="CCO",
    )
    assert res["run_status"] == "success"
    payload = res["payload"]
    assert payload["status"] == "mocked"
    assert payload["source"] == "ADMETAI_predict_toxicity"
    assert payload["smiles"] == "CCO"
    assert payload["predictions"] is None


def test_local_client_real_call_path_for_wired_wrapper():
    """alphafold_get_prediction has a real wrapper; in non-_live mode it
    returns a mocked dict, proving the binding is callable."""
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="alphafold_get_prediction",
        uniprot="P00533",
    )
    assert res["run_status"] == "success"
    assert res["payload"]["uniprot"] == "P00533"


def test_local_client_list_tools_respects_agent_scope():
    client = LocalMCPClient()
    sup_tools = client.list_tools(agent_name="supervisor_agent", step_id="step_05")
    assert sup_tools == []  # supervisor is not mapped to step_05
    cca_tools = client.list_tools(agent_name="candidate_context_agent", step_id="step_05")
    assert "ChEMBL_search_molecules" in cca_tools


def test_local_client_europepmc_callable_for_patent_ip_step_14():
    # EuropePMC is exposed to patent_ip_agent/step_14 (Enola literature/prior-art
    # evidence) via the scope override; the binding is reachable end-to-end.
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="patent_ip_agent", step_id="step_14",
        tool_name="EuropePMC_search_articles", query="HER2 ADC prior art",
    )
    assert res["run_status"] == "success"
    assert res["payload"]["source"] == "EuropePMC_search_articles"
    assert res["payload"]["query"] == "HER2 ADC prior art"
