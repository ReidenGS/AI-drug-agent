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
    client = LocalMCPClient()
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ChEMBL_search_molecules",
        query="MMAE",
    )
    # ChEMBL_search_molecules wrapper is unimplemented → dependency_unavailable.
    assert res["run_status"] == "dependency_unavailable"


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
