from __future__ import annotations

import json
import inspect

import pytest

from app.mcp.client import FastMCPClient, LocalMCPClient
from app.mcp.outcome import (
    MULTI_AGENT_REQUIRED_AGENTS,
    normalize_mcp_outcome,
)
from app.mcp.tools import evidence


def _local(tool_name: str, envelope: object) -> dict:
    client = LocalMCPClient(bindings={tool_name: lambda **_kwargs: envelope})
    return client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name=tool_name,
    )


class _Response:
    is_error = False

    def __init__(self, envelope: object) -> None:
        self.content = [{"type": "text", "text": json.dumps(envelope)}]


def _fast(tool_name: str, envelope: object) -> dict:
    client = object.__new__(FastMCPClient)
    client._fastmcp = object()
    client._remote = None
    client.inventory = None
    client._scope_ok = lambda **_kwargs: True
    client._dispatch_sync = lambda _tool_name, _kwargs: _Response(envelope)
    return client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name=tool_name,
    )


@pytest.mark.parametrize(
    ("envelope", "expected"),
    [
        ({"status": "ok", "payload": {"results": [1]}}, "success"),
        ({"status": "empty", "payload": {"results": []}}, "success"),
        ({"status": "upstream_error", "error_message": "maintenance"}, "failed"),
        ({"status": "error", "error": "maintenance"}, "failed"),
        ({"status": "failed", "error": "maintenance"}, "failed"),
        ({"status": "dependency_unavailable", "reason_code": "missing"}, "dependency_unavailable"),
        ({"status": "mystery"}, "failed"),
        ({"status": "mocked", "executor": "mock"}, "failed"),
        ({"payload": {"results": [1]}}, "failed"),
        ({"status": "ok", "error_message": "contradictory"}, "failed"),
        ({"status": "ok", "payload": {"success": False}}, "failed"),
    ],
)
def test_local_and_fastmcp_share_exact_envelope_mapping(envelope, expected):
    local = _local("EuropePMC_search_articles", envelope)
    fast = _fast("EuropePMC_search_articles", envelope)
    assert local["run_status"] == fast["run_status"] == expected
    assert local["envelope_status"] == fast["envelope_status"]
    assert local["payload"] == fast["payload"] == envelope
    assert local["attempted_execution_count"] == fast["attempted_execution_count"]
    assert local["successful_execution_count"] == fast["successful_execution_count"]


def test_pubtator_maintenance_envelope_remains_failed_at_both_layers():
    raw = {
        "status": "upstream_error",
        "source": "PubTator3_LiteratureSearch",
        "error_message": "PubTator maintenance",
        "payload": {"results": []},
    }
    result = _local("PubTator3_LiteratureSearch", raw)
    assert result["run_status"] == "failed"
    assert result["envelope_status"] == "upstream_error"
    assert result["payload"] == raw


def test_literature_reviewer_execution_record_is_required_for_success():
    missing = {
        "status": "ok",
        "payload": {"summary": "looks successful", "papers": []},
    }
    rejected = normalize_mcp_outcome(
        tool_name="LiteratureSearchTool", envelope=missing
    )
    assert rejected["run_status"] == "failed"
    assert rejected["reason"] == "medical_literature_reviewer_not_executed"
    assert rejected["attempted_execution_count"] == 1
    assert rejected["successful_execution_count"] == 0

    records = [
        {
            "tool_name": "MedicalLiteratureReviewer",
            "run_status": "success",
        }
    ]
    accepted = normalize_mcp_outcome(
        tool_name="LiteratureSearchTool",
        envelope={
            "status": "ok",
            "payload": {
                "summary": "reviewed",
                "papers": [],
                "tool_call_records": records,
            },
        },
    )
    assert accepted["run_status"] == "success"
    assert accepted["actual_execution_count"] == 1
    assert accepted["successful_execution_count"] == 1
    assert accepted["successful_tool_call_record_count"] == 1
    assert accepted["tool_call_records"] == records


def _multi_raw(
    *, plans: int = 1, include_agents: bool = True, include_search: bool = True
) -> dict:
    records = []
    if include_agents:
        records.extend(
            {"agent_name": name, "run_status": "success"}
            for name in sorted(MULTI_AGENT_REQUIRED_AGENTS)
        )
    if include_search:
        records.append(
            {
                "tool_name": "EuropePMC_search_articles",
                "run_status": "success",
            }
        )
    return {
        "status": "ok",
        "payload": {
            "success": True,
            "search_plans": [{} for _ in range(plans)],
            "results": {"papers": []},
            "tool_call_records": records,
        },
    }


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (_multi_raw(include_agents=False), "multi_agent_internal_agents_not_executed"),
        (_multi_raw(plans=0), "multi_agent_search_plans_zero"),
        (_multi_raw(include_search=False), "multi_agent_search_not_executed"),
    ],
)
def test_multi_agent_incomplete_raw_shapes_fail_closed(raw, reason):
    result = normalize_mcp_outcome(
        tool_name="MultiAgentLiteratureSearch", envelope=raw
    )
    assert result["run_status"] == "failed"
    assert result["reason"] == reason


def test_multi_agent_synthetic_records_do_not_claim_official_success():
    """There is no controlled call_tool instrumentation in this runtime."""
    raw = _multi_raw()
    result = normalize_mcp_outcome(
        tool_name="MultiAgentLiteratureSearch", envelope=raw
    )
    assert result["run_status"] == "failed"
    assert result["reason"] == "multi_agent_execution_not_instrumented"
    assert result["actual_execution_count"] == 1
    assert result["successful_tool_call_record_count"] == 6
    assert result["tool_call_records"] == raw["payload"]["tool_call_records"]


@pytest.mark.parametrize(
    ("tool_name", "wrapper", "kwargs", "reason"),
    [
        (
            "LiteratureSearchTool",
            evidence.LiteratureSearchTool,
            {"query": "HER2", "_live": True},
            "medical_literature_reviewer_outside_approved_inventory",
        ),
        (
            "MultiAgentLiteratureSearch",
            evidence.MultiAgentLiteratureSearch,
            {"query": "HER2", "_live": True},
            "uncontained_tooluniverse_full_discovery",
        ),
    ],
)
def test_unsafe_compose_tools_are_scope_blocked_before_tooluniverse(
    monkeypatch, tool_name, wrapper, kwargs, reason
):
    monkeypatch.setattr(
        evidence,
        "_tu",
        lambda *_args, **_kwargs: pytest.fail("ToolUniverse must not execute"),
    )
    envelope = wrapper(**kwargs)
    result = normalize_mcp_outcome(tool_name=tool_name, envelope=envelope)
    assert result["run_status"] == "dependency_unavailable"
    assert result["reason"] == reason
    assert result["actual_execution_count"] == 0
    assert result["tool_call_records"] == []


def test_installed_multi_agent_compose_contains_unscoped_autoload_operations():
    from tooluniverse.compose_scripts import multi_agent_literature_search

    source = inspect.getsource(multi_agent_literature_search.compose)
    assert "tooluniverse.force_full_discovery()" in source
    assert "tooluniverse.load_tools()" in source


@pytest.mark.parametrize(
    ("tool_name", "inner", "source_type", "source_path", "limitation_token"),
    [
        (
            "PubChem_get_associated_patents_by_CID",
            {"data": {"Record": {"Reference": [{"title": "reference", "id": "PMID1"}]}}},
            "pubchem_associated_reference",
            "payload.data.Record.Reference",
            "not confirmed patent records",
        ),
        (
            "FDA_OrangeBook_get_patent_info",
            {"data": {"drugs": [{"application_number": "NDA123"}]}},
            "fda_orange_book_application_row",
            "payload.data.drugs",
            "not confirmed patent records",
        ),
    ],
)
def test_real_nested_reference_and_application_rows_are_not_confirmed_patents(
    tool_name, inner, source_type, source_path, limitation_token
):
    result = normalize_mcp_outcome(
        tool_name=tool_name,
        envelope={"status": "ok", "payload": inner},
    )
    assert result["run_status"] == "success"
    normalized = result["normalized_output"]
    assert normalized["source_type"] == source_type
    assert normalized["source_path"] == source_path
    assert normalized["record_count"] == 1
    assert normalized["confirmed_patent_records"] == []
    assert limitation_token in normalized["functional_limitation"]


@pytest.mark.parametrize(
    ("tool_name", "inner", "source_path"),
    [
        (
            "PubChem_get_associated_patents_by_CID",
            {"references": [{"id": "legacy"}]},
            "payload.references",
        ),
        (
            "FDA_OrangeBook_get_patent_info",
            {"application_rows": [{"application_number": "legacy"}]},
            "payload.application_rows",
        ),
        (
            "FDA_OrangeBook_get_patent_info",
            {"records": [{"application_number": "legacy"}]},
            "payload.records",
        ),
    ],
)
def test_patent_normalization_preserves_only_explicit_compatibility_shapes(
    tool_name, inner, source_path
):
    result = normalize_mcp_outcome(
        tool_name=tool_name, envelope={"status": "ok", "payload": inner}
    )
    assert result["normalized_output"]["source_path"] == source_path


def test_local_wrapper_exception_is_compact_and_redacted():
    secret = "https://user:credential@example.invalid?q=HER2&payload=raw"

    def _raise(**_kwargs):
        raise RuntimeError(secret)

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _raise})
    result = client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
    )
    rendered = json.dumps(result)
    assert result["run_status"] == "failed"
    assert result["reason"] == "wrapper_exception"
    assert result["attempted_execution_count"] == 1
    assert secret not in rendered
    assert "credential" not in rendered


def test_fastmcp_transport_exception_is_compact_and_redacted():
    secret = "https://user:credential@example.invalid?q=HER2&payload=raw"
    client = object.__new__(FastMCPClient)
    client._fastmcp = object()
    client._remote = None
    client.inventory = None
    client._scope_ok = lambda **_kwargs: True

    def _raise(_tool_name, _kwargs):
        raise RuntimeError(secret)

    client._dispatch_sync = _raise
    result = client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
    )
    rendered = json.dumps(result)
    assert result["run_status"] == "failed"
    assert result["reason"] == "fastmcp_transport_exception"
    assert result["attempted_execution_count"] == 1
    assert secret not in rendered
    assert "credential" not in rendered


def test_fastmcp_error_response_does_not_expose_raw_content():
    response = _Response({"status": "ok"})
    response.is_error = True
    response.content = [{"type": "text", "text": "credential endpoint query raw payload"}]
    client = object.__new__(FastMCPClient)
    client._fastmcp = object()
    client._remote = None
    client.inventory = None
    client._scope_ok = lambda **_kwargs: True
    client._dispatch_sync = lambda _tool_name, _kwargs: response
    result = client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
    )
    rendered = json.dumps(result)
    assert result["reason"] == "fastmcp_transport_error"
    assert "endpoint query raw payload" not in rendered


def test_fastmcp_invalid_envelope_does_not_expose_raw_content():
    response = _Response({"status": "ok"})
    response.content = [{"type": "text", "text": "credential endpoint query raw payload"}]
    client = object.__new__(FastMCPClient)
    client._fastmcp = object()
    client._remote = None
    client.inventory = None
    client._scope_ok = lambda **_kwargs: True
    client._dispatch_sync = lambda _tool_name, _kwargs: response
    result = client.call_tool(
        agent_name="patent_evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
    )
    rendered = json.dumps(result)
    assert result["run_status"] == "failed"
    assert result["reason"] == "fastmcp_invalid_envelope"
    assert "endpoint query raw payload" not in rendered
