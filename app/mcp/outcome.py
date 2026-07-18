"""Shared fail-closed MCP/wrapper outcome authority.

Local transport, FastMCP transport, and future callers must normalize wrapper
envelopes through :func:`normalize_mcp_outcome`.  Raw envelopes remain under
``payload`` for persistence/audit; a successful Python return is never enough
to claim a successful tool execution.
"""

from __future__ import annotations

from typing import Any, Callable

from ..schemas.patent_evidence_contract import PATENT_EVIDENCE_MULTI_SEARCH_BACKENDS


SUCCESS_ENVELOPE_STATUSES = frozenset({"ok", "empty"})
FAILED_ENVELOPE_STATUSES = frozenset({"upstream_error", "error", "failed"})
DEPENDENCY_ENVELOPE_STATUS = "dependency_unavailable"

MULTI_AGENT_REQUIRED_AGENTS = frozenset(
    {
        "IntentAnalyzerAgent",
        "KeywordExtractorAgent",
        "ResultSummarizerAgent",
        "QualityCheckerAgent",
        "OverallSummaryAgent",
    }
)
MULTI_AGENT_SEARCH_BACKENDS = PATENT_EVIDENCE_MULTI_SEARCH_BACKENDS
CONTROLLED_TOOL_CALL_INSTRUMENTATION = "patent_evidence_call_tool_v1"

PATENT_EVIDENCE_COMPOSITE_RUNTIME_AVAILABILITY: dict[str, dict[str, Any]] = {
    "LiteratureSearchTool": {
        "status": "dependency_unavailable",
        "can_execute": False,
        "reason_code": "medical_literature_reviewer_outside_approved_inventory",
    },
    "MultiAgentLiteratureSearch": {
        "status": "scope_blocked",
        "can_execute": False,
        "reason_code": "uncontained_tooluniverse_full_discovery",
    },
}


def composite_scope_block_envelope(tool_name: str) -> dict[str, Any] | None:
    availability = PATENT_EVIDENCE_COMPOSITE_RUNTIME_AVAILABILITY.get(tool_name)
    if availability is None:
        return None
    details: dict[str, Any] = {
        "approved_inventory_enforced": True,
        "runtime_status": availability["status"],
    }
    if tool_name == "LiteratureSearchTool":
        details["missing_required_tools"] = ["MedicalLiteratureReviewer"]
    else:
        details.update(
            {
                "unsafe_operations": ["force_full_discovery", "load_tools()"],
                "scope_policy_version": "patent_evidence_compose_scope_v1",
                "approved_search_backends": sorted(MULTI_AGENT_SEARCH_BACKENDS),
            }
        )
    return dependency_unavailable_envelope(
        tool_name=tool_name,
        reason_code=str(availability["reason_code"]),
        details=details,
    )


def dependency_unavailable_envelope(
    *, tool_name: str, reason_code: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "status": DEPENDENCY_ENVELOPE_STATUS,
        "source": tool_name,
        "executor": "deferred",
        "reason_code": reason_code,
        "attempted_execution_count": 0,
        "successful_execution_count": 0,
        "actual_execution_count": 0,
        "tool_call_records": [],
        **({"scope_audit": details} if details else {}),
    }


def failed_envelope(*, tool_name: str, reason_code: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "source": tool_name,
        "executor": "error",
        "reason_code": reason_code,
        "error_message": reason_code,
        "attempted_execution_count": 1,
        "successful_execution_count": 0,
        "actual_execution_count": 1,
        "tool_call_records": [],
    }


def invoke_wrapper(
    *, tool_name: str, wrapper: Callable[..., Any], kwargs: dict[str, Any]
) -> Any:
    """Invoke a wrapper and turn exceptions into explicit envelopes."""
    try:
        return wrapper(**kwargs)
    except NotImplementedError:
        return dependency_unavailable_envelope(
            tool_name=tool_name, reason_code="wrapper_not_wired"
        )
    except Exception:  # noqa: BLE001 - public boundary emits compact code only
        return failed_envelope(tool_name=tool_name, reason_code="wrapper_exception")


def _inner_payload(envelope: dict[str, Any]) -> Any:
    return envelope.get("payload") if "payload" in envelope else envelope


def _tool_call_records(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    inner = _inner_payload(envelope)
    raw = None
    if isinstance(inner, dict):
        raw = inner.get("tool_call_records")
    if raw is None:
        raw = envelope.get("tool_call_records")
    return [dict(item) for item in raw or [] if isinstance(item, dict)]


def _record_name(record: dict[str, Any]) -> str:
    return str(record.get("tool_name") or record.get("agent_name") or "")


def _record_succeeded(record: dict[str, Any]) -> bool:
    return str(record.get("run_status") or record.get("status") or "").lower() in {
        "success",
        "ok",
        "empty",
    }


def _successful_record_names(records: list[dict[str, Any]]) -> set[str]:
    return {_record_name(record) for record in records if _record_succeeded(record)}


def _explicit_count(envelope: dict[str, Any], key: str) -> int | None:
    inner = _inner_payload(envelope)
    for candidate in (inner, envelope):
        if isinstance(candidate, dict):
            value = candidate.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def _success_contradiction(envelope: dict[str, Any]) -> str | None:
    if any(envelope.get(key) not in (None, "", [], {}) for key in ("error", "error_message")):
        return "success_envelope_contains_error"
    inner = _inner_payload(envelope)
    if inner is envelope or not isinstance(inner, dict):
        return None
    if inner.get("success") is False:
        return "nested_success_false"
    if str(inner.get("status") or "").lower() in FAILED_ENVELOPE_STATUSES:
        return "nested_failure_status"
    if inner.get("error") not in (None, "", [], {}):
        return "nested_error_present"
    return None


def _literature_integrity_error(
    envelope: dict[str, Any], records: list[dict[str, Any]]
) -> str | None:
    names = _successful_record_names(records)
    if "MedicalLiteratureReviewer" not in names:
        return "medical_literature_reviewer_not_executed"
    return None


def _multi_agent_integrity_error(
    envelope: dict[str, Any], records: list[dict[str, Any]]
) -> str | None:
    inner = _inner_payload(envelope)
    if not isinstance(inner, dict) or inner.get("success") is not True:
        return "multi_agent_success_not_confirmed"
    names = _successful_record_names(records)
    missing_agents = sorted(MULTI_AGENT_REQUIRED_AGENTS - names)
    if missing_agents:
        return "multi_agent_internal_agents_not_executed"
    results = inner.get("results") if isinstance(inner.get("results"), dict) else {}
    search_metadata = (
        results.get("search_metadata")
        if isinstance(results.get("search_metadata"), dict)
        else {}
    )
    plans = inner.get("search_plans")
    if isinstance(plans, list):
        plan_count = len(plans)
    else:
        plan_count = search_metadata.get("total_plans")
    if not isinstance(plan_count, int) or isinstance(plan_count, bool) or plan_count <= 0:
        return "multi_agent_search_plans_zero"
    if not (names & MULTI_AGENT_SEARCH_BACKENDS):
        return "multi_agent_search_not_executed"
    if inner.get("tool_call_instrumentation") != CONTROLLED_TOOL_CALL_INSTRUMENTATION:
        return "multi_agent_execution_not_instrumented"
    return None


def _patent_source_normalization(tool_name: str, envelope: dict[str, Any]) -> dict[str, Any] | None:
    inner = _inner_payload(envelope)
    if not isinstance(inner, dict):
        return None
    if tool_name == "PubChem_get_associated_patents_by_CID":
        references = inner.get("references")
        source_path = "payload.references"
        data = inner.get("data")
        record = data.get("Record") if isinstance(data, dict) else None
        if isinstance(record, dict) and isinstance(record.get("Reference"), list):
            references = record["Reference"]
            source_path = "payload.data.Record.Reference"
        if isinstance(references, list):
            return {
                "source_type": "pubchem_associated_reference",
                "source_path": source_path,
                "record_count": len(references),
                "confirmed_patent_records": [],
                "functional_limitation": (
                    "PubChem associated references are discovery leads, not "
                    "confirmed patent records"
                ),
            }
    if tool_name == "FDA_OrangeBook_get_patent_info":
        rows = inner.get("application_rows")
        source_path = "payload.application_rows"
        if not isinstance(rows, list):
            rows = inner.get("records")
            source_path = "payload.records"
        data = inner.get("data")
        if isinstance(data, dict) and isinstance(data.get("drugs"), list):
            rows = data["drugs"]
            source_path = "payload.data.drugs"
        if isinstance(rows, list):
            return {
                "source_type": "fda_orange_book_application_row",
                "source_path": source_path,
                "record_count": len(rows),
                "confirmed_patent_records": [],
                "functional_limitation": (
                    "FDA application rows are regulatory lookup results, not "
                    "confirmed patent records"
                ),
            }
    return None


def normalize_mcp_outcome(*, tool_name: str, envelope: Any) -> dict[str, Any]:
    """Map one exact wrapper envelope to a canonical outer MCP outcome."""
    if not isinstance(envelope, dict):
        return {
            "run_status": "failed",
            "tool_name": tool_name,
            "payload": envelope,
            "envelope_status": "invalid",
            "executor": "unknown",
            "attempted_execution_count": 1,
            "successful_execution_count": 0,
            "actual_execution_count": 1,
            "tool_call_records": [],
            "reason": "invalid_wrapper_envelope",
            "error_message": "invalid_wrapper_envelope",
        }
    status = envelope.get("status")
    if not isinstance(status, str) or not status:
        status = "missing"
    status = status.lower()
    records = _tool_call_records(envelope)

    reason_code: str | None = None
    if status in SUCCESS_ENVELOPE_STATUSES:
        reason_code = _success_contradiction(envelope)
        if reason_code is None and tool_name == "LiteratureSearchTool":
            reason_code = _literature_integrity_error(envelope, records)
        if reason_code is None and tool_name == "MultiAgentLiteratureSearch":
            reason_code = _multi_agent_integrity_error(envelope, records)
        run_status = "success" if reason_code is None else "failed"
    elif status in FAILED_ENVELOPE_STATUSES:
        run_status = "failed"
    elif status == DEPENDENCY_ENVELOPE_STATUS:
        run_status = "dependency_unavailable"
    elif status == "missing":
        run_status = "failed"
        reason_code = "missing_wrapper_status"
    else:
        run_status = "failed"
        reason_code = "unknown_wrapper_status"

    normalized_output = _patent_source_normalization(tool_name, envelope)
    attempted_count = _explicit_count(envelope, "attempted_execution_count")
    if attempted_count is None:
        attempted_count = _explicit_count(envelope, "actual_execution_count")
    if attempted_count is None:
        attempted_count = 1
    # One normalized result represents one outer dispatch. A failed or
    # dependency-unavailable result can never carry a successful outer count,
    # even if an upstream envelope claims otherwise.
    successful_count = 1 if run_status == "success" else 0
    successful_record_count = sum(1 for record in records if _record_succeeded(record))
    result: dict[str, Any] = {
        "run_status": run_status,
        "tool_name": tool_name,
        "payload": envelope,
        "envelope_status": status,
        "executor": str(
            envelope.get("executor")
            or ("mock" if status == "mocked" else "unknown")
        ),
        # `actual_execution_count` is retained as the backward-compatible name
        # for outer dispatch attempts. Nested composite records are reported
        # separately and never inflate the outer call count.
        "attempted_execution_count": attempted_count,
        "successful_execution_count": successful_count,
        "actual_execution_count": attempted_count,
        "successful_tool_call_record_count": successful_record_count,
        "tool_call_records": records,
    }
    if normalized_output is not None:
        result["normalized_output"] = normalized_output
    compact_reason = reason_code or envelope.get("reason_code")
    if compact_reason:
        result["reason"] = str(compact_reason)
    if run_status == "failed":
        result["error_message"] = str(
            envelope.get("error_message")
            or envelope.get("error")
            or compact_reason
            or status
        )
    return result
