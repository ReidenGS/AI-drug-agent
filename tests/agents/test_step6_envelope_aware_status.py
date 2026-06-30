"""Step 6 lane status/summary is envelope-aware for upstream_error.

A tool whose OUTER ToolCallRecord.run_status is success but whose output
envelope status is upstream_error (e.g. a live IEDB call that exhausted
transient retries) must make the lane partial, not clean ok, and surface
the upstream_error tool in the lane summary.
"""

from __future__ import annotations

from app.agents.developability_agent import (
    _aggregate_lane_run_status,
    _aggregate_lane_summary,
)
from app.schemas.common import ToolCallRecord


def _rec(tool_name: str, run_status: str, envelope_status: str | None) -> ToolCallRecord:
    summary = {}
    if envelope_status is not None:
        summary["output_envelope_status"] = envelope_status
    return ToolCallRecord(
        tool_call_id=f"tc_{tool_name}",
        tool_name=tool_name,
        run_status=run_status,
        tool_input_summary=summary,
    )


def test_lane_partial_when_envelope_upstream_error_despite_success_run_status():
    records = [
        _rec("PROSITE_scan_sequence", "success", "ok"),
        # Outer run_status success but envelope upstream_error (retries exhausted).
        _rec("IEDB_predict_mhci_binding", "success", "upstream_error"),
    ]
    assert _aggregate_lane_run_status(records) == "partial"
    summary = _aggregate_lane_summary(records)
    assert "upstream_error" in summary
    assert "IEDB_predict_mhci_binding" in summary


def test_lane_ok_when_all_envelopes_clean():
    records = [
        _rec("PROSITE_scan_sequence", "success", "ok"),
        _rec("IEDB_predict_mhci_binding", "success", "ok"),
    ]
    assert _aggregate_lane_run_status(records) == "ok"


def test_lane_failed_when_upstream_error_and_no_success_outer_status():
    # Outer run_status is not success and the envelope is upstream_error.
    records = [_rec("IEDB_predict_mhci_binding", "failed", "upstream_error")]
    assert _aggregate_lane_run_status(records) == "failed"


def test_lone_upstream_error_with_success_outer_is_partial_not_clean():
    # Even a single tool whose outer run_status is success but whose envelope
    # is upstream_error must NOT report clean ok — it is partial.
    records = [_rec("IEDB_predict_mhci_binding", "success", "upstream_error")]
    assert _aggregate_lane_run_status(records) == "partial"
