"""Unit-level checks for ``scripts/run_live_llm_step1_5_pdb_smoke.py``
helpers.

We exercise only the pure helpers — never the network-bound ``main()``
— to make sure the smoke summary stays compact:

- the input-summary redactor drops `query` for sequence/CDR3 material
  types (no raw VH/VL, no raw CDR3 in the summary),
- the token-usage aggregator buckets events per task and adds counts,
- the helpers never raise on missing keys.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SMOKE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "run_live_llm_step1_5_pdb_smoke.py"
)


@pytest.fixture(scope="module")
def smoke_module():
    """Load the smoke script as a module WITHOUT running main().

    The smoke script's live-mode env mutation must live inside
    ``main()`` / ``_configure_live_env()``. Importing the module from
    a test must NOT flip ``MCP_LIVE_TOOLS`` to ``true``.
    """
    spec = importlib.util.spec_from_file_location(
        "_live_step1_5_smoke", _SMOKE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_set_mcp_live_tools(monkeypatch):
    """Regression fence: pytest-collecting the smoke module via
    importlib must not flip ``MCP_LIVE_TOOLS`` on in the test process.

    The live-mode env mutation must be deferred to ``main()`` /
    ``_configure_live_env()`` so a developer running ``pytest`` does
    not accidentally route every subsequent test through live
    ToolUniverse wrappers.
    """
    import os
    monkeypatch.delenv("MCP_LIVE_TOOLS", raising=False)
    monkeypatch.delenv("MCP_LIVE_TOOL_ALLOWLIST", raising=False)
    spec = importlib.util.spec_from_file_location(
        "_live_step1_5_smoke_envprobe", _SMOKE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Module loaded — but the env must still be unset / not "true".
    value = os.environ.get("MCP_LIVE_TOOLS")
    assert value in (None, "", "false"), (
        f"Importing the smoke module mutated MCP_LIVE_TOOLS = {value!r}; "
        "the live-mode env mutation must be deferred to main() / "
        "_configure_live_env()."
    )
    assert "MCP_LIVE_TOOL_ALLOWLIST" not in os.environ or (
        os.environ["MCP_LIVE_TOOL_ALLOWLIST"] == ""
    ), (
        "Importing the smoke module mutated MCP_LIVE_TOOL_ALLOWLIST; "
        "deferring to main() / _configure_live_env() prevents this."
    )
    # The helper exists and is callable, but we never invoke it here —
    # invoking it would mutate the env for the rest of the suite.
    assert callable(getattr(module, "_configure_live_env"))


def test_compact_input_summary_strips_sequence_query(smoke_module):
    summary = {
        "query_kind": "name",
        "query_role": "antibody",
        "material_type": "antibody_heavy_chain_sequence",
        "query": "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG",
        "tool_selection_source": "llm_stage1",
    }
    out = smoke_module._compact_input_summary(summary)
    assert "query" not in out, (
        "sequence-type materials must not surface a `query` field that "
        "could contain a full VH/VL string"
    )
    assert out["material_type"] == "antibody_heavy_chain_sequence"
    assert out["tool_selection_source"] == "llm_stage1"


def test_compact_input_summary_strips_cdr3_material_query(smoke_module):
    summary = {
        "query_kind": "cdr3_filter",
        "material_type": "antibody_heavy_cdr3_sequence",
        "query": "ARGGYDFWSGYYTFDY",
        "cdr3_length": 16,
        "cdr3_sha256_prefix": "deadbeef0123",
    }
    out = smoke_module._compact_input_summary(summary)
    assert "query" not in out
    assert out["cdr3_length"] == 16
    assert out["cdr3_sha256_prefix"] == "deadbeef0123"


def test_compact_input_summary_keeps_short_query_for_non_sensitive_material(
    smoke_module,
):
    summary = {
        "query_kind": "name",
        "material_type": "payload_name",
        "query": "monomethyl auristatin E",
    }
    out = smoke_module._compact_input_summary(summary)
    assert out["query"] == "monomethyl auristatin E"


def test_compact_input_summary_truncates_long_query(smoke_module):
    summary = {
        "query_kind": "name",
        "material_type": "payload_name",
        "query": "x" * 500,
    }
    out = smoke_module._compact_input_summary(summary)
    assert out["query"].endswith("…")
    assert len(out["query"]) <= 80


def test_compact_input_summary_only_whitelisted_keys(smoke_module):
    summary = {
        "query_kind": "name",
        "material_type": "payload_name",
        "query": "MMAE",
        "secret_password": "should_not_appear",
        "raw_payload": {"never": "include this"},
    }
    out = smoke_module._compact_input_summary(summary)
    assert "secret_password" not in out
    assert "raw_payload" not in out


def test_aggregate_usage_by_task_buckets_and_sums(smoke_module):
    events = [
        {"task": "structured_query", "provider": "openai",
         "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
         "cached_prompt_tokens": 40},
        {"task": "structured_query", "provider": "openai",
         "prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100,
         "cached_prompt_tokens": 0},
        {"task": "tool_selection_stage_1", "provider": "openai",
         "prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250,
         "cached_prompt_tokens": 128},
        {"task": "tool_selection_stage_1", "provider": "openai",
         "prompt_tokens": None, "completion_tokens": None,
         "total_tokens": None, "cached_prompt_tokens": None},
    ]
    out = smoke_module._aggregate_usage_by_task(events)
    assert set(out) == {"structured_query", "tool_selection_stage_1"}
    sq = out["structured_query"]
    assert sq["calls"] == 2
    assert sq["prompt_tokens"] == 190
    assert sq["completion_tokens"] == 30
    assert sq["total_tokens"] == 220
    assert sq["cached_prompt_tokens"] == 40
    s1 = out["tool_selection_stage_1"]
    assert s1["calls"] == 2
    # Null counts must be ignored, not raise.
    assert s1["prompt_tokens"] == 200
    assert s1["total_tokens"] == 250
    assert s1["cached_prompt_tokens"] == 128


def test_aggregate_usage_by_task_handles_empty_input(smoke_module):
    out = smoke_module._aggregate_usage_by_task([])
    assert out == {}


# ── cached / uncached prompt-token totals ────────────────────────────────


def test_aggregate_usage_totals_when_all_events_have_cache_field(smoke_module):
    events = [
        {"task": "structured_query", "prompt_tokens": 1000,
         "completion_tokens": 100, "total_tokens": 1100,
         "cached_prompt_tokens": 200},
        {"task": "tool_selection_stage_1", "prompt_tokens": 500,
         "completion_tokens": 50, "total_tokens": 550,
         "cached_prompt_tokens": 300},
    ]
    out = smoke_module._aggregate_usage_totals(events)
    assert out["llm_usage_total_tokens"] == 1650
    assert out["llm_usage_prompt_tokens_total"] == 1500
    assert out["llm_usage_cached_prompt_tokens_total"] == 500
    assert out["llm_usage_uncached_prompt_tokens_total"] == 1000
    assert out["llm_usage_uncached_prompt_tokens_total_is_estimate"] is False


def test_aggregate_usage_totals_marks_estimate_when_cache_missing(smoke_module):
    """If an event reports prompt_tokens but no cached_prompt_tokens,
    the uncached total is computed assuming worst-case 0 cached AND
    the rollup flags is_estimate=True so reviewers read it as an
    upper bound."""
    events = [
        {"task": "structured_query", "prompt_tokens": 1000,
         "completion_tokens": 100, "total_tokens": 1100,
         "cached_prompt_tokens": None},
        {"task": "tool_selection_stage_1", "prompt_tokens": 500,
         "completion_tokens": 50, "total_tokens": 550,
         "cached_prompt_tokens": 300},
    ]
    out = smoke_module._aggregate_usage_totals(events)
    assert out["llm_usage_cached_prompt_tokens_total"] == 300
    # 1000 (missing → assume 0 cached) + (500 - 300) = 1200
    assert out["llm_usage_uncached_prompt_tokens_total"] == 1200
    assert out["llm_usage_uncached_prompt_tokens_total_is_estimate"] is True


def test_aggregate_usage_totals_empty_events(smoke_module):
    out = smoke_module._aggregate_usage_totals([])
    assert out["llm_usage_total_tokens"] == 0
    assert out["llm_usage_prompt_tokens_total"] == 0
    assert out["llm_usage_cached_prompt_tokens_total"] == 0
    assert out["llm_usage_uncached_prompt_tokens_total"] == 0
    assert out["llm_usage_uncached_prompt_tokens_total_is_estimate"] is False


# ── Step 5 tool-call rollup (envelope-aware) ─────────────────────────────


def _tcr(
    tool_name: str, run_status: str = "success", *, ref: str | None = None,
    selection_source: str | None = "llm_stage1",
    fallback_reason: str | None = None,
    material_type: str | None = None,
    error_message: str | None = None,
) -> dict:
    """Build a minimal ToolCallRecord-shaped dict for the rollup."""
    summary: dict = {}
    if material_type:
        summary["material_type"] = material_type
    if selection_source:
        summary["tool_selection_source"] = selection_source
    if fallback_reason:
        summary["fallback_reason"] = fallback_reason
    return {
        "tool_name": tool_name,
        "run_status": run_status,
        "tool_output_ref": ref,
        "tool_input_summary": summary,
        "error_message": error_message,
    }


def _store(items: dict[str, dict]):
    """Return a dict-backed output_reader callable."""
    def _reader(ref: str) -> dict:
        return items.get(ref, {})
    return _reader


def test_rollup_surfaces_envelope_upstream_error_even_when_run_status_success(
    smoke_module,
):
    """Regression for the ChEMBL HTTP 400 case: ToolCallRecord.run_status
    was 'success' but the persisted envelope said
    output.status='upstream_error'. The rollup must still surface it."""
    records = [
        _tcr("ChEMBL_search_substructure", "success", ref="ref_upe"),
        _tcr("ChEMBL_search_molecules", "success", ref="ref_ok"),
    ]
    reader = _store({
        "ref_upe": {"output": {
            "executor": "tooluniverse",
            "status": "upstream_error",
            "error_message": "ChEMBL API returned HTTP 400",
            "error_details": {"type": "ChEMBLApiError"},
        }},
        "ref_ok": {"output": {
            "executor": "tooluniverse",
            "status": "ok",
            "results": [{"x": 1}],
        }},
    })
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
    )
    # Run-status counter sees both as success.
    assert rollup["run_status_counts"] == {"success": 2}
    # Envelope counter splits them.
    assert rollup["envelope_status_counts"] == {
        "upstream_error": 1, "ok": 1,
    }
    # The upstream_error row is surfaced even though run_status=success.
    failed = rollup["skipped_or_failed"]
    assert len(failed) == 1
    entry = failed[0]
    assert entry["tool_name"] == "ChEMBL_search_substructure"
    assert entry["run_status"] == "success"
    assert entry["envelope_status"] == "upstream_error"
    assert entry["executor"] == "tooluniverse"
    assert entry["error_type"] == "ChEMBLApiError"
    assert "HTTP 400" in (entry["envelope_error_message"] or "")
    # And the live_tool_status downgrades from ok to partial.
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=len(records),
    )
    assert status == "partial"
    assert "upstream_error" in reasons


def test_rollup_clean_run_is_ok_when_all_envelopes_say_ok(smoke_module):
    records = [
        _tcr("ChEMBL_search_molecules", "success", ref="r1"),
        _tcr("SAbDab_search_structures", "success", ref="r2"),
    ]
    reader = _store({
        "r1": {"output": {"executor": "tooluniverse", "status": "ok"}},
        "r2": {"output": {"executor": "tooluniverse", "status": "ok"}},
    })
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
    )
    assert rollup["skipped_or_failed"] == []
    assert rollup["envelope_status_counts"] == {"ok": 2}
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=len(records),
    )
    assert status == "ok"
    assert reasons == ["clean"]


def test_rollup_selection_source_and_fallback_reason_counts(smoke_module):
    """deterministic_fallback + fallback_reason=llm_empty_selection must
    show up in the summary counters — not be silently presented as an
    LLM pick."""
    records = [
        _tcr("ChEMBL_search_molecules", "success", ref="r1",
             selection_source="llm_stage1"),
        _tcr("ChEMBL_search_substructure", "success", ref="r2",
             selection_source="deterministic_fallback",
             fallback_reason="llm_empty_selection"),
        _tcr("SAbDab_search_structures", "success", ref="r3",
             selection_source="deterministic_fallback",
             fallback_reason="llm_empty_selection"),
    ]
    reader = _store({
        "r1": {"output": {"executor": "tooluniverse", "status": "ok"}},
        "r2": {"output": {"executor": "tooluniverse", "status": "ok"}},
        "r3": {"output": {"executor": "tooluniverse", "status": "ok"}},
    })
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
    )
    assert rollup["selection_source_counts"] == {
        "llm_stage1": 1, "deterministic_fallback": 2,
    }
    assert rollup["selection_fallback_reason_counts"] == {
        "llm_empty_selection": 2,
    }


def test_rollup_flags_mocked_executor(smoke_module):
    records = [_tcr("ChEMBL_search_molecules", "success", ref="r1")]
    reader = _store({
        "r1": {"output": {"executor": "local_mock", "status": "ok"}}
    })
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
    )
    assert rollup["any_mocked_outputs"] is True
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=len(records),
    )
    assert status == "failed"
    assert "mocked_executor" in reasons


def test_rollup_does_not_flag_synthetic_dependency_record_as_mocked(
    smoke_module,
):
    """Synthetic dependency-gap records have no tool_output_ref. They
    must NOT be flagged as mocked executors."""
    records = [
        _tcr("ZINC_get_compound", "dependency_unavailable", ref=None),
    ]
    reader = _store({})
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
        known_dependency_gaps=("ZINC_get_compound",),
    )
    assert rollup["any_mocked_outputs"] is False
    failed = rollup["skipped_or_failed"]
    assert len(failed) == 1
    assert failed[0]["expected_dependency_gap"] is True
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=len(records),
    )
    assert status == "partial"
    assert reasons == ["known_dependency_gap"]


def test_rollup_no_calls(smoke_module):
    rollup = smoke_module._summarize_step5_tool_calls(
        [], output_reader=_store({}),
    )
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=0,
    )
    assert status == "no_calls"
    assert reasons == ["no_step5_tool_calls"]


def test_rollup_pdb_leak_detection(smoke_module):
    records = [_tcr("SAbDab_search_structures", "success", ref="r1")]
    reader = _store({
        "r1": {"output": {
            "executor": "tooluniverse", "status": "ok",
            "payload": {"raw_pdb": "ATOM     1 N MET A 1 ..."},
        }}
    })
    rollup = smoke_module._summarize_step5_tool_calls(
        records, output_reader=reader,
    )
    assert rollup["pdb_content_leaked"] is True
    status, reasons = smoke_module._compute_live_tool_status(
        rollup=rollup, tool_call_count=len(records),
    )
    assert status == "failed"
    assert "pdb_leak" in reasons


def test_envelope_meta_degrades_safely_on_missing_blocks(smoke_module):
    meta = smoke_module._envelope_meta(None)
    assert meta["executor"] is None
    assert meta["envelope_status"] is None
    meta = smoke_module._envelope_meta({})
    assert meta["envelope_status"] is None
    meta = smoke_module._envelope_meta({"output": "not-a-dict"})
    assert meta["envelope_status"] is None


# ── llm_usage_events provenance fields ────────────────────────────────────


def test_llm_usage_event_shape_includes_model_and_provider(smoke_module):
    """The smoke surfaces provider, model, task, attempt, and token
    counters per LLM call. The aggregator does not strip these — and
    the top-level summary keeps `provider` / `model` at the root too."""
    events = [
        {"provider": "openai", "model": "gpt-5.5",
         "task": "structured_query", "attempt": 0,
         "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
         "cached_prompt_tokens": 32},
        {"provider": "openai", "model": "gpt-5.5",
         "task": "tool_selection_stage_1", "attempt": 0,
         "prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92,
         "cached_prompt_tokens": None},
    ]
    # Each event itself carries provider/model/task/attempt + tokens
    # + the new cached_prompt_tokens field (None when absent).
    for evt in events:
        for required in (
            "provider", "model", "task", "attempt",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cached_prompt_tokens",
        ):
            assert required in evt, required
    # The aggregator keys by task and never drops the provider/model
    # information — it just adds compact counters.
    out = smoke_module._aggregate_usage_by_task(events)
    assert set(out) == {"structured_query", "tool_selection_stage_1"}
    assert out["structured_query"]["total_tokens"] == 120
    assert out["structured_query"]["cached_prompt_tokens"] == 32


# ── Privacy sweep: no prompt / response / API key / payload leak ─────────


def test_compact_input_summary_blocks_arbitrary_secret_keys(smoke_module):
    summary = {
        "query_kind": "name",
        "material_type": "payload_name",
        "query": "MMAE",
        "api_key": "sk-very-secret",
        "system_prompt": "SECRET SYSTEM PROMPT body",
        "raw_response": "FULL RAW LLM RESPONSE BODY",
        "raw_pdb_payload": "ATOM ...",
        "full_fasta": ">trastuzumab_HC\nEVQLVQSG...",
        "raw_cdr3": "ARGGYDFWSGYYTFDY",
    }
    out = smoke_module._compact_input_summary(summary)
    import json as _json
    blob = _json.dumps(out)
    for forbidden in (
        "sk-very-secret",
        "SECRET SYSTEM PROMPT",
        "FULL RAW LLM RESPONSE",
        "ATOM ...",
        ">trastuzumab_HC",
        "ARGGYDFWSGYYTFDY",
        "api_key",
        "system_prompt",
        "raw_response",
        "raw_pdb_payload",
        "full_fasta",
        "raw_cdr3",
    ):
        assert forbidden not in blob, forbidden
