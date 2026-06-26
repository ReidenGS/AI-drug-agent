"""Live-ToolUniverse Step 1→5 smoke with a real uploaded PDB + LLM
token-usage accounting.

Scope is intentionally narrower than ``run_live_llm_step1_6_pdb_smoke.py``:

- runs Step 1 intake → Step 2 structured query (configured real LLM) →
  Step 3 readiness → Step 4 workflow setup → Step 5 candidate context
  (live ToolUniverse where allowed, real LLM Stage-1 selection),
- emits a compact JSON summary including a ``llm_usage_events`` block
  populated from the configured provider's ``usage_events`` list.

Hard rules (compact-only output):

- Never prints raw PDB / FASTA contents, raw provider response bodies,
  full prompts, full ToolUniverse payloads, full antibody sequences,
  raw CDR3, or API keys.
- Sets ``MCP_LIVE_TOOLS=true`` and a narrow ``MCP_LIVE_TOOL_ALLOWLIST``
  scoped to Step 5 wrappers only — the production MCP catalog is NOT
  filtered by this script.
- Does NOT modify ``.env``, the MCP registry, the ToolUniverse
  inventory, or the agent runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Live-mode config: applied by ``main()`` BEFORE any settings /
#    deps / agent import. Module import does NOT mutate the
#    environment, so pytest collecting this module via importlib does
#    not flip the production process into live mode.
#
#    The allowlist names Step 5 wrappers only. It is a runtime
#    ``_live=True`` injection policy and never filters the production
#    MCP catalog or the Stage-1 compact_catalog the LLM sees.
LIVE_ALLOWLIST = (
    "SAbDab_search_structures",
    "SAbDab_get_structure",
    "TheraSAbDab_search_by_target",
    "TheraSAbDab_search_therapeutics",
    "ChEMBL_get_molecule",
    "ChEMBL_search_molecules",
    "ChEMBL_search_substructure",
    "ChEMBL_search_similarity",
    "iedb_search_bcr_sequences",
)


def _configure_live_env() -> None:
    """Set the live-mode environment variables.

    Called by ``main()`` only; importing this module does not call it.
    Idempotent: ``STORAGE_MODE`` is only filled when unset.
    """
    os.environ["MCP_LIVE_TOOLS"] = "true"
    os.environ["MCP_LIVE_TOOL_ALLOWLIST"] = ",".join(LIVE_ALLOWLIST)
    os.environ.setdefault("STORAGE_MODE", "local")


DEFAULT_PDB = Path(
    "/Users/jackiewen/Desktop/desk/实习工作/国外ai医药/程序/data/pdb/S1.pdb"
)


_UNAVAILABLE_TOKENS = (
    "429", "503", "quota", "rate limit", "unavailable",
    "timeout", "timed out", "server disconnect", "connection reset",
    "remote disconnected", "remoteprotocolerror", "service unavailable",
    "insufficient_quota",
)


# Documented dependency gap: registry-side known-unavailable Step 5
# tools whose appearance as ``dependency_unavailable`` is expected,
# not a real upstream failure. Reused for the audit rollup only.
KNOWN_LIVE_DEPENDENCY_GAPS = (
    "ZINC_get_compound",
    "ZINC_search_by_smiles",
    "ZINC_search_compounds",
    "ZINC_get_purchasable",
    "ZINC_search_by_properties",
)


def _is_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in _UNAVAILABLE_TOKENS):
        return True
    cls = type(exc).__name__.lower()
    return any(tok in cls for tok in (
        "ratelimit", "timeout", "apiconnection", "apierror"
    ))


def _summary(d: dict) -> None:
    print(json.dumps(d, indent=2, sort_keys=True, default=str))


def _short(value, limit: int = 160):
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    return value


def _compact_input_summary(summary: dict | None) -> dict:
    """Return a redacted view of the tool_input_summary safe for the
    smoke output. Drops full antibody sequences, raw CDR3, and any
    unbounded string values — keeps only short metadata keys."""
    if not isinstance(summary, dict):
        return {}
    safe_keys = (
        "query_kind", "query_role", "material_type", "capability_type",
        "output_extractor_type", "fallback_group", "provenance_policy",
        "confidence_policy", "tool_selection_source", "selection_reason",
        "argument_construction_source", "execution_semantics",
        "selection_policy_version", "fallback_reason",
        "eligible_count", "real_selected_count", "skipped_eligible_count",
        "known_unavailable_count",
        # CDR3 / IEDB redacted fields (no raw sequence).
        "iedb_filter_key", "cdr3_chain_type", "cdr3_length",
        "cdr3_sha256_prefix", "cdr3_numbering_scheme", "cdr3_backend",
        "cdr3_source_material_id", "select_columns_count",
    )
    out: dict = {}
    for k in safe_keys:
        if k in summary:
            v = summary[k]
            out[k] = _short(v) if isinstance(v, str) else v
    # `query` (if present and short / not a SMILES) is included
    # truncated; deliberately omitted for material_type values that
    # indicate sequence/CDR3 inputs.
    sensitive = summary.get("material_type") in (
        "antibody_heavy_chain_sequence", "antibody_light_chain_sequence",
        "antibody_heavy_cdr3_sequence", "antibody_light_cdr3_sequence",
        "antibody_sequence_reference",
    )
    if not sensitive and isinstance(summary.get("query"), str):
        out["query"] = _short(summary["query"], 80)
    return out


_USAGE_COUNTER_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cached_prompt_tokens",
)


def _aggregate_usage_by_task(events: list[dict]) -> dict:
    by_task: dict[str, dict[str, int]] = {}
    for evt in events:
        task = str(evt.get("task") or "")
        bucket = by_task.setdefault(task, {
            "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0,
            "cached_prompt_tokens": 0,
        })
        bucket["calls"] += 1
        for key in _USAGE_COUNTER_KEYS:
            value = evt.get(key)
            if isinstance(value, int):
                bucket[key] += value
    return by_task


def _aggregate_usage_totals(events: list[dict]) -> dict:
    """Top-level usage rollup with cached / uncached prompt token split.

    Conservative semantics:

    - ``llm_usage_total_tokens`` sums ``total_tokens`` over events that
      reported it.
    - ``llm_usage_prompt_tokens_total`` and
      ``llm_usage_cached_prompt_tokens_total`` sum the matching fields.
    - ``llm_usage_uncached_prompt_tokens_total`` is computed as
      ``prompt_tokens - cached_prompt_tokens`` per event when BOTH
      fields are integers; events that report a ``prompt_tokens`` but
      no ``cached_prompt_tokens`` contribute their full
      ``prompt_tokens`` to the uncached total AND mark the rollup as
      an estimate.
    - ``llm_usage_uncached_prompt_tokens_total_is_estimate`` is
      ``True`` whenever at least one event lacked
      ``cached_prompt_tokens``. Reviewers can then read the field as
      an upper bound rather than a true uncached count.
    """
    total_tokens = 0
    prompt_total = 0
    cached_total = 0
    uncached_total = 0
    is_estimate = False
    saw_event = False
    for evt in events:
        if not isinstance(evt, dict):
            continue
        saw_event = True
        if isinstance(evt.get("total_tokens"), int):
            total_tokens += evt["total_tokens"]
        prompt = evt.get("prompt_tokens")
        cached = evt.get("cached_prompt_tokens")
        if isinstance(prompt, int):
            prompt_total += prompt
        if isinstance(cached, int):
            cached_total += cached
        if isinstance(prompt, int) and isinstance(cached, int):
            uncached_total += max(0, prompt - cached)
        elif isinstance(prompt, int):
            # Missing cached info — assume worst case 0 cached.
            uncached_total += prompt
            is_estimate = True
    return {
        "llm_usage_total_tokens": total_tokens,
        "llm_usage_prompt_tokens_total": prompt_total,
        "llm_usage_cached_prompt_tokens_total": cached_total,
        "llm_usage_uncached_prompt_tokens_total": uncached_total,
        "llm_usage_uncached_prompt_tokens_total_is_estimate": (
            is_estimate if saw_event else False
        ),
    }


# ── Step 5 tool-call rollup helpers (pure, testable) ───────────────────────


_PDB_LINE_SIGNATURES = ("ATOM ", "HETATM", "HEADER")


def _envelope_meta(raw_record: dict | None) -> dict:
    """Compact metadata about a persisted ``tool_output_ref`` envelope.

    Reads ``output.executor`` / ``output.status`` /
    ``output.error_message`` / ``output.error_details.type``. Missing
    or non-dict ``output`` blocks degrade to ``None`` values — never
    raises. Never includes the raw payload in the return.
    """
    out = {
        "executor": None,
        "source": None,
        "envelope_status": None,
        "error_type": None,
        "error_message": None,
    }
    if not isinstance(raw_record, dict):
        return out
    output = raw_record.get("output")
    if not isinstance(output, dict):
        return out
    out["executor"] = output.get("executor")
    out["source"] = output.get("source")
    out["envelope_status"] = output.get("status")
    err_msg = output.get("error_message")
    if isinstance(err_msg, str):
        out["error_message"] = _short(err_msg, 200)
    error_details = output.get("error_details")
    if isinstance(error_details, dict):
        out["error_type"] = error_details.get("type")
    return out


def _detect_pdb_content_leak(raw_record: dict | None) -> bool:
    """True iff the persisted output contains raw PDB-style lines."""
    if not isinstance(raw_record, dict):
        return False
    output = raw_record.get("output")
    if not isinstance(output, (dict, list)):
        return False
    blob = json.dumps(output)
    return any(sig in blob for sig in _PDB_LINE_SIGNATURES)


def _summarize_step5_tool_calls(
    tool_call_records: list[dict],
    *,
    output_reader,
    known_dependency_gaps: tuple[str, ...] = KNOWN_LIVE_DEPENDENCY_GAPS,
) -> dict:
    """Pure rollup over Step 5 ``tool_call_records``.

    ``output_reader(ref)`` returns the persisted JSON dict for a
    ``tool_output_ref`` (or ``{}``). Splitting the I/O lets the tests
    pass a dict-backed reader so no storage is required.

    Returns a dict with every field the smoke summary needs:

    - ``tools_called`` — sorted set of tool names that were called
    - ``run_status_counts`` — counter over ``ToolCallRecord.run_status``
    - ``envelope_status_counts`` — counter over persisted
      ``output.status`` (e.g. ``ok`` / ``upstream_error`` / ``empty``)
    - ``selection_source_counts`` — counter over
      ``tool_input_summary.tool_selection_source``
    - ``selection_fallback_reason_counts`` — counter over
      ``tool_input_summary.fallback_reason`` (only non-empty entries
      are counted; the canonical "no fallback" case is omitted)
    - ``skipped_or_failed`` — list of compact entries for every call
      whose ``run_status != "success"`` OR whose persisted
      ``envelope_status`` is set AND not ``ok``. A ``ToolCallRecord``
      with ``run_status="success"`` but
      ``envelope_status="upstream_error"`` is therefore surfaced.
    - ``any_mocked_outputs`` — True if any persisted envelope's
      ``executor`` is neither ``None`` nor ``"tooluniverse"``.
      Synthetic dependency-gap records (no ``tool_output_ref``) are
      NOT counted as mocked.
    - ``pdb_content_leaked`` — True if any persisted output contains
      raw PDB-style header / ATOM / HETATM lines.
    """
    tools_called: list[str] = []
    run_status_counts: dict[str, int] = {}
    envelope_status_counts: dict[str, int] = {}
    selection_source_counts: dict[str, int] = {}
    selection_fallback_reason_counts: dict[str, int] = {}
    skipped_or_failed: list[dict] = []
    any_mocked_outputs = False
    pdb_content_leaked = False

    for tc in tool_call_records:
        if not isinstance(tc, dict):
            continue
        tn = tc.get("tool_name", "")
        rs = tc.get("run_status", "")
        tools_called.append(tn)
        run_status_counts[rs] = run_status_counts.get(rs, 0) + 1

        input_summary = tc.get("tool_input_summary") or {}
        sel_source = input_summary.get("tool_selection_source")
        if isinstance(sel_source, str) and sel_source:
            selection_source_counts[sel_source] = (
                selection_source_counts.get(sel_source, 0) + 1
            )
        fb_reason = input_summary.get("fallback_reason")
        if isinstance(fb_reason, str) and fb_reason:
            selection_fallback_reason_counts[fb_reason] = (
                selection_fallback_reason_counts.get(fb_reason, 0) + 1
            )

        ref = tc.get("tool_output_ref")
        meta = {"executor": None, "source": None, "envelope_status": None,
                "error_type": None, "error_message": None}
        if ref:
            try:
                raw_record = output_reader(ref) or {}
            except Exception:  # noqa: BLE001
                raw_record = {}
            meta = _envelope_meta(raw_record)
            if meta["executor"] not in (None, "tooluniverse"):
                any_mocked_outputs = True
            if _detect_pdb_content_leak(raw_record):
                pdb_content_leaked = True

        env_status = meta["envelope_status"]
        if isinstance(env_status, str) and env_status:
            envelope_status_counts[env_status] = (
                envelope_status_counts.get(env_status, 0) + 1
            )

        # A call is surfaced if EITHER the runtime marked it non-success
        # OR the persisted envelope says it failed upstream. The latter
        # catches the documented case where ``run_status="success"``
        # but ``output.status="upstream_error"`` (e.g. ChEMBL HTTP 400).
        non_success_run = rs != "success"
        envelope_problem = (
            isinstance(env_status, str)
            and env_status not in ("", "ok")
        )
        if non_success_run or envelope_problem:
            entry = {
                "tool_name": tn,
                "run_status": rs,
                "envelope_status": env_status,
                "executor": meta["executor"],
                "error_type": meta["error_type"],
                "envelope_error_message": meta["error_message"],
                "tool_call_error_message": _short(
                    tc.get("error_message"), 200
                ),
                "input_summary": _compact_input_summary(input_summary),
                "expected_dependency_gap": tn in known_dependency_gaps,
            }
            skipped_or_failed.append(entry)

    return {
        "tools_called": sorted(set(tools_called)),
        "run_status_counts": run_status_counts,
        "envelope_status_counts": envelope_status_counts,
        "selection_source_counts": selection_source_counts,
        "selection_fallback_reason_counts": selection_fallback_reason_counts,
        "skipped_or_failed": skipped_or_failed,
        "any_mocked_outputs": any_mocked_outputs,
        "pdb_content_leaked": pdb_content_leaked,
    }


def _compute_live_tool_status(
    *,
    rollup: dict,
    tool_call_count: int,
) -> tuple[str, list[str]]:
    """Decide the rollup-level ``live_tool_status`` + a reason list.

    Priority:

    1. ``failed`` if a non-tooluniverse executor was seen OR raw PDB
       lines leaked into a persisted output. Reasons include
       ``mocked_executor`` / ``pdb_leak``.
    2. ``partial`` if any ``skipped_or_failed`` row is NOT a known
       dependency gap. Reasons include ``upstream_error`` /
       ``failed_run_status`` depending on what was seen.
    3. ``partial`` (reason ``known_dependency_gap``) if every
       non-success row IS a known dependency gap.
    4. ``ok`` if at least one tool call ran and no problems surfaced.
       Reason is ``clean``.
    5. ``no_calls`` if no tool calls fired. Reason is
       ``no_step5_tool_calls``.
    """
    reasons: list[str] = []
    skipped_or_failed = rollup.get("skipped_or_failed") or []
    if rollup.get("any_mocked_outputs"):
        reasons.append("mocked_executor")
    if rollup.get("pdb_content_leaked"):
        reasons.append("pdb_leak")
    if reasons:
        return "failed", reasons

    unexpected = []
    has_known_gap = False
    saw_upstream_error = False
    saw_failed_run = False
    for s in skipped_or_failed:
        if s.get("expected_dependency_gap"):
            has_known_gap = True
            continue
        unexpected.append(s)
        if s.get("envelope_status") and s["envelope_status"] != "ok":
            saw_upstream_error = True
        if s.get("run_status") and s["run_status"] not in ("success",):
            saw_failed_run = True

    if unexpected:
        if saw_upstream_error:
            reasons.append("upstream_error")
        if saw_failed_run:
            reasons.append("failed_run_status")
        return "partial", reasons or ["unexpected"]

    if has_known_gap:
        return "partial", ["known_dependency_gap"]

    if tool_call_count > 0:
        return "ok", ["clean"]

    return "no_calls", ["no_step5_tool_calls"]


def main() -> int:
    # Apply live env BEFORE any app.settings / app.deps / agent import.
    _configure_live_env()
    print('{"smoke_phase":"main_start"}', flush=True)
    from app.settings import get_settings  # noqa: PLC0415
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    settings = get_settings()
    provider_name = settings.llm_provider
    print('{"smoke_phase":"settings_loaded"}', flush=True)

    if provider_name == "gemini":
        if not settings.gemini_api_key:
            _summary({"status": "SKIP", "reason": "GEMINI_API_KEY unset"})
            return 0
        model_name = settings.gemini_model
    elif provider_name == "openai":
        if not settings.openai_api_key:
            _summary({"status": "SKIP", "reason": "OPENAI_API_KEY unset"})
            return 0
        model_name = settings.openai_model
    else:
        _summary({"status": "SKIP",
                  "reason": f"unsupported LLM_PROVIDER={provider_name!r}"})
        return 0

    pdb_path = Path(os.environ.get("ADC_SMOKE_PDB", str(DEFAULT_PDB)))
    if not pdb_path.exists():
        _summary({"status": "SKIP", "reason": "PDB file not found",
                  "pdb_path": str(pdb_path)})
        return 0
    pdb_bytes = pdb_path.read_bytes()
    pdb_sha256 = hashlib.sha256(pdb_bytes).hexdigest()

    from app.agents.candidate_context_agent import CandidateContextAgent  # noqa: PLC0415
    from app.deps import (  # noqa: PLC0415
        get_llm_provider, get_mcp_client, get_registry_service,
        get_storage, get_workflow_state_service,
    )
    from app.graph.adc_graph import build_minimal_graph  # noqa: PLC0415
    from app.utils.ids import new_artifact_id, new_run_id  # noqa: PLC0415
    print('{"smoke_phase":"pipeline_imports_loaded"}', flush=True)

    for fn in (get_storage, get_registry_service, get_workflow_state_service,
               get_mcp_client, get_llm_provider):
        try:
            fn.cache_clear()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    storage = get_storage()
    registry = get_registry_service()
    workflow_state = get_workflow_state_service()
    try:
        llm = get_llm_provider()
    except Exception as exc:  # noqa: BLE001
        _summary({"status": "SKIP",
                  "reason": f"get_llm_provider failed: {type(exc).__name__}"})
        return 0

    # If the provider does not surface a usage_events list (Mock /
    # legacy), we still produce the compact summary — just empty.
    usage_events_ref = getattr(llm, "usage_events", None)
    if not isinstance(usage_events_ref, list):
        usage_events_ref = []

    run_id = new_run_id()
    file_id = new_artifact_id("uploaded_file")
    storage_path = storage.run_key(run_id, "inputs", "files", pdb_path.name)
    if hasattr(storage, "write_bytes"):
        storage.write_bytes(storage_path, pdb_bytes)
    elif hasattr(storage, "write"):
        storage.write(storage_path, pdb_bytes)

    findings: dict = {
        "pipeline_status": "PASS",
        "live_tool_status": "unknown",
        "provider": provider_name,
        "model": model_name,
        "run_id": run_id,
        "pdb_filename": pdb_path.name,
        "pdb_sha256_prefix": pdb_sha256[:12],
        "mcp_live_config": {
            "mcp_live_tools": settings.mcp_live_tools,
            "mcp_live_tool_allowlist_count": len(
                settings.live_tool_allowlist_set()
            ),
            "mcp_live_tool_allowlist": sorted(
                settings.live_tool_allowlist_set()
            ),
        },
    }

    intake_request = {
        "run_id": run_id,
        "raw_user_query": (
            "HER2 ADC Step 5 enrichment smoke. target HER2 / ERBB2 / "
            "UniProt P04626. payload-linker vc-MMAE. "
            "payload SMILES CCO. linker SMILES NCC(=O)O. "
            "Use uploaded PDB for antigen structure context."
        ),
        "user_provided_context": {
            "target_or_antigen_text": "HER2 (UniProt P04626)",
            "candidate_text": "trastuzumab analog",
            "payload_linker_text": (
                "vc-MMAE (payload SMILES CCO; linker SMILES NCC(=O)O)"
            ),
        },
        "uploaded_files": [
            {
                "file_id": file_id,
                "original_filename": pdb_path.name,
                "storage_path": storage_path,
                "content_type": "chemical/x-pdb",
                "sha256": pdb_sha256,
                "size_bytes": len(pdb_bytes),
            }
        ],
    }

    # ── Steps 1-4 via the minimal LangGraph ──────────────────────────────
    graph = build_minimal_graph(
        storage=storage, registry=registry,
        workflow_state=workflow_state, llm=llm,
    )
    try:
        final = graph.invoke({"intake_request": intake_request})
    except Exception as exc:  # noqa: BLE001
        if _is_unavailable_error(exc):
            partial = list(usage_events_ref)
            _summary({
                **findings,
                "pipeline_status": "LLM_UNAVAILABLE",
                "live_tool_status": "failed",
                "reason": f"{provider_name} quota/timeout/disconnect",
                "stage": "graph.invoke (Step 1-4)",
                "llm_usage_events": partial,
                "llm_usage_totals_by_task": _aggregate_usage_by_task(partial),
                **_aggregate_usage_totals(partial),
            })
            return 0
        _summary({
            **findings,
            "pipeline_status": "FAIL",
            "live_tool_status": "failed",
            "stage": "graph.invoke (Step 1-4)",
            "error_type": type(exc).__name__,
            "llm_usage_events": list(usage_events_ref),
        })
        raise
    run_id = final.get("run_id", run_id)
    findings["run_id"] = run_id
    findings["step_02_artifact"] = (
        (final.get("artifacts") or {}).get("structured_query")
    )

    # ── Step 5 ───────────────────────────────────────────────────────────
    try:
        CandidateContextAgent(
            storage=storage, registry=registry,
            workflow_state=workflow_state,
            mcp_client=get_mcp_client(), llm=llm,
        ).run(run_id)
    except Exception as exc:  # noqa: BLE001
        if _is_unavailable_error(exc):
            partial = list(usage_events_ref)
            _summary({
                **findings,
                "pipeline_status": "LLM_UNAVAILABLE",
                "live_tool_status": "failed",
                "stage": "step_05",
                "reason": f"{provider_name} quota/timeout/disconnect",
                "llm_usage_events": partial,
                "llm_usage_totals_by_task": _aggregate_usage_by_task(partial),
                **_aggregate_usage_totals(partial),
            })
            return 0
        _summary({
            **findings,
            "pipeline_status": "FAIL",
            "live_tool_status": "failed",
            "stage": "step_05",
            "error_type": type(exc).__name__,
            "llm_usage_events": list(usage_events_ref),
        })
        raise

    findings["step_05_artifact"] = (
        registry.get(run_id).active_artifacts.candidate_context_table_id
    )

    cct_key = storage.run_key(run_id, "candidate_context_table.json")
    cct = storage.read_json(cct_key)
    candidate_records = cct.get("candidate_records") or []
    tool_call_records = cct.get("tool_call_records") or []
    findings["step_05_candidate_count"] = len(candidate_records)
    findings["step_05_tool_call_count"] = len(tool_call_records)

    def _read_output_ref(ref: str) -> dict:
        try:
            return storage.read_json(ref) or {}
        except Exception:  # noqa: BLE001
            return {}

    rollup = _summarize_step5_tool_calls(
        tool_call_records, output_reader=_read_output_ref,
    )
    findings["step_05_tools_called"] = rollup["tools_called"]
    # Two distinct status views: the ToolCallRecord.run_status counter
    # and the persisted envelope status counter. ChEMBL HTTP 400-style
    # upstream errors land in the envelope counter even when the
    # runtime marked the call success.
    findings["step_05_tool_call_run_status_counts"] = rollup["run_status_counts"]
    findings["step_05_tool_output_envelope_status_counts"] = (
        rollup["envelope_status_counts"]
    )
    findings["step_05_skipped_or_failed_tool_summary"] = rollup["skipped_or_failed"]
    findings["step_05_selection_source_counts"] = rollup["selection_source_counts"]
    findings["step_05_selection_fallback_reason_counts"] = (
        rollup["selection_fallback_reason_counts"]
    )
    findings["any_mocked_tool_outputs"] = rollup["any_mocked_outputs"]
    findings["pdb_content_leaked"] = rollup["pdb_content_leaked"]

    # ── llm_usage rollup ─────────────────────────────────────────────────
    usage_events = list(usage_events_ref)
    findings["llm_usage_events"] = usage_events
    findings["llm_usage_totals_by_task"] = _aggregate_usage_by_task(usage_events)
    findings.update(_aggregate_usage_totals(usage_events))

    # ── live_tool_status rollup ──────────────────────────────────────────
    status, reasons = _compute_live_tool_status(
        rollup=rollup, tool_call_count=len(tool_call_records),
    )
    findings["live_tool_status"] = status
    findings["live_tool_status_reason"] = reasons

    _summary(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
