"""Live-ToolUniverse Step 1→6 smoke with a real uploaded PDB.

Same shape as ``run_llm_step1_6_pdb_smoke.py`` but ALSO flips
``MCP_LIVE_TOOLS=true`` and supplies a narrow ``MCP_LIVE_TOOL_ALLOWLIST``
so the Step 5 / Step 6 wrappers actually route through
``ToolUniverseAdapter`` live instead of returning the deterministic mock
envelope.

The allowlist is restricted to Step 5 / Step 6 scoped tools. We do NOT
register new MCP tools, do NOT widen the inventory, do NOT touch
``.env``.

Compact-only output: never prints raw PDB bytes, raw provider response
bodies, the system prompt, full ToolUniverse payloads, or the API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Live-mode config: set BEFORE any settings import so pydantic-settings
#    sees the live opt-in flag and the narrow allowlist. We never write to
#    ``.env`` — env vars only.
LIVE_ALLOWLIST = (
    "SAbDab_search_structures",
    "ChEMBL_search_molecules",
    "ChEMBL_search_substructure",
    "ChEMBL_search_activities",
    "EBIProteins_get_features",
    "EBIProteins_get_epitopes",
    "ProteinsPlus_profile_structure_quality",
    "DrugProps_pains_filter",
    "DrugProps_lipinski_filter",
    "SwissADME_calculate_adme",
)
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


# Tools whose live wrapper is intentionally not wired yet. Surfacing them
# under a named field keeps `live_tool_status` honest without letting
# their `dependency_unavailable` poison the rollup.
KNOWN_LIVE_DEPENDENCY_GAPS = (
    "ProteinsPlus_profile_structure_quality",
)


class _AllowlistMCPClient:
    """Smoke-local MCP wrapper that filters ``list_tools`` to the live
    allowlist before the Step 6 LLM ever sees the catalog.

    Only ``list_tools`` is narrowed; ``call_tool`` still delegates to the
    real inventory-backed client. This guarantees:

    1. Stage 1 catalog presented to OpenAI contains ONLY the live
       allowlist, so the LLM cannot pick a Step 6 tool that would later
       fail `attempted_live=true` (per request item 1).
    2. Production agents that build their own client via
       ``deps.get_mcp_client()`` are untouched (per request item 2).

    Scope, registry, inventory, and bindings are all reused — we add no
    MCP tools and do not widen any agent/step's scope.
    """

    def __init__(self, inner, allowlist):
        self._inner = inner
        self._allowlist = set(allowlist)

    def list_tools(self, *, agent_name, step_id):
        return [
            t for t in self._inner.list_tools(agent_name=agent_name, step_id=step_id)
            if t in self._allowlist
        ]

    def call_tool(self, *, agent_name, step_id, tool_name, **kwargs):
        return self._inner.call_tool(
            agent_name=agent_name, step_id=step_id, tool_name=tool_name, **kwargs
        )


def _is_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in _UNAVAILABLE_TOKENS):
        return True
    cls = type(exc).__name__.lower()
    return any(tok in cls for tok in ("ratelimit", "timeout", "apiconnection", "apierror"))


def _summary(d: dict) -> None:
    print(json.dumps(d, indent=2, sort_keys=True, default=str))


def _short(value, limit: int = 120):
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    return value


def _hit_count(payload: dict) -> int:
    """Best-effort count of result rows from common TU/live envelopes."""
    if not isinstance(payload, dict):
        return 0
    raw = payload.get("payload")
    if isinstance(raw, dict):
        for key in ("results", "items", "records", "activities", "features",
                    "epitopes", "molecules", "patents", "hits", "data"):
            v = raw.get(key)
            if isinstance(v, list):
                return len(v)
    for key in ("molecules", "results", "hits", "features", "epitopes",
                "alerts", "violations", "records"):
        v = payload.get(key)
        if isinstance(v, list):
            return len(v)
    return 0


def _extract_chembl_ids_and_smiles(payload: dict) -> tuple[int, int, set[str]]:
    """Walk a payload counting chembl_id / smiles fields.

    Returns ``(occurrence_count, smiles_count, unique_chembl_id_set)``.
    Unique IDs are the *strings* themselves so the caller can compute a
    cross-call deduped count — but we never include the IDs in compact
    smoke output, only their count.
    """
    occurrences = 0
    smiles = 0
    unique_ids: set[str] = set()

    def walk(obj):
        nonlocal occurrences, smiles
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in ("chembl_id", "molecule_chembl_id") and isinstance(v, str) and v.strip():
                    occurrences += 1
                    unique_ids.add(v.strip())
                if key in ("smiles", "canonical_smiles") and isinstance(v, str) and v.strip():
                    smiles += 1
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return occurrences, smiles, unique_ids


def _read_tool_output_payload(storage, ref: str | None) -> dict:
    """Return raw {input, output, tool_name, ...} record from a tool_output_ref."""
    if not ref:
        return {}
    try:
        return storage.read_json(ref) or {}
    except Exception:  # noqa: BLE001
        return {}


def _envelope_meta(raw_record: dict) -> dict:
    output = (raw_record or {}).get("output")
    if not isinstance(output, dict):
        return {"executor": None, "source": None, "envelope_status": None, "error_type": None,
                "error_message": None}
    err = output.get("error_message")
    return {
        "executor": output.get("executor"),
        "source": output.get("source"),
        "envelope_status": output.get("status"),
        "error_type": output.get("error_details", {}).get("type") if isinstance(output.get("error_details"), dict) else None,
        "error_message": _short(err) if isinstance(err, str) else None,
    }


def main() -> int:
    from app.settings import get_settings  # noqa: PLC0415
    # Settings caches via lru_cache; we just set env vars so first call
    # already reads them. Defensive cache clear in case anything imported
    # us via another path.
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    settings = get_settings()
    provider_name = settings.llm_provider
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
        _summary({"status": "SKIP", "reason": "PDB file not found", "pdb_path": str(pdb_path)})
        return 0
    pdb_bytes = pdb_path.read_bytes()
    pdb_sha256 = hashlib.sha256(pdb_bytes).hexdigest()

    from app.agents.candidate_context_agent import CandidateContextAgent  # noqa: PLC0415
    from app.agents.developability_agent import DevelopabilityAgent  # noqa: PLC0415
    from app.deps import (  # noqa: PLC0415
        get_llm_provider, get_mcp_client, get_registry_service,
        get_storage, get_workflow_state_service,
    )
    from app.graph.adc_graph import build_minimal_graph  # noqa: PLC0415
    from app.utils.ids import new_artifact_id, new_run_id  # noqa: PLC0415

    # Clear dep caches too so our live env vars are honored by every getter.
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
        _summary({"status": "SKIP", "reason": f"get_llm_provider failed: {type(exc).__name__}"})
        return 0

    candidate_limit = int(os.environ.get("ADC_SMOKE_CANDIDATE_LIMIT", "5") or "5")
    run_id = new_run_id()
    file_id = new_artifact_id("uploaded_file")
    storage_path = storage.run_key(run_id, "inputs", "files", pdb_path.name)
    if hasattr(storage, "write_bytes"):
        storage.write_bytes(storage_path, pdb_bytes)
    elif hasattr(storage, "write"):
        storage.write(storage_path, pdb_bytes)

    findings: dict = {
        # `pipeline_status` is whether Step 1-6 ran end-to-end. It does NOT
        # mean every live tool succeeded — see `live_tool_status` below.
        "pipeline_status": "PASS",
        "live_tool_status": "unknown",  # filled in at the end
        "provider": provider_name,
        "model": model_name,
        "run_id": run_id,
        "pdb_filename": pdb_path.name,
        "pdb_sha256_prefix": pdb_sha256[:12],
        "candidate_limit": candidate_limit,
        "mcp_live_config": {
            "mcp_live_tools": settings.mcp_live_tools,
            "mcp_live_tool_allowlist_count": len(settings.live_tool_allowlist_set()),
            "mcp_live_tool_allowlist": sorted(settings.live_tool_allowlist_set()),
        },
    }

    intake_request = {
        "run_id": run_id,
        "raw_user_query": (
            "HER2 ADC developability pre-filter. target HER2 / ERBB2 / "
            "UniProt P04626. payload-linker vc-MMAE. payload SMILES CCO. "
            "linker SMILES NCC(=O)O."
        ),
        "user_provided_context": {
            "target_or_antigen_text": "HER2 (UniProt P04626)",
            "candidate_text": "trastuzumab analog",
            "payload_linker_text": "vc-MMAE (payload SMILES CCO; linker SMILES NCC(=O)O)",
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

    graph = build_minimal_graph(
        storage=storage, registry=registry,
        workflow_state=workflow_state, llm=llm,
    )
    try:
        final = graph.invoke({"intake_request": intake_request})
    except Exception as exc:  # noqa: BLE001
        if _is_unavailable_error(exc):
            _summary({**findings, "pipeline_status": "LLM_UNAVAILABLE",
                      "live_tool_status": "failed",
                      "reason": f"{provider_name} quota/timeout/disconnect",
                      "stage": "graph.invoke (Step 1-4)"})
            return 0
        _summary({**findings, "pipeline_status": "FAIL",
                  "live_tool_status": "failed",
                  "stage": "graph.invoke (Step 1-4)",
                  "error_type": type(exc).__name__})
        raise
    run_id = final.get("run_id", run_id)
    findings["run_id"] = run_id
    findings["step_02_artifact"] = (final.get("artifacts") or {}).get("structured_query")

    # ── Step 5 ────────────────────────────────────────────────────────────
    try:
        CandidateContextAgent(
            storage=storage, registry=registry,
            workflow_state=workflow_state, mcp_client=get_mcp_client(),
        ).run(run_id)
    except Exception as exc:  # noqa: BLE001
        _summary({**findings, "pipeline_status": "FAIL",
                  "live_tool_status": "failed",
                  "stage": "step_05",
                  "error_type": type(exc).__name__})
        raise
    findings["step_05_artifact"] = registry.get(run_id).active_artifacts.candidate_context_table_id

    cct_key = storage.run_key(run_id, "candidate_context_table.json")
    cct = storage.read_json(cct_key)

    # Step 5 ChEMBL audit
    step5_audit: list[dict] = []
    chembl_occurrence_total = 0
    chembl_unique_ids: set[str] = set()
    for tc in cct.get("tool_call_records") or []:
        tn = tc.get("tool_name", "")
        if not tn.startswith("ChEMBL_"):
            continue
        raw = _read_tool_output_payload(storage, tc.get("tool_output_ref"))
        meta = _envelope_meta(raw)
        ids, smiles, this_unique = _extract_chembl_ids_and_smiles(raw.get("output") or {})
        chembl_occurrence_total += ids
        chembl_unique_ids.update(this_unique)
        attempted_live = tn in settings.live_tool_allowlist_set()
        live_success = (
            attempted_live
            and meta["executor"] == "tooluniverse"
            and meta["envelope_status"] == "ok"
            and tc.get("run_status") == "success"
        )
        step5_audit.append({
            "tool_name": tn,
            "query": _short((tc.get("tool_input_summary") or {}).get("query")
                            or (tc.get("tool_input_summary") or {}).get("arg_value")),
            "query_kind": (tc.get("tool_input_summary") or {}).get("query_kind"),
            "query_role": (tc.get("tool_input_summary") or {}).get("query_role"),
            "material_type": (tc.get("tool_input_summary") or {}).get("material_type"),
            "run_status": tc.get("run_status"),
            "attempted_live": attempted_live,
            "executor": meta["executor"],
            "envelope_status": meta["envelope_status"],
            "live_success": live_success,
            "source": meta["source"],
            "error_message": meta["error_message"],
            "hit_count": _hit_count(raw.get("output") or {}),
            "extracted_chembl_id_count": ids,
            "extracted_smiles_count": smiles,
        })
    findings["step_05_chembl_audit"] = step5_audit
    # `occurrence_count` = every chembl_id string we saw in the raw envelope
    # (the same ID may be repeated under multiple nested keys per hit row).
    # `unique_chembl_id_count` = dedup by string. NEITHER counts surviving
    # into the normalized Step 5 record — that's `normalized_chembl_id_count`.
    findings["step_05_raw_chembl_id_occurrence_count"] = chembl_occurrence_total
    findings["step_05_raw_unique_chembl_id_count"] = len(chembl_unique_ids)
    # Back-compat alias used by earlier runs/docs.
    findings["step_05_raw_chembl_id_seen_total"] = chembl_occurrence_total

    # Count chembl_id identifiers that actually landed inside the
    # normalized candidate_records[].identifiers (i.e. what Step 6 will see).
    normalized_chembl_count = 0
    for cand in cct.get("candidate_records") or []:
        for ident in cand.get("identifiers") or []:
            if (ident.get("id_type") or "").lower() == "chembl_id" and ident.get("id_value"):
                normalized_chembl_count += 1
    findings["step_05_normalized_chembl_id_count"] = normalized_chembl_count
    findings["step_05_chembl_normalization_discrepancy"] = (
        len(chembl_unique_ids) > 0 and normalized_chembl_count == 0
    )

    structure_mats = [
        m for cand in cct.get("candidate_records") or []
        for m in (cand.get("materials") or [])
        if (m.get("material_type") or "").endswith("structure_reference")
        or m.get("material_type") in {"structure_file", "structure_ref"}
    ]
    findings["step_05_structure_material_count"] = len(structure_mats)
    original_candidate_count = len(cct.get("candidate_records") or [])
    findings["step_05_candidate_count"] = original_candidate_count

    # Trim cct to candidate_limit using the structure-then-compound preference.
    if candidate_limit > 0 and original_candidate_count > candidate_limit:
        candidates = cct["candidate_records"]
        STRUCTURE_TYPES = {"target_antigen", "adc_construct"}
        COMPOUND_TYPES = {"compound_component"}

        def _pick(types: set[str]) -> dict | None:
            for c in candidates:
                if c.get("candidate_type") in types:
                    return c
            return None

        selected: list[dict] = []
        for picker in (_pick(STRUCTURE_TYPES), _pick(COMPOUND_TYPES)):
            if picker is not None and picker not in selected:
                selected.append(picker)
            if len(selected) >= candidate_limit:
                break
        for c in candidates:
            if len(selected) >= candidate_limit:
                break
            if c not in selected:
                selected.append(c)
        cct["candidate_records"] = selected[:candidate_limit]
        storage.write_json(cct_key, cct)
    findings["step_06_input_candidate_count"] = len(cct["candidate_records"])

    # ── Step 6 ────────────────────────────────────────────────────────────
    try:
        DevelopabilityAgent(
            storage=storage, registry=registry,
            workflow_state=workflow_state,
            mcp_client=_AllowlistMCPClient(get_mcp_client(), LIVE_ALLOWLIST),
            llm=llm,
        ).run(run_id)
    except Exception as exc:  # noqa: BLE001
        if _is_unavailable_error(exc):
            _summary({**findings, "pipeline_status": "LLM_UNAVAILABLE",
                      "live_tool_status": "failed",
                      "reason": f"{provider_name} quota/timeout/disconnect",
                      "stage": "step_06"})
            return 0
        _summary({**findings, "pipeline_status": "FAIL",
                  "live_tool_status": "failed",
                  "stage": "step_06",
                  "error_type": type(exc).__name__})
        raise
    findings["step_06_artifact"] = registry.get(run_id).active_artifacts.structured_liability_summary_id

    step6 = storage.read_json(storage.run_key(run_id, "structured_liability_summary.json"))

    step6_audit: list[dict] = []
    tool_names_seen: set[str] = set()
    mocked_tools: set[str] = set()
    upstream_error_tools: set[str] = set()
    dep_unavail_tools: set[str] = set()
    bioactivity_lane_ran = False

    for cidx, cand in enumerate(step6.get("candidate_liability_results") or []):
        for lane in cand.get("lane_results") or []:
            lane_type = lane["lane_type"]
            if lane_type == "compound_bioactivity_prior_context" and lane.get("run_status") not in {"skipped"}:
                bioactivity_lane_ran = True
            for tc in lane.get("tool_call_records") or []:
                tn = tc.get("tool_name", "")
                tool_names_seen.add(tn)
                raw = _read_tool_output_payload(storage, tc.get("tool_output_ref"))
                meta = _envelope_meta(raw)
                if meta["envelope_status"] == "upstream_error":
                    upstream_error_tools.add(tn)
                if (raw.get("output") or {}).get("status") == "mocked":
                    mocked_tools.add(tn)
                if tc.get("run_status") == "dependency_unavailable":
                    dep_unavail_tools.add(tn)
                input_summary = tc.get("tool_input_summary") or {}
                # Compact-only: keep input keys without the long values.
                compact_input = {k: _short(v, 80) for k, v in input_summary.items()
                                 if k not in ("validation_warnings",)}
                attempted_live = tn in settings.live_tool_allowlist_set()
                live_success = (
                    attempted_live
                    and meta["executor"] == "tooluniverse"
                    and meta["envelope_status"] == "ok"
                    and tc.get("run_status") == "success"
                )
                step6_audit.append({
                    "candidate_index": cidx,
                    "candidate_id": cand.get("candidate_id"),
                    "lane_type": lane_type,
                    "tool_name": tn,
                    "selected_by": input_summary.get("selected_by"),
                    "run_status": tc.get("run_status"),
                    "attempted_live": attempted_live,
                    "executor": meta["executor"],
                    "envelope_status": meta["envelope_status"],
                    "live_success": live_success,
                    "source": meta["source"],
                    "error_message": meta["error_message"],
                    "input_summary": compact_input,
                    "liability_flags_count": len(lane.get("liability_flags") or []),
                })
    findings["step_06_audit"] = step6_audit
    live_success_tools = {a["tool_name"] for a in step6_audit if a.get("live_success")}
    findings["step_06_selected_proteins_plus"] = (
        "ProteinsPlus_profile_structure_quality" in tool_names_seen
    )
    findings["step_06_proteins_plus_live_success"] = (
        "ProteinsPlus_profile_structure_quality" in live_success_tools
    )
    findings["step_06_selected_ebi_features"] = (
        "EBIProteins_get_features" in tool_names_seen
        or "EBIProteins_get_epitopes" in tool_names_seen
    )
    findings["step_06_ebi_features_live_success"] = bool(
        live_success_tools & {"EBIProteins_get_features", "EBIProteins_get_epitopes"}
    )
    forbidden = {
        "EuropePMC_search_articles", "LiteratureSearchTool",
        "MultiAgentLiteratureSearch", "PubTator3_LiteratureSearch",
        "SemanticScholar_search_papers", "openalex_search_works",
        "PubChem_get_associated_patents_by_CID",
        "drugbank_get_drug_references_by_drug_name_or_id",
        "FDA_OrangeBook_get_patent_info",
    }
    findings["step_13_or_14_tools_called"] = sorted(tool_names_seen & forbidden)

    normalized_blob = json.dumps(step6)
    findings["pdb_content_leaked"] = any(
        tok in normalized_blob for tok in ("ATOM  ", "HETATM", "HEADER")
    )

    # Also track upstream_error tools in Step 5 ChEMBL audit.
    for entry in step5_audit:
        if entry.get("envelope_status") == "upstream_error":
            upstream_error_tools.add(entry["tool_name"])

    # ── final conclusion fields ──────────────────────────────────────────
    all_in_scope_tools = set(LIVE_ALLOWLIST)
    invoked_in_scope_tools = (
        {a["tool_name"] for a in step5_audit if a.get("attempted_live")}
        | {a["tool_name"] for a in step6_audit if a.get("attempted_live")}
    )
    # `all_step5_6_tools_attempted_live` is True iff every Step 5/Step 6
    # tool invocation in this run was attempted with `_live=True` (no
    # silent mock-fallback). It is NOT a claim that every live attempt
    # succeeded — see `live_tool_status` and per-tool `live_success`.
    findings["all_step5_6_tools_attempted_live"] = (
        bool(invoked_in_scope_tools)
        and all(
            (a.get("attempted_live") for a in step5_audit) if step5_audit else [True]
        )
        and all(
            (a.get("attempted_live") for a in step6_audit) if step6_audit else [True]
        )
        and not mocked_tools
    )
    findings["any_mocked_tool_outputs"] = bool(mocked_tools)
    findings["mocked_tool_names"] = sorted(mocked_tools)
    findings["upstream_error_tool_names"] = sorted(upstream_error_tools)
    findings["dependency_unavailable_tool_names"] = sorted(dep_unavail_tools)
    # Surface "known dependency gaps" so a known-not-wired wrapper does NOT
    # get conflated with silent failures or off-allowlist selections.
    known_gap_tools = set(KNOWN_LIVE_DEPENDENCY_GAPS) & dep_unavail_tools
    findings["known_dependency_gap_tool_names"] = sorted(known_gap_tools)
    findings["unknown_dependency_unavailable_tool_names"] = sorted(
        dep_unavail_tools - set(KNOWN_LIVE_DEPENDENCY_GAPS)
    )
    # `chembl_id_extracted` now strictly reflects what Step 6 actually sees:
    # at least one chembl_id identifier present in normalized Step 5 records.
    # Raw-only sightings are reported via `step_05_raw_chembl_id_occurrence_count`
    # and `step_05_chembl_normalization_discrepancy`.
    findings["chembl_id_extracted"] = normalized_chembl_count > 0
    findings["compound_bioactivity_prior_context_ran"] = bioactivity_lane_ran

    # Per-tool live-success rollup (Step 5 + Step 6 union).
    live_success_tool_names = sorted(
        {a["tool_name"] for a in (step5_audit + step6_audit) if a.get("live_success")}
    )
    findings["live_success_tool_names"] = live_success_tool_names

    # ── live_tool_status taxonomy ─────────────────────────────────────────
    # - "all_live_success": no mocked outputs, no upstream_error, no
    #   dependency_unavailable across in-scope tools invoked this run.
    # - "partial_live": no mocked outputs but at least one upstream_error
    #   or dependency_unavailable surfaced. When ALL dependency_unavailable
    #   tools fall under `KNOWN_LIVE_DEPENDENCY_GAPS`, the status is still
    #   `partial_live` and `live_tool_status_reason="known_dependency_gap"`
    #   is set so reviewers see the known-not-wired wrappers are not silent
    #   failures.
    # - "mocked_or_not_live": any mocked output or any in-scope tool whose
    #   invocation did NOT carry `_live=True` (which the smoke's
    #   ``_AllowlistMCPClient`` should prevent for Step 6 in this script).
    # - "failed": the pipeline itself failed (set earlier in error paths).
    any_attempt_not_live = any(
        not a.get("attempted_live") for a in (step5_audit + step6_audit)
    )
    unknown_dep_tools = dep_unavail_tools - set(KNOWN_LIVE_DEPENDENCY_GAPS)
    reasons: list[str] = []
    if mocked_tools or any_attempt_not_live:
        findings["live_tool_status"] = "mocked_or_not_live"
        if mocked_tools:
            reasons.append("mocked_outputs")
        if any_attempt_not_live:
            reasons.append("off_allowlist_selection")
    elif upstream_error_tools or unknown_dep_tools:
        findings["live_tool_status"] = "partial_live"
        if upstream_error_tools:
            reasons.append("upstream_error")
        if unknown_dep_tools:
            reasons.append("unknown_dependency_unavailable")
        if known_gap_tools:
            reasons.append("known_dependency_gap")
    elif known_gap_tools:
        findings["live_tool_status"] = "partial_live"
        reasons.append("known_dependency_gap")
    elif invoked_in_scope_tools:
        findings["live_tool_status"] = "all_live_success"
    else:
        findings["live_tool_status"] = "partial_live"
        reasons.append("no_in_scope_invocation")
    findings["live_tool_status_reason"] = reasons or ["clean"]

    _summary(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
