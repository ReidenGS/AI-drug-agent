"""Provider-agnostic Step 1→6 smoke with a real uploaded PDB.

Reads ``LLM_PROVIDER`` from settings. Currently supports:
- ``gemini`` (needs ``GEMINI_API_KEY``)
- ``openai`` (needs ``OPENAI_API_KEY``)
- anything else → ``SKIP``.

Reports ``LLM_UNAVAILABLE`` (without raising) when the upstream provider
returns quota / 429 / 503 / disconnect / timeout errors, so quota-limited
runs are not mistaken for correctness regressions.

Compact-only output: never prints raw PDB bytes, raw provider response
bodies, the system prompt, or the API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PDB = Path(
    "/Users/jackiewen/Desktop/desk/实习工作/国外ai医药/程序/data/pdb/S1.pdb"
)


_UNAVAILABLE_TOKENS = (
    "429", "503", "quota", "rate limit", "unavailable",
    "timeout", "timed out", "server disconnect", "connection reset",
    "remote disconnected", "remoteprotocolerror", "service unavailable",
    "insufficient_quota",
)


def _is_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in _UNAVAILABLE_TOKENS):
        return True
    cls = type(exc).__name__.lower()
    return any(tok in cls for tok in ("ratelimit", "timeout", "apiconnection", "apierror"))


def _summary(d: dict) -> None:
    print(json.dumps(d, indent=2, sort_keys=True))


def main() -> int:
    os.environ.setdefault("STORAGE_MODE", "local")
    from app.settings import get_settings  # noqa: PLC0415

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
        _summary({
            "status": "SKIP",
            "reason": f"unsupported LLM_PROVIDER={provider_name!r}; expected gemini or openai",
        })
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
        get_llm_provider,
        get_mcp_client,
        get_registry_service,
        get_storage,
        get_workflow_state_service,
    )
    from app.graph.adc_graph import build_minimal_graph  # noqa: PLC0415
    from app.utils.ids import new_artifact_id, new_run_id  # noqa: PLC0415

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
        "status": "PASS",
        "provider": provider_name,
        "model": model_name,
        "run_id": run_id,
        "pdb_filename": pdb_path.name,
        "pdb_sha256_prefix": pdb_sha256[:12],
        "candidate_limit": candidate_limit,
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
            _summary({**findings, "status": "LLM_UNAVAILABLE",
                      "reason": f"{provider_name} quota/timeout/disconnect",
                      "stage": "graph.invoke (Step 1-4)"})
            return 0
        _summary({**findings, "status": "FAIL",
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
        _summary({**findings, "status": "FAIL", "stage": "step_05",
                  "error_type": type(exc).__name__})
        raise
    findings["step_05_artifact"] = registry.get(run_id).active_artifacts.candidate_context_table_id

    cct_key = storage.run_key(run_id, "candidate_context_table.json")
    cct = storage.read_json(cct_key)
    structure_mats = [
        m for cand in cct.get("candidate_records") or []
        for m in (cand.get("materials") or [])
        if (m.get("material_type") or "").endswith("structure_reference")
        or m.get("material_type") in {"structure_file", "structure_ref"}
    ]
    findings["step_05_structure_material_count"] = len(structure_mats)
    original_candidate_count = len(cct.get("candidate_records") or [])
    findings["step_05_candidate_count"] = original_candidate_count

    # Trim to representative candidates so Step 6 LLM call budget stays small.
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
            mcp_client=get_mcp_client(), llm=llm,
        ).run(run_id)
    except Exception as exc:  # noqa: BLE001
        if _is_unavailable_error(exc):
            _summary({**findings, "status": "LLM_UNAVAILABLE",
                      "reason": f"{provider_name} quota/timeout/disconnect",
                      "stage": "step_06"})
            return 0
        _summary({**findings, "status": "FAIL", "stage": "step_06",
                  "error_type": type(exc).__name__})
        raise
    findings["step_06_artifact"] = registry.get(run_id).active_artifacts.structured_liability_summary_id

    try:
        step6 = storage.read_json(storage.run_key(run_id, "structured_liability_summary.json"))
        tool_names: set[str] = set()
        lane_summary: dict[str, str] = {}
        for cand in step6.get("candidate_liability_results") or []:
            for lane in cand.get("lane_results") or []:
                lane_summary.setdefault(lane["lane_type"], lane.get("run_status", "?"))
                for tc in lane.get("tool_call_records") or []:
                    tool_names.add(tc.get("tool_name", ""))
        findings["step_06_lane_run_status_sample"] = lane_summary
        findings["step_06_called_proteins_plus"] = "ProteinsPlus_profile_structure_quality" in tool_names
        findings["step_06_called_ebi_features"] = (
            "EBIProteins_get_features" in tool_names
            or "EBIProteins_get_epitopes" in tool_names
        )
        forbidden = {
            "EuropePMC_search_articles", "LiteratureSearchTool",
            "MultiAgentLiteratureSearch", "PubTator3_LiteratureSearch",
            "SemanticScholar_search_papers", "openalex_search_works",
            "PubChem_get_associated_patents_by_CID",
            "drugbank_get_drug_references_by_drug_name_or_id",
            "FDA_OrangeBook_get_patent_info",
        }
        leaked = sorted(tool_names & forbidden)
        findings["step_06_step13_or_14_tools_called"] = leaked
        # Raw PDB content isolation: PDB header signature must not appear.
        normalized_blob = json.dumps(step6)
        findings["step_06_pdb_content_leaked"] = any(
            tok in normalized_blob for tok in ("ATOM  ", "HETATM", "HEADER")
        )
        # Step 6 LLM `selected_by` distribution (so we can see if the live
        # provider actually drove tool selection).
        selected_by_counts: dict[str, int] = {}
        for cand in step6.get("candidate_liability_results") or []:
            for lane in cand.get("lane_results") or []:
                for tc in lane.get("tool_call_records") or []:
                    src = (tc.get("tool_input_summary") or {}).get("selected_by") or "unknown"
                    selected_by_counts[src] = selected_by_counts.get(src, 0) + 1
        findings["step_06_selected_by_distribution"] = selected_by_counts
    except Exception as exc:  # noqa: BLE001
        findings["step_06_check_error"] = type(exc).__name__

    _summary(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
