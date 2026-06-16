"""Payload builders and validators for optional Gemini live smoke scripts."""

from __future__ import annotations

from typing import Any

from app.services.storage_service import Storage


def structured_query_payload() -> tuple[str, dict]:
    return (
        "Parse this ADC request: Design a HER2 ADC with vc-MMAE payload.",
        {
            "raw_request_record": {
                "run_id": "gemini_smoke",
                "run_artifact_registry_id": "registry_smoke",
                "raw_user_query": "Design a HER2 ADC with vc-MMAE payload.",
                "user_provided_context": {},
            }
        },
    )


def stage1_payload() -> tuple[str, dict]:
    catalog = [
        {
            "tool_name": "DrugProps_pains_filter",
            "short_description": "Pan-assay interference substructure filter",
            "capability_tags": ["small_molecule", "liability_filter"],
            "coarse_input_requirements": ["smiles", "compound_name"],
            "step_id": "step_06",
            "agent_name": "developability_agent",
        },
        {
            "tool_name": "PROSITE_scan_sequence",
            "short_description": "PROSITE motif scan over a protein sequence",
            "capability_tags": ["protein_sequence", "motif"],
            "coarse_input_requirements": ["protein_sequence"],
            "step_id": "step_06",
            "agent_name": "developability_agent",
        },
        {
            "tool_name": "ProteinsPlus_profile_structure_quality",
            "short_description": "ProteinsPlus structure quality profiling",
            "capability_tags": ["structure_quality"],
            "coarse_input_requirements": ["pdb_id", "structure_file"],
            "step_id": "step_06",
            "agent_name": "developability_agent",
        },
    ]
    return (
        "Pick tools from the compact catalog that match the context.",
        {
            "task": "tool_selection_stage_1",
            "agent_name": "developability_agent",
            "step_id": "step_06",
            "compact_catalog": catalog,
            "context": {"signals": {"smiles": True}, "note": "payload/linker SMILES available"},
        },
    )


def stage2_payload() -> tuple[str, dict]:
    return (
        "Construct arguments for the selected tool.",
        {
            "task": "tool_selection_stage_2",
            "agent_name": "developability_agent",
            "step_id": "step_06",
            "tool_name": "DrugProps_pains_filter",
            "full_schema": {
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
                "required": ["smiles"],
            },
            "context": {"arg_hints": {"smiles": "CCO"}, "note": "ethanol smoke input"},
        },
    )


def validate_structured_query(out: Any) -> dict:
    if not isinstance(out, dict):
        raise AssertionError("structured_query output must be a dict")
    _require_keys(out, ["task_intent", "mentioned_entities", "referenced_inputs", "parse_warnings"])
    if not isinstance(out["task_intent"], dict):
        raise AssertionError("structured_query.task_intent must be a dict")
    if not isinstance(out["mentioned_entities"], dict):
        raise AssertionError("structured_query.mentioned_entities must be a dict")
    if not isinstance(out["referenced_inputs"], list):
        raise AssertionError("structured_query.referenced_inputs must be a list")
    if not isinstance(out["parse_warnings"], list):
        raise AssertionError("structured_query.parse_warnings must be a list")
    return out


def validate_stage1(out: Any, schema: dict) -> dict:
    if not isinstance(out, dict):
        raise AssertionError("stage1 output must be a dict")
    selections = out.get("selections")
    if not isinstance(selections, list):
        raise AssertionError("stage1.selections must be a list")
    allowed = {entry["tool_name"] for entry in schema["compact_catalog"]}
    for i, entry in enumerate(selections):
        if not isinstance(entry, dict):
            raise AssertionError(f"stage1.selections[{i}] must be a dict")
        tool_name = entry.get("tool_name")
        if tool_name not in allowed:
            raise AssertionError(f"stage1 selected out-of-catalog tool: {tool_name}")
    return out


def validate_stage2(out: Any) -> dict:
    if not isinstance(out, dict):
        raise AssertionError("stage2 output must be a dict")
    if not isinstance(out.get("arguments"), dict):
        raise AssertionError("stage2.arguments must be a dict")
    return out


def top_level_keys(out: dict) -> list[str]:
    return sorted(out.keys())


def validate_step1_6_artifacts(*, storage: Storage, run_id: str, artifacts: dict) -> dict:
    structured_query_id = artifacts.get("structured_query")
    liability_id = artifacts.get("structured_liability_summary")
    if not structured_query_id:
        raise AssertionError("Step 2 structured_query artifact id is missing")
    if not liability_id:
        raise AssertionError("Step 6 structured_liability_summary artifact id is missing")

    structured_query = storage.read_json(storage.run_key(run_id, "inputs", "structured_query.json"))
    liability = storage.read_json(storage.run_key(run_id, "structured_liability_summary.json"))
    if structured_query.get("artifact_id") != structured_query_id:
        raise AssertionError("Step 2 structured_query artifact id mismatch")
    if liability.get("artifact_id") != liability_id:
        raise AssertionError("Step 6 structured_liability_summary artifact id mismatch")
    if not has_selection_policy_version(liability):
        raise AssertionError(
            "Step 6 expected at least one tool_call_records[].tool_input_summary "
            "with selection_policy_version"
        )
    return {"structured_query": structured_query, "liability": liability}


def has_selection_policy_version(liability: dict) -> bool:
    for candidate in liability.get("candidate_liability_results") or []:
        for lane in candidate.get("lane_results") or []:
            for record in lane.get("tool_call_records") or []:
                summary = record.get("tool_input_summary") or {}
                if summary.get("selection_policy_version"):
                    return True
    return False


def _require_keys(out: dict, keys: list[str]) -> None:
    missing = [key for key in keys if key not in out]
    if missing:
        raise AssertionError(f"missing required keys: {missing}")
