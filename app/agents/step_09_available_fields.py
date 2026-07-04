"""Step 9 readiness projection + hard-gate helpers.

This module is pure projection logic: it reads normalized artifacts from prior
steps and computes compact readiness metadata consumed by Step 9 runtime and
future selector layers.
"""

from __future__ import annotations

import re
from typing import Any

from ..agents.tool_selection_policy import signature_schema_for
from ..schemas.step_09_structure_variant_and_compound_screening import (
    Step9AvailableField,
    Step9HardGateAllowedTool,
    Step9HardGateBlockedTool,
    Step9LaneReadinessProfile,
    Step9LaneStatus,
    Step9ReadinessSummary,
)


_COMPOUND_SMILES_MATERIAL_TYPES = {
    "payload_smiles",
    "linker_smiles",
    "compound_smiles",
}
_COMPOUND_NAME_MATERIAL_TYPES = {
    "payload_name",
    "linker_name",
    "compound_name",
}
_COMPOUND_IDENTIFIER_TYPES = {"zinc_id", "chembl_id", "pubchem_cid"}
_PROTEIN_SEQUENCE_MATERIAL_TYPES = {
    "target_sequence",
    "target_antigen_sequence",
    "antibody_heavy_chain_sequence",
    "antibody_light_chain_sequence",
}
_EXPLICIT_PROTEIN_VARIANT_TYPES = {
    "protein_variant",
    "variant",
    "variant_sequence",
    "mutation",
}
_UNIPROT_IDENTIFIER_TYPES = {"uniprot_id", "uniprot"}

_PROTEIN_DESIGN_TOOLS = {
    "NvidiaNIM_rfdiffusion",
    "NvidiaNIM_proteinmpnn",
    "AlphaMissense_get_variant_score",
    "DynaMut2_predict_stability",
    "ESM_score_variant_sae_batch",
    "ESM_generate_protein_sequence",
}
_STRUCTURE_TOOLS = {"NvidiaNIM_rfdiffusion", "NvidiaNIM_proteinmpnn"}

_COMPOUND_TOOLS = {
    "ChEMBL_search_molecules",
    "ChEMBL_search_similarity",
    "ChEMBL_search_substructure",
    "ZINC_get_compound",
    "ZINC_get_purchasable",
    "ZINC_search_by_properties",
    "ZINC_search_by_smiles",
    "ZINC_search_compounds",
}

_VARIANT_PATTERN = re.compile(r"\b(?:p\.)?[A-Z][A-Za-z0-9]{0,7}[0-9]{1,6}[A-Z][A-Za-z0-9]{0,7}\b")


_TOOL_REQUIRED_ARGS_CACHE: dict[str, list[str]] = {}


def _schema_required_args(tool_name: str) -> list[str]:
    if tool_name in _TOOL_REQUIRED_ARGS_CACHE:
        return _TOOL_REQUIRED_ARGS_CACHE[tool_name]

    schema = signature_schema_for(tool_name) or {}
    required = schema.get("required") or []
    normalized: list[str] = []
    for name in required:
        if not isinstance(name, str):
            continue
        arg = name.strip()
        if not arg or arg.startswith("_"):
            continue
        normalized.append(arg)
    _TOOL_REQUIRED_ARGS_CACHE[tool_name] = normalized
    return normalized


def _infer_schema_arg_readiness(
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    arg_name: str,
) -> bool:
    """Best-effort mapping from TU argument name -> compact input readiness."""
    arg = arg_name.lower().strip()
    if not arg:
        return False

    # Candidate-provided sequence-like args.
    if "variant" in arg or "mutation" in arg:
        return bool(_extract_explicit_variants(candidate, ""))

    if "uniprot" in arg and "id" in arg:
        return _candidate_id_types(candidate, _UNIPROT_IDENTIFIER_TYPES)

    if "sequence" in arg:
        return _candidate_step9_sequence_presence(candidate, step7_seq_refs)

    if "structure" in arg or "pdb" in arg or "pdb_id" in arg:
        return _step8_structure_reference_available(step8_result, str(candidate.get("candidate_id") or ""))

    if arg in {"operation", "mode", "task", "input_mode", "output_format"}:
        return True

    return False


def _required_args_missing(
    tool_name: str,
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
) -> list[str]:
    """Return schema-required args for `tool_name` that are not currently ready."""
    missing = []
    for arg in _schema_required_args(tool_name):
        if not _infer_schema_arg_readiness(candidate, step7_seq_refs, step8_result, arg):
            missing.append(arg)
    return missing


def _candidate_value_types(candidate: dict, material_types: set[str]) -> list[dict[str, Any]]:
    return [m for m in (candidate.get("materials") or []) if isinstance(m, dict) and m.get("material_type") in material_types]


def _candidate_ids(candidate: dict, id_types: set[str]) -> list[str]:
    out: list[str] = []
    for ident in candidate.get("identifiers") or []:
        if not isinstance(ident, dict):
            continue
        if ident.get("id_type") in id_types:
            value = ident.get("id_value")
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
    return out


def _candidate_id_types(candidate: dict, id_types: set[str]) -> bool:
    return bool(_candidate_ids(candidate, id_types))


def _extract_explicit_variants(candidate: dict, readiness_text: str) -> list[str]:
    variants: list[str] = []
    for ident in candidate.get("identifiers") or []:
        if isinstance(ident, dict) and ident.get("id_type") in _EXPLICIT_PROTEIN_VARIANT_TYPES:
            value = ident.get("id_value")
            if isinstance(value, str) and value.strip():
                variants.append(value.strip())
    for mat in candidate.get("materials") or []:
        if not isinstance(mat, dict):
            continue
        if mat.get("material_type") in _EXPLICIT_PROTEIN_VARIANT_TYPES:
            value = mat.get("value")
            if isinstance(value, str) and value.strip():
                variants.append(value.strip())
    if isinstance(readiness_text, str):
        variants.extend(_VARIANT_PATTERN.findall(readiness_text))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _sequence_readiness_label(seq_ref: dict[str, Any]) -> str:
    status = str(seq_ref.get("sequence_value_status") or "").lower()
    if status in {"inline", "referenced"}:
        return "ready"
    if status == "identifier_only":
        return "identifier_only"
    return "unavailable"


def _readiness_text_hint_for_sequence_generation(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "generate sequence",
            "sequence generation",
            "design sequence",
            "generate a sequence",
            "predict sequence",
            "optimize sequence",
        )
    )


def _build_step7_sequence_refs(prepared_inputs: list[dict], candidate_id: str) -> list[dict[str, Any]]:
    out = []
    for sin in prepared_inputs:
        if not isinstance(sin, dict):
            continue
        if str(sin.get("candidate_id") or "") != candidate_id:
            continue
        for seq_ref in sin.get("sequence_refs_for_prediction") or []:
            if not isinstance(seq_ref, dict):
                continue
            out.append(
                {
                    "sequence_id": seq_ref.get("sequence_id"),
                    "candidate_id": candidate_id,
                    "field_ref": f"step7_sequence:{seq_ref.get('sequence_id')}",
                    "chain_role": seq_ref.get("chain_role"),
                    "sequence_value_status": seq_ref.get("sequence_value_status"),
                    "source_ref": seq_ref.get("source_ref"),
                    "prediction_input_kind": seq_ref.get("prediction_input_kind"),
                    "sequence_length": seq_ref.get("sequence_length"),
                    "sha256_prefix": seq_ref.get("sha256_prefix"),
                }
            )
    return out


def _step8_candidate_results(step8_result: dict | None, candidate_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in step8_result.get("candidate_structure_results") or [] if isinstance(step8_result, dict) else []:
        if isinstance(item, dict) and str(item.get("candidate_id") or "") == str(candidate_id):
            out.append(item)
    return out


def _step8_complex_refs(step8_result: dict | None, candidate_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in _step8_candidate_results(step8_result, candidate_id):
        for ref in result.get("complex_structure_refs") or []:
            if isinstance(ref, dict):
                out.append(ref)
    return out


def _is_step9_design_complex_ref(ref: dict[str, Any]) -> bool:
    source_kind = str(ref.get("source_kind") or "").lower()
    return source_kind in {"existing_pdb_complex", "predicted_complex"}


def _design_complex_ready(step8_result: dict | None, candidate_id: str) -> tuple[bool, list[str]]:
    complex_refs = _step8_complex_refs(step8_result, candidate_id)
    for ref in complex_refs:
        if _is_step9_design_complex_ref(ref):
            if str(ref.get("storage_ref") or ref.get("pdb_id") or ref.get("source_ref") or "").strip():
                return True, []
    return False, ["complex_structure_missing"]


def _candidate_step9_sequence_presence(candidate: dict, step7_seq_refs: list[dict[str, Any]]) -> bool:
    for material in _candidate_value_types(candidate, _PROTEIN_SEQUENCE_MATERIAL_TYPES):
        value = material.get("value")
        if isinstance(value, str) and value.strip():
            return True
    for seq_ref in step7_seq_refs:
        if str(seq_ref.get("sequence_value_status") or "") in {"inline", "referenced"}:
            return True
    return False


def _handoff_bool(step8_result: dict, candidate_id: str, field: str) -> bool:
    for cr in _step8_candidate_results(step8_result, candidate_id):
        handoff = cr.get("downstream_handoff")
        if isinstance(handoff, dict):
            value = handoff.get(field)
            if isinstance(value, bool):
                return value
    return False


def _handoff_value(step8_result: dict, candidate_id: str, field: str) -> list[str] | None | str:
    for cr in _step8_candidate_results(step8_result, candidate_id):
        handoff = cr.get("downstream_handoff")
        if not isinstance(handoff, dict):
            continue
        if field in handoff:
            return handoff.get(field)
    return None


def _step8_structure_reference_available(step8_result: dict, candidate_id: str) -> bool:
    for cr in _step8_candidate_results(step8_result, candidate_id):
        handoff = cr.get("downstream_handoff") if isinstance(cr, dict) else None
        if not isinstance(handoff, dict):
            continue
        if handoff.get("validated_structure_ref") or handoff.get("structure_for_variant_generation_ref"):
            return True
    return False


def _protein_design_gate(
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    readiness_text: str,
) -> tuple[str, list[str], list[Step9HardGateAllowedTool], list[Step9HardGateBlockedTool]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    allowed: list[Step9HardGateAllowedTool] = []
    blocked: list[Step9HardGateBlockedTool] = []

    has_uniprot = _candidate_id_types(candidate, _UNIPROT_IDENTIFIER_TYPES)
    variants = _extract_explicit_variants(candidate, readiness_text)
    has_variants = bool(variants)
    has_seq = _candidate_step9_sequence_presence(candidate, step7_seq_refs)
    has_structure_ref = _step8_structure_reference_available(step8_result, candidate_id)
    complex_ready, complex_missing = _design_complex_ready(step8_result, candidate_id)

    def _reason_for_missing(missing: list[str]) -> str:
        if not missing:
            return ""
        return "schema_required:" + ",".join(sorted(missing))

    def _allow_or_block(tool_name: str, fallback_reason: str) -> None:
        nonlocal allowed, blocked
        missing_from_schema = _required_args_missing(
            tool_name, candidate, step7_seq_refs, step8_result
        )
        if missing_from_schema:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="protein_design",
                    reason=_reason_for_missing(missing_from_schema),
                    rationale="required official TU args are missing",
                )
            )
            return
        if fallback_reason:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="protein_design",
                    reason=fallback_reason,
                    rationale="requires input for runtime handoff and tool contract",
                )
            )
            return
        allowed.append(
            Step9HardGateAllowedTool(
                candidate_id=candidate_id,
                tool_name=tool_name,
                lane_type="protein_design",
                rationale="required inputs available",
            )
        )

    # RFdiffusion / ProteinMPNN
    for tool_name in sorted({"NvidiaNIM_rfdiffusion", "NvidiaNIM_proteinmpnn"}):
        if not complex_ready:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="protein_design",
                    reason=complex_missing[0],
                    rationale="requires true protein complex evidence",
                )
            )
            continue
        _allow_or_block(tool_name, fallback_reason="")

    # AlphaMissense_get_variant_score
    _allow_or_block(
        "AlphaMissense_get_variant_score",
        fallback_reason=(
            "explicit_variant_missing" if not has_variants else
            "uniprot_id_missing" if not has_uniprot else ""
        ),
    )

    # DynaMut2_predict_stability
    _allow_or_block(
        "DynaMut2_predict_stability",
        fallback_reason=(
            "mutation_missing" if not has_variants else
            "structure_reference_missing" if not has_structure_ref else ""
        ),
    )

    # ESM_score_variant_sae_batch
    _allow_or_block(
        "ESM_score_variant_sae_batch",
        fallback_reason=(
            "explicit_variant_missing" if not has_variants else
            "sequence_value_unavailable" if not has_seq else ""
        ),
    )

    # ESM_generate_protein_sequence
    _allow_or_block(
        "ESM_generate_protein_sequence",
        fallback_reason="" if _readiness_text_hint_for_sequence_generation(readiness_text) else "intent_not_sequence_generation",
    )

    blocked_missing = [tool.reason for tool in blocked]
    status = "ready" if allowed else "blocked"
    return status, blocked_missing, allowed, blocked


def _compound_gate(
    candidate: dict,
) -> tuple[str, list[str], list[Step9HardGateAllowedTool], list[Step9HardGateBlockedTool]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    allowed: list[Step9HardGateAllowedTool] = []
    blocked: list[Step9HardGateBlockedTool] = []

    has_smiles = bool(_candidate_value_types(candidate, _COMPOUND_SMILES_MATERIAL_TYPES))
    has_name = bool(_candidate_value_types(candidate, _COMPOUND_NAME_MATERIAL_TYPES))
    has_identifier = _candidate_id_types(candidate, _COMPOUND_IDENTIFIER_TYPES)

    if not (has_smiles or has_name or has_identifier):
        blocked.append(
            Step9HardGateBlockedTool(
                candidate_id=candidate_id,
                tool_name="compound_screening",
                lane_type="compound_screening",
                reason="compound_input_missing",
                rationale="compound evidence not present",
            )
        )
        return "not_applicable", ["compound_input_missing"], allowed, blocked

    for tool in sorted(_COMPOUND_TOOLS):
        if (
            (tool == "ZINC_search_by_smiles" and has_smiles)
            or (tool == "ZINC_get_compound" and has_identifier)
            or (tool in {"ZINC_search_compounds", "ZINC_search_by_properties", "ZINC_get_purchasable"} and has_name)
            or (tool.startswith("ChEMBL") and _candidate_id_types(candidate, {"chembl_id"}))
        ):
            allowed.append(
                Step9HardGateAllowedTool(
                    candidate_id=candidate_id,
                    tool_name=tool,
                    lane_type="compound_screening",
                    rationale="compound evidence available",
                )
            )
    if not allowed:
        return "blocked", ["compound_tool_evidence_gap"], allowed, blocked
    return "ready", [], allowed, blocked


def _available_fields_for_compound(candidate: dict) -> list[Step9AvailableField]:
    out: list[Step9AvailableField] = []
    candidate_id = str(candidate.get("candidate_id") or "")
    for material in candidate.get("materials") or []:
        if not isinstance(material, dict):
            continue
        m_type = material.get("material_type")
        mat_id = material.get("material_id")
        if not isinstance(mat_id, str):
            continue
        if m_type in _COMPOUND_SMILES_MATERIAL_TYPES:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"material:{mat_id}",
                    provider="step_05",
                    field_type="compound",
                    value_kind="smiles",
                )
            )
        if m_type in _COMPOUND_NAME_MATERIAL_TYPES:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"material:{mat_id}",
                    provider="step_05",
                    field_type="compound",
                    value_kind="name",
                )
            )
        if m_type in _COMPOUND_IDENTIFIER_TYPES:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"identifier:{m_type}:{mat_id}",
                    provider="step_05",
                    field_type="compound_identifier",
                    value_kind=m_type,
                )
            )
    return out


def _available_fields_for_protein_candidate(
    candidate: dict, step7_prepared_inputs: list[dict]
) -> list[Step9AvailableField]:
    out: list[Step9AvailableField] = []
    candidate_id = str(candidate.get("candidate_id") or "")
    for material in _candidate_value_types(candidate, _PROTEIN_SEQUENCE_MATERIAL_TYPES):
        mat_id = material.get("material_id")
        if isinstance(mat_id, str) and mat_id:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"material:{mat_id}",
                    provider="step_05",
                    field_type="protein_sequence",
                    value_kind="sequence_material",
                    status="available",
                )
            )
    for ident in candidate.get("identifiers") or []:
        if (
            isinstance(ident, dict)
            and str(ident.get("id_type") or "") in _UNIPROT_IDENTIFIER_TYPES
            and isinstance(ident.get("id_value"), str)
            and ident.get("id_value")
        ):
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"identifier:{ident.get('id_type')}:{ident.get('id_value')}",
                    provider="step_05",
                    field_type="identifier",
                    value_kind="uniprot_id",
                )
            )
    for seq_ref in _build_step7_sequence_refs(step7_prepared_inputs, candidate_id):
        status = _sequence_readiness_label(seq_ref)
        out.append(
            Step9AvailableField(
                candidate_id=candidate_id,
                field_ref=seq_ref["field_ref"],
                provider="step_07",
                field_type="protein_sequence",
                value_kind=str(seq_ref.get("prediction_input_kind") or "fasta_ref"),
                source_ref=seq_ref.get("source_ref"),
                status="available" if status in {"ready", "identifier_only"} else "blocked",
            )
        )
    return out


def _aggregate_readiness_profile(
    lane_statuses: list[Step9LaneStatus], lane_type: str
) -> Step9LaneReadinessProfile:
    lanes = [lane for lane in lane_statuses if lane.lane_type == lane_type]
    if not lanes:
        return Step9LaneReadinessProfile(
            status="not_applicable",
            ready_tool_count=0,
            blocked_tool_count=0,
        )

    ready = len([lane for lane in lanes if lane.status == "ready"])
    blocked = len([lane for lane in lanes if lane.status == "blocked"])
    not_applicable = len([lane for lane in lanes if lane.status == "not_applicable"])
    missing_requirements = list(
        dict.fromkeys(
            req
            for lane in lanes
            for req in lane.missing_requirements
            if req
        )
    )
    allowed_tools = list(dict.fromkeys(tool for lane in lanes for tool in lane.allowed_tools))
    blocked_tools = list(dict.fromkeys(tool for lane in lanes for tool in lane.blocked_tools))

    if ready:
        status = "ready"
    elif blocked:
        status = "blocked"
    elif not_applicable:
        status = "not_applicable"
    else:
        status = "not_applicable"

    return Step9LaneReadinessProfile(
        status=status,
        ready_tool_count=ready,
        blocked_tool_count=blocked,
        missing_requirements=missing_requirements,
        allowed_tools=allowed_tools,
        blocked_tools=blocked_tools,
    )


def _lane_status(
    candidate: dict,
    lane_type: str,
    allowed: list[Step9HardGateAllowedTool],
    blocked: list[Step9HardGateBlockedTool],
    fields: list[Step9AvailableField],
    status_default: str,
) -> Step9LaneStatus:
    refs = [f.field_ref for f in fields]
    blocked_tool_names = [tool.tool_name for tool in blocked]
    allowed_tool_names = [tool.tool_name for tool in allowed]
    reasons = [tool.rationale for tool in blocked if isinstance(tool.rationale, str)]
    return Step9LaneStatus(
        lane_type=lane_type,  # type: ignore[arg-type]
        candidate_id=str(candidate.get("candidate_id") or ""),
        candidate_type=str(candidate.get("candidate_type") or "unknown"),
        status=status_default,  # type: ignore[arg-type]
        allowed_tools=allowed_tool_names,
        blocked_tools=blocked_tool_names,
        missing_requirements=list(dict.fromkeys([tool.reason for tool in blocked if tool.reason])),
        gate_reasons=list(dict.fromkeys([r for r in reasons if r])),
        available_field_refs=refs,
    )


def project_step9_readiness(
    *,
    candidate_context_table: dict,
    prepared_structure_input_package: list[dict] | dict | None,
    structure_prediction_and_interface_results: dict | None,
    compound_context_text: str,
) -> dict[str, Any]:
    candidate_records = candidate_context_table.get("candidate_records") or []
    if not isinstance(candidate_records, list):
        candidate_records = []

    if isinstance(prepared_structure_input_package, dict):
        prepared_inputs = prepared_structure_input_package.get("prepared_structure_inputs") or []
    else:
        prepared_inputs = prepared_structure_input_package or []
    if not isinstance(prepared_inputs, list):
        prepared_inputs = []

    available_fields: list[Step9AvailableField] = []
    allowed_tools: list[Step9HardGateAllowedTool] = []
    blocked_tools: list[Step9HardGateBlockedTool] = []
    lane_statuses: list[Step9LaneStatus] = []

    step8_data = structure_prediction_and_interface_results or {}

    for candidate in candidate_records:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue

        candidate_type = str(candidate.get("candidate_type") or "unknown")
        step7_refs = _build_step7_sequence_refs(prepared_inputs, candidate_id)

        if candidate_type == "compound_component":
            fields = _available_fields_for_compound(candidate)
            status, missing, allowed, blocked = _compound_gate(candidate)
            # Compound candidates may also expose referenced compounds in Step 5 identifiers.
            for ident in candidate.get("identifiers") or []:
                if (
                    isinstance(ident, dict)
                    and ident.get("id_type") in _COMPOUND_IDENTIFIER_TYPES
                    and isinstance(ident.get("id_value"), str)
                    and ident.get("id_value")
                ):
                    fields.append(
                        Step9AvailableField(
                            candidate_id=candidate_id,
                            field_ref=f"identifier:{ident.get('id_type')}:{ident.get('id_value')}",
                            provider="step_05",
                            field_type="identifier",
                            value_kind=str(ident.get("id_type")),
                        )
                    )
            status_value = (
                "ready" if any(tool.tool_name != "compound_screening" for tool in allowed) else ("not_applicable" if missing else "partial")
            )
            lane_statuses.append(
                _lane_status(candidate, "compound_screening", allowed, blocked, fields, status_value)
            )
            available_fields.extend(fields)
            allowed_tools.extend(allowed)
            blocked_tools.extend(blocked)
            continue

        fields = _available_fields_for_protein_candidate(candidate, prepared_inputs)
        status, _, allowed, blocked = _protein_design_gate(candidate, step7_refs, step8_data, compound_context_text)
        lane_statuses.append(
            _lane_status(candidate, "protein_design", allowed, blocked, fields, status)
        )
        available_fields.extend(fields)
        allowed_tools.extend(allowed)
        blocked_tools.extend(blocked)

    summary = Step9ReadinessSummary(
        total_candidates=len(candidate_records),
        protein_design_candidates=sum(
            1 for candidate in candidate_records
            if isinstance(candidate, dict) and str(candidate.get("candidate_type") or "") != "compound_component"
        ),
        protein_design_ready_candidates=sum(
            1 for lane in lane_statuses if lane.lane_type == "protein_design" and lane.status == "ready"
        ),
        protein_design_blocked_candidates=sum(
            1 for lane in lane_statuses if lane.lane_type == "protein_design" and lane.status == "blocked"
        ),
        protein_design_not_applicable_candidates=sum(
            1 for lane in lane_statuses if lane.lane_type == "protein_design" and lane.status == "not_applicable"
        ),
        compound_candidates=sum(
            1 for candidate in candidate_records
            if isinstance(candidate, dict) and candidate.get("candidate_type") == "compound_component"
        ),
        compound_candidate_with_tools=sum(
            1 for lane in lane_statuses if lane.lane_type == "compound_screening" and lane.status == "ready"
        ),
        hard_gate_allowed_tool_count=len(allowed_tools),
        hard_gate_blocked_tool_count=len(blocked_tools),
    )

    all_missing = list(dict.fromkeys(
        reason
        for lane in lane_statuses
        for reason in lane.missing_requirements
        if reason
    ))

    return {
        "step9_available_fields": available_fields,
        "step9_readiness_summary": summary,
        "step9_lane_statuses": lane_statuses,
        "protein_design_readiness": _aggregate_readiness_profile(lane_statuses, "protein_design"),
        "compound_screening_readiness": _aggregate_readiness_profile(
            lane_statuses, "compound_screening"
        ),
        "variant_evaluation_readiness": Step9LaneReadinessProfile(status="not_applicable"),
        "step9_hard_gate_allowed_tools": allowed_tools,
        "step9_hard_gate_blocked_tools_with_reason": blocked_tools,
        "step9_missing_inputs": all_missing,
    }
