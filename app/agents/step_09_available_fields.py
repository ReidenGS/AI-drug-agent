"""Step 9 readiness projection + hard-gate helpers.

This module reads artifacts from previous steps and computes compact readiness
signals for Step 9 selector/runtime input.
"""

from __future__ import annotations

from typing import Any

import re

from ..agents.tool_selection_policy import signature_schema_for
from ..schemas.step_09_structure_variant_and_compound_screening import (
    Step9AvailableField,
    Step9HardGateAllowedTool,
    Step9HardGateBlockedTool,
    Step9LaneReadinessProfile,
    Step9LaneStatus,
    Step9ReadinessSummary,
    Step9ToolSchemaRequirement,
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

_PDB_ID_RE = re.compile(r"[0-9][A-Za-z0-9]{3}")

_PROTEIN_DESIGN_TOOLS = {
    "NvidiaNIM_rfdiffusion",
    "NvidiaNIM_proteinmpnn",
    "ESM_generate_protein_sequence",
}
_VARIANT_EVALUATION_TOOLS = {
    "AlphaMissense_get_variant_score",
    "DynaMut2_predict_stability",
    "ESM_score_variant_sae_batch",
}
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


_TOOL_REQUIRED_ARGS_CACHE: dict[str, list[str]] = {}
_TOOL_SCHEMA_SOURCE_CACHE: dict[str, str] = {}


def _schema_required_args(tool_name: str) -> tuple[list[str], str]:
    """Return required parameters and where the schema came from."""
    if tool_name in _TOOL_REQUIRED_ARGS_CACHE:
        return (
            _TOOL_REQUIRED_ARGS_CACHE[tool_name],
            _TOOL_SCHEMA_SOURCE_CACHE.get(tool_name, "unavailable"),
        )

    schema = signature_schema_for(tool_name)
    if not isinstance(schema, dict):
        _TOOL_REQUIRED_ARGS_CACHE[tool_name] = []
        _TOOL_SCHEMA_SOURCE_CACHE[tool_name] = "unavailable"
        return [], "unavailable"

    required = schema.get("required") or []
    normalized: list[str] = []
    for name in required:
        if not isinstance(name, str):
            continue
        arg = name.strip()
        if not arg or arg.startswith("_"):
            continue
        normalized.append(arg)

    properties = schema.get("properties")
    schema_source = "signature"
    has_tooluniverse_spec = False
    try:
        from ..mcp import tooluniverse_adapter

        has_tooluniverse_spec = tooluniverse_adapter.get_tool_specification(tool_name) is not None
        if has_tooluniverse_spec:
            schema_source = "tooluniverse_or_signature"
    except Exception:  # noqa: BLE001
        schema_source = "signature"

    _TOOL_REQUIRED_ARGS_CACHE[tool_name] = normalized
    _TOOL_SCHEMA_SOURCE_CACHE[tool_name] = schema_source
    return normalized, schema_source


def _schema_props(tool_name: str) -> dict[str, dict[str, Any]]:
    schema = signature_schema_for(tool_name) or {}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return properties
    return {}


def _candidate_value_types(candidate: dict, material_types: set[str]) -> list[dict[str, Any]]:
    return [
        m
        for m in (candidate.get("materials") or [])
        if isinstance(m, dict) and m.get("material_type") in material_types
    ]


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


def _extract_explicit_variants(candidate: dict) -> list[str]:
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
    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _candidate_has_compound_smiles(candidate: dict) -> bool:
    return bool(_candidate_value_types(candidate, _COMPOUND_SMILES_MATERIAL_TYPES))


def _candidate_has_compound_name(candidate: dict) -> bool:
    for material in candidate.get("materials") or []:
        if (
            isinstance(material, dict)
            and material.get("material_type") in _COMPOUND_NAME_MATERIAL_TYPES
            and isinstance(material.get("value"), str)
            and material.get("value").strip()
        ):
            return True
    return False


def _candidate_has_smiles_like_query_text(candidate: dict) -> bool:
    # Preserve deterministic behavior: explicit string fields can be used as a
    # query-like hint. This intentionally does not scan raw user query.
    return _candidate_has_compound_name(candidate)


def _candidate_has_zinc_id(candidate: dict) -> bool:
    return _candidate_id_types(candidate, {"zinc_id"})


def _candidate_has_chembl_id(candidate: dict) -> bool:
    return _candidate_id_types(candidate, {"chembl_id"})


def _candidate_has_pubchem_id(candidate: dict) -> bool:
    return _candidate_id_types(candidate, {"pubchem_cid"})


def _candidate_has_uniprot_id(candidate: dict) -> bool:
    return _candidate_id_types(candidate, _UNIPROT_IDENTIFIER_TYPES)


def _candidate_identifier_values(candidate: dict, id_types: set[str]) -> list[str]:
    return [
        value.strip()
        for value in _candidate_ids(candidate, id_types)
        if isinstance(value, str) and value.strip()
    ]


def _looks_like_pdb_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped or len(stripped) != 4:
        return False
    return bool(_PDB_ID_RE.fullmatch(stripped))


def _candidate_has_real_pdb_id(candidate: dict, step8_result: dict | None, candidate_id: str) -> bool:
    if any(_looks_like_pdb_id(v) for v in _candidate_identifier_values(candidate, {"pdb_id"})):
        return True

    for material in candidate.get("materials") or []:
        if not isinstance(material, dict):
            continue
        if material.get("material_type") != "pdb_id":
            continue
        value = material.get("value")
        if _looks_like_pdb_id(value):
            return True

    for ref in _step8_complex_refs(step8_result, candidate_id):
        kind = str(ref.get("source_kind") or "").lower()
        if kind not in {"existing_pdb_complex", "existing_pdb_structure"}:
            continue
        if _looks_like_pdb_id(ref.get("pdb_id")) or _looks_like_pdb_id(ref.get("source_ref")):
            return True

    if candidate_id:
        for cr in _step8_candidate_results(step8_result, candidate_id):
            handoff = cr.get("downstream_handoff")
            if not isinstance(handoff, dict):
                continue
            if _looks_like_pdb_id(handoff.get("structure_for_variant_generation_ref")):
                return True
    return False


def _candidate_has_variant_structure_input(candidate: dict, step8_result: dict | None, candidate_id: str) -> bool:
    if candidate_id:
        for result in _step8_candidate_results(step8_result, candidate_id):
            handoff = result.get("downstream_handoff") if isinstance(result, dict) else None
            if not isinstance(handoff, dict):
                continue
            if handoff.get("structure_for_variant_generation_ref"):
                return True

    for ref in _step8_complex_refs(step8_result, candidate_id):
        kind = str(ref.get("source_kind") or "").lower()
        if kind in {"existing_pdb_complex", "predicted_complex", "existing_pdb_structure"}:
            if str(ref.get("storage_ref") or ref.get("source_ref") or ref.get("pdb_id") or "").strip():
                return True
    return False


def _candidate_has_structured_variants(candidate: dict) -> bool:
    return bool(_extract_explicit_variants(candidate))


def _candidate_has_chain(candidate: dict) -> bool:
    for ident in candidate.get("identifiers") or []:
        if isinstance(ident, dict) and str(ident.get("id_type") or "").lower() in {
            "chain_id",
            "chain",
        }:
            if isinstance(ident.get("id_value"), str) and ident.get("id_value").strip():
                return True
    for mat in candidate.get("materials") or []:
        if not isinstance(mat, dict):
            continue
        role = str(mat.get("chain_role") or "").strip().lower()
        if role:
            return True
    return False


def _candidate_has_contigs(candidate: dict) -> bool:
    for material in _candidate_value_types(candidate, {"design_contigs", "contigs"}):
        if isinstance(material.get("value"), str) and material.get("value").strip():
            return True
    return False


def _sequence_generation_intent_text(intent_text: str | None) -> str:
    if not isinstance(intent_text, str):
        return ""
    return " ".join(intent_text.lower().strip().split())


_SEQUENCE_GENERATION_HINTS = (
    "generate sequence",
    "generate protein sequence",
    "generate aa sequence",
    "protein sequence generation",
    "design sequence",
    "design protein sequence",
    "sequence generation",
    "predict sequence",
)


def _is_sequence_generation_intent(intent_text: str) -> bool:
    normalized = _sequence_generation_intent_text(intent_text)
    if not normalized:
        return False
    return any(hint in normalized for hint in _SEQUENCE_GENERATION_HINTS)


def _structure_arg_ready(
    arg: str,
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    lane_type: str,
    candidate_id: str,
) -> bool:
    if arg in {"input_pdb", "structure", "structure_ref", "backbone", "path", "pdb_file"}:
        if lane_type == "protein_design":
            return _candidate_has_design_structure_input(candidate, step7_seq_refs, step8_result)
        if lane_type == "variant_evaluation":
            return _candidate_has_variant_structure_input(candidate, step8_result, candidate_id)
        return False
    return False


def _candidate_step9_sequence_presence(candidate: dict, step7_seq_refs: list[dict[str, Any]]) -> bool:
    for material in _candidate_value_types(candidate, _PROTEIN_SEQUENCE_MATERIAL_TYPES):
        value = material.get("value")
        if isinstance(value, str) and value.strip():
            return True
    for seq_ref in step7_seq_refs:
        if str(seq_ref.get("sequence_value_status") or "").lower() in {
            "inline",
            "referenced",
        }:
            return True
    return False


def _candidate_has_prompt_sequence(candidate: dict, step7_seq_refs: list[dict[str, Any]], _: dict | None) -> bool:
    return _candidate_step9_sequence_presence(candidate, step7_seq_refs)


def _step8_candidate_results(step8_result: dict | None, candidate_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(step8_result, dict):
        return out
    for item in step8_result.get("candidate_structure_results") or []:
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


def _step8_structure_reference_available(step8_result: dict, candidate_id: str) -> bool:
    for cr in _step8_candidate_results(step8_result, candidate_id):
        handoff = cr.get("downstream_handoff") if isinstance(cr, dict) else None
        if not isinstance(handoff, dict):
            continue
        if handoff.get("validated_structure_ref") or handoff.get("structure_for_variant_generation_ref"):
            return True
    return False


def _candidate_has_complex_structure_ref(candidate_id: str, step8_result: dict | None) -> bool:
    ready, _ = _design_complex_ready(step8_result, candidate_id)
    return ready


def _sequence_readiness_label(seq_ref: dict[str, Any]) -> str:
    status = str(seq_ref.get("sequence_value_status") or "").lower()
    if status in {"inline", "referenced"}:
        return "ready"
    if status == "identifier_only":
        return "identifier_only"
    return "unavailable"


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


def _candidate_has_contigs_hint(candidate: dict) -> bool:
    return _candidate_has_contigs(candidate)


def _candidate_has_design_structure_input(candidate: dict, step7_seq_refs: list[dict[str, Any]], step8_result: dict | None) -> bool:
    return _candidate_has_complex_structure_ref(str(candidate.get("candidate_id") or ""), step8_result)


def _infer_schema_arg_readiness(
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    arg_name: str,
    lane_type: str,
    arg_schema: dict[str, Any] | None,
) -> bool:
    """Map schema arg name -> compact readiness check."""
    arg = arg_name.lower().strip()
    if not arg:
        return False

    # Enum singleton args are fixed constants in official schema.
    if isinstance(arg_schema, dict):
        enum_values = arg_schema.get("enum")
        if isinstance(enum_values, list) and enum_values and len(enum_values) == 1:
            return True

    arg_readiness_map: dict[str, Any] = {
        "operation": lambda: True,
        "mode": lambda: True,
        "output_format": lambda: True,
        "task": lambda: True,
        "input_mode": lambda: True,
        "smiles": lambda: _candidate_has_compound_smiles(candidate),
        "query": lambda: _candidate_has_smiles_like_query_text(candidate),
        "zinc_id": lambda: _candidate_has_zinc_id(candidate),
        "molecule_chembl_id": lambda: _candidate_has_chembl_id(candidate),
        "chembl_id": lambda: _candidate_has_chembl_id(candidate),
        "pubchem_cid": lambda: _candidate_has_pubchem_id(candidate),
        "uniprot_id": lambda: _candidate_has_uniprot_id(candidate),
        "variant": lambda: _candidate_has_structured_variants(candidate),
        "variants": lambda: bool(_extract_explicit_variants(candidate)),
        "mutation": lambda: _candidate_has_structured_variants(candidate),
        "chain": lambda: _candidate_has_chain(candidate),
        "contigs": lambda: _candidate_has_contigs_hint(candidate),
        "input_pdb": lambda: _structure_arg_ready(
            "input_pdb",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "pdb_id": lambda: _candidate_has_real_pdb_id(
            candidate,
            step8_result,
            str(candidate.get("candidate_id") or ""),
        ),
        "structure": lambda: _structure_arg_ready(
            "structure",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "structure_ref": lambda: _structure_arg_ready(
            "structure_ref",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "backbone": lambda: _structure_arg_ready(
            "backbone",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "path": lambda: _structure_arg_ready(
            "path",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "pdb_file": lambda: _structure_arg_ready(
            "pdb_file",
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ),
        "sequence": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "prompt_sequence": lambda: _candidate_has_prompt_sequence(candidate, step7_seq_refs, step8_result),
        "sequence_value": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "sequence_1": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "sequence_2": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "sequence_3": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "sequence_a": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
        "sequence_b": lambda: _candidate_step9_sequence_presence(candidate, step7_seq_refs),
    }

    helper = arg_readiness_map.get(arg)
    if helper is None:
        # Conservative default for opaque arguments.
        if arg.startswith("sequence"):
            return _candidate_step9_sequence_presence(candidate, step7_seq_refs)
        if _structure_arg_ready(
            arg,
            candidate,
            step7_seq_refs,
            step8_result,
            lane_type,
            str(candidate.get("candidate_id") or ""),
        ):
            return True
        if arg.endswith("_pdb") and lane_type == "variant_evaluation":
            return _candidate_has_real_pdb_id(
                candidate,
                step8_result,
                str(candidate.get("candidate_id") or ""),
            )
        return False
    return bool(helper())


def _required_args_missing(
    tool_name: str,
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    lane_type: str = "protein_design",
    schema_props: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    required, source = _schema_required_args(tool_name)
    if source == "unavailable":
        if required:
            return required, [], required
        return [], [], ["tool_schema_unavailable"]

    if not required:
        return [], [], []

    if schema_props is None:
        schema_props = _schema_props(tool_name)

    satisfiable: list[str] = []
    missing: list[str] = []
    for arg in required:
        ready = _infer_schema_arg_readiness(
            candidate,
            step7_seq_refs,
            step8_result,
            arg,
            lane_type,
            schema_props.get(arg) if isinstance(schema_props.get(arg), dict) else None,
        )
        if ready:
            satisfiable.append(arg)
        else:
            missing.append(arg)
    return required, satisfiable, missing


def _required_args_missing_only(
    tool_name: str,
    candidate: dict,
    step7_seq_refs: list[dict[str, Any]],
    step8_result: dict | None,
    lane_type: str = "protein_design",
    schema_props: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    _, satisfiable, missing = _required_args_missing(
        tool_name, candidate, step7_seq_refs, step8_result, lane_type, schema_props
    )
    return [arg for arg in missing if arg not in satisfiable]


def _tool_schema_requirement_record(
    candidate: dict,
    candidate_id: str,
    lane_type: str,
    tool_name: str,
    required: list[str],
    satisfiable: list[str],
    missing: list[str],
    schema_source: str,
) -> Step9ToolSchemaRequirement:
    decision = "allowed" if not missing else "blocked"
    if missing and missing == ["tool_schema_unavailable"]:
        reason = "tool_schema_unavailable"
    elif missing:
        reason = f"schema_required:{','.join(sorted(missing))}"
    else:
        reason = "schema_requirements_satisfied"
    return Step9ToolSchemaRequirement(
        candidate_id=candidate_id,
        tool_name=tool_name,
        lane_type=lane_type,
        required_fields=required,
        schema_source=schema_source,
        satisfiable_required_fields=satisfiable,
        missing_required_fields=missing,
        hard_gate_decision=decision,
        reason=reason,
    )


def _protein_design_gate(
    candidate: dict,
    step7_refs: list[dict[str, Any]],
    step8_result: dict | None,
    sequence_generation_intent: str,
) -> tuple[str, list[str], list[Step9HardGateAllowedTool], list[Step9HardGateBlockedTool], list[Step9ToolSchemaRequirement]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    allowed: list[Step9HardGateAllowedTool] = []
    blocked: list[Step9HardGateBlockedTool] = []
    schema_requirements: list[Step9ToolSchemaRequirement] = []

    sequence_generation_intent_flag = _is_sequence_generation_intent(sequence_generation_intent)
    complex_ready, complex_missing = _design_complex_ready(step8_result, candidate_id)

    for tool_name in sorted(_PROTEIN_DESIGN_TOOLS):
        required, satisfiable, missing = _required_args_missing(
            tool_name,
            candidate,
            step7_refs,
            step8_result,
            "protein_design",
        )
        schema_requirements.append(
                _tool_schema_requirement_record(
                    candidate,
                    candidate_id,
                    "protein_design",
                    tool_name,
                    required,
                    satisfiable,
                    missing,
                    _schema_required_args(tool_name)[1],
            )
        )
        if missing:
            if tool_name in {"NvidiaNIM_rfdiffusion", "NvidiaNIM_proteinmpnn"}:
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
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="protein_design",
                    reason=(
                        "tool_schema_unavailable"
                        if missing == ["tool_schema_unavailable"]
                        else "schema_required:" + ",".join(sorted(missing))
                    ),
                    rationale="required official TU args are missing",
                )
            )
            continue

        if tool_name in {"NvidiaNIM_rfdiffusion", "NvidiaNIM_proteinmpnn"}:
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

            if tool_name == "NvidiaNIM_rfdiffusion" and "contigs" not in satisfiable:
                blocked.append(
                    Step9HardGateBlockedTool(
                        candidate_id=candidate_id,
                        tool_name=tool_name,
                        lane_type="protein_design",
                        reason="schema_required:contigs",
                        rationale="requires deterministic contig template input",
                    )
                )
                continue

        if tool_name == "ESM_generate_protein_sequence":
            if not sequence_generation_intent_flag:
                blocked.append(
                    Step9HardGateBlockedTool(
                        candidate_id=candidate_id,
                        tool_name=tool_name,
                        lane_type="protein_design",
                        reason="sequence_generation_intent_missing",
                        rationale="sequence-generation intent not asserted in canonical/ raw query",
                    )
                )
                continue

            # Prompt sequence requirement is already handled by official required args.
        allowed.append(
            Step9HardGateAllowedTool(
                candidate_id=candidate_id,
                tool_name=tool_name,
                lane_type="protein_design",
                rationale=(
                    "true complex and schema inputs available for protein design"
                    if tool_name in {"NvidiaNIM_rfdiffusion", "NvidiaNIM_proteinmpnn"}
                    else "sequence-generation intent and official args are available"
                ),
            )
        )

    blocked_missing = [tool.reason for tool in blocked]
    has_any_tool_evidence = bool(_PROTEIN_DESIGN_TOOLS)
    status = "ready" if allowed else ("not_applicable" if has_any_tool_evidence and not blocked else "blocked")
    return status, blocked_missing, allowed, blocked, schema_requirements


def _map_variant_missing_reason(tool_name: str, missing: list[str], missing_set: set[str]) -> str:
    if "pdb_id" in missing_set:
        return "pdb_id_missing"
    if "variant" in missing_set or "variants" in missing_set or "mutation" in missing_set:
        return "variant_missing"
    if "chain" in missing_set:
        return "chain_missing"
    if "uniprot_id" in missing_set:
        return "uniprot_id_missing"
    if {"input_pdb", "structure", "structure_ref", "backbone", "path", "pdb_file"} & missing_set:
        return "complex_structure_missing"
    if "contigs" in missing_set:
        return "contigs_missing"
    if any(arg.endswith("_pdb") for arg in missing_set):
        return "complex_structure_missing"
    return "schema_required:" + ",".join(sorted(missing))


def _variant_evaluation_gate(
    candidate: dict,
    step7_refs: list[dict[str, Any]],
    step8_result: dict | None,
) -> tuple[str, list[str], list[Step9HardGateAllowedTool], list[Step9HardGateBlockedTool], list[Step9ToolSchemaRequirement]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    allowed: list[Step9HardGateAllowedTool] = []
    blocked: list[Step9HardGateBlockedTool] = []
    schema_requirements: list[Step9ToolSchemaRequirement] = []

    for tool_name in sorted(_VARIANT_EVALUATION_TOOLS):
        required, satisfiable, missing = _required_args_missing(
            tool_name,
            candidate,
            step7_refs,
            step8_result,
            "variant_evaluation",
            _schema_props(tool_name),
        )
        schema_source = _schema_required_args(tool_name)[1]
        schema_requirements.append(
            _tool_schema_requirement_record(
                candidate,
                candidate_id,
                "variant_evaluation",
                tool_name,
                required,
                satisfiable,
                missing,
                schema_source,
            )
        )

        if missing:
            missing_set = set(missing)
            reason = _map_variant_missing_reason(tool_name, missing, missing_set)
            if tool_name == "AlphaMissense_get_variant_score" and "variant" in missing_set and "uniprot_id" in missing_set:
                if has := _candidate_has_structured_variants(candidate):
                    if "uniprot_id" in missing_set:
                        reason = "uniprot_id_missing"
                elif "uniprot_id" in missing_set:
                    reason = "uniprot_id_missing"
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="variant_evaluation",
                    reason=reason,
                    rationale="required official TU args are missing",
                )
            )
            continue

        # For schema that returns satisfiable args, allow execution.
        blocked_reason = ""
        if tool_name == "AlphaMissense_get_variant_score":
            if "variant" in required and not _extract_explicit_variants(candidate):
                blocked_reason = "variant_missing"
            elif "uniprot_id" in required and not _candidate_has_uniprot_id(candidate):
                blocked_reason = "uniprot_id_missing"

        elif tool_name == "DynaMut2_predict_stability":
            if "pdb_id" in required and not _candidate_has_real_pdb_id(
                candidate, step8_result, candidate_id
            ):
                blocked_reason = "pdb_id_missing"
            elif "chain" in required and not _candidate_has_chain(candidate):
                blocked_reason = "chain_missing"
            elif "mutation" in required and not _extract_explicit_variants(candidate):
                blocked_reason = "variant_missing"

        elif tool_name == "ESM_score_variant_sae_batch":
            if (
                "variants" in required
                and not _extract_explicit_variants(candidate)
            ):
                blocked_reason = "variant_missing"
            elif "sequence" in required and not _candidate_step9_sequence_presence(candidate, step7_refs):
                blocked_reason = "sequence_missing"

        if blocked_reason:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool_name,
                    lane_type="variant_evaluation",
                    reason=blocked_reason,
                    rationale="required official TU args are missing",
                )
            )
            continue

        allowed.append(
            Step9HardGateAllowedTool(
                candidate_id=candidate_id,
                tool_name=tool_name,
                lane_type="variant_evaluation",
                rationale="required official TU inputs available for variant evaluation",
            )
        )

    status = "ready" if allowed else ("not_applicable" if not blocked else "blocked")
    blocked_missing = [entry.reason for entry in blocked]
    return status, blocked_missing, allowed, blocked, schema_requirements


def _compound_gate(
    candidate: dict,
) -> tuple[str, list[str], list[Step9HardGateAllowedTool], list[Step9HardGateBlockedTool], list[Step9ToolSchemaRequirement]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    allowed: list[Step9HardGateAllowedTool] = []
    blocked: list[Step9HardGateBlockedTool] = []
    schema_requirements: list[Step9ToolSchemaRequirement] = []

    has_smiles = _candidate_has_compound_smiles(candidate)
    has_name = _candidate_has_compound_name(candidate)
    has_identifier = _candidate_id_types(candidate, _COMPOUND_IDENTIFIER_TYPES)

    if not (has_smiles or has_name or has_identifier):
        for tool in sorted(_COMPOUND_TOOLS):
            required, satisfiable, missing = _required_args_missing(tool, candidate, [], None)
            schema_source = _schema_required_args(tool)[1]
            schema_requirements.append(
                _tool_schema_requirement_record(
                    candidate,
                    candidate_id,
                    "compound_screening",
                    tool,
                    required,
                    satisfiable,
                    missing,
                    schema_source,
                )
            )
            # Keep existing semantic: no compound evidence => not applicable.
        return "not_applicable", ["compound_input_missing"], allowed, blocked, schema_requirements

    for tool in sorted(_COMPOUND_TOOLS):
        required, satisfiable, missing = _required_args_missing(tool, candidate, [], None)
        schema_source = _schema_required_args(tool)[1]
        schema_requirements.append(
            _tool_schema_requirement_record(
                candidate,
                candidate_id,
                "compound_screening",
                tool,
                required,
                satisfiable,
                missing,
                schema_source,
            )
        )
        if missing:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool,
                    lane_type="compound_screening",
                    reason=(
                        "tool_schema_unavailable"
                        if missing == ["tool_schema_unavailable"]
                        else "schema_required:" + ",".join(sorted(missing))
                    ),
                    rationale="required official TU args are missing",
                )
            )
            continue

        required_set = set(required)
        tool_is_allowed = (
            (
                "zinc_id" in required_set
                and _candidate_has_zinc_id(candidate)
            )
            or (
                "smiles" in required_set
                and has_smiles
            )
            or (
                "query" in required_set
                and _candidate_has_smiles_like_query_text(candidate)
            )
            or (
                # No explicit query evidence required by schema (example:
                # ChEMBL_search_molecules), but keep existing expectation:
                # only attempt tools when compound context is present.
                not required_set
                and (has_smiles or has_name or has_identifier)
            )
            or (
                # For schema-required tier cases, keep generic compound
                # evidence gating and let threshold/tier missing be schema
                # blocking if not provided.
                "tier" in required_set
                and (has_smiles or has_name or has_identifier)
            )
        )
        if not tool_is_allowed:
            blocked.append(
                Step9HardGateBlockedTool(
                    candidate_id=candidate_id,
                    tool_name=tool,
                    lane_type="compound_screening",
                    reason="compound_input_missing",
                    rationale="compound lane evidence not sufficient for this tool",
                )
            )
            continue

        allowed.append(
            Step9HardGateAllowedTool(
                candidate_id=candidate_id,
                tool_name=tool,
                lane_type="compound_screening",
                rationale="compound evidence available",
            )
        )

    if not allowed and not blocked:
        return "blocked", ["compound_tool_evidence_gap"], allowed, blocked, schema_requirements
    return ("ready" if allowed else "blocked"), ([tool.reason for tool in blocked if tool.reason]), allowed, blocked, schema_requirements


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
    candidate: dict, step7_prepared_inputs: list[dict], step8_result: dict | None = None
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
    for material in _candidate_value_types(candidate, _EXPLICIT_PROTEIN_VARIANT_TYPES):
        mat_id = material.get("material_id")
        if isinstance(mat_id, str) and mat_id:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"material:{mat_id}",
                    provider="step_05",
                    field_type="variant",
                    value_kind=str(material.get("material_type") or "variant"),
                    status="available",
                )
            )
    for material in _candidate_value_types(candidate, {"design_contigs", "contigs"}):
        mat_id = material.get("material_id")
        if isinstance(mat_id, str) and mat_id:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"material:{mat_id}",
                    provider="step_05",
                    field_type="design_constraint",
                    value_kind="design_contigs",
                    status="available",
                )
            )
    for ident in candidate.get("identifiers") or []:
        if not isinstance(ident, dict):
            continue
        id_type = str(ident.get("id_type") or "").lower()
        value = ident.get("id_value")
        if not isinstance(value, str) or not value.strip():
            continue
        if id_type in _EXPLICIT_PROTEIN_VARIANT_TYPES:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"identifier:{id_type}:{value.strip()}",
                    provider="step_05",
                    field_type="variant",
                    value_kind=id_type,
                    status="available",
                )
            )
        if id_type in {"chain_id", "chain"}:
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"identifier:{id_type}:{value.strip()}",
                    provider="step_05",
                    field_type="chain",
                    value_kind="chain_id",
                    status="available",
                )
            )
        if id_type == "pdb_id" and _looks_like_pdb_id(value):
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"identifier:pdb_id:{value.strip()}",
                    provider="step_05",
                    field_type="structure_identifier",
                    value_kind="pdb_id",
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
    if isinstance(step8_result, dict):
        for i, ref in enumerate(_step8_complex_refs(step8_result, candidate_id)):
            if not isinstance(ref, dict):
                continue
            source_kind = str(ref.get("source_kind") or "").lower()
            if source_kind not in {"existing_pdb_complex", "predicted_complex", "existing_pdb_structure"}:
                continue
            source_ref = str(
                ref.get("storage_ref") or ref.get("pdb_id") or ref.get("source_ref") or ""
            ).strip()
            if not source_ref:
                continue
            out.append(
                Step9AvailableField(
                    candidate_id=candidate_id,
                    field_ref=f"step8_complex_ref:{candidate_id}:{i}",
                    provider="step_08",
                    field_type="structure",
                    value_kind="complex_structure_ref",
                    source_ref=source_ref,
                    status="available",
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

    ready_tool_count = len(
        list(dict.fromkeys(tool for lane in lanes for tool in lane.allowed_tools))
    )
    blocked_tool_count = len(
        list(dict.fromkeys(tool for lane in lanes for tool in lane.blocked_tools))
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
        ready_tool_count=ready_tool_count,
        blocked_tool_count=blocked_tool_count,
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
    # Keep production behavior deterministic in long-lived test/daemon processes.
    # Cache entries can carry monkeypatched schema state across tests.
    global _TOOL_REQUIRED_ARGS_CACHE, _TOOL_SCHEMA_SOURCE_CACHE
    _TOOL_REQUIRED_ARGS_CACHE = {}
    _TOOL_SCHEMA_SOURCE_CACHE = {}

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
    schema_requirements: list[Step9ToolSchemaRequirement] = []
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
            status, _, allowed, blocked, reqs = _compound_gate(candidate)
            schema_requirements.extend(reqs)
            # Keep legacy status semantics for no-evidence compound candidates.
            status_value = (
                "ready" if any(tool.tool_name != "compound_screening" for tool in allowed) else ("not_applicable" if status == "not_applicable" else "partial")
            )
            lane_statuses.append(
                _lane_status(candidate, "compound_screening", allowed, blocked, fields, status_value)
            )
            available_fields.extend(fields)
            allowed_tools.extend(allowed)
            blocked_tools.extend(blocked)
            continue

        fields = _available_fields_for_protein_candidate(candidate, prepared_inputs, step8_data)
        sequence_generation_intent = _sequence_generation_intent_text(compound_context_text)
        status, _, allowed, blocked, reqs = _protein_design_gate(
            candidate, step7_refs, step8_data, sequence_generation_intent
        )
        schema_requirements.extend(reqs)
        lane_statuses.append(
            _lane_status(candidate, "protein_design", allowed, blocked, fields, status)
        )
        available_fields.extend(fields)
        allowed_tools.extend(allowed)
        blocked_tools.extend(blocked)

        variant_status, _, variant_allowed, variant_blocked, variant_reqs = _variant_evaluation_gate(
            candidate, step7_refs, step8_data
        )
        schema_requirements.extend(variant_reqs)
        lane_statuses.append(
            _lane_status(candidate, "variant_evaluation", variant_allowed, variant_blocked, fields, variant_status)
        )
        allowed_tools.extend(variant_allowed)
        blocked_tools.extend(variant_blocked)

    summary = Step9ReadinessSummary(
        total_candidates=len(candidate_records),
        protein_design_candidates=sum(
            1 for candidate in candidate_records if isinstance(candidate, dict) and str(candidate.get("candidate_type") or "") != "compound_component"
        ),
        variant_evaluation_candidates=sum(
            1 for candidate in candidate_records if isinstance(candidate, dict) and str(candidate.get("candidate_type") or "") != "compound_component"
        ),
        variant_evaluation_ready_candidates=sum(
            1 for lane in lane_statuses
            if lane.lane_type == "variant_evaluation" and lane.status == "ready"
        ),
        variant_evaluation_blocked_candidates=sum(
            1 for lane in lane_statuses
            if lane.lane_type == "variant_evaluation" and lane.status == "blocked"
        ),
        variant_evaluation_not_applicable_candidates=sum(
            1 for lane in lane_statuses
            if lane.lane_type == "variant_evaluation" and lane.status == "not_applicable"
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
        "variant_evaluation_readiness": _aggregate_readiness_profile(lane_statuses, "variant_evaluation"),
        "step9_hard_gate_allowed_tools": allowed_tools,
        "step9_hard_gate_blocked_tools_with_reason": blocked_tools,
        "step9_tool_schema_requirements": schema_requirements,
        "step9_missing_inputs": all_missing,
    }
