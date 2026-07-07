"""Step 9 runtime planning resolver for protein/variant lanes.

This module resolves Stage 2 schema-to-field mappings into a compact,
raw-safe execution plan. It consumes ONLY `Step9InputProjection.input_fields`
(via `input_fields=`) and Stage 2 mapped tools — it never inspects Step 5/7/8
raw artifacts directly. Any real value resolution at execution time must go
through a field's `runtime_lookup` description; this module never calls Step
9 MCP tools.
"""

from __future__ import annotations

from typing import Any

from .step_09_input_projection import (
    MASKED_PROMPT_SEQUENCE_VALUE_KIND,
    assert_unique_input_field_refs,
)


STEP9_RUNTIME_EXECUTION_MODE = "planning_only"

_ACTIVE_LANES = {"protein_design", "variant_evaluation"}
_ACTIVE_TOOLS = {
    "NvidiaNIM_rfdiffusion",
    "NvidiaNIM_proteinmpnn",
    "ESM_generate_protein_sequence",
    "DynaMut2_predict_stability",
    "AlphaMissense_get_variant_score",
    "ESM_score_variant_sae_batch",
}

_STRUCTURE_ARGS = {"input_pdb", "pdb_file", "structure_ref", "structure", "backbone", "path"}
_SEQUENCE_ARGS = {
    "prompt_sequence",
    "sequence",
    "sequence_value",
    "sequence_1",
    "sequence_2",
    "sequence_3",
    "sequence_a",
    "sequence_b",
}
_VARIANT_ARGS = {"variant", "variants", "mutation", "mutations"}
_STRUCTURE_FIELD_TYPES = {"structure", "complex_structure"}


def plan_step9_runtime_execution(
    *,
    mapped_tools: list[Any],
    input_fields: list[Any],
) -> dict[str, Any]:
    """Resolve Stage 2 mapped tools into raw-safe Step 9 runtime plans.

    `input_fields` is `Step9InputProjection.input_fields` — the single
    field shape produced by `step_09_input_projection`. This function never
    reads Step 5/7/8 artifacts itself.
    """

    dumped_input_fields = _model_dump_list(input_fields)
    # `Step9InputProjection.input_fields` is contractually field_ref-unique
    # (see `step_09_input_projection._merge_duplicate_field_refs`). A
    # duplicate reaching here means an upstream contract violation; fail
    # fast instead of letting the dict comprehension below silently keep
    # whichever entry appears last in the list.
    assert_unique_input_field_refs(dumped_input_fields)
    fields_by_ref = {
        str(field.get("field_ref") or ""): field
        for field in dumped_input_fields
        if str(field.get("field_ref") or "")
    }
    execution_plan: list[dict[str, Any]] = []
    resolver_audit: list[dict[str, Any]] = []
    kwargs_contracts: list[dict[str, Any]] = []
    kwargs_contract_audit: list[dict[str, Any]] = []

    for mapped in _model_dump_list(mapped_tools):
        tool_name = str(mapped.get("tool_name") or "")
        lane_type = str(mapped.get("lane_type") or "")
        if lane_type not in _ACTIVE_LANES or tool_name not in _ACTIVE_TOOLS:
            continue

        plan_entry, audit_entries = _plan_tool(mapped, fields_by_ref)
        contract_entry, contract_audit_entries = _kwargs_contract_for_plan(plan_entry)
        execution_plan.append(plan_entry)
        resolver_audit.extend(audit_entries)
        kwargs_contracts.append(contract_entry)
        kwargs_contract_audit.extend(contract_audit_entries)

    resolved = [entry for entry in execution_plan if entry.get("can_resolve") is True]
    unresolved = [entry for entry in execution_plan if entry.get("can_resolve") is not True]
    return {
        "step9_runtime_execution_plan": execution_plan,
        "step9_runtime_resolved_tools": resolved,
        "step9_runtime_unresolved_tools": unresolved,
        "step9_runtime_resolver_audit": resolver_audit,
        "step9_runtime_kwargs_contracts": kwargs_contracts,
        "step9_runtime_kwargs_contract_audit": kwargs_contract_audit,
        "step9_runtime_execution_mode": STEP9_RUNTIME_EXECUTION_MODE,
    }


def _plan_tool(
    mapped: dict[str, Any],
    fields_by_ref: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool_name = str(mapped.get("tool_name") or "")
    lane_type = str(mapped.get("lane_type") or "")
    missing_required = [str(arg) for arg in mapped.get("missing_required_fields") or [] if str(arg)]
    unresolved_reasons: list[str] = []
    argument_plan: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    resolved_schema_args: set[str] = set()

    if mapped.get("can_invoke") is not True:
        unresolved_reasons.append("stage2_can_invoke_false")
    if missing_required:
        unresolved_reasons.append("stage2_missing_required_fields")

    for pair in mapped.get("argument_mappings") or []:
        if not isinstance(pair, dict):
            continue
        schema_arg = str(pair.get("schema_arg") or "")
        field_ref = str(pair.get("field_ref") or "")
        field = fields_by_ref.get(field_ref)
        arg_reasons: list[str] = []
        source_metadata: dict[str, Any] = {}
        if not schema_arg:
            arg_reasons.append("schema_arg_missing")
        if not field_ref:
            arg_reasons.append("field_ref_missing")
        if field is None:
            arg_reasons.append("field_ref_not_available")
        else:
            source_metadata = _source_metadata(field)
            if str(field.get("status") or "") != "available":
                arg_reasons.append("field_ref_not_available_status")
            if not field.get("can_resolve_at_runtime"):
                arg_reasons.append("field_not_runtime_resolvable")
            contract_reason = _contract_reason(tool_name, schema_arg, field)
            if contract_reason:
                arg_reasons.append(contract_reason)

        resolved = not arg_reasons
        if resolved:
            resolved_schema_args.add(schema_arg)
        else:
            unresolved_reasons.extend(arg_reasons)
        argument_plan.append(
            {
                "schema_arg": schema_arg,
                "source": "field_ref",
                "field_ref": field_ref,
                "literal_present": False,
                "source_metadata": source_metadata,
                "resolve_status": "resolved" if resolved else "unresolved",
                "unresolved_reasons": sorted(set(arg_reasons)),
            }
        )
        audit.append(
            _audit_entry(
                tool_name=tool_name,
                lane_type=lane_type,
                schema_arg=schema_arg,
                field_ref=field_ref,
                source="field_ref",
                status="resolved" if resolved else "unresolved",
                reason=",".join(sorted(set(arg_reasons))),
            )
        )

    for pair in mapped.get("argument_literals") or []:
        if not isinstance(pair, dict):
            continue
        schema_arg = str(pair.get("schema_arg") or "")
        arg_reasons = [] if schema_arg else ["schema_arg_missing"]
        resolved = not arg_reasons
        if resolved:
            resolved_schema_args.add(schema_arg)
        else:
            unresolved_reasons.extend(arg_reasons)
        argument_plan.append(
            {
                "schema_arg": schema_arg,
                "source": "literal",
                "literal_present": True,
                "literal_value": pair.get("literal_value"),
                "source_metadata": {"literal_source": "official_schema_literal"},
                "resolve_status": "resolved" if resolved else "unresolved",
                "unresolved_reasons": sorted(set(arg_reasons)),
            }
        )
        audit.append(
            _audit_entry(
                tool_name=tool_name,
                lane_type=lane_type,
                schema_arg=schema_arg,
                field_ref="",
                source="literal",
                status="resolved" if resolved else "unresolved",
                reason=",".join(sorted(set(arg_reasons))),
            )
        )

    contract_missing = _contract_missing_groups(tool_name, resolved_schema_args)
    if contract_missing:
        missing_required = sorted(set(missing_required) | set(contract_missing))
        unresolved_reasons.extend(_contract_missing_reasons(tool_name, contract_missing))

    unresolved_reasons = sorted(set(reason for reason in unresolved_reasons if reason))
    can_resolve = not unresolved_reasons
    skip_reason = "" if can_resolve else "runtime_resolution_failed"
    return (
        {
            "tool_name": tool_name,
            "lane_type": lane_type,
            "can_resolve": can_resolve,
            "would_execute": False,
            "execution_mode": STEP9_RUNTIME_EXECUTION_MODE,
            "argument_keys": sorted(arg for arg in resolved_schema_args if arg),
            "argument_plan": argument_plan,
            "missing_required_fields": missing_required,
            "unresolved_reasons": unresolved_reasons,
            "skip_reason": skip_reason,
        },
        audit,
    )


def _kwargs_contract_for_plan(plan_entry: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool_name = str(plan_entry.get("tool_name") or "")
    lane_type = str(plan_entry.get("lane_type") or "")
    can_build = plan_entry.get("can_resolve") is True
    kwargs_plan: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_args: set[str] = set()

    for item in plan_entry.get("argument_plan") or []:
        if not isinstance(item, dict):
            continue
        schema_arg = str(item.get("schema_arg") or "")
        if not schema_arg:
            continue
        runtime_arg = schema_arg
        seen_args.add(schema_arg)
        source = str(item.get("source") or "")
        resolve_status = str(item.get("resolve_status") or "unresolved")
        reason = ",".join(str(r) for r in (item.get("unresolved_reasons") or []) if str(r))
        if source == "field_ref":
            kwargs_plan.append(
                {
                    "runtime_arg": runtime_arg,
                    "source": "field_ref",
                    "schema_arg": schema_arg,
                    "field_ref": item.get("field_ref"),
                    "value_placeholder": "<resolved_at_execution_time>",
                    "source_metadata": item.get("source_metadata") or {},
                }
            )
        elif source == "literal":
            kwargs_plan.append(
                {
                    "runtime_arg": runtime_arg,
                    "source": "official_schema_literal",
                    "schema_arg": schema_arg,
                    "literal_value": item.get("literal_value"),
                }
            )
        else:
            kwargs_plan.append(
                {
                    "runtime_arg": runtime_arg,
                    "source": "unresolved",
                    "schema_arg": schema_arg,
                }
            )
        audit.append(
            _kwargs_audit_entry(
                tool_name=tool_name,
                lane_type=lane_type,
                schema_arg=schema_arg,
                runtime_arg=runtime_arg,
                source=kwargs_plan[-1]["source"],
                status=resolve_status,
                reason=reason,
            )
        )

    for missing_arg in plan_entry.get("missing_required_fields") or []:
        schema_arg = str(missing_arg or "")
        if not schema_arg or schema_arg in seen_args:
            continue
        kwargs_plan.append(
            {
                "runtime_arg": schema_arg,
                "source": "unresolved",
                "schema_arg": schema_arg,
            }
        )
        audit.append(
            _kwargs_audit_entry(
                tool_name=tool_name,
                lane_type=lane_type,
                schema_arg=schema_arg,
                runtime_arg=schema_arg,
                source="unresolved",
                status="unresolved",
                reason="missing_required_field",
            )
        )

    return (
        {
            "tool_name": tool_name,
            "lane_type": lane_type,
            "can_build_kwargs": can_build,
            "execution_mode": STEP9_RUNTIME_EXECUTION_MODE,
            "kwargs_keys": sorted(
                str(item.get("runtime_arg") or "")
                for item in kwargs_plan
                if str(item.get("runtime_arg") or "")
            ),
            "kwargs_plan": kwargs_plan,
            "unresolved_reasons": list(plan_entry.get("unresolved_reasons") or []),
        },
        audit,
    )


def _contract_reason(tool_name: str, schema_arg: str, field: dict[str, Any]) -> str:
    """Tool-specific runtime-contract checks, evaluated against the
    Step9InputField shape (`field_type` / `value_kind` / `supports_tool_args`).

    Stage 2 already gates `supports_tool_args` membership before a mapping
    reaches here; these checks catch tool-specific semantic requirements
    (e.g. DynaMut2's `pdb_id` arg must come from a true `pdb_id` field, not a
    generic complex-structure ref) that a bare `supports_tool_args` match
    does not fully capture.
    """
    arg = schema_arg.lower().strip()
    field_type = str(field.get("field_type") or "").lower()
    value_kind = str(field.get("value_kind") or "").lower()
    supports = {str(a).lower() for a in (field.get("supports_tool_args") or [])}

    if tool_name in {"NvidiaNIM_proteinmpnn", "NvidiaNIM_rfdiffusion"}:
        if arg in _STRUCTURE_ARGS and field_type not in _STRUCTURE_FIELD_TYPES:
            return "input_pdb_requires_true_complex_structure_ref"

    if tool_name == "NvidiaNIM_rfdiffusion":
        if arg == "contigs" and field_type != "design_constraint":
            return "contigs_missing_or_not_validated"

    if tool_name == "ESM_generate_protein_sequence":
        # `prompt_sequence` is ToolUniverse's masked GENERATION PROMPT arg,
        # never an ordinary complete heavy/light/target chain — a field must
        # be explicitly marked as a masked prompt (`value_kind ==
        # MASKED_PROMPT_SEQUENCE_VALUE_KIND`) to satisfy it. This is checked
        # independently of Stage 2's `supports_tool_args` gate (defense in
        # depth against a stale/incorrect field or a misbehaving LLM
        # mapping), and independently of `Step9InputProjection`, which today
        # never emits that value_kind at all.
        if arg == "prompt_sequence" and (
            field_type != "protein_sequence" or value_kind != MASKED_PROMPT_SEQUENCE_VALUE_KIND
        ):
            return "prompt_sequence_requires_masked_generation_prompt"

    if tool_name in {"ESM_generate_protein_sequence", "ESM_score_variant_sae_batch"}:
        if arg in _SEQUENCE_ARGS and field_type != "protein_sequence":
            return "sequence_field_ref_required"

    if tool_name == "AlphaMissense_get_variant_score":
        if arg in {"uniprot_id", "accession", "uniprot_accession"} and value_kind != "uniprot_id":
            return "uniprot_id_field_ref_required"
        if arg in _VARIANT_ARGS and field_type != "variant":
            return "variant_field_ref_required"

    if tool_name == "DynaMut2_predict_stability":
        if arg == "pdb_id" and value_kind != "pdb_id":
            return "true_pdb_id_field_ref_required"
        if arg == "chain" and field_type != "chain":
            return "chain_field_ref_required"
        if arg in _VARIANT_ARGS and field_type != "variant":
            return "variant_field_ref_required"

    if tool_name == "ESM_score_variant_sae_batch":
        if arg in _VARIANT_ARGS and field_type != "variant":
            return "variant_field_ref_required"

    if arg not in supports:
        return f"field_does_not_support_arg:{arg}"

    return ""


def _contract_missing_groups(tool_name: str, resolved_args: set[str]) -> list[str]:
    lowered = {arg.lower() for arg in resolved_args}
    missing: list[str] = []
    if tool_name == "NvidiaNIM_proteinmpnn":
        if not lowered & _STRUCTURE_ARGS:
            missing.append("input_pdb")
    elif tool_name == "NvidiaNIM_rfdiffusion":
        if not lowered & _STRUCTURE_ARGS:
            missing.append("input_pdb")
        if "contigs" not in lowered:
            missing.append("contigs")
    elif tool_name == "ESM_generate_protein_sequence":
        if not lowered & _SEQUENCE_ARGS:
            missing.append("prompt_sequence")
    elif tool_name == "AlphaMissense_get_variant_score":
        if not lowered & {"uniprot_id", "accession", "uniprot_accession"}:
            missing.append("uniprot_id")
        if not lowered & _VARIANT_ARGS:
            missing.append("variant")
    elif tool_name == "DynaMut2_predict_stability":
        for arg in ("pdb_id", "chain"):
            if arg not in lowered:
                missing.append(arg)
        if not lowered & _VARIANT_ARGS:
            missing.append("mutation")
    elif tool_name == "ESM_score_variant_sae_batch":
        if not lowered & _SEQUENCE_ARGS:
            missing.append("sequence")
        if not lowered & _VARIANT_ARGS:
            missing.append("variants")
    return missing


def _contract_missing_reasons(tool_name: str, missing: list[str]) -> list[str]:
    reasons: list[str] = []
    for arg in missing:
        if tool_name == "NvidiaNIM_rfdiffusion" and arg == "contigs":
            reasons.append("contigs_missing_or_not_validated")
        elif tool_name == "DynaMut2_predict_stability" and arg == "pdb_id":
            reasons.append("true_pdb_id_missing")
        elif arg in {"variant", "variants", "mutation"}:
            reasons.append("variant_missing")
        else:
            reasons.append(f"{arg}_missing")
    return reasons


def _source_metadata(field: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidate_id": field.get("candidate_id"),
        "field_ref": field.get("field_ref"),
        "source_step": field.get("source_step"),
        "field_type": field.get("field_type"),
        "value_kind": field.get("value_kind"),
        "status": field.get("status"),
    }
    runtime_lookup = field.get("runtime_lookup")
    if isinstance(runtime_lookup, dict) and runtime_lookup:
        out["runtime_lookup"] = runtime_lookup
    return {key: value for key, value in out.items() if value not in (None, "", {})}


def _audit_entry(
    *,
    tool_name: str,
    lane_type: str,
    schema_arg: str,
    field_ref: str,
    source: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "lane_type": lane_type,
        "schema_arg": schema_arg,
        "source": source,
        "field_ref": field_ref,
        "resolve_status": status,
        "candidate_value_persisted": False,
        "reason": reason,
    }


def _kwargs_audit_entry(
    *,
    tool_name: str,
    lane_type: str,
    schema_arg: str,
    runtime_arg: str,
    source: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "lane_type": lane_type,
        "schema_arg": schema_arg,
        "runtime_arg": runtime_arg,
        "source": source,
        "resolve_status": status,
        "reason": reason,
        "candidate_value_persisted": False,
    }


def _model_dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out
