"""Step 6 progressive disclosure + schema-to-field-ref selector.

Turn B keeps this separate from the shared selector used by other steps:
Step 6 now lets the LLM map official tool schema arguments to LLM-safe
candidate field refs, while raw values are resolved only at runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..llm.provider import LLMProvider
from ..mcp.client import MCPClient
from .step_06_available_fields import AvailableField, CandidateModalitySummary
from .step_06_capability_registry import (
    STEP_06_CAPABILITY_BY_TOOL,
    STEP_06_CAPABILITY_REGISTRY,
)
from .tool_selection_policy import (
    SELECTION_POLICY_VERSION,
    ToolInvocationPlan,
    build_compact_catalog,
    signature_schema_for,
)

logger = logging.getLogger(__name__)


STEP6_STAGE1_SCHEMA_MAPPING_SYSTEM_PROMPT = """You are selecting relevant Step 6 tools from a disclosed catalog.

Rules:
1. Use ONLY the disclosed compact_catalog. It is already Step 6 scoped and
   modality-filtered. Do not name tools outside it.
2. Select zero or more relevant tool_name values. A valid empty selection is allowed.
3. Do NOT construct arguments. Do NOT infer raw values. Candidate fields are
   digests and field refs only.
4. Return exactly JSON: {"selections":[{"tool_name":"...","selection_reason":"..."}]}.
No prose, no markdown, no tool calls.
""".strip()


STEP6_STAGE1_SCHEMA_MAPPING_USER_PROMPT = (
    "Select relevant Step 6 tools from the disclosed catalog for this candidate."
)


STEP6_STAGE2_SCHEMA_MAPPING_SYSTEM_PROMPT = """You are mapping selected Step 6 tool schemas to candidate field refs.

Rules:
1. Use only each tool's official full_schema and candidate_available_fields.
2. Output argument_mapping as schema_arg -> field_ref. Do not output raw values.
3. If required args cannot be satisfied by field refs, set can_invoke=false
   and list missing_required_fields. Do not guess identifiers or convert
   uploaded paths into pdb_id.
4. Return exactly JSON: {"tools":[{"tool_name":"...","can_invoke":true|false,
   "argument_mapping":{},"missing_required_fields":[],"argument_mapping_reason":"..."}]}.
No prose, no markdown, no tool calls.
""".strip()


STEP6_STAGE2_SCHEMA_MAPPING_USER_PROMPT = (
    "Map required schema arguments to candidate field refs for the selected tools."
)


class DisclosureResult(BaseModel):
    scoped_tool_names: list[str] = Field(default_factory=list)
    disclosed_tool_names: list[str] = Field(default_factory=list)
    hidden_tools_with_reason: list[dict] = Field(default_factory=list)
    disclosure_tags: list[str] = Field(default_factory=list)
    disclosure_summary: dict[str, Any] = Field(default_factory=dict)


def disclose_step6_tools(
    *,
    scoped_tool_names: set[str],
    modality_summary: CandidateModalitySummary,
    available_fields: list[AvailableField],
    user_query_summary: str = "",
) -> DisclosureResult:
    """Broad modality progressive disclosure, not final eligibility."""
    scoped = sorted(scoped_tool_names)
    tags = set(modality_summary.modality_tags or [])
    hidden: list[dict] = []
    disclose_tags: set[str] = set()
    lower_query = (user_query_summary or "").lower()

    small = any([
        modality_summary.has_payload_smiles,
        modality_summary.has_linker_smiles,
        modality_summary.has_compound_smiles,
        modality_summary.has_compound_identifier,
        bool(tags & {"payload_smiles", "linker_smiles", "compound_smiles", "compound_identifier"}),
        any(k in lower_query for k in ("admet", "drug-likeness", "druglikeness", "qed", "pains")),
    ])
    protein = any([
        modality_summary.has_antibody_heavy_sequence,
        modality_summary.has_antibody_light_sequence,
        modality_summary.has_antibody_sequence,
        modality_summary.has_antigen_sequence,
        modality_summary.has_protein_sequence,
        modality_summary.has_uploaded_fasta_ref,
        modality_summary.has_cdr3_ref_or_marker,
        bool(tags & {"antibody_sequence", "protein_sequence", "fasta_ref", "cdr3_marker"}),
        any(k in lower_query for k in ("sequence motif", "motif", "immunogenicity")),
    ])
    uniprot = any([
        modality_summary.has_uniprot_id,
        any(f.id_type == "uniprot_id" for f in available_fields),
        any(k in lower_query for k in ("epitope", "glycosylation", "glycoprotein", "ptm")),
    ])
    structure = any([
        modality_summary.has_pdb_id,
        modality_summary.has_uploaded_structure_ref,
        any(f.value_kind == "structure_ref" or f.id_type == "pdb_id" for f in available_fields),
        any(k in lower_query for k in ("interface quality", "interface", "structure quality", "pdb")),
    ])

    if small:
        disclose_tags.add("small_molecule")
    if protein:
        disclose_tags.add("protein_sequence")
    if uniprot:
        disclose_tags.add("uniprot_antigen")
    if structure:
        disclose_tags.add("structure")

    fail_open = modality_summary.ambiguous_or_unknown or not disclose_tags
    if fail_open:
        disclose_tags.add("ambiguous_modality_fail_open")

    disclosed: set[str] = set()
    for tool_name in scoped:
        cap = STEP_06_CAPABILITY_BY_TOOL.get(tool_name)
        if cap is None:
            hidden.append({"tool_name": tool_name, "reason": "not_in_step6_capability_registry"})
            continue
        if cap.lane_type is None:
            hidden.append({"tool_name": tool_name, "reason": cap.runtime_policy})
            continue
        if fail_open:
            disclosed.add(tool_name)
            continue
        if _cap_matches_modalities(cap, small=small, protein=protein, uniprot=uniprot, structure=structure):
            disclosed.add(tool_name)
        else:
            hidden.append({"tool_name": tool_name, "reason": "modality_not_present"})

    return DisclosureResult(
        scoped_tool_names=scoped,
        disclosed_tool_names=sorted(disclosed),
        hidden_tools_with_reason=hidden,
        disclosure_tags=sorted(disclose_tags),
        disclosure_summary={
            "small_molecule": small,
            "protein_sequence": protein,
            "uniprot_antigen": uniprot,
            "structure": structure,
            "fail_open": fail_open,
            "available_field_count": len(available_fields),
        },
    )


def select_step6_schema_mapped_invocations(
    *,
    agent_name: str,
    step_id: str,
    mcp_client: MCPClient,
    llm: LLMProvider,
    candidate_id: str,
    available_fields: list[AvailableField],
    modality_summary: CandidateModalitySummary,
    user_query_summary: str = "",
    deterministic_fallback: Callable[[list[str]], list[ToolInvocationPlan]] | None = None,
) -> tuple[dict[str, list[ToolInvocationPlan]], DisclosureResult, dict[str, Any]]:
    scoped = set(mcp_client.list_tools(agent_name=agent_name, step_id=step_id))
    disclosure = disclose_step6_tools(
        scoped_tool_names=scoped,
        modality_summary=modality_summary,
        available_fields=available_fields,
        user_query_summary=user_query_summary,
    )
    audit = {
        "stage1_call_status": "not_called",
        "stage2_call_status": "not_called",
        "stage1_selected_tools": [],
        "stage2_schema_survivors": [],
        "stage2_mapped_tools": [],
        "stage2_uninvokable_tools": [],
        "fallback_reason": None,
    }
    if not disclosure.disclosed_tool_names:
        return {}, disclosure, audit

    catalog = [
        entry for entry in build_compact_catalog(
            mcp_client=mcp_client, agent_name=agent_name, step_id=step_id
        )
        if entry.tool_name in set(disclosure.disclosed_tool_names)
    ]
    stage1_payload = {
        "task": "step6_schema_mapping_stage_1",
        "agent_name": agent_name,
        "step_id": step_id,
        "candidate_id": candidate_id,
        "compact_catalog": [entry.model_dump() for entry in catalog],
        "candidate_modality_summary": modality_summary.model_dump(),
        "candidate_available_fields": [field.model_dump() for field in available_fields],
        "user_query_summary": user_query_summary,
        "disclosure_tags": disclosure.disclosure_tags,
    }
    try:
        stage1 = llm.generate_json(
            STEP6_STAGE1_SCHEMA_MAPPING_USER_PROMPT,
            schema=stage1_payload,
            system=STEP6_STAGE1_SCHEMA_MAPPING_SYSTEM_PROMPT,
        )
        audit["stage1_call_status"] = "ok"
        raw_selections = (stage1 or {}).get("selections")
        if not isinstance(raw_selections, list):
            raise ValueError("stage1 selections missing or not list")
    except Exception as exc:  # noqa: BLE001
        audit["stage1_call_status"] = "fallback"
        audit["fallback_reason"] = f"stage1_provider_or_parse_error:{exc}"
        plans = deterministic_fallback(disclosure.disclosed_tool_names) if deterministic_fallback else []
        return _group_by_lane(plans), disclosure, audit

    selected_entries = _clean_stage1(raw_selections, set(disclosure.disclosed_tool_names))
    if not selected_entries:
        return {}, disclosure, audit
    audit["stage1_selected_tools"] = [entry["tool_name"] for entry in selected_entries]

    stage2_items: list[tuple[dict, dict]] = []
    schema_skipped: list[ToolInvocationPlan] = []
    for entry in selected_entries:
        tool_name = entry["tool_name"]
        schema = signature_schema_for(tool_name)
        if schema is None:
            schema_skipped.append(_skipped_plan(
                tool_name=tool_name,
                reason=entry.get("selection_reason") or "selected",
                warnings=["no callable signature found"],
                source="llm_stage1",
                missing=[],
            ))
        else:
            stage2_items.append((entry, schema))
    audit["stage2_schema_survivors"] = [entry["tool_name"] for entry, _schema in stage2_items]

    stage2_response_tools: dict[str, dict] = {}
    if stage2_items:
        stage2_payload = {
            "task": "step6_schema_mapping_stage_2",
            "agent_name": agent_name,
            "step_id": step_id,
            "candidate_id": candidate_id,
            "candidate_available_fields": [field.model_dump() for field in available_fields],
            "tools": [
                {
                    "tool_name": entry["tool_name"],
                    "full_schema": schema,
                    "selection_reason": entry.get("selection_reason") or "",
                }
                for entry, schema in stage2_items
            ],
        }
        try:
            stage2 = llm.generate_json(
                STEP6_STAGE2_SCHEMA_MAPPING_USER_PROMPT,
                schema=stage2_payload,
                system=STEP6_STAGE2_SCHEMA_MAPPING_SYSTEM_PROMPT,
            )
            audit["stage2_call_status"] = "ok"
            tools = (stage2 or {}).get("tools")
            if not isinstance(tools, list):
                raise ValueError("stage2 tools missing or not list")
            stage2_response_tools = {
                item.get("tool_name"): item for item in tools
                if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
            }
        except Exception as exc:  # noqa: BLE001
            audit["stage2_call_status"] = "fallback"
            audit["fallback_reason"] = f"stage2_provider_or_parse_error:{exc}"
            stage2_response_tools = {
                entry["tool_name"]: _deterministic_mapping_response(entry["tool_name"], schema, available_fields)
                for entry, schema in stage2_items
            }

    plans = list(schema_skipped)
    for entry, schema in stage2_items:
        tool_name = entry["tool_name"]
        response = stage2_response_tools.get(tool_name)
        if response is None:
            response = {
                "tool_name": tool_name,
                "can_invoke": False,
                "argument_mapping": {},
                "missing_required_fields": list(schema.get("required") or []),
                "argument_mapping_reason": "stage2 omitted selected tool",
            }
        plan = _plan_from_stage2_response(
            entry=entry,
            schema=schema,
            response=response,
            available_fields=available_fields,
            source="llm_stage2" if audit["stage2_call_status"] == "ok" else "deterministic_mapping",
        )
        if plan.validation_status == "skipped":
            audit["stage2_uninvokable_tools"].append(tool_name)
        else:
            audit["stage2_mapped_tools"].append(tool_name)
        plans.append(plan)

    return _group_by_lane(plans), disclosure, audit


def _cap_matches_modalities(cap: Any, *, small: bool, protein: bool, uniprot: bool, structure: bool) -> bool:
    if small and cap.lane_type in {"payload_linker_compound_liability", "compound_bioactivity_prior_context"}:
        return True
    if protein and cap.lane_type == "antibody_protein_sequence_liability":
        return True
    if protein and cap.tool_name in {"iPTMnet_get_ptm_sites", "IEDB_predict_mhci_binding"}:
        return True
    if uniprot and cap.lane_type == "antigen_protein_feature_context":
        return True
    if structure and cap.lane_type == "structure_interface_quality":
        return True
    return False


def _clean_stage1(raw: list[dict], allowed: set[str]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name")
        if not isinstance(name, str) or name not in allowed or name in seen:
            continue
        seen.add(name)
        out.append(item)
    return out


def _plan_from_stage2_response(
    *,
    entry: dict,
    schema: dict,
    response: dict,
    available_fields: list[AvailableField],
    source: Literal["llm_stage2", "deterministic_mapping"],
) -> ToolInvocationPlan:
    tool_name = entry["tool_name"]
    raw_mapping = response.get("argument_mapping") or {}
    mapping = {
        str(arg): str(ref)
        for arg, ref in raw_mapping.items()
        if isinstance(arg, str) and isinstance(ref, str)
    } if isinstance(raw_mapping, dict) else {}
    valid_mapping, warnings = _validate_field_mapping(mapping, schema, available_fields)
    required = list(schema.get("required") or [])
    missing = [
        arg for arg in required
        if arg not in valid_mapping or not valid_mapping.get(arg)
    ]
    response_missing = [
        str(x) for x in (response.get("missing_required_fields") or [])
        if isinstance(x, str)
    ]
    missing = sorted(set(missing + response_missing))
    can_invoke = bool(response.get("can_invoke")) and not missing and bool(valid_mapping)
    if not can_invoke:
        status: Literal["ok", "warning", "skipped"] = "skipped"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    audit = [
        {
            "tool_name": tool_name,
            "schema_arg": arg,
            "field_ref": ref,
            "mapping_source": source,
            "argument_mapping_reason": response.get("argument_mapping_reason") or "",
        }
        for arg, ref in sorted(valid_mapping.items())
    ]
    try:
        return ToolInvocationPlan(
            tool_name=tool_name,
            selection_reason=entry.get("selection_reason") or "",
            arguments={},
            argument_field_refs=valid_mapping,
            argument_mapping_audit=audit,
            argument_construction_reason=response.get("argument_mapping_reason") or "",
            selected_by="llm" if source == "llm_stage2" else "deterministic_fallback",
            tool_selection_source="llm_stage1",
            argument_construction_source=source,
            validation_status=status,
            validation_warnings=warnings,
            missing_required_fields=missing,
        )
    except ValidationError:
        return _skipped_plan(
            tool_name=tool_name,
            reason="model_validation_failed",
            warnings=warnings,
            source="llm_stage1",
            missing=missing,
        )


def _validate_field_mapping(
    mapping: dict[str, str], schema: dict, fields: list[AvailableField]
) -> tuple[dict[str, str], list[str]]:
    field_by_ref = {field.field_ref: field for field in fields}
    properties = schema.get("properties") or {}
    out: dict[str, str] = {}
    warnings: list[str] = []
    for arg, field_ref in mapping.items():
        if arg not in properties:
            warnings.append(f"argument `{arg}` not in schema; dropping")
            continue
        field = field_by_ref.get(field_ref)
        if field is None:
            warnings.append(f"field_ref for `{arg}` not available; dropping")
            continue
        if not _field_can_satisfy_arg(arg, field):
            warnings.append(f"field_ref for `{arg}` has incompatible value_kind; dropping")
            continue
        out[arg] = field_ref
    return out, warnings


def _field_can_satisfy_arg(arg: str, field: AvailableField) -> bool:
    lowered = arg.lower()
    if lowered in {"smiles", "canonical_smiles"}:
        return field.value_kind == "smiles"
    if lowered in {"sequence", "protein_sequence"}:
        return field.field_type == "protein_sequence" and field.value_kind == "protein_sequence"
    if lowered in {"accession", "uniprot_id", "uniprot_accession", "uniprot_ac"}:
        return field.id_type == "uniprot_id"
    if lowered in {"molecule_chembl_id", "chembl_id"}:
        return field.id_type == "chembl_id"
    if lowered in {"pdb_id", "pdb"}:
        return field.id_type == "pdb_id"
    if lowered in {"pdb_id_or_path", "structure_file", "structure_ref"}:
        return field.id_type == "pdb_id" or field.value_kind == "structure_ref"
    if lowered in {"query"}:
        return field.value_kind in {"smiles", "uniprot_id", "chembl_id", "pdb_id", "protein_sequence"}
    return False


def _deterministic_mapping_response(tool_name: str, schema: dict, fields: list[AvailableField]) -> dict:
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for arg in schema.get("required") or []:
        match = next((f for f in fields if _field_can_satisfy_arg(str(arg), f)), None)
        if match is None:
            missing.append(str(arg))
        else:
            mapping[str(arg)] = match.field_ref
    if not schema.get("required"):
        for arg in (schema.get("properties") or {}):
            match = next((f for f in fields if _field_can_satisfy_arg(str(arg), f)), None)
            if match is not None:
                mapping[str(arg)] = match.field_ref
    return {
        "tool_name": tool_name,
        "can_invoke": not missing and bool(mapping),
        "argument_mapping": mapping,
        "missing_required_fields": missing,
        "argument_mapping_reason": "deterministic fallback after malformed Stage 2",
    }


def _skipped_plan(
    *,
    tool_name: str,
    reason: str,
    warnings: list[str],
    source: str,
    missing: list[str],
) -> ToolInvocationPlan:
    return ToolInvocationPlan(
        tool_name=tool_name,
        selection_reason=reason,
        arguments={},
        selected_by="llm",
        tool_selection_source="llm_stage1",
        argument_construction_source="none",
        validation_status="skipped",
        validation_warnings=warnings,
        missing_required_fields=missing,
        argument_mapping_audit=[{
            "tool_name": tool_name,
            "mapping_source": source,
            "can_invoke": False,
            "missing_required_fields": missing,
        }],
    )


def _group_by_lane(plans: list[ToolInvocationPlan]) -> dict[str, list[ToolInvocationPlan]]:
    out: dict[str, list[ToolInvocationPlan]] = {}
    for plan in plans:
        cap = STEP_06_CAPABILITY_BY_TOOL.get(plan.tool_name)
        if cap is None or cap.lane_type is None:
            continue
        if cap.runtime_policy in {"future", "unsupported_for_adc_step6"}:
            skipped = plan.model_copy(update={
                "validation_status": "skipped",
                "validation_warnings": [
                    *plan.validation_warnings,
                    f"runtime_policy={cap.runtime_policy}",
                ],
            })
            out.setdefault(cap.lane_type, []).append(skipped)
            continue
        out.setdefault(cap.lane_type, []).append(plan)
    return out


def step6_live_capability_fallback(tool_names: list[str]) -> list[ToolInvocationPlan]:
    plans: list[ToolInvocationPlan] = []
    for name in tool_names:
        cap = STEP_06_CAPABILITY_BY_TOOL.get(name)
        if cap is None or cap.lane_type is None:
            continue
        if cap.runtime_policy != "live_wired":
            continue
        plans.append(
            ToolInvocationPlan(
                tool_name=name,
                selection_reason="deterministic fallback after provider error",
                arguments={},
                selected_by="deterministic_fallback",
                tool_selection_source="deterministic_fallback",
                argument_construction_source="none",
                validation_status="skipped",
                validation_warnings=["fallback selection requires Stage 2 mapping before execution"],
            )
        )
    return plans
