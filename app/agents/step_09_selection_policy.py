"""Step 9 LLM relevance selection and schema mapping.

These selectors are audit-only for the current Step 9 iteration. Stage 1
selects relevant tools from the FULL active Step 9 tool catalog (all 6 active
tools, independent of any hard-gate/readiness computation). Stage 2 maps
selected tool schemas to `Step9InputProjection.input_fields` field refs /
official schema literals. Neither stage executes Step 9 protein/variant
tools.

Architecture note: prior iterations built the Stage 1 catalog from a
per-candidate hard-gate `step9_hard_gate_allowed_tools` list, and Stage 2
consumed `step9_available_fields` computed by the same hard gate. Both stages
now consume ONLY the centralized `step_09_input_projection` output — the hard
gate (`step_09_available_fields.py`) remains for backward-compatible audit
fields only and no longer drives catalog/selection/mapping.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm.provider import LLMProvider
from .step_09_input_projection import _compact_text, assert_unique_input_field_refs
from .tool_selection_policy import signature_schema_for


STEP9_STAGE1_PROMPT_CACHE_LAYOUT_VERSION = "step9_stage1_v2"
STEP9_STAGE2_PROMPT_CACHE_LAYOUT_VERSION = "step9_stage2_v2"


# ── Active Step 9 tool catalog (production authority; NOT hard-gate driven) ─

ACTIVE_STEP9_TOOLS: dict[str, dict[str, str]] = {
    "NvidiaNIM_rfdiffusion": {
        "lane_type": "protein_design",
        "short_description": "RFdiffusion backbone-conditioned structure generation",
    },
    "NvidiaNIM_proteinmpnn": {
        "lane_type": "protein_design",
        "short_description": "ProteinMPNN backbone-conditioned sequence design",
    },
    "ESM_generate_protein_sequence": {
        "lane_type": "protein_design",
        "short_description": "ESM protein sequence completion/generation",
    },
    "DynaMut2_predict_stability": {
        "lane_type": "variant_evaluation",
        "short_description": "DynaMut2 mutation stability impact prediction",
    },
    "AlphaMissense_get_variant_score": {
        "lane_type": "variant_evaluation",
        "short_description": "AlphaMissense variant pathogenicity scoring",
    },
    "ESM_score_variant_sae_batch": {
        "lane_type": "variant_evaluation",
        "short_description": "ESM SAE batch variant scoring",
    },
}


STEP9_STAGE1_SYSTEM_PROMPT = """You select relevant Step 9 tools.

Rules:
1. Select only from the given catalog (the full active Step 9 tool set).
2. A tool being in the catalog does not mean its inputs are ready yet —
   judge relevance to the user's request, not input availability.
3. Use the dynamic query_summary / projection overview only to judge relevance.
4. Do not construct tool arguments.
5. Do not invent variants, mutations, contigs, sequences, PDB IDs, structures, compounds, thresholds, or tiers.
6. Avoid redundant tools; if tools answer the same question, choose the best one.
7. Return exactly one valid JSON object with this shape:
{
  "selections": [
    {
      "tool_name": "string from the catalog",
      "lane_type": "protein_design or variant_evaluation",
      "selection_reason": "short reason"
    }
  ]
}
8. If no catalog tool is relevant, return {"selections": []}.

Relevant-tool example:
Input situation: catalog contains AlphaMissense_get_variant_score; query_summary asks to evaluate a protein variant; projection overview shows UniProt and variant input fields.
Output:
{
  "selections": [
    {
      "tool_name": "AlphaMissense_get_variant_score",
      "lane_type": "variant_evaluation",
      "selection_reason": "Variant scoring is relevant and required inputs are available."
    }
  ]
}

No-relevant-tool example:
Input situation: no catalog tool is relevant.
Output:
{"selections": []}
""".strip()


STEP9_STAGE1_USER_PROMPT = (
    "Select relevant Step 9 tools from the active tool catalog. Return only "
    "tool_name, lane_type, and selection_reason; do not construct arguments."
)


STEP9_STAGE2_SYSTEM_PROMPT = """You map selected Step 9 tool schemas to Step9InputProjection field refs.

Rules:
1. Use only selected tools.
2. Use only official full_schema / required_fields and step9_input_fields.
3. Do not output raw values.
4. Do not invent variants, mutations, contigs, thresholds, tiers, PDB IDs, structures, sequences, or compounds.
5. Output schema_arg -> field_ref using list-of-pairs.
6. Use argument_literals for official schema literals — enum/singleton/default scalars AND any argument whose official schema type is an array or object. Encode every literal value as a JSON string in `literal_value_json` (a JSON scalar, array, or object). If the official schema requires an array/object argument, do NOT map an ordinary scalar field_ref to it; supply the official value as a JSON array/object in `literal_value_json`.
7. A field_ref only satisfies a schema_arg when the field's supports_tool_args includes that schema_arg.
8. If required args cannot be satisfied, can_invoke=false and list missing_required_fields.
9. Return exactly one valid JSON object with this shape:
{
  "tools": [
    {
      "tool_name": "string from selected tools",
      "lane_type": "protein_design or variant_evaluation",
      "can_invoke": true,
      "argument_mappings": [
        {"schema_arg": "official schema arg", "field_ref": "step9_input_fields field_ref"}
      ],
      "argument_literals": [
        {"schema_arg": "official schema arg", "literal_value_json": "<official value as JSON text>"}
      ],
      "missing_required_fields": [],
      "skip_reason": "",
      "argument_mapping_reason": "short reason"
    }
  ]
}
10. Return one tools[] item for every selected tool.
11. Do not map storage paths or uploaded file paths as PDB IDs.
12. ESM_generate_protein_sequence's prompt_sequence is a masked GENERATION PROMPT (contains "_" mask positions to complete), not an ordinary complete protein sequence. Do not map an ordinary heavy/light/target protein_sequence field to prompt_sequence; leave it in missing_required_fields instead.

Example:
Input situation: selected tool DynaMut2_predict_stability requires pdb_id, chain, mutation. step9_input_fields includes {"field_ref": "identifier:mutation:V777L", "field_type": "variant", "value_kind": "mutation", "supports_tool_args": ["variant","variants","mutation","mutations"]}. No field's supports_tool_args includes "pdb_id" or "chain".
Output:
{
  "tools": [
    {
      "tool_name": "DynaMut2_predict_stability",
      "lane_type": "variant_evaluation",
      "can_invoke": false,
      "argument_mappings": [
        {"schema_arg": "mutation", "field_ref": "identifier:mutation:V777L"}
      ],
      "argument_literals": [],
      "missing_required_fields": ["pdb_id", "chain"],
      "skip_reason": "missing_required_fields",
      "argument_mapping_reason": "Mutation is available, but no field supports pdb_id or chain."
    }
  ]
}

Example (array/object schema argument as a JSON literal):
Input situation: selected tool ESM_score_variant_sae_batch; official schema requires `sequence` as a string and `variants` as an array of objects with `position`, `ref_aa`, `alt_aa`; available fields include {"field_ref": "step7_sequence:file_her2_p04626_fasta", "field_type": "protein_sequence", "supports_tool_args": ["sequence"]} and {"field_ref": "identifier:variant:V777L", "field_type": "variant", "supports_tool_args": ["variant","variants","mutation","mutations"]}.
Output:
{
  "tools": [
    {
      "tool_name": "ESM_score_variant_sae_batch",
      "lane_type": "variant_evaluation",
      "can_invoke": true,
      "argument_mappings": [
        {"schema_arg": "sequence", "field_ref": "step7_sequence:file_her2_p04626_fasta"}
      ],
      "argument_literals": [
        {"schema_arg": "variants", "literal_value_json": "[{\\"position\\":777,\\"ref_aa\\":\\"V\\",\\"alt_aa\\":\\"L\\"}]"},
        {"schema_arg": "model", "literal_value_json": "\\"esmc-6b-2024-12\\""}
      ],
      "missing_required_fields": [],
      "skip_reason": "",
      "argument_mapping_reason": "sequence is mapped from the projected protein sequence field; variants is supplied as the official array-of-objects JSON literal required by the tool schema."
    }
  ]
}
""".strip()


STEP9_STAGE2_USER_PROMPT = (
    "Map selected Step 9 tool schemas to step9_input_fields field refs and "
    "official schema literals. Return list-of-pairs only."
)


class Step9Stage1SelectionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    lane_type: str
    selection_reason: str = ""


class Step9Stage1SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_tool_names: list[str] = Field(default_factory=list)
    selected_tools: list[Step9Stage1SelectionAudit] = Field(default_factory=list)
    rejected_tools_with_reason: list[dict[str, str]] = Field(default_factory=list)
    selection_source: str = "llm_stage1"
    prompt_cache_layout_version: str = STEP9_STAGE1_PROMPT_CACHE_LAYOUT_VERSION


class Step9Stage2ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    field_ref: str


class Step9Stage2ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    # The resolved official-schema literal. Scalars come from the legacy
    # enum/const/default path; array/object values come from a validated
    # `literal_value_json` (parsed once, in `validate_step9_stage2_mapping`).
    # This is the post-validation artifact model, not an LLM strict-output
    # schema, so `Any` is acceptable here.
    literal_value: Any = None


class Step9Stage2MappedTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    lane_type: str
    can_invoke: bool
    argument_mappings: list[Step9Stage2ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[Step9Stage2ArgumentLiteral] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    skip_reason: str = ""
    argument_mapping_reason: str = ""


class Step9Stage2MappingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_survivors: list[str] = Field(default_factory=list)
    mapped_tools: list[Step9Stage2MappedTool] = Field(default_factory=list)
    uninvokable_tools: list[str] = Field(default_factory=list)
    uninvokable_tool_details: list[dict[str, Any]] = Field(default_factory=list)
    argument_mapping_audit: list[dict[str, Any]] = Field(default_factory=list)
    prompt_cache_layout_version: str = STEP9_STAGE2_PROMPT_CACHE_LAYOUT_VERSION


# `ESM_generate_protein_sequence` currently has no official ToolUniverse spec
# loaded in this environment's tool inventory, so `signature_schema_for` falls
# back to introspecting its local Python binding — which gives every arg
# (including `prompt_sequence`) a default value, so the signature-fallback
# schema reports `required: []`. Without this override, Stage 2 would report
# `can_invoke=True` with zero argument mappings whenever no field can satisfy
# `prompt_sequence` (e.g. only an ordinary heavy/light/target sequence is
# available), instead of the correct `can_invoke=False` /
# `missing_required_fields=["prompt_sequence"]`. Scoped to exactly this one
# tool/arg — it does not change required-field computation for any other
# Step 9 tool, and it never marks a field as satisfying the arg (that gate is
# still `_step9_field_can_satisfy_arg`'s `supports_tool_args` check).
_FALLBACK_REQUIRED_ARGS_OVERRIDE: dict[str, list[str]] = {
    "ESM_generate_protein_sequence": ["prompt_sequence"],
}

_ESM_SCORE_VARIANTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["position", "ref_aa", "alt_aa"],
        "properties": {
            "position": {"type": "integer"},
            "ref_aa": {"type": "string"},
            "alt_aa": {"type": "string"},
        },
    },
}


def _step9_signature_schema_for(tool_name: str) -> dict[str, Any]:
    schema = signature_schema_for(tool_name) or {}
    return _apply_step9_schema_overrides(tool_name, schema)


def _apply_step9_schema_overrides(tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "ESM_score_variant_sae_batch":
        return schema
    out = dict(schema)
    properties = dict(out.get("properties") if isinstance(out.get("properties"), dict) else {})
    existing_variants = (
        dict(properties.get("variants")) if isinstance(properties.get("variants"), dict) else {}
    )
    variants_schema = dict(_ESM_SCORE_VARIANTS_SCHEMA)
    variants_schema["items"] = dict(_ESM_SCORE_VARIANTS_SCHEMA["items"])
    variants_schema["items"]["properties"] = dict(_ESM_SCORE_VARIANTS_SCHEMA["items"]["properties"])
    if "description" in existing_variants:
        variants_schema["description"] = existing_variants["description"]
    properties["variants"] = variants_schema
    out["properties"] = properties
    required = [str(arg) for arg in (out.get("required") or []) if isinstance(arg, str)]
    if "variants" not in required:
        required.append("variants")
    out["required"] = required
    return out


def _required_fields_with_fallback(tool_name: str, schema: dict[str, Any]) -> list[str]:
    required = [str(arg) for arg in (schema.get("required") or []) if isinstance(arg, str)]
    if not required:
        required = list(_FALLBACK_REQUIRED_ARGS_OVERRIDE.get(tool_name, []))
    return required


def _tool_schema_source(tool_name: str) -> str:
    try:
        from ..mcp import tooluniverse_adapter

        if tooluniverse_adapter.get_tool_specification(tool_name) is not None:
            return "tooluniverse_or_signature"
    except Exception:  # noqa: BLE001
        pass
    return "signature"


def _official_descriptions(tool_names: list[str]) -> dict[str, str]:
    """Best-effort official ToolUniverse descriptions for Step 9 Stage 1.

    Stage 1 should describe tools with ToolUniverse's own wording when
    available. Any metadata failure falls back to the local active-tool
    description without changing the visible tool set.
    """
    if not tool_names:
        return {}
    try:
        from ..mcp import tooluniverse_adapter

        specs = tooluniverse_adapter.get_tool_specifications(tool_names)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for name, spec in (specs or {}).items():
        desc = (spec or {}).get("description")
        if isinstance(desc, str) and desc.strip():
            out[name] = desc.strip()
    return out


def build_step9_stage1_catalog() -> list[dict[str, Any]]:
    """Return the FULL active Step 9 tool catalog — always all 6 active
    tools, sorted by lane then tool name for a stable prompt-cache prefix.

    This is independent of any per-candidate hard-gate/readiness state; the
    hard gate no longer drives which tools Stage 1 sees.
    """
    sorted_names = sorted(
        ACTIVE_STEP9_TOOLS,
        key=lambda name: (ACTIVE_STEP9_TOOLS[name]["lane_type"], name),
    )
    official_descriptions = _official_descriptions(sorted_names)
    catalog: list[dict[str, Any]] = []
    for tool_name in sorted_names:
        meta = ACTIVE_STEP9_TOOLS[tool_name]
        schema = _step9_signature_schema_for(tool_name)
        required = _required_fields_with_fallback(tool_name, schema)
        short_description = official_descriptions.get(tool_name) or meta["short_description"]
        catalog.append(
            {
                "tool_name": tool_name,
                "lane_type": meta["lane_type"],
                "short_description": short_description,
                "required_fields": required,
                "schema_source": _tool_schema_source(tool_name),
            }
        )
    return catalog


def build_step9_stage1_payload(
    *,
    candidate_id: str,
    catalog: list[dict[str, Any]],
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the schema payload for the Step 9 Stage 1 LLM call."""

    return {
        "task": "step9_tool_selection_stage_1",
        "compact_catalog": catalog,
        "candidate_id": candidate_id,
        "query_summary": _redact_query_summary(projection.get("query_summary") or {}),
        "projection_handoff_summary": projection.get("handoff_summary") or {},
        "projection_missing_inputs": list(projection.get("missing_inputs") or []),
        "projection_input_overview": _compact_input_field_overview(
            projection.get("input_fields") or []
        ),
    }


def select_step9_stage1_tools(
    *,
    llm: LLMProvider,
    projection: dict[str, Any],
    candidate_id: str = "all_candidates",
) -> Step9Stage1SelectionResult:
    """Run Stage 1 and validate selections against the active tool catalog."""

    catalog = build_step9_stage1_catalog()
    allowed = {(entry["tool_name"], entry["lane_type"]) for entry in catalog}
    allowed_names = {entry["tool_name"] for entry in catalog}
    payload = build_step9_stage1_payload(
        candidate_id=candidate_id,
        catalog=catalog,
        projection=projection,
    )
    response = llm.generate_json(
        STEP9_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE1_SYSTEM_PROMPT,
    )

    selected: list[Step9Stage1SelectionAudit] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, entry in enumerate((response or {}).get("selections") or []):
        if not isinstance(entry, dict):
            rejected.append({"tool_name": f"index:{i}", "reason": "selection_not_object"})
            continue
        tool_name = entry.get("tool_name")
        lane_type = entry.get("lane_type")
        if not isinstance(tool_name, str) or not tool_name:
            rejected.append({"tool_name": f"index:{i}", "reason": "missing_tool_name"})
            continue
        if not isinstance(lane_type, str) or not lane_type:
            rejected.append({"tool_name": tool_name, "reason": "missing_lane_type"})
            continue
        key = (tool_name, lane_type)
        if key not in allowed:
            reason = (
                "tool_not_in_active_catalog"
                if tool_name not in allowed_names
                else "tool_lane_not_in_active_catalog"
            )
            rejected.append({"tool_name": tool_name, "reason": reason})
            continue
        if key in seen:
            rejected.append({"tool_name": tool_name, "reason": "duplicate_selection"})
            continue
        seen.add(key)
        selected.append(
            Step9Stage1SelectionAudit(
                tool_name=tool_name,
                lane_type=lane_type,
                selection_reason=str(entry.get("selection_reason") or ""),
            )
        )

    return Step9Stage1SelectionResult(
        catalog_tool_names=[entry["tool_name"] for entry in catalog],
        selected_tools=selected,
        rejected_tools_with_reason=rejected,
        selection_source="llm_stage1",
    )


def build_step9_stage2_payload(
    *,
    candidate_id: str,
    selected_tools: list[Step9Stage1SelectionAudit] | list[dict[str, Any]],
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the Stage 2 schema payload.

    The stable selected-tool schema block is ``tools``. Candidate-specific
    field refs and Stage 1 reasons stay as dynamic payload keys; the shared
    prompt renderer splits them accordingly.
    """

    selected = [_selection_dict(item) for item in selected_tools]
    selected_pairs = {
        (item["tool_name"], item["lane_type"])
        for item in selected
        if item.get("tool_name") and item.get("lane_type")
    }
    tools: list[dict[str, Any]] = []
    for tool_name, lane_type in sorted(selected_pairs, key=lambda p: (p[1], p[0])):
        schema = _step9_signature_schema_for(tool_name)
        required = _required_fields_with_fallback(tool_name, schema)
        tools.append(
            {
                "tool_name": tool_name,
                "lane_type": lane_type,
                "full_schema": _compact_schema(schema, tool_name=tool_name),
                "required_fields": required,
                "schema_source": _tool_schema_source(tool_name),
            }
        )

    return {
        "task": "step9_tool_schema_mapping_stage_2",
        "candidate_id": candidate_id,
        "tools": tools,
        "selected_tools": selected,
        "step9_input_fields": _compact_input_fields_for_stage2(
            projection.get("input_fields") or []
        ),
        "query_summary": _redact_query_summary(projection.get("query_summary") or {}),
    }


def select_step9_stage2_mappings(
    *,
    llm: LLMProvider,
    projection: dict[str, Any],
    selected_tools: list[Step9Stage1SelectionAudit],
    candidate_id: str = "all_candidates",
) -> Step9Stage2MappingResult:
    payload = build_step9_stage2_payload(
        candidate_id=candidate_id,
        selected_tools=selected_tools,
        projection=projection,
    )
    stage2_tools = [tool for tool in payload["tools"] if isinstance(tool, dict)]
    schema_survivors = [str(tool.get("tool_name") or "") for tool in stage2_tools]
    if not stage2_tools:
        return Step9Stage2MappingResult(schema_survivors=[])

    try:
        response = llm.generate_json(
            STEP9_STAGE2_USER_PROMPT,
            schema=payload,
            system=STEP9_STAGE2_SYSTEM_PROMPT,
        )
        response_tools = [
            item for item in (response or {}).get("tools") or [] if isinstance(item, dict)
        ]
    except Exception as exc:  # noqa: BLE001
        response_tools = [
            {
                "tool_name": tool.get("tool_name"),
                "lane_type": tool.get("lane_type"),
                "can_invoke": False,
                "argument_mappings": [],
                "argument_literals": [],
                "missing_required_fields": list(tool.get("required_fields") or []),
                "skip_reason": f"stage2_provider_or_parse_error:{type(exc).__name__}",
                "argument_mapping_reason": "Stage 2 provider/parse error; no mapping trusted",
            }
            for tool in stage2_tools
        ]

    response_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for item in response_tools:
        key = (str(item.get("tool_name") or ""), str(item.get("lane_type") or ""))
        response_by_pair.setdefault(key, item)

    fields = _model_dump_list(projection.get("input_fields") or [])
    mapped: list[Step9Stage2MappedTool] = []
    uninvokable: list[str] = []
    details: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for tool in stage2_tools:
        key = (str(tool.get("tool_name") or ""), str(tool.get("lane_type") or ""))
        response_item = response_by_pair.get(key) or {
            "tool_name": key[0],
            "lane_type": key[1],
            "can_invoke": False,
            "argument_mappings": [],
            "argument_literals": [],
            "missing_required_fields": list(tool.get("required_fields") or []),
            "skip_reason": "stage2_omitted_selected_tool",
            "argument_mapping_reason": "Stage 2 omitted selected tool",
        }
        validated = validate_step9_stage2_mapping(
            response_item=response_item,
            selected_tool=tool,
            available_fields=fields,
        )
        mapped.append(validated)
        audit.extend(_mapping_audit_entries(validated))
        if not validated.can_invoke:
            uninvokable.append(validated.tool_name)
            details.append(
                {
                    "tool_name": validated.tool_name,
                    "lane_type": validated.lane_type,
                    "missing_required_fields": list(validated.missing_required_fields),
                    "skip_reason": validated.skip_reason,
                }
            )

    return Step9Stage2MappingResult(
        schema_survivors=schema_survivors,
        mapped_tools=mapped,
        uninvokable_tools=uninvokable,
        uninvokable_tool_details=details,
        argument_mapping_audit=audit,
    )


def validate_step9_stage2_mapping(
    *,
    response_item: dict[str, Any],
    selected_tool: dict[str, Any],
    available_fields: list[dict[str, Any]],
) -> Step9Stage2MappedTool:
    tool_name = str(selected_tool.get("tool_name") or response_item.get("tool_name") or "")
    lane_type = str(selected_tool.get("lane_type") or response_item.get("lane_type") or "")
    schema = selected_tool.get("full_schema") if isinstance(selected_tool.get("full_schema"), dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = [str(arg) for arg in (selected_tool.get("required_fields") or schema.get("required") or [])]
    if not required:
        required = list(_FALLBACK_REQUIRED_ARGS_OVERRIDE.get(tool_name, []))

    valid_mappings: list[Step9Stage2ArgumentMapping] = []
    valid_literals: list[Step9Stage2ArgumentLiteral] = []
    missing: set[str] = {
        str(arg) for arg in (response_item.get("missing_required_fields") or []) if isinstance(arg, str)
    }
    # `Step9InputProjection.input_fields` is contractually field_ref-unique
    # (see `step_09_input_projection._merge_duplicate_field_refs`). A
    # duplicate reaching here means an upstream contract violation; fail
    # fast instead of letting the dict comprehension below silently keep
    # whichever entry appears last in the list.
    assert_unique_input_field_refs(available_fields)
    field_refs = {str(field.get("field_ref") or ""): field for field in available_fields}
    seen_args: set[str] = set()
    duplicate_schema_args: set[str] = set()
    warnings: list[str] = []

    for pair in response_item.get("argument_mappings") or []:
        if not isinstance(pair, dict):
            warnings.append("argument_mapping_not_object")
            continue
        arg = str(pair.get("schema_arg") or "")
        ref = str(pair.get("field_ref") or "")
        if not arg or not ref:
            warnings.append("argument_mapping_missing_schema_arg_or_field_ref")
            continue
        if arg in seen_args:
            warnings.append(f"duplicate_schema_arg:{arg}")
            duplicate_schema_args.add(arg)
            continue
        if arg not in properties:
            warnings.append(f"schema_arg_not_in_full_schema:{arg}")
            continue
        field = field_refs.get(ref)
        if field is None:
            warnings.append(f"field_ref_not_available:{arg}")
            continue
        if not _step9_field_can_satisfy_arg(arg, field):
            warnings.append(f"field_ref_incompatible:{arg}")
            continue
        seen_args.add(arg)
        valid_mappings.append(Step9Stage2ArgumentMapping(schema_arg=arg, field_ref=ref))

    # Set when a provided argument literal is structurally unusable. Invalid
    # JSON and schema-invalid JSON literals both make the whole tool
    # uninvokable — never a silent drop or a string fallback — so the LLM's
    # bad structured value can be surfaced and retried, not executed.
    hard_literal_json_rejections: set[str] = set()
    hard_literal_schema_rejections: set[str] = set()

    for pair in _normalize_stage2_argument_literals(response_item.get("argument_literals")):
        if not isinstance(pair, dict):
            warnings.append("argument_literal_not_object")
            continue
        arg = str(pair.get("schema_arg") or "")
        if not arg:
            warnings.append("argument_literal_missing_schema_arg")
            continue
        if arg in seen_args:
            # Duplicate across argument_mappings/argument_literals or within
            # argument_literals: audit and keep the first — never overwrite.
            warnings.append(f"duplicate_schema_arg:{arg}")
            duplicate_schema_args.add(arg)
            continue
        prop = properties.get(arg)
        if not isinstance(prop, dict):
            warnings.append(f"literal_schema_arg_not_in_full_schema:{arg}")
            continue
        # New path: the model supplies the official value as a JSON string in
        # `literal_value_json` (lets it express array/object literals such as
        # ESM's `variants`). We only `json.loads` the model's own text — never
        # parse a domain value (e.g. "V777L") ourselves. Invalid JSON makes
        # the tool uninvokable.
        if "literal_value_json" in pair:
            raw_json = pair.get("literal_value_json")
            if not isinstance(raw_json, str):
                warnings.append(f"literal_value_json_not_string:{arg}")
                hard_literal_json_rejections.add(arg)
                continue
            try:
                parsed_literal = json.loads(raw_json)
            except (ValueError, TypeError):
                warnings.append(f"literal_value_json_invalid:{arg}")
                hard_literal_json_rejections.add(arg)
                continue
            if not _json_literal_allowed_by_schema(parsed_literal, prop):
                if arg == "variants":
                    warnings.append(f"invalid_variants_shape:{arg}")
                else:
                    warnings.append(f"literal_json_schema_invalid:{arg}")
                hard_literal_schema_rejections.add(arg)
                continue
            seen_args.add(arg)
            valid_literals.append(
                Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=parsed_literal)
            )
            continue
        parsed_literal = pair.get("literal_value")
        if isinstance(parsed_literal, (list, dict)):
            if not _json_literal_allowed_by_schema(parsed_literal, prop):
                if arg == "variants":
                    warnings.append(f"invalid_variants_shape:{arg}")
                else:
                    warnings.append(f"literal_json_schema_invalid:{arg}")
                hard_literal_schema_rejections.add(arg)
                continue
            seen_args.add(arg)
            valid_literals.append(
                Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=parsed_literal)
            )
            continue
        # Legacy path: a scalar official literal gated to the schema's
        # enum/const/default vocabulary.
        ok, literal = _literal_allowed_by_schema(parsed_literal, prop)
        if not ok:
            warnings.append(f"literal_not_allowed:{arg}")
            continue
        seen_args.add(arg)
        valid_literals.append(Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=literal))

    for arg in required:
        if arg in seen_args:
            missing.discard(arg)
            continue
        literal = _deterministic_step9_literal_if_allowed(arg, schema)
        if literal is not None:
            valid_literals.append(Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=literal))
            seen_args.add(arg)
            missing.discard(arg)
            continue
        missing.add(arg)

    can_invoke = (
        bool(response_item.get("can_invoke"))
        and not missing
        and not hard_literal_json_rejections
        and not hard_literal_schema_rejections
        and not duplicate_schema_args
    )
    skip_reason = str(response_item.get("skip_reason") or "")
    if not can_invoke and not skip_reason:
        if duplicate_schema_args:
            skip_reason = "duplicate_schema_arg"
        elif hard_literal_json_rejections:
            skip_reason = "invalid_argument_literal_json"
        elif hard_literal_schema_rejections:
            skip_reason = "invalid_argument_literal_schema"
        elif missing:
            skip_reason = "missing_required_fields"
        else:
            skip_reason = "mapping_rejected"
    elif duplicate_schema_args and skip_reason != "duplicate_schema_arg":
        skip_reason = "duplicate_schema_arg"
    elif hard_literal_json_rejections and skip_reason != "invalid_argument_literal_json":
        # Preserve the explicit reason even when the model also set skip_reason.
        skip_reason = "invalid_argument_literal_json"
    elif hard_literal_schema_rejections and skip_reason != "invalid_argument_literal_schema":
        skip_reason = "invalid_argument_literal_schema"
    reason = str(response_item.get("argument_mapping_reason") or "")
    if warnings:
        reason = (reason + "; " if reason else "") + "warnings=" + ",".join(warnings)
    return Step9Stage2MappedTool(
        tool_name=tool_name,
        lane_type=lane_type,
        can_invoke=can_invoke,
        argument_mappings=valid_mappings,
        argument_literals=valid_literals,
        missing_required_fields=sorted(missing),
        skip_reason=skip_reason,
        argument_mapping_reason=reason,
    )


def _model_dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _selection_dict(item: Step9Stage1SelectionAudit | dict[str, Any]) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return dict(item)
    return {}


def _compact_input_field_overview(items: list[Any]) -> list[dict[str, Any]]:
    fields = _model_dump_list(items)
    return [
        {
            "candidate_id": field.get("candidate_id"),
            "field_ref": field.get("field_ref"),
            "field_type": field.get("field_type"),
            "value_kind": field.get("value_kind"),
            "status": field.get("status"),
        }
        for field in fields
    ]


def _compact_input_fields_for_stage2(items: list[Any]) -> list[dict[str, Any]]:
    fields = _model_dump_list(items)
    return [
        {
            "candidate_id": field.get("candidate_id"),
            "field_ref": field.get("field_ref"),
            "field_type": field.get("field_type"),
            "value_kind": field.get("value_kind"),
            "supports_tool_args": list(field.get("supports_tool_args") or []),
            "status": field.get("status"),
        }
        for field in fields
    ]


def _redact_query_summary(query_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_query": _compact_text(str(query_summary.get("canonical_query") or "")),
        "raw_user_query": _compact_text(str(query_summary.get("raw_user_query") or "")),
    }


def _compact_schema(schema: dict[str, Any], *, tool_name: str = "") -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = _required_fields_with_fallback(tool_name, schema)
    compact_props: dict[str, Any] = {}
    for name, prop in sorted(properties.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or name.startswith("_") or not isinstance(prop, dict):
            continue
        compact_props[name] = _compact_schema_property(prop)
    return {
        "type": "object",
        "properties": compact_props,
        "required": sorted(required),
    }


def _compact_schema_property(prop: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("type", "enum", "const", "default", "description"):
        if key in prop:
            compact[key] = prop[key]
    if isinstance(prop.get("required"), list):
        compact["required"] = [str(item) for item in prop["required"] if isinstance(item, str)]
    if isinstance(prop.get("properties"), dict):
        nested_props: dict[str, Any] = {}
        for name, nested in sorted(prop["properties"].items(), key=lambda item: str(item[0])):
            if isinstance(name, str) and isinstance(nested, dict):
                nested_props[name] = _compact_schema_property(nested)
        compact["properties"] = nested_props
    if isinstance(prop.get("items"), dict):
        compact["items"] = _compact_schema_property(prop["items"])
    return compact


def _mapping_audit_entries(tool: Step9Stage2MappedTool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pair in tool.argument_mappings:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "schema_arg": pair.schema_arg,
                "field_ref": pair.field_ref,
                "argument_value_source": "mapped_from_field_ref",
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    for pair in tool.argument_literals:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "schema_arg": pair.schema_arg,
                "literal_value": pair.literal_value,
                "argument_value_source": "mapped_from_official_schema_literal",
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    if not entries:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    return entries


def _step9_field_can_satisfy_arg(arg: str, field: dict[str, Any]) -> bool:
    """A field satisfies a schema_arg only when the projection layer already
    marked that arg as supported (`supports_tool_args`), computed once,
    deterministically, from the raw Step 5/7/8 shapes — Stage 2 never
    re-derives compatibility from raw value_kind/provider heuristics."""
    lowered = arg.lower().strip()
    supports = {str(a).lower() for a in (field.get("supports_tool_args") or [])}
    return lowered in supports


def _literal_allowed_by_schema(value: Any, prop: dict[str, Any]) -> tuple[bool, Any]:
    if "const" in prop:
        const = prop.get("const")
        return value == const or str(value) == str(const), const
    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values:
        for allowed in enum_values:
            if value == allowed or str(value) == str(allowed):
                return True, allowed
        return False, value
    if "default" in prop:
        default = prop.get("default")
        return value == default or str(value) == str(default), default
    return False, value


def _json_literal_allowed_by_schema(value: Any, prop: dict[str, Any]) -> bool:
    expected = prop.get("type")
    expected_types = {str(item) for item in expected} if isinstance(expected, list) else {str(expected)}
    if isinstance(value, list):
        if "array" not in expected_types:
            return False
        items_schema = prop.get("items")
        if not isinstance(items_schema, dict):
            return True
        return all(_json_literal_allowed_by_schema(item, items_schema) for item in value)
    if isinstance(value, dict):
        if "object" not in expected_types:
            return False
        required = [str(item) for item in (prop.get("required") or []) if isinstance(item, str)]
        if any(key not in value for key in required):
            return False
        nested_props = prop.get("properties") if isinstance(prop.get("properties"), dict) else {}
        for key, nested_prop in nested_props.items():
            if key in value and isinstance(nested_prop, dict):
                if not _json_literal_allowed_by_schema(value[key], nested_prop):
                    return False
        return True
    if isinstance(value, bool):
        return "boolean" in expected_types
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer" in expected_types or "number" in expected_types
    if isinstance(value, float):
        return "number" in expected_types
    if isinstance(value, str):
        return "string" in expected_types
    return "null" in expected_types and value is None


def _normalize_stage2_argument_literals(raw: Any) -> list[Any]:
    """Accept both Stage 2 literal shapes:

    - json_object/mock path: list-of-pairs with `literal_value_json`.
    - OpenAI strict parser external path: dict `schema_arg -> parsed literal`.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            {"schema_arg": str(schema_arg), "literal_value": literal_value}
            for schema_arg, literal_value in raw.items()
        ]
    if isinstance(raw, list):
        return raw
    return [raw]


def _deterministic_step9_literal_if_allowed(arg: str, schema: dict[str, Any]) -> Any | None:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    prop = properties.get(arg)
    if not isinstance(prop, dict):
        return None
    if "const" in prop:
        return prop.get("const")
    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and len(enum_values) == 1:
        return enum_values[0]
    if "default" in prop:
        return prop.get("default")
    return None
