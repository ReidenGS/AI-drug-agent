"""Shared JSON-task prompt + validation + normalization for LLM providers.

Single source of truth for:
- the per-task shape hint baked into the prompt body,
- the per-task top-level validator that rejects drift,
- the Step 2 ``requested_outputs`` normalization with alias mapping.

Both ``GeminiProvider`` and ``OpenAIProvider`` consume these helpers. They
pass their own provider-specific error class as ``error_factory`` so
existing ``pytest.raises(GeminiProviderError, match=...)`` patterns and the
analogous OpenAI ones both keep working.

The module never logs API keys, never logs raw response bodies, never
returns prompt text in raised errors.
"""

from __future__ import annotations

import json
from typing import Any, Callable

ErrorFactory = Callable[[str], BaseException]


# ── Prompt construction ────────────────────────────────────────────────────

def build_json_prompt(*, prompt: str, schema: dict, system: str | None) -> str:
    """Compose the user/system prompt body sent to the JSON-only LLM call."""
    task = (schema or {}).get("task") or "structured_query"
    shape = shape_instruction(task)
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str)
    system_block = f"System instructions:\n{system}\n\n" if system else ""
    return (
        f"{system_block}"
        "Return exactly one valid JSON object. Do not include markdown fences, "
        "comments, prose, or tool calls.\n"
        f"Expected top-level shape:\n{shape}\n\n"
        f"User/developer task:\n{prompt}\n\n"
        f"Input schema/context JSON:\n{payload}"
    )


def shape_instruction(task: str) -> str:
    if task == "tool_selection_stage_1":
        return (
            '{"selections":[{"tool_name":"string","selection_reason":"string",'
            '"priority":1,"required_context":["string"]}],'
            '"selection_metadata":{"strategy":"string"}}'
        )
    if task == "tool_selection_stage_2":
        return (
            '{"arguments":{"parameter_name":"value"},'
            '"argument_construction_reason":"string","missing_fields":["string"]}'
        )
    if task == "tool_selection_stage_1_multi_lane":
        return (
            '{"selections":[{"lane_type":"string","tool_name":"string",'
            '"selection_reason":"string","priority":1,"required_context":["string"]}],'
            '"selection_metadata":{"strategy":"string"}}'
        )
    if task == "tool_selection_stage_2_multi_tool":
        return (
            '{"tools":[{"lane_type":"string","tool_name":"string",'
            '"arguments":{"parameter_name":"value"},'
            '"argument_construction_reason":"string","missing_fields":["string"]}]}'
        )
    if task == "step6_schema_mapping_stage_1":
        return (
            '{"selections":[{"tool_name":"string",'
            '"selection_reason":"string"}]}'
        )
    if task == "step6_schema_mapping_stage_2":
        return (
            '{"tools":[{"tool_name":"string","can_invoke":true,'
            '"argument_mapping":{"schema_arg":"field_ref"},'
            '"missing_required_fields":["string"],'
            '"argument_mapping_reason":"string"}]}'
        )
    return (
        '{"task_intent":{"task_type":"adc_design","primary_intent":'
        '"new_adc_design","secondary_intents":[]},"mentioned_entities":{},'
        '"referenced_inputs":[],'
        '"requested_outputs":["ranked_candidates"],'
        '"user_constraints":[],"parse_warnings":[],'
        '"normalized_entities":[],"entity_decompositions":[],'
        '"clarification_questions":[]}\n'
        "`task_intent.primary_intent` MUST be one of "
        '"new_adc_design", "existing_adc_evaluation", '
        '"developability_assessment", "structure_analysis", '
        '"compound_screening", "literature_review", "patent_ip_review", '
        '"optimization", "unclear_or_needs_clarification". '
        "`task_intent.secondary_intents` is a list drawn from the same enum.\n"
        "`requested_outputs` MUST be a JSON array of plain strings drawn from "
        "this enum only: "
        '"ranked_candidates", "report", "evidence_summary", '
        '"literature_review_summary", "patent_or_ip_summary", '
        '"optimization_suggestions", "developability_summary", '
        '"structure_validation_report", "compound_screening_results", '
        '"entity_normalization_summary", "workflow_recommendation", '
        '"data_gap_summary", "case_study_summary". '
        "Do not return objects, do not invent new keys, do not return entity "
        "names inside `requested_outputs`. Omit values you are unsure about."
    )


# ── JSON extraction from raw text ──────────────────────────────────────────

def parse_text_to_json_dict(
    text: str, *, error_factory: ErrorFactory, provider_label: str
) -> dict:
    """Strip optional markdown fence, extract first JSON object, decode."""
    json_text = extract_json_object(text, error_factory=error_factory, provider_label=provider_label)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise error_factory(
            f"{provider_label} returned malformed JSON at line {exc.lineno} "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise error_factory(
            f"{provider_label} JSON response must be an object, "
            f"got {type(data).__name__}"
        )
    return data


def extract_json_object(
    text: str, *, error_factory: ErrorFactory, provider_label: str
) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise error_factory(f"{provider_label} response did not contain a JSON object")

    in_string = False
    escaped = False
    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    raise error_factory(f"{provider_label} response contained an unterminated JSON object")


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


# ── Per-task validator ────────────────────────────────────────────────────

def validate_task_shape(data: dict, task: str, *, error_factory: ErrorFactory) -> dict:
    if task == "tool_selection_stage_1":
        selections = data.get("selections")
        if not isinstance(selections, list):
            raise error_factory("tool_selection_stage_1 response requires list `selections`")
        for i, entry in enumerate(selections):
            if not isinstance(entry, dict):
                raise error_factory(
                    f"tool_selection_stage_1 selections[{i}] must be an object"
                )
            if "tool_name" not in entry or not isinstance(entry.get("tool_name"), str):
                raise error_factory(
                    f"tool_selection_stage_1 selections[{i}] requires string `tool_name`"
                )
        metadata = data.get("selection_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise error_factory("tool_selection_stage_1 `selection_metadata` must be an object")
        return data

    if task == "tool_selection_stage_2":
        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            raise error_factory("tool_selection_stage_2 response requires object `arguments`")
        if "missing_fields" in data and not isinstance(data["missing_fields"], list):
            raise error_factory("tool_selection_stage_2 `missing_fields` must be a list")
        return data

    if task == "tool_selection_stage_1_multi_lane":
        selections = data.get("selections")
        if not isinstance(selections, list):
            raise error_factory(
                "tool_selection_stage_1_multi_lane response requires list `selections`"
            )
        for i, entry in enumerate(selections):
            if not isinstance(entry, dict):
                raise error_factory(
                    f"tool_selection_stage_1_multi_lane selections[{i}] must be an object"
                )
            if not isinstance(entry.get("lane_type"), str) or not entry.get("lane_type"):
                raise error_factory(
                    f"tool_selection_stage_1_multi_lane selections[{i}] requires string `lane_type`"
                )
            if not isinstance(entry.get("tool_name"), str) or not entry.get("tool_name"):
                raise error_factory(
                    f"tool_selection_stage_1_multi_lane selections[{i}] requires string `tool_name`"
                )
            if "selection_reason" in entry and not isinstance(entry["selection_reason"], str):
                raise error_factory(
                    f"tool_selection_stage_1_multi_lane selections[{i}] `selection_reason` must be a string"
                )
            if "required_context" in entry and not isinstance(entry["required_context"], list):
                raise error_factory(
                    f"tool_selection_stage_1_multi_lane selections[{i}] `required_context` must be a list"
                )
        metadata = data.get("selection_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise error_factory(
                "tool_selection_stage_1_multi_lane `selection_metadata` must be an object"
            )
        return data

    if task == "tool_selection_stage_2_multi_tool":
        tools = data.get("tools")
        if not isinstance(tools, list):
            raise error_factory(
                "tool_selection_stage_2_multi_tool response requires list `tools`"
            )
        for i, entry in enumerate(tools):
            if not isinstance(entry, dict):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] must be an object"
                )
            if not isinstance(entry.get("lane_type"), str) or not entry.get("lane_type"):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] requires string `lane_type`"
                )
            if not isinstance(entry.get("tool_name"), str) or not entry.get("tool_name"):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] requires string `tool_name`"
                )
            if not isinstance(entry.get("arguments"), dict):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] requires object `arguments`"
                )
            if "argument_construction_reason" in entry and not isinstance(
                entry["argument_construction_reason"], str
            ):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] `argument_construction_reason` must be a string"
                )
            if "missing_fields" in entry and not isinstance(entry["missing_fields"], list):
                raise error_factory(
                    f"tool_selection_stage_2_multi_tool tools[{i}] `missing_fields` must be a list"
                )
        return data

    if task == "step6_schema_mapping_stage_1":
        selections = data.get("selections")
        if not isinstance(selections, list):
            raise error_factory(
                "step6_schema_mapping_stage_1 response requires list `selections`"
            )
        for i, entry in enumerate(selections):
            if not isinstance(entry, dict):
                raise error_factory(
                    f"step6_schema_mapping_stage_1 selections[{i}] must be an object"
                )
            if not isinstance(entry.get("tool_name"), str) or not entry.get("tool_name"):
                raise error_factory(
                    f"step6_schema_mapping_stage_1 selections[{i}] requires string `tool_name`"
                )
            if "selection_reason" in entry and not isinstance(entry["selection_reason"], str):
                raise error_factory(
                    f"step6_schema_mapping_stage_1 selections[{i}] `selection_reason` must be a string"
                )
        return data

    if task == "step6_schema_mapping_stage_2":
        tools = data.get("tools")
        if not isinstance(tools, list):
            raise error_factory(
                "step6_schema_mapping_stage_2 response requires list `tools`"
            )
        for i, entry in enumerate(tools):
            if not isinstance(entry, dict):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] must be an object"
                )
            if not isinstance(entry.get("tool_name"), str) or not entry.get("tool_name"):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] requires string `tool_name`"
                )
            if not isinstance(entry.get("can_invoke"), bool):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] requires boolean `can_invoke`"
                )
            if not isinstance(entry.get("argument_mapping"), dict):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] requires object `argument_mapping`"
                )
            if not isinstance(entry.get("missing_required_fields"), list):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] requires list `missing_required_fields`"
                )
            if "argument_mapping_reason" in entry and not isinstance(
                entry["argument_mapping_reason"], str
            ):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] `argument_mapping_reason` must be a string"
                )
        return data

    if not isinstance(data.get("task_intent"), dict):
        raise error_factory("structured_query response requires object `task_intent`")
    return _validate_structured_query_rest(data, error_factory=error_factory)


def _validate_structured_query_rest(data: dict, *, error_factory: ErrorFactory) -> dict:
    if not isinstance(data.get("mentioned_entities"), dict):
        raise error_factory("structured_query response requires object `mentioned_entities`")
    for key in (
        "referenced_inputs",
        "requested_outputs",
        "user_constraints",
        "parse_warnings",
        "normalized_entities",
        "entity_decompositions",
        "clarification_questions",
    ):
        if key not in data:
            data[key] = []
        if not isinstance(data[key], list):
            raise error_factory(f"structured_query `{key}` must be a list")
    return data


# ── Step 2 ``requested_outputs`` normalization ─────────────────────────────

_REQUESTED_OUTPUTS_ENUM = frozenset(
    {
        "ranked_candidates",
        "report",
        "evidence_summary",
        "literature_review_summary",
        "patent_or_ip_summary",
        "optimization_suggestions",
        "developability_summary",
        "structure_validation_report",
        "compound_screening_results",
        "entity_normalization_summary",
        "workflow_recommendation",
        "data_gap_summary",
        "case_study_summary",
    }
)

_REQUESTED_OUTPUTS_ALIASES = {
    "adc_candidate": "ranked_candidates",
    "adc_candidates": "ranked_candidates",
    "candidate": "ranked_candidates",
    "candidates": "ranked_candidates",
    "candidate_shortlist": "ranked_candidates",
    "final_ranking": "ranked_candidates",
    "ranking": "ranked_candidates",
    "ranked_candidate": "ranked_candidates",
    "evidence": "evidence_summary",
    "literature": "literature_review_summary",
    "literature_summary": "literature_review_summary",
    "literature_review": "literature_review_summary",
    "patent": "patent_or_ip_summary",
    "ip_summary": "patent_or_ip_summary",
    "patent_summary": "patent_or_ip_summary",
    "optimization": "optimization_suggestions",
    "optimization_suggestion": "optimization_suggestions",
    "developability": "developability_summary",
    "developability_report": "developability_summary",
    "structure_report": "structure_validation_report",
    "structure_summary": "structure_validation_report",
    "compound_screening": "compound_screening_results",
    "screening_results": "compound_screening_results",
    "compound_screen": "compound_screening_results",
    "entity_normalization": "entity_normalization_summary",
    "normalization_summary": "entity_normalization_summary",
    "workflow": "workflow_recommendation",
    "workflow_suggestion": "workflow_recommendation",
    "gap_analysis": "data_gap_summary",
    "gap_summary": "data_gap_summary",
    "missing_inputs": "data_gap_summary",
    "case_study": "case_study_summary",
    "benchmark": "case_study_summary",
    "benchmark_summary": "case_study_summary",
}


def normalize_structured_query(data: dict) -> dict:
    _normalize_entity_decomposition_components(data)

    raw = data.get("requested_outputs")
    if not isinstance(raw, list):
        return data

    warnings = data.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        candidate = _candidate_label(item)
        if candidate is None:
            warnings.append(f"dropped non-string requested_outputs entry: {item!r}")
            continue
        mapped = _REQUESTED_OUTPUTS_ALIASES.get(candidate, candidate)
        if mapped in _REQUESTED_OUTPUTS_ENUM:
            if mapped not in seen:
                normalized.append(mapped)
                seen.add(mapped)
        else:
            warnings.append(f"dropped unknown requested_outputs value: {item!r}")

    data["requested_outputs"] = normalized
    data["parse_warnings"] = warnings
    return data


_COMPONENT_NAME_ALIASES = ("component_name", "name", "label", "value")
_COMPONENT_ROLE_VALUES = {"antibody", "payload", "linker", "linker_payload", "other"}


def _normalize_entity_decomposition_components(data: dict) -> None:
    decompositions = data.get("entity_decompositions")
    if not isinstance(decompositions, list):
        return

    warnings = data.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    data["parse_warnings"] = warnings

    for decomp_index, decomp in enumerate(decompositions):
        if not isinstance(decomp, dict):
            warnings.append(
                f"dropped entity_decompositions[{decomp_index}]: expected object"
            )
            continue
        components = decomp.get("components")
        if not isinstance(components, list):
            if components is not None:
                warnings.append(
                    f"dropped entity_decompositions[{decomp_index}].components: expected list"
                )
            decomp["components"] = []
            continue

        normalized_components: list[dict[str, Any]] = []
        for comp_index, component in enumerate(components):
            if not isinstance(component, dict):
                warnings.append(
                    "dropped "
                    f"entity_decompositions[{decomp_index}].components[{comp_index}]: "
                    "expected object"
                )
                continue

            canonical_name = _component_canonical_name(component)
            if canonical_name is None:
                warnings.append(
                    "dropped "
                    f"entity_decompositions[{decomp_index}].components[{comp_index}]: "
                    "missing canonical_name"
                )
                continue

            out = dict(component)
            out["canonical_name"] = canonical_name
            component_type = out.get("component_type")
            if (
                "role" not in out
                and isinstance(component_type, str)
                and component_type in _COMPONENT_ROLE_VALUES
            ):
                out["role"] = component_type
            normalized_components.append(out)

        decomp["components"] = normalized_components


def _component_canonical_name(component: dict[str, Any]) -> str | None:
    value = component.get("canonical_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for key in _COMPONENT_NAME_ALIASES:
        value = component.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_label(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip().lower() or None
    if isinstance(item, dict):
        for key in ("entity_type", "name", "type", "value"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
    return None
