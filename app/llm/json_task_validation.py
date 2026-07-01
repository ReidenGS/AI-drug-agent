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
    stable_prefix, dynamic_suffix = build_json_prompt_sections(
        prompt=prompt, schema=schema, system=system,
    )
    return stable_prefix + dynamic_suffix


def build_json_prompt_sections(
    *, prompt: str, schema: dict, system: str | None,
) -> tuple[str, str]:
    """Split ``build_json_prompt``'s output into a stable prefix and a
    run-specific dynamic suffix (prompt-cache-friendly layout).

    Stable prefix: the system block (if any), the fixed JSON-only
    instruction, the per-task shape hint, and the task/developer
    instruction text. All of these are constant for a given
    ``(system, task, prompt)`` triple and never depend on run data — for
    the Step 2 ``structured_query`` task both ``system``
    (``SUPERVISOR_SYSTEM_PROMPT``) and ``prompt``
    (``build_supervisor_user_prompt``'s fixed instruction text) are
    themselves constant across runs, so the whole prefix is byte-identical
    across different queries / uploaded files / clarification turns for the
    same code version.

    Dynamic suffix: for most tasks this is the "Input schema/context JSON"
    block. For the Step 2 ``structured_query`` task it renders ONLY
    ``schema["task"]`` and ``schema["prompt_inputs"]`` — never
    ``schema["raw_request_record"]``. ``SupervisorAgent`` keeps
    ``raw_request_record`` on the schema dict solely for
    ``MockLLMProvider``'s in-process rule-based path (that provider reads
    the Python dict directly and never renders it to text); dumping it
    into a real provider's prompt text would leak ``storage_path`` /
    ``run_id`` / timestamps.

    Step 5 split: for the Step 5 ``tool_selection_stage_1`` payload (a
    ``tool_selection_stage_1`` task carrying ``context.candidate``), the
    stable tool catalog + rules metadata (``task`` / ``agent_name`` /
    ``step_id`` / ``compact_catalog`` / ``context.note``) is rendered into
    the stable prefix's "Input schema/context JSON" block, and only the
    candidate/run-specific portion (``context.candidate`` +
    ``context.signals``) is rendered into a trailing
    "Candidate/run-specific context JSON" block. The MockLLMProvider /
    selection policy read ``schema`` from the Python dict, so this text-only
    relocation changes nothing about selection. Non-Step-5
    ``tool_selection_stage_1`` payloads (Step 9/13/14 single-lane, which
    carry no ``context.candidate``) and all other tasks keep dumping the
    full ``schema`` dict unchanged.
    """
    schema = schema or {}
    task = schema.get("task") or "structured_query"
    shape = shape_instruction(task)
    system_block = f"System instructions:\n{system}\n\n" if system else ""
    header = (
        f"{system_block}"
        "Return exactly one valid JSON object. Do not include markdown fences, "
        "comments, prose, or tool calls.\n"
        f"Expected top-level shape:\n{shape}\n\n"
        f"User/developer task:\n{prompt}\n\n"
    )
    stable_schema, dynamic_schema = _split_prompt_schema(schema, task)
    if stable_schema is None:
        # Whole schema is dynamic (Step 2 structured_query + every
        # non-split task): keep the single "Input schema/context JSON"
        # block in the dynamic suffix exactly as before.
        payload = json.dumps(dynamic_schema, ensure_ascii=False, sort_keys=True, default=str)
        return header, f"Input schema/context JSON:\n{payload}"
    # Split layout (Step 5): stable catalog/rules block lives in the
    # prefix; candidate/run-specific block trails in the suffix.
    stable_payload = json.dumps(stable_schema, ensure_ascii=False, sort_keys=True, default=str)
    stable_prefix = f"{header}Input schema/context JSON:\n{stable_payload}"
    dynamic_payload = json.dumps(dynamic_schema, ensure_ascii=False, sort_keys=True, default=str)
    dynamic_suffix = f"\n\nCandidate/run-specific context JSON:\n{dynamic_payload}"
    return stable_prefix, dynamic_suffix


# Keys inside a Step 5 ``tool_selection_stage_1`` ``context`` block that are
# candidate/run-specific and therefore belong in the dynamic suffix. Every
# other ``context`` key (currently only ``note``, a fixed English string)
# is stable and stays in the prefix.
_STEP5_DYNAMIC_CONTEXT_KEYS: tuple[str, ...] = ("candidate", "signals")


def _split_prompt_schema(schema: dict, task: str) -> tuple[dict | None, dict]:
    """Return ``(stable_schema, dynamic_schema)`` for prompt rendering.

    ``stable_schema is None`` means "no split — the whole
    ``dynamic_schema`` goes into the single dynamic-suffix block" (Step 2
    and every non-split task). A non-None ``stable_schema`` means the
    caller renders ``stable_schema`` into the prefix and ``dynamic_schema``
    into a trailing candidate block (Step 5).
    """
    # Step 5 stage-1 selection: split stable catalog/rules from the
    # candidate/run-specific context. Keyed on the Step-5-specific
    # ``context.candidate`` shape so Step 9/13/14 single-lane
    # ``tool_selection_stage_1`` payloads (no ``context.candidate``) are
    # untouched.
    if task == "tool_selection_stage_1":
        context = schema.get("context")
        if isinstance(context, dict) and "candidate" in context:
            dynamic_context = {
                k: context[k]
                for k in _STEP5_DYNAMIC_CONTEXT_KEYS
                if k in context
            }
            stable_context = {
                k: v
                for k, v in context.items()
                if k not in _STEP5_DYNAMIC_CONTEXT_KEYS
            }
            stable_schema = {k: v for k, v in schema.items() if k != "context"}
            if stable_context:
                stable_schema["context"] = stable_context
            return stable_schema, {"context": dynamic_context}

    # Step 2 structured_query: trim ``raw_request_record`` out of the
    # dynamic dump so it never reaches a real provider's prompt text.
    if task == "structured_query" and "prompt_inputs" in schema:
        return None, {
            "task": schema.get("task", "structured_query"),
            "prompt_inputs": schema.get("prompt_inputs"),
        }

    return None, schema


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
            '"argument_literals":{"schema_arg":"official_schema_literal"},'
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
        '"clarification_questions":[],'
        '"missing_slots":[],"response":null,"canonical_query":null}\n'
        "You MUST write the canonical, normalized natural-language "
        "description of the CURRENT task into `canonical_query` (<= 800 "
        "chars). First turn: normalize the user's request. If "
        "`user_provided_context` carries `previous_canonical_query`, "
        "`previous_task_intent`, `previous_missing_slots`, "
        "`previous_clarification_requests`, or `clarification_answers`, "
        "UPDATE `canonical_query` from `previous_canonical_query` + the "
        "answers; keep `previous_task_intent` unless the user clearly changes "
        "the task; do not treat a short answer like \"HER2\" as a new task; "
        "do not invent unanswered fields (leave them 'unspecified'). Do NOT "
        "create alternative query fields — do not output `working_query`, "
        "`normalized_query`, `final_query`, `rewritten_query`, "
        "`user_query_summary`, `query_for_downstream`, `canonical_task`, "
        "`task_summary`, `query_summary`, or any other query-like field. "
        "`canonical_query` must never contain prompts, API keys, raw payloads, "
        "or full sequences.\n"
        "`missing_slots` is a JSON array of required-slot gaps you judged "
        "against the inferred task intent and the user's query / context / "
        "uploaded-file metadata. Each item: `slot_name` "
        '("target_or_antigen", "antibody", "payload", "linker", '
        '"structure_or_sequence", "sequence_role", "pdb_id", "uniprot_id", '
        '"smiles", "task_intent", "constraint", "other"), `slot_category` ("target", '
        '"antibody", "payload", "linker", "structure", "sequence", '
        '"identifier", "task_intent", "constraint", "other"), `severity` '
        '("blocking", "warning", "optional"), `required_for` (list of '
        "intents), `reason`, and an optional one-line `suggested_question`. "
        "Only list slots that are genuinely missing; omit a slot when an "
        "equivalent typed input already satisfies it.\n"
        "You MUST output `missing_slots` as structured data. If "
        "`missing_slots` is non-empty, ALSO write `response` as a concise, "
        "natural user-facing message that asks for the missing information; "
        "prioritize blocking slots, combine multiple warning slots compactly, "
        "phrase it for an end user (do not expose internal schema names "
        "unless useful), and keep it short. If `missing_slots` is empty, set "
        '`response` to null or "". `response` must never contain prompts, API '
        "keys, raw file content, or full sequences.\n"
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
            if "argument_literals" in entry and not isinstance(entry["argument_literals"], dict):
                raise error_factory(
                    f"step6_schema_mapping_stage_2 tools[{i}] `argument_literals` must be an object"
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
    # `missing_slots` is drift-tolerant: a single malformed item must never
    # fail the whole Step 2 parse, so we coerce it into a clean list BEFORE
    # the strict list checks below rather than raising on shape drift.
    normalize_missing_slots(data)
    # `response` is a drift-tolerant scalar string (or None); coerce rather
    # than raise so a non-string never fails the whole parse.
    normalize_response(data)
    # `canonical_query` is the stable working-query field; coerce + promote
    # any wrong query-like alias into it (and drop the alias).
    normalize_canonical_query(data)
    for key in (
        "referenced_inputs",
        "requested_outputs",
        "user_constraints",
        "parse_warnings",
        "normalized_entities",
        "entity_decompositions",
        "clarification_questions",
        "missing_slots",
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
    normalize_missing_slots(data)
    normalize_response(data)
    normalize_canonical_query(data)

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


# ── Step 2 ``missing_slots`` normalization ─────────────────────────────────
#
# Single source of truth for missing_slot drift handling. Both the shared
# provider normalizer (`normalize_structured_query`) and the SupervisorAgent
# boundary coercer call `normalize_missing_slots(data)` so OpenAI / Gemini /
# Qwen / Mock all agree on the cleaned shape. The rule: never crash on a
# single malformed item — coerce what we can, drop the rest, and record a
# compact parse_warning. Unknown enum values map to the safe `other` /
# `warning` defaults rather than failing validation.

_MISSING_SLOT_NAMES = frozenset(
    {
        "target_or_antigen",
        "antibody",
        "payload",
        "linker",
        "structure_or_sequence",
        "sequence_role",
        "pdb_id",
        "uniprot_id",
        "smiles",
        "task_intent",
        "constraint",
        "other",
    }
)

_MISSING_SLOT_CATEGORIES = frozenset(
    {
        "target",
        "antibody",
        "payload",
        "linker",
        "structure",
        "sequence",
        "identifier",
        "task_intent",
        "constraint",
        "other",
    }
)

_MISSING_SLOT_SEVERITIES = frozenset({"blocking", "warning", "optional"})

# Best-effort slot_name → slot_category default when the LLM omitted or
# drifted the category. Keeps the two fields internally consistent.
_MISSING_SLOT_NAME_TO_CATEGORY = {
    "target_or_antigen": "target",
    "antibody": "antibody",
    "payload": "payload",
    "linker": "linker",
    "structure_or_sequence": "structure",
    "sequence_role": "sequence",
    "pdb_id": "identifier",
    "uniprot_id": "identifier",
    "smiles": "identifier",
    "task_intent": "task_intent",
    "constraint": "constraint",
    "other": "other",
}

_MISSING_SLOT_NAME_ALIASES = {
    "target": "target_or_antigen",
    "antigen": "target_or_antigen",
    "target_antigen": "target_or_antigen",
    "antibody_candidate": "antibody",
    "structure": "structure_or_sequence",
    "sequence": "structure_or_sequence",
    "fasta_role": "sequence_role",
    "sequence_file_role": "sequence_role",
    "uploaded_sequence_role": "sequence_role",
    "structure_or_sequence_input": "structure_or_sequence",
    "uniprot": "uniprot_id",
    "pdb": "pdb_id",
    "task": "task_intent",
    "intent": "task_intent",
    "constraints": "constraint",
}


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item is not None and not isinstance(item, (list, tuple, dict)):
                out.append(str(item))
        return out
    return []


def _coerce_missing_slot_item(item: Any) -> dict | None:
    """Coerce one missing_slots entry into the canonical shape, or drop it.

    Returns ``None`` for entries that carry no usable signal (so the caller
    drops them). Unknown enum values degrade to the safe defaults; a bare
    string becomes an ``other`` slot whose ``reason`` is the string.
    """
    if item is None:
        return None
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        return {
            "slot_name": "other",
            "slot_category": "other",
            "severity": "warning",
            "required_for": [],
            "reason": s,
            "suggested_question": None,
            "evidence": None,
        }
    if not isinstance(item, dict):
        return None

    raw_name = item.get("slot_name")
    name = raw_name.strip().lower() if isinstance(raw_name, str) else ""
    name = _MISSING_SLOT_NAME_ALIASES.get(name, name)
    raw_category = item.get("slot_category")
    category = raw_category.strip().lower() if isinstance(raw_category, str) else ""
    reason_probe = " ".join(
        str(item.get(k) or "")
        for k in ("reason", "suggested_question", "evidence")
    ).lower()
    if name == "other" and category == "sequence" and "fasta" in reason_probe and "role" in reason_probe:
        name = "sequence_role"
    if name not in _MISSING_SLOT_NAMES:
        name = "other"

    if category not in _MISSING_SLOT_CATEGORIES:
        category = _MISSING_SLOT_NAME_TO_CATEGORY.get(name, "other")

    raw_sev = item.get("severity")
    severity = raw_sev.strip().lower() if isinstance(raw_sev, str) else ""
    if severity not in _MISSING_SLOT_SEVERITIES:
        severity = "warning"

    suggested = item.get("suggested_question")
    if suggested is not None and not isinstance(suggested, str):
        suggested = str(suggested)
    if isinstance(suggested, str):
        suggested = suggested.strip() or None

    evidence = item.get("evidence")
    if evidence is not None and not isinstance(evidence, str):
        evidence = str(evidence)
    if isinstance(evidence, str):
        evidence = evidence.strip() or None

    reason = item.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ("" if reason is None else str(reason))

    return {
        "slot_name": name,
        "slot_category": category,
        "severity": severity,
        "required_for": _coerce_str_list(item.get("required_for")),
        "reason": reason,
        "suggested_question": suggested,
        "evidence": evidence,
    }


def normalize_missing_slots(data: dict) -> dict:
    """Coerce ``data['missing_slots']`` into a clean ``list[dict]`` in place.

    Accepts absent / dict / string / list inputs (real LLM drift shapes),
    drops unusable entries, and records a single compact parse_warning when
    any entry was dropped or the container shape itself was coerced.
    """
    if not isinstance(data, dict):
        return data
    raw = data.get("missing_slots")
    if raw is None or raw == "":
        data["missing_slots"] = []
        return data

    warnings = data.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    data["parse_warnings"] = warnings

    container_coerced = False
    if isinstance(raw, dict):
        items: list[Any] = [raw]
        container_coerced = True
    elif isinstance(raw, str):
        items = [raw]
        container_coerced = True
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        data["missing_slots"] = []
        warnings.append(
            f"dropped malformed missing_slots: expected list, got {type(raw).__name__}"
        )
        return data

    cleaned: list[dict] = []
    dropped = 0
    for entry in items:
        coerced = _coerce_missing_slot_item(entry)
        if coerced is None:
            dropped += 1
            continue
        cleaned.append(coerced)

    data["missing_slots"] = cleaned
    if dropped:
        warnings.append(f"dropped {dropped} malformed missing_slots entr{'y' if dropped == 1 else 'ies'}")
    if container_coerced:
        warnings.append("normalized missing_slots container to a list")
    return data


# ── Step 2 ``response`` normalization ───────────────────────────────────────
#
# `response` is the user-facing follow-up message the Step 2 LLM writes when
# missing_slots is non-empty. The program only passes it through, so the
# normalizer just guarantees it is a trimmed string (or None) and never
# crashes on shape drift. It never stores prompts / keys / raw payloads.

_RESPONSE_MAX_LEN = 500


def normalize_response(data: dict) -> dict:
    """Coerce ``data['response']`` into a trimmed string or ``None`` in place."""
    if not isinstance(data, dict):
        return data
    if "response" not in data:
        data["response"] = None
        return data
    raw = data.get("response")

    if raw is None:
        data["response"] = None
        return data

    warnings = data.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    data["parse_warnings"] = warnings

    if isinstance(raw, str):
        text: str | None = raw.strip()
    elif isinstance(raw, bool):
        text = str(raw)
        warnings.append("coerced non-string response to string")
    elif isinstance(raw, (int, float)):
        text = str(raw)
        warnings.append("coerced non-string response to string")
    elif isinstance(raw, (list, tuple)):
        parts = [str(x).strip() for x in raw if x not in (None, "", [], {})]
        text = " ".join(p for p in parts if p) or None
        warnings.append("compacted list response into a string")
    elif isinstance(raw, dict):
        candidate = None
        for key in ("response", "message", "text", "question"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                candidate = val.strip()
                break
        text = candidate
        warnings.append("compacted dict response into a string")
    else:
        text = None
        warnings.append(f"dropped malformed response: unexpected {type(raw).__name__}")

    if isinstance(text, str) and len(text) > _RESPONSE_MAX_LEN:
        text = text[:_RESPONSE_MAX_LEN].rstrip()
        warnings.append(f"truncated response to {_RESPONSE_MAX_LEN} chars")

    data["response"] = text
    return data


# ── Step 2 ``canonical_query`` normalization ────────────────────────────────
#
# `canonical_query` is the single stable working-query field. Real LLMs
# sometimes emit a differently-named query field; we promote the first such
# alias into `canonical_query`, then DELETE every alias key so none leak into
# the StructuredQuery artifact. The value itself is coerced to a trimmed
# string (or None) and capped, mirroring `normalize_response`.

_CANONICAL_QUERY_MAX_LEN = 800

_CANONICAL_QUERY_ALIASES = (
    "working_query",
    "normalized_query",
    "final_query",
    "rewritten_query",
    "user_query_summary",
    "query_for_downstream",
    "canonical_task",
    "task_summary",
    "query_summary",
)


def _coerce_query_text(raw: object) -> tuple[Optional[str], Optional[str]]:
    """Return (text_or_None, warning_or_None) for a query-like value."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        return raw.strip() or None, None
    if isinstance(raw, bool) or isinstance(raw, (int, float)):
        return str(raw), "coerced non-string canonical_query to string"
    if isinstance(raw, (list, tuple)):
        parts = [str(x).strip() for x in raw if x not in (None, "", [], {})]
        return (" ".join(p for p in parts if p) or None), "compacted list canonical_query into a string"
    if isinstance(raw, dict):
        for key in ("canonical_query", "text", "summary", "description"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip(), "compacted dict canonical_query into a string"
        return None, "compacted dict canonical_query into a string"
    return None, f"dropped malformed canonical_query: unexpected {type(raw).__name__}"


def normalize_canonical_query(data: dict) -> dict:
    """Coerce ``data['canonical_query']`` and promote any wrong alias in place."""
    if not isinstance(data, dict):
        return data

    warnings = data.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []

    raw = data.get("canonical_query")
    promoted_from: Optional[str] = None
    # If canonical_query is absent/empty, adopt the first present alias value.
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        for alias in _CANONICAL_QUERY_ALIASES:
            if alias in data and data.get(alias) not in (None, "", [], {}):
                raw = data.get(alias)
                promoted_from = alias
                break

    # Always strip every alias key so none reach the artifact.
    removed_aliases = [a for a in _CANONICAL_QUERY_ALIASES if a in data]
    for alias in removed_aliases:
        data.pop(alias, None)

    text, note = _coerce_query_text(raw)
    if note:
        warnings.append(note)
    if isinstance(text, str) and len(text) > _CANONICAL_QUERY_MAX_LEN:
        text = text[:_CANONICAL_QUERY_MAX_LEN].rstrip()
        warnings.append(f"truncated canonical_query to {_CANONICAL_QUERY_MAX_LEN} chars")
    if promoted_from:
        warnings.append("promoted query alias to canonical_query")
    elif removed_aliases:
        # Aliases existed but canonical_query was already set — drop them.
        warnings.append("removed query alias in favor of canonical_query")

    data["canonical_query"] = text
    if warnings:
        data["parse_warnings"] = warnings
    return data
