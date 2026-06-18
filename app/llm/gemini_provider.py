"""Gemini provider — wraps `google-genai`.

This is the SINGLE place where `google.genai` is allowed to be imported. API
endpoints and agents must depend on the `LLMProvider` Protocol, never on this
class directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GeminiProviderError(RuntimeError):
    """Raised when Gemini cannot produce a usable JSON object."""


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        *,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiProvider requires a non-empty api_key")
        self.api_key = api_key
        self.model = model
        self.max_retries = max(0, max_retries)
        self._client: Any | None = None

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError("GeminiProvider.generate not wired yet")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        """Generate a JSON object and validate the expected top-level shape.

        Gemini is never allowed to call MCP tools. It only returns structured
        JSON for the existing LLMProvider call sites:
        - tool_selection_stage_1
        - tool_selection_stage_2
        - SupervisorAgent structured-query parsing
        """
        task = (schema or {}).get("task") or "structured_query"
        base_prompt = _build_json_prompt(prompt=prompt, schema=schema or {}, system=system)
        errors: list[str] = []

        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nYour previous response could not be parsed or validated as the "
                    f"required JSON object. Error: {errors[-1]}. Return corrected JSON only."
                )
            response = self._generate_content(base_prompt + retry_note)
            try:
                parsed = _response_to_dict(response)
                validated = _validate_task_shape(parsed, task)
                if task == "structured_query":
                    validated = _normalize_structured_query(validated)
                return validated
            except GeminiProviderError as exc:
                errors.append(str(exc))
                logger.warning(
                    "Gemini JSON generation failed for task=%s attempt=%s/%s: %s",
                    task,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

        joined = " | ".join(errors) if errors else "unknown error"
        raise GeminiProviderError(
            f"GeminiProvider.generate_json failed for task `{task}` after "
            f"{self.max_retries + 1} attempt(s): {joined}"
        )

    def _generate_content(self, prompt: str) -> Any:
        client = self._get_client()
        return client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=_generation_config(),
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai  # type: ignore[import-not-found]
            except ImportError as exc:
                raise GeminiProviderError(
                    "google-genai is not installed; install project dependencies "
                    "before using LLM_PROVIDER=gemini"
                ) from exc

            self._client = genai.Client(api_key=self.api_key)
        return self._client


def _generation_config() -> Any:
    """Use Gemini JSON mode when the installed SDK exposes config types."""
    try:
        from google.genai import types  # type: ignore[import-not-found]

        return types.GenerateContentConfig(response_mime_type="application/json")
    except Exception:  # pragma: no cover - exercised only by SDK/version drift
        return {"response_mime_type": "application/json"}


def _build_json_prompt(*, prompt: str, schema: dict, system: str | None) -> str:
    task = schema.get("task") or "structured_query"
    shape = _shape_instruction(task)
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


def _shape_instruction(task: str) -> str:
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


def _response_to_dict(response: Any) -> dict:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None:
        raise GeminiProviderError(
            f"Gemini parsed response must be a JSON object, got {type(parsed).__name__}"
        )

    text = _response_text(response)
    json_text = _extract_json_object(text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise GeminiProviderError(
            f"Gemini returned malformed JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise GeminiProviderError(
            f"Gemini JSON response must be an object, got {type(data).__name__}"
        )
    return data


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts_text.append(part_text)
    joined = "\n".join(parts_text).strip()
    if joined:
        return joined
    raise GeminiProviderError("Gemini response did not include text or parsed JSON")


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise GeminiProviderError("Gemini response did not contain a JSON object")

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
    raise GeminiProviderError("Gemini response contained an unterminated JSON object")


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _validate_task_shape(data: dict, task: str) -> dict:
    if task == "tool_selection_stage_1":
        selections = data.get("selections")
        if not isinstance(selections, list):
            raise GeminiProviderError("tool_selection_stage_1 response requires list `selections`")
        for i, entry in enumerate(selections):
            if not isinstance(entry, dict):
                raise GeminiProviderError(
                    f"tool_selection_stage_1 selections[{i}] must be an object"
                )
            if "tool_name" not in entry or not isinstance(entry.get("tool_name"), str):
                raise GeminiProviderError(
                    f"tool_selection_stage_1 selections[{i}] requires string `tool_name`"
                )
        metadata = data.get("selection_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise GeminiProviderError("tool_selection_stage_1 `selection_metadata` must be an object")
        return data

    if task == "tool_selection_stage_2":
        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            raise GeminiProviderError("tool_selection_stage_2 response requires object `arguments`")
        if "missing_fields" in data and not isinstance(data["missing_fields"], list):
            raise GeminiProviderError("tool_selection_stage_2 `missing_fields` must be a list")
        return data

    if not isinstance(data.get("task_intent"), dict):
        raise GeminiProviderError("structured_query response requires object `task_intent`")
    return _validate_structured_query_rest(data)


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

# Map shorthand / entity-shaped labels Gemini sometimes returns to the
# canonical Step 2 enum. Only add aliases the canonical doc semantically
# supports — never invent new enum values.
_REQUESTED_OUTPUTS_ALIASES = {
    # ranked_candidates
    "adc_candidate": "ranked_candidates",
    "adc_candidates": "ranked_candidates",
    "candidate": "ranked_candidates",
    "candidates": "ranked_candidates",
    "candidate_shortlist": "ranked_candidates",
    "final_ranking": "ranked_candidates",
    "ranking": "ranked_candidates",
    "ranked_candidate": "ranked_candidates",
    # evidence / literature
    "evidence": "evidence_summary",
    "literature": "literature_review_summary",
    "literature_summary": "literature_review_summary",
    "literature_review": "literature_review_summary",
    # patent / IP
    "patent": "patent_or_ip_summary",
    "ip_summary": "patent_or_ip_summary",
    "patent_summary": "patent_or_ip_summary",
    # optimization
    "optimization": "optimization_suggestions",
    "optimization_suggestion": "optimization_suggestions",
    # developability
    "developability": "developability_summary",
    "developability_report": "developability_summary",
    # structure
    "structure_report": "structure_validation_report",
    "structure_summary": "structure_validation_report",
    # compound screening
    "compound_screening": "compound_screening_results",
    "screening_results": "compound_screening_results",
    "compound_screen": "compound_screening_results",
    # entity normalization
    "entity_normalization": "entity_normalization_summary",
    "normalization_summary": "entity_normalization_summary",
    # workflow / gap / case study
    "workflow": "workflow_recommendation",
    "workflow_suggestion": "workflow_recommendation",
    "gap_analysis": "data_gap_summary",
    "gap_summary": "data_gap_summary",
    "missing_inputs": "data_gap_summary",
    "case_study": "case_study_summary",
    "benchmark": "case_study_summary",
    "benchmark_summary": "case_study_summary",
}


def _normalize_structured_query(data: dict) -> dict:
    """Coerce loose Gemini output for `requested_outputs` into Step 2 schema.

    Gemini occasionally returns entries like `{"entity_type": "adc_candidate"}`
    or stray strings outside the canonical enum. We map known aliases to the
    canonical enum, drop anything we cannot map, and record dropped items in
    `parse_warnings` so the provenance is preserved.
    """
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
            warnings.append(
                f"dropped non-string requested_outputs entry: {item!r}"
            )
            continue
        mapped = _REQUESTED_OUTPUTS_ALIASES.get(candidate, candidate)
        if mapped in _REQUESTED_OUTPUTS_ENUM:
            if mapped not in seen:
                normalized.append(mapped)
                seen.add(mapped)
        else:
            warnings.append(
                f"dropped unknown requested_outputs value: {item!r}"
            )

    data["requested_outputs"] = normalized
    data["parse_warnings"] = warnings
    return data


def _candidate_label(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip().lower() or None
    if isinstance(item, dict):
        for key in ("entity_type", "name", "type", "value"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
    return None


def _validate_structured_query_rest(data: dict) -> dict:
    if not isinstance(data.get("mentioned_entities"), dict):
        raise GeminiProviderError("structured_query response requires object `mentioned_entities`")
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
            raise GeminiProviderError(f"structured_query `{key}` must be a list")
    return data
