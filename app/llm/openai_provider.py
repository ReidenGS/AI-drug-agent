"""OpenAI provider — wraps the official `openai` Python SDK.

This is the SINGLE place where `openai` is allowed to be imported in app
code. Agents and API routes must only depend on the ``LLMProvider``
Protocol.

JSON-prompt construction, JSON extraction, per-task shape validation, and
Step 2 ``requested_outputs`` normalization are reused from
``app.llm.json_task_validation`` so the OpenAI surface never drifts from
the Gemini one. The OpenAI provider is JSON-only — it never calls MCP
tools, never accesses biomedical APIs, and never logs raw response
bodies, prompts, or API keys.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .json_task_validation import (
    build_json_prompt as _build_json_prompt,
    normalize_structured_query as _normalize_structured_query,
    parse_text_to_json_dict,
    validate_task_shape as _shared_validate_task_shape,
)

logger = logging.getLogger(__name__)


class OpenAIProviderError(RuntimeError):
    """Raised when OpenAI cannot produce a usable JSON object."""


def _validate_task_shape(data: dict, task: str) -> dict:
    """Thin wrapper for monkeypatch-friendly testing."""
    return _shared_validate_task_shape(data, task, error_factory=OpenAIProviderError)


# ── Structured-output response models ──────────────────────────────────────
#
# These are STRICT-compliant response models for the OpenAI structured-output
# parser (`…completions.parse`). Strict structured output requires, at every
# object level, `additionalProperties: false` (→ `extra="forbid"`) and fully
# CONSTRAINED array items (→ typed list elements, never a bare `list`/`dict`).
# `tests/llm/test_openai_provider.py` proves each model's
# `to_strict_json_schema(...)` output has no `additionalProperties: true` and
# no unconstrained `items`.
#
# Tasks whose output is naturally fixed-key are modeled directly:
#   - tool_selection_stage_1
#   - step6_schema_mapping_stage_1
#
# step6_schema_mapping_stage_2 has a strict LIST-OF-PAIRS parser model
# (`_Step6SchemaMappingStage2ParserResponse` + `to_external_dict`) that folds
# dynamic `schema_arg -> value` maps back to the dynamic-dict external shape.
# It is TEMPORARILY DISABLED (not registered in `_RESPONSE_MODEL_FOR_TASK`)
# because the current Step 6 Stage 2 prompt still asks for the legacy
# dynamic-dict shape; the task therefore runs on the json_object path until
# the prompt/shared validation path is migrated to list-of-pairs.
#
# The authoritative per-task validation still runs afterward on the external
# dict.
#
# Tasks that are NOT strict-parser-friendly stay on the json_object path:
#   - structured_query: deep, many optional list-of-object / variant fields;
#     a faithful strict schema would over-constrain real LLM output. Keeping
#     it on json_object avoids "faking strict" with an open dict.


class _SelectionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Optional[str] = None


class _Stage1Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    selection_reason: Optional[str] = None
    priority: Optional[int] = None
    required_context: Optional[list[str]] = None


class _ToolSelectionStage1Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selections: list[_Stage1Selection] = Field(default_factory=list)
    selection_metadata: Optional[_SelectionMetadata] = None


class _Step6Stage1Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    selection_reason: Optional[str] = None


class _Step6SchemaMappingStage1Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selections: list[_Step6Stage1Selection] = Field(default_factory=list)


class _OrchestratorRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    priority: Literal["low", "normal", "high"]


class _OrchestratorRoutingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loop_decision: Literal[
        "dispatch_next_workers",
        "wait_for_dependencies",
        "route_to_final_response",
        "request_user_input",
        "repair_or_retry",
        "stop_cannot_satisfy",
    ]
    decisions: list[_OrchestratorRouteDecision] = Field(default_factory=list)
    decision_summary: str = Field(min_length=1)


class _Step9Stage1Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    lane_type: Literal[
        "protein_design",
        "variant_evaluation",
        "compound_screening",
    ]
    selection_reason: str


class _Step9ToolSelectionStage1Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selections: list[_Step9Stage1Selection] = Field(default_factory=list)


class _Step9Stage2ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    field_ref: str


class _Step9Stage2ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    # The official-schema literal value, encoded as a JSON *string*. Encoding
    # as text (not a typed union or an arbitrary dict/list) keeps the parser
    # schema strict — no `additionalProperties=true`, no unconstrained array
    # items — while still letting the model express any official value shape
    # (string / number / bool / null / array / object). The shared Stage 2
    # validator (`validate_step9_stage2_mapping`) is the single place that
    # `json.loads` this text; invalid JSON makes the tool uninvokable there
    # rather than being silently coerced to a string here.
    literal_value_json: str


class _Step9Stage2ToolForParser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    lane_type: Literal[
        "protein_design",
        "variant_evaluation",
        "compound_screening",
    ]
    can_invoke: bool
    argument_mappings: list[_Step9Stage2ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[_Step9Stage2ArgumentLiteral] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    skip_reason: Optional[str] = None
    argument_mapping_reason: Optional[str] = None


class _Step9SchemaMappingStage2Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[_Step9Stage2ToolForParser] = Field(default_factory=list)

    def to_external_dict(self) -> dict:
        tools_out: list[dict[str, Any]] = []
        for tool in self.tools:
            mapping_seen: set[str] = set()
            literal_seen: set[str] = set()
            mappings: list[dict[str, str]] = []
            literals: dict[str, Any] = {}
            for pair in tool.argument_mappings:
                if pair.schema_arg in mapping_seen or pair.schema_arg in literal_seen:
                    raise OpenAIProviderError(
                        "step9_tool_schema_mapping_stage_2 duplicate schema_arg "
                        f"`{pair.schema_arg}`"
                    )
                mapping_seen.add(pair.schema_arg)
                mappings.append(pair.model_dump())
            for pair in tool.argument_literals:
                if pair.schema_arg in literal_seen or pair.schema_arg in mapping_seen:
                    raise OpenAIProviderError(
                        "step9_tool_schema_mapping_stage_2 duplicate schema_arg "
                        f"`{pair.schema_arg}`"
                    )
                literal_seen.add(pair.schema_arg)
                try:
                    literals[pair.schema_arg] = json.loads(pair.literal_value_json)
                except (TypeError, ValueError) as exc:
                    raise OpenAIProviderError(
                        "step9_tool_schema_mapping_stage_2 invalid literal_value_json "
                        f"for schema_arg `{pair.schema_arg}`"
                    ) from exc
            out: dict[str, Any] = {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "can_invoke": tool.can_invoke,
                "argument_mappings": mappings,
                "argument_literals": literals,
                "missing_required_fields": list(tool.missing_required_fields),
            }
            if tool.skip_reason is not None:
                out["skip_reason"] = tool.skip_reason
            if tool.argument_mapping_reason is not None:
                out["argument_mapping_reason"] = tool.argument_mapping_reason
            tools_out.append(out)
        return {"tools": tools_out}


# ── Step 6 Stage 2: strict list-of-pairs parser shape (TEMPORARILY DISABLED) ─
#
# Temporarily disabled because the current Step 6 Stage 2 prompt still asks
# for the legacy dynamic-dict shape. Re-enable only after the prompt/shared
# validation path is migrated to list-of-pairs. These models + to_external_dict
# are intentionally KEPT (not deleted) for that future re-enable; they are just
# not registered in `_RESPONSE_MODEL_FOR_TASK`, so step6_schema_mapping_stage_2
# currently runs through the json_object path.
#
# The external step6_schema_mapping_stage_2 shape has dynamic `schema_arg`
# keys; the strict parser model expresses those as explicit
# `{schema_arg, field_ref}` / `{schema_arg, literal_value}` pairs, then
# `to_external_dict()` folds them back to the dynamic-dict shape the Step 6
# selector already consumes. Duplicate `schema_arg` values raise
# `OpenAIProviderError` (never a silent overwrite) so the retry loop fires.


class _Step6Stage2ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    field_ref: str


class _Step6Stage2ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    # Official-schema literals are usually enum/const strings, but numeric /
    # boolean defaults occur too. The scalar union is strict-schema-compatible
    # (an `anyOf` of typed scalars + null); we intentionally do NOT use
    # `dict`/`list` here. Object/array literals are out of scope for the parser
    # path and, if a tool ever needed them, would arrive via json_object.
    literal_value: Optional[Union[str, int, float, bool]] = None


class _Step6Stage2ToolForParser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    can_invoke: bool
    argument_mappings: list[_Step6Stage2ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[_Step6Stage2ArgumentLiteral] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    argument_mapping_reason: Optional[str] = None


class _Step6SchemaMappingStage2ParserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[_Step6Stage2ToolForParser] = Field(default_factory=list)

    def to_external_dict(self) -> dict:
        """Fold the strict list-of-pairs shape back into the external
        step6_schema_mapping_stage_2 dict shape the Step 6 selector consumes.

        - ``argument_mappings`` → ``argument_mapping`` dict ``schema_arg → field_ref``.
        - ``argument_literals`` → ``argument_literals`` dict ``schema_arg → literal``
          (always present as a dict; ``[]`` → ``{}``).
        - Duplicate ``schema_arg`` in either list raises ``OpenAIProviderError``
          (no silent overwrite → the generate_json retry loop reacts).
        - ``missing_required_fields`` stays ``list[str]``.
        """
        tools_out: list[dict] = []
        for tool in self.tools:
            mapping: dict[str, str] = {}
            for pair in tool.argument_mappings:
                if pair.schema_arg in mapping:
                    raise OpenAIProviderError(
                        "step6_schema_mapping_stage_2 duplicate argument_mapping "
                        f"schema_arg `{pair.schema_arg}`"
                    )
                mapping[pair.schema_arg] = pair.field_ref
            literals: dict[str, Any] = {}
            for pair in tool.argument_literals:
                if pair.schema_arg in literals:
                    raise OpenAIProviderError(
                        "step6_schema_mapping_stage_2 duplicate argument_literal "
                        f"schema_arg `{pair.schema_arg}`"
                    )
                literals[pair.schema_arg] = pair.literal_value
            tool_out: dict[str, Any] = {
                "tool_name": tool.tool_name,
                "can_invoke": tool.can_invoke,
                "argument_mapping": mapping,
                "argument_literals": literals,
                "missing_required_fields": list(tool.missing_required_fields),
            }
            if tool.argument_mapping_reason is not None:
                tool_out["argument_mapping_reason"] = tool.argument_mapping_reason
            tools_out.append(tool_out)
        return {"tools": tools_out}


# ── Step 14 patent tool planning (single-stage): strict parser shape ────────


class _Step14ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    input_ref_id: str


class _Step14ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    # The static config literal, encoded as a JSON *string* (like Step 9) so the
    # parser schema stays strict — no unconstrained dict/list/Any. The shared
    # validator (`plan_step14_tool_calls`) is the single place that gates it
    # against the sourced schema; invalid JSON makes the literal drop out.
    literal_value_json: str


class _Step14ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    can_invoke: bool
    argument_mappings: list[_Step14ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[_Step14ArgumentLiteral] = Field(default_factory=list)
    missing_required_args: list[str] = Field(default_factory=list)
    selection_reason: Optional[str] = None


class _Step14PatentToolSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_plans: list[_Step14ToolPlan] = Field(default_factory=list)

    def to_external_dict(self) -> dict:
        plans_out: list[dict[str, Any]] = []
        for plan in self.tool_plans:
            literals: list[dict[str, Any]] = []
            for pair in plan.argument_literals:
                try:
                    value = json.loads(pair.literal_value_json)
                except (TypeError, ValueError) as exc:
                    raise OpenAIProviderError(
                        "step14_patent_tool_selection invalid literal_value_json "
                        f"for schema_arg `{pair.schema_arg}`"
                    ) from exc
                literals.append({"schema_arg": pair.schema_arg, "literal_value": value})
            out: dict[str, Any] = {
                "tool_name": plan.tool_name,
                "can_invoke": plan.can_invoke,
                "argument_mappings": [m.model_dump() for m in plan.argument_mappings],
                "argument_literals": literals,
                "missing_required_args": list(plan.missing_required_args),
            }
            if plan.selection_reason is not None:
                out["selection_reason"] = plan.selection_reason
            plans_out.append(out)
        return {"tool_plans": plans_out}


# Tasks whose output uses the official structured-output parser. Every other
# task (structured_query, tool_selection_stage_2, *_multi_lane, *_multi_tool)
# keeps the json_object path unchanged.
_RESPONSE_MODEL_FOR_TASK: dict[str, type[BaseModel]] = {
    "tool_selection_stage_1": _ToolSelectionStage1Response,
    "step6_schema_mapping_stage_1": _Step6SchemaMappingStage1Response,
    "orchestrator_worker_routing": _OrchestratorRoutingResponse,
    "step14_patent_tool_selection": _Step14PatentToolSelectionResponse,
    # step6_schema_mapping_stage_2 is TEMPORARILY DISABLED from the official
    # parser path. The current Step 6 Stage 2 prompt still asks for the legacy
    # dynamic-dict shape ({"argument_mapping": {schema_arg: field_ref},
    # "argument_literals": {schema_arg: literal}}), so routing it through the
    # list-of-pairs parser model would make the parser schema and the prompt
    # output requirement inconsistent. Until the prompt + shared validation
    # path are migrated to list-of-pairs, this task goes through the
    # json_object path (the legacy dict shape). Re-enable by restoring:
    #     "step6_schema_mapping_stage_2": _Step6SchemaMappingStage2ParserResponse,
    # The parser models below (_Step6Stage2ArgumentMapping /
    # _Step6Stage2ArgumentLiteral / _Step6Stage2ToolForParser /
    # _Step6SchemaMappingStage2ParserResponse + to_external_dict) are kept for
    # that future re-enable.
    "step9_tool_selection_stage_1": _Step9ToolSelectionStage1Response,
    "step9_tool_schema_mapping_stage_2": _Step9SchemaMappingStage2Response,
}


# Signals that the parser path is unavailable / incompatible with the running
# SDK+backend, so we fall back to the json_object path instead of failing.
_PARSER_INCOMPAT_TOKENS = (
    "response_format",
    "json_schema",
    "structured output",
    "structured_output",
    "additionalproperties",
    "not supported",
    "does not support",
    "unsupported",
    "invalid schema",
)


def _resolve_parse_fn(client: Any) -> Any | None:
    """Return the SDK structured-output parser callable, or ``None``.

    Prefers the official ``client.beta.chat.completions.parse`` and degrades
    to ``client.chat.completions.parse`` if a newer SDK moved it. ``None``
    means "no parser API in this SDK" → caller falls back to json_object.
    """
    for path in (("beta", "chat", "completions", "parse"), ("chat", "completions", "parse")):
        node: Any = client
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if callable(node):
            return node
    return None


def _is_parser_incompatibility(exc: BaseException) -> bool:
    """Classify an exception as a parser-availability/compat issue (→ fallback).

    Genuine transport/auth/rate-limit errors are NOT treated as incompat, so
    they propagate exactly like the json_object path (no silent success)."""
    if isinstance(exc, (TypeError, NotImplementedError, AttributeError)):
        return True
    lowered = str(exc).lower()
    return any(token in lowered for token in _PARSER_INCOMPAT_TOKENS)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        *,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIProvider requires a non-empty api_key")
        self.api_key = api_key
        self.model = model
        self.max_retries = max(0, max_retries)
        self.timeout = float(timeout)
        self._client: Any | None = None
        # The task of the in-flight generate_json call. Read by the real
        # ``_generate_content`` to choose the structured-output parser model.
        # Never carries prompt/response content.
        self._active_task: str | None = None
        # Compact per-attempt usage log. Each entry holds only
        # provider / model / task / attempt + token counts — never
        # prompt, system, schema, response content, or API key.
        # Callers (e.g. live smoke scripts) may read or reset this
        # list to attribute token spend to a workflow phase.
        self.usage_events: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError("OpenAIProvider.generate not wired yet")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        """Generate a JSON object and validate the expected top-level shape.

        OpenAI is JSON-only here. For the tasks in ``_RESPONSE_MODEL_FOR_TASK``
        (currently tool_selection_stage_1, step6_schema_mapping_stage_1, and the
        step9 parser tasks) it prefers the official structured-output parser
        (``…completions.parse``) and falls back to the json_object path when the
        running SDK/backend does not support it. ``step6_schema_mapping_stage_2``
        is TEMPORARILY DISABLED from the parser path (its list-of-pairs model is
        kept for future re-enable) and therefore runs on the json_object path
        with the legacy dynamic-dict shape. ``structured_query`` (deeply nested /
        variant fields) also stays on the json_object path. The external
        ``generate_json`` contract is unchanged: same args, same ``dict``
        return, same validation + Step 2 normalization.
        """
        task = (schema or {}).get("task") or "structured_query"
        # Task drives the parser-model choice inside `_generate_content`; it
        # never carries prompt/response content.
        self._active_task = task
        # Build the user-message body WITHOUT embedding the system block.
        # Chat Completions takes ``system`` as a dedicated role="system"
        # message via ``_generate_content``; ``_build_json_prompt`` would
        # otherwise prepend the same text into the user message, sending
        # the system prompt twice on every call (a measurable Step 2 /
        # Step 5 prompt-token waste). Gemini still passes ``system`` into
        # ``_build_json_prompt`` because that path has no separate
        # system role today.
        base_prompt = _build_json_prompt(
            prompt=prompt, schema=schema or {}, system=None,
        )
        errors: list[str] = []

        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nYour previous response could not be parsed or validated as the "
                    f"required JSON object. Error: {errors[-1]}. Return corrected JSON only."
                )
            response = self._generate_content(base_prompt + retry_note, system=system)
            # Record per-attempt token usage as soon as a response object
            # exists, regardless of whether parsing later succeeds — a
            # retry on a malformed response still cost real tokens.
            self.usage_events.append(
                _build_usage_event(
                    provider=self.name, model=self.model,
                    task=task, attempt=attempt, response=response,
                )
            )
            try:
                parsed = _response_to_dict(response)
                validated = _validate_task_shape(parsed, task)
                if task == "structured_query":
                    validated = _normalize_structured_query(validated)
                return validated
            except OpenAIProviderError as exc:
                errors.append(str(exc))
                logger.warning(
                    "OpenAI JSON generation failed for task=%s attempt=%s/%s: %s",
                    task,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

        joined = " | ".join(errors) if errors else "unknown error"
        raise OpenAIProviderError(
            f"OpenAIProvider.generate_json failed for task `{task}` after "
            f"{self.max_retries + 1} attempt(s): {joined}"
        )

    def _generate_content(self, prompt: str, *, system: str | None) -> Any:
        """Produce one raw SDK response for the in-flight task.

        For a task with a structured-output model, tries the official parser
        first and falls back to the json_object path on parser
        unavailability/incompatibility. All other tasks go straight to
        json_object. Returns the raw SDK response object (either the parsed or
        the content shape); ``_response_to_dict`` normalizes both.
        """
        response_model = _RESPONSE_MODEL_FOR_TASK.get(self._active_task or "")
        if response_model is not None:
            parsed_response = self._try_structured_parse(
                prompt, system=system, response_model=response_model,
            )
            if parsed_response is not None:
                return parsed_response
        return self._create_json_object(prompt, system=system)

    def _messages(self, prompt: str, *, system: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _try_structured_parse(
        self, prompt: str, *, system: str | None, response_model: type[BaseModel],
    ) -> Any | None:
        """Attempt the official structured-output parser.

        Returns the raw SDK response on success, or ``None`` to signal the
        caller should fall back to the json_object path (parser API missing or
        an explicit incompatibility). A compact warning is logged on fallback —
        never a silent failure, and never the prompt / response body.
        """
        task = self._active_task
        client = self._get_client()
        parse_fn = _resolve_parse_fn(client)
        if parse_fn is None:
            logger.warning(
                "OpenAI SDK exposes no structured-output parser; task=%s falls "
                "back to json_object mode",
                task,
            )
            return None
        try:
            return parse_fn(
                model=self.model,
                messages=self._messages(prompt, system=system),
                response_format=response_model,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — classify then fall back or re-raise
            if _is_parser_incompatibility(exc):
                logger.warning(
                    "OpenAI structured-output parser unavailable/incompatible "
                    "for task=%s (%s); falling back to json_object mode",
                    task,
                    type(exc).__name__,
                )
                return None
            raise

    def _create_json_object(self, prompt: str, *, system: str | None) -> Any:
        """Call OpenAI Chat Completions with JSON-object response mode.

        Chat Completions + ``response_format={"type": "json_object"}`` is the
        stable JSON contract across SDK versions we depend on, and the fallback
        for tasks/SDKs where the structured-output parser is unavailable.
        """
        client = self._get_client()
        return client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system=system),
            response_format={"type": "json_object"},
            timeout=self.timeout,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except ImportError as exc:
                raise OpenAIProviderError(
                    "openai SDK is not installed; install project dependencies "
                    "before using LLM_PROVIDER=openai"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client


def _response_to_dict(response: Any) -> dict:
    """Extract a JSON dict from a Chat Completions / parsed response object.

    Two shapes are supported and normalized to the same dict:
    - structured-output parser: ``choices[0].message.parsed`` is a Pydantic
      model (or dict); we ``model_dump()`` it.
    - json_object mode: ``choices[0].message.content`` is a JSON string; we
      decode it. We never log either body — only the dict is returned upward.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise OpenAIProviderError("OpenAI response had no choices")
    message = getattr(choices[0], "message", None)

    # Structured-output parser path: `message.parsed` is the validated model.
    parsed = getattr(message, "parsed", None) if message is not None else None
    if parsed is not None:
        if isinstance(parsed, BaseModel):
            # Models whose strict parser shape differs from the external shape
            # (e.g. Step 6 Stage 2 list-of-pairs) provide `to_external_dict`;
            # otherwise `exclude_none` so unset optional fields are absent
            # (matching what a json_object LLM emits) rather than explicit
            # `null`, which the shared per-task validator would reject.
            converter = getattr(parsed, "to_external_dict", None)
            if callable(converter):
                return converter()
            return parsed.model_dump(exclude_none=True)
        if isinstance(parsed, dict):
            return parsed
        raise OpenAIProviderError(
            f"OpenAI parsed response was not a JSON object (got {type(parsed).__name__})"
        )
    refusal = getattr(message, "refusal", None) if message is not None else None
    if isinstance(refusal, str) and refusal.strip():
        # The model refused via structured outputs — surface a compact error
        # (never the refusal body) so the retry/fallback loop can react.
        raise OpenAIProviderError("OpenAI structured-output response was a refusal")

    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        return parse_text_to_json_dict(
            content, error_factory=OpenAIProviderError, provider_label="OpenAI"
        )
    if isinstance(content, list):
        # Some SDK paths return content as a list of segments; concatenate text segments only.
        text_parts: list[str] = []
        for part in content:
            text_val = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
            if isinstance(text_val, str):
                text_parts.append(text_val)
        joined = "\n".join(text_parts).strip()
        if joined:
            return parse_text_to_json_dict(
                joined, error_factory=OpenAIProviderError, provider_label="OpenAI"
            )
    raise OpenAIProviderError("OpenAI response did not include JSON content")


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _build_usage_event(
    *, provider: str, model: str, task: str, attempt: int, response: Any,
) -> dict[str, Any]:
    """Compact usage event built from a Chat Completions response.

    Reads ``response.usage`` ({prompt_tokens, completion_tokens,
    total_tokens}). Missing fields degrade to ``None`` so the event
    is always present even when the SDK version drifts. NEVER reads
    or stores prompt / system / schema / response content / API key.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    def _get(key: str) -> Any:
        if usage is None:
            return None
        return getattr(usage, key, None) if not isinstance(usage, dict) else usage.get(key)

    # OpenAI Chat Completions exposes the cached-prompt token count
    # under ``usage.prompt_tokens_details.cached_tokens`` (auto prompt
    # caching). Degrades to ``None`` when the SDK / proxy / older
    # backend does not surface that block.
    details = _get("prompt_tokens_details")

    def _detail(key: str) -> Any:
        if details is None:
            return None
        if isinstance(details, dict):
            return details.get(key)
        return getattr(details, key, None)

    return {
        "provider": provider,
        "model": model,
        "task": task,
        "attempt": attempt,
        "prompt_tokens": _coerce_int(_get("prompt_tokens")),
        "completion_tokens": _coerce_int(_get("completion_tokens")),
        "total_tokens": _coerce_int(_get("total_tokens")),
        "cached_prompt_tokens": _coerce_int(_detail("cached_tokens")),
    }
