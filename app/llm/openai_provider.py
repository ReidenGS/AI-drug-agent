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
from typing import Any

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

        OpenAI is JSON-only here. Supported tasks (same set as Gemini):

        - tool_selection_stage_1
        - tool_selection_stage_2
        - tool_selection_stage_1_multi_lane (Step 6 per-candidate Stage 1)
        - tool_selection_stage_2_multi_tool (Step 6 per-candidate Stage 2)
        - SupervisorAgent structured-query parsing
        """
        task = (schema or {}).get("task") or "structured_query"
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
        """Call OpenAI Chat Completions with JSON-object response mode.

        Chat Completions + ``response_format={"type": "json_object"}`` is the
        stable JSON contract across SDK versions we depend on. The Responses
        API is intentionally NOT used here to keep the surface narrow.
        """
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return client.chat.completions.create(
            model=self.model,
            messages=messages,
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
    """Extract JSON dict from a Chat Completions response object.

    Modern SDK returns choices with ``message.content`` being a JSON string
    (because ``response_format=json_object`` was requested). We never log
    that content body — only the validated dict is returned upward.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise OpenAIProviderError("OpenAI response had no choices")
    message = getattr(choices[0], "message", None)
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
