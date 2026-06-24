"""Gemini provider — wraps `google-genai`.

This is the SINGLE place where `google.genai` is allowed to be imported. API
endpoints and agents must depend on the `LLMProvider` Protocol, never on this
class directly.

JSON prompt construction, JSON extraction, per-task shape validation, and
Step 2 ``requested_outputs`` normalization live in
``app.llm.json_task_validation`` and are reused unchanged here so the Gemini
and OpenAI surfaces cannot drift apart.
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


class GeminiProviderError(RuntimeError):
    """Raised when Gemini cannot produce a usable JSON object."""


def _validate_task_shape(data: dict, task: str) -> dict:
    """Backwards-compatible wrapper kept for tests that monkeypatch it."""
    return _shared_validate_task_shape(data, task, error_factory=GeminiProviderError)


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
        - tool_selection_stage_1_multi_lane (Step 6 per-candidate Stage 1)
        - tool_selection_stage_2_multi_tool (Step 6 per-candidate Stage 2)
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


def _response_to_dict(response: Any) -> dict:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None:
        raise GeminiProviderError(
            f"Gemini parsed response must be a JSON object, got {type(parsed).__name__}"
        )
    return parse_text_to_json_dict(
        _response_text(response),
        error_factory=GeminiProviderError,
        provider_label="Gemini",
    )


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
