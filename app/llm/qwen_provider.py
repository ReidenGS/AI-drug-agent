"""Qwen provider via DashScope's OpenAI-compatible Chat Completions API.

This provider is JSON-only and intentionally mirrors ``OpenAIProvider`` for
prompt construction, response parsing, task-shape validation, and compact
usage accounting. It never logs raw prompts, raw responses, schemas, or API
keys.
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
from .openai_provider import _build_usage_event

logger = logging.getLogger(__name__)


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenProviderError(RuntimeError):
    """Raised when Qwen cannot produce a usable JSON object."""


def _validate_task_shape(data: dict, task: str) -> dict:
    """Thin wrapper for monkeypatch-friendly testing."""
    return _shared_validate_task_shape(data, task, error_factory=QwenProviderError)


class QwenProvider:
    name = "qwen"

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        *,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("QwenProvider requires a non-empty api_key")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max(0, max_retries)
        self.timeout = float(timeout)
        self._client: Any | None = None
        self.usage_events: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError("QwenProvider.generate not wired yet")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        task = (schema or {}).get("task") or "structured_query"
        base_prompt = _build_json_prompt(
            prompt=prompt,
            schema=schema or {},
            system=None,
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
            self.usage_events.append(
                _build_usage_event(
                    provider=self.name,
                    model=self.model,
                    task=task,
                    attempt=attempt,
                    response=response,
                )
            )
            try:
                parsed = _response_to_dict(response)
                validated = _validate_task_shape(parsed, task)
                if task == "structured_query":
                    validated = _normalize_structured_query(validated)
                return validated
            except QwenProviderError as exc:
                errors.append(str(exc))
                logger.warning(
                    "Qwen JSON generation failed for task=%s attempt=%s/%s: %s",
                    task,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

        joined = " | ".join(errors) if errors else "unknown error"
        raise QwenProviderError(
            f"QwenProvider.generate_json failed for task `{task}` after "
            f"{self.max_retries + 1} attempt(s): {joined}"
        )

    def _generate_content(self, prompt: str, *, system: str | None) -> Any:
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
                raise QwenProviderError(
                    "openai SDK is not installed; install project dependencies "
                    "before using LLM_PROVIDER=qwen"
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client


def _response_to_dict(response: Any) -> dict:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise QwenProviderError("Qwen response had no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        return parse_text_to_json_dict(
            content,
            error_factory=QwenProviderError,
            provider_label="Qwen",
        )
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            text_val = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if isinstance(text_val, str):
                text_parts.append(text_val)
        joined = "\n".join(text_parts).strip()
        if joined:
            return parse_text_to_json_dict(
                joined,
                error_factory=QwenProviderError,
                provider_label="Qwen",
            )
    raise QwenProviderError("Qwen response did not include JSON content")
