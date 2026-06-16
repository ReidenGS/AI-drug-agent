"""Gemini provider — wraps `google-genai`.

This is the SINGLE place where `google.genai` is allowed to be imported. API
endpoints and agents must depend on the `LLMProvider` Protocol, never on this
class directly.

Current state: instantiation is legal so wiring can be exercised end-to-end,
but `generate*` raises NotImplementedError until the production client is
wired (model selection, retries, structured-output config, usage logging).
"""

from __future__ import annotations

from typing import Any


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro") -> None:
        if not api_key:
            raise ValueError("GeminiProvider requires a non-empty api_key")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        # TODO: from google import genai; client = genai.Client(api_key=self.api_key);
        #       client.models.generate_content(model=self.model, contents=...)
        raise NotImplementedError("GeminiProvider.generate not wired yet")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        # TODO: use response_mime_type="application/json" + response_schema
        raise NotImplementedError("GeminiProvider.generate_json not wired yet")
