"""Base agent: holds LLM provider, A2A card, and a step-scoped MCP tool subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.provider import LLMProvider


@dataclass
class BaseAgent:
    name: str
    llm: LLMProvider
    mcp_client: Any | None = None  # python-a2a MCP client; wired by runtime

    def run(self, *, run_id: str, step_id: str, payload: dict) -> dict:
        raise NotImplementedError
