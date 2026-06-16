"""A2A runtime — assembles python-a2a Server/Client and agent registry.

HARD CONSTRAINT (README_FOR_CLAUDE.md): do NOT handwrite A2A protocol envelopes
or task frames. Use only the primitives provided by `python-a2a`.
"""

from __future__ import annotations

from typing import Any


class A2ARuntime:
    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> Any:
        return self._agents[name]

    def start_server(self) -> None:
        # Wire python-a2a Server here; intentionally not implemented in the skeleton.
        raise NotImplementedError("Wire python-a2a Server in real deployment")
