"""Additive LangGraph checkpoint seam for compact Orchestrator state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from app.a2a.orchestrator_execution_state import recompute_aggregate_state
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState, RunId


class _CheckpointConfigurable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: RunId


class _CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configurable: _CheckpointConfigurable


class OrchestratorExecutionGraphError(ValueError):
    """Compact fail-closed graph facade error."""


def _checkpoint_compact_state(
    state: OrchestratorExecutionState,
) -> dict[str, Any]:
    """Validate the compact schema; retain no command or transport payload."""
    try:
        checked = OrchestratorExecutionState.model_validate(state.model_dump())
    except ValidationError:
        raise OrchestratorExecutionGraphError("checkpoint_state_invalid") from None
    if recompute_aggregate_state(checked) != checked:
        raise ValueError("orchestrator_execution_aggregate_state_invalid")
    return checked.model_dump(mode="json")


class _ValidatedExecutionGraph:
    """Explicit facade that never exposes raw compiled mutation/batch methods."""

    def __init__(self, compiled: Any) -> None:
        self.__compiled = compiled

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        checked, sanitized_config = self._prepare_input(input, config)
        return self.__compiled.invoke(checked, config=sanitized_config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        checked, sanitized_config = await self._prepare_async_input(input, config)
        return await self.__compiled.ainvoke(
            checked, config=sanitized_config, **kwargs
        )

    def stream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Iterator[Any]:
        checked, sanitized_config = self._prepare_input(input, config)
        return self.__compiled.stream(checked, config=sanitized_config, **kwargs)

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        checked, sanitized_config = await self._prepare_async_input(input, config)
        async for item in self.__compiled.astream(
            checked, config=sanitized_config, **kwargs
        ):
            yield item

    def get_state(self, config: Any, **kwargs: Any) -> Any:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        snapshot = self.__compiled.get_state(sanitized_config, **kwargs)
        _validate_snapshot_identity(snapshot, thread_id, require_state=False)
        return snapshot

    async def aget_state(self, config: Any, **kwargs: Any) -> Any:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        snapshot = await self.__compiled.aget_state(sanitized_config, **kwargs)
        _validate_snapshot_identity(snapshot, thread_id, require_state=False)
        return snapshot

    def get_state_history(
        self, config: Any, **kwargs: Any
    ) -> Iterator[Any]:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        for snapshot in self.__compiled.get_state_history(
            sanitized_config, **kwargs
        ):
            _validate_snapshot_identity(snapshot, thread_id, require_state=False)
            yield snapshot

    async def aget_state_history(
        self, config: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        async for snapshot in self.__compiled.aget_state_history(
            sanitized_config, **kwargs
        ):
            _validate_snapshot_identity(snapshot, thread_id, require_state=False)
            yield snapshot

    def update_state(self, *args: Any, **kwargs: Any) -> Any:
        raise OrchestratorExecutionGraphError("external_state_mutation_unsupported")

    async def aupdate_state(self, *args: Any, **kwargs: Any) -> Any:
        raise OrchestratorExecutionGraphError("external_state_mutation_unsupported")

    def bulk_update_state(self, *args: Any, **kwargs: Any) -> Any:
        raise OrchestratorExecutionGraphError("external_state_mutation_unsupported")

    async def abulk_update_state(self, *args: Any, **kwargs: Any) -> Any:
        raise OrchestratorExecutionGraphError("external_state_mutation_unsupported")

    def _prepare_input(
        self, input: Any, config: Any
    ) -> tuple[dict[str, Any] | None, dict[str, dict[str, str]]]:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        if input is None:
            snapshot = self.__compiled.get_state(sanitized_config)
            _validate_snapshot_identity(snapshot, thread_id, require_state=True)
            return None, sanitized_config
        return _validated_graph_input(input, thread_id), sanitized_config

    async def _prepare_async_input(
        self, input: Any, config: Any
    ) -> tuple[dict[str, Any] | None, dict[str, dict[str, str]]]:
        sanitized_config, thread_id = _sanitized_checkpoint_config(config)
        if input is None:
            snapshot = await self.__compiled.aget_state(sanitized_config)
            _validate_snapshot_identity(snapshot, thread_id, require_state=True)
            return None, sanitized_config
        return _validated_graph_input(input, thread_id), sanitized_config


def _validated_graph_input(input: Any, thread_id: str) -> dict[str, Any]:
    payload = (
        input.model_dump()
        if isinstance(input, OrchestratorExecutionState)
        else input
    )
    try:
        checked = OrchestratorExecutionState.model_validate(payload)
    except ValidationError:
        raise OrchestratorExecutionGraphError(
            "checkpoint_state_input_invalid"
        ) from None
    if checked.run_id != thread_id:
        raise OrchestratorExecutionGraphError("checkpoint_thread_identity_mismatch")
    if recompute_aggregate_state(checked) != checked:
        raise OrchestratorExecutionGraphError(
            "orchestrator_execution_aggregate_state_invalid"
        )
    return checked.model_dump(mode="json")


def _sanitized_checkpoint_config(
    config: Any,
) -> tuple[dict[str, dict[str, str]], str]:
    try:
        checked = _CheckpointConfig.model_validate(config)
    except ValidationError:
        raise OrchestratorExecutionGraphError("checkpoint_config_invalid") from None
    sanitized = checked.model_dump(mode="json")
    return sanitized, checked.configurable.thread_id


def _validate_snapshot_identity(
    snapshot: Any, thread_id: str, *, require_state: bool
) -> None:
    values = getattr(snapshot, "values", None)
    if not values:
        if require_state:
            raise OrchestratorExecutionGraphError("checkpoint_state_missing")
        return
    try:
        checked = OrchestratorExecutionState.model_validate(values)
    except ValidationError:
        raise OrchestratorExecutionGraphError("checkpoint_state_invalid") from None
    if checked.run_id != thread_id:
        raise OrchestratorExecutionGraphError("checkpoint_thread_identity_mismatch")


def build_orchestrator_execution_graph(*, checkpointer: Any) -> Any:
    """Compile the isolated execution-state graph with an injected saver."""
    if checkpointer is None:
        raise ValueError("orchestrator_execution_checkpointer_required")
    builder = StateGraph(OrchestratorExecutionState)
    builder.add_node("checkpoint_execution_state", _checkpoint_compact_state)
    builder.add_edge(START, "checkpoint_execution_state")
    builder.add_edge("checkpoint_execution_state", END)
    return _ValidatedExecutionGraph(builder.compile(checkpointer=checkpointer))


def execution_graph_config(run_id: str) -> dict[str, dict[str, str]]:
    """Return the required run-scoped LangGraph checkpoint identity."""
    try:
        checked = _CheckpointConfig.model_validate(
            {"configurable": {"thread_id": run_id}}
        )
    except ValidationError:
        raise OrchestratorExecutionGraphError(
            "orchestrator_execution_run_id_invalid"
        ) from None
    return checked.model_dump(mode="json")


__all__ = [
    "OrchestratorExecutionGraphError",
    "build_orchestrator_execution_graph",
    "execution_graph_config",
]
