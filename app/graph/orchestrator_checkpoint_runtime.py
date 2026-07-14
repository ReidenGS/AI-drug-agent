"""Production lifecycle for the durable Orchestrator Postgres checkpointer."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import SecretStr

from app.settings import Settings

from .orchestrator_execution_graph import build_orchestrator_execution_graph

_MIGRATION_LOCK_KEYS = (0x414443, 0xF2C)


class OrchestratorCheckpointRuntimeError(RuntimeError):
    """Compact error that never exposes the checkpoint DSN or raw DB error."""


class OrchestratorPostgresCheckpointRuntime:
    """Own one AsyncPostgresSaver connection and its compiled graph.

    ``startup`` is the explicit migration boundary. It is called once per
    runtime instance; database-level migration idempotence remains the saver's
    responsibility. There is deliberately no InMemory fallback.
    """

    def __init__(
        self,
        database_url: SecretStr | str,
        *,
        startup_timeout_seconds: float = 30.0,
        saver_context_factory: Callable[..., AbstractAsyncContextManager[Any]] = (
            AsyncPostgresSaver.from_conn_string
        ),
    ) -> None:
        raw = (
            database_url.get_secret_value()
            if isinstance(database_url, SecretStr)
            else database_url
        )
        if not isinstance(raw, str) or not raw.strip():
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_database_url_required"
            )
        if not raw.startswith(("postgresql://", "postgres://")):
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_database_url_invalid"
            )
        self.__database_url = raw
        try:
            timeout = float(startup_timeout_seconds)
        except (TypeError, ValueError):
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_startup_timeout_invalid"
            ) from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_startup_timeout_invalid"
            )
        self.__startup_timeout = timeout
        self.__factory = saver_context_factory
        self.__context: AbstractAsyncContextManager[Any] | None = None
        self.__saver: Any | None = None
        self.__graph: Any | None = None

    @classmethod
    def from_settings(
        cls, settings: Settings, **kwargs: Any
    ) -> OrchestratorPostgresCheckpointRuntime:
        dsn = settings.langgraph_checkpoint_database_url
        if dsn is None or not dsn.get_secret_value().strip():
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_database_url_required"
            )
        kwargs.setdefault(
            "startup_timeout_seconds",
            settings.langgraph_checkpoint_startup_timeout_seconds,
        )
        return cls(dsn, **kwargs)

    @property
    def saver(self) -> Any:
        if self.__saver is None:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_not_started"
            )
        return self.__saver

    @property
    def graph(self) -> Any:
        if self.__graph is None:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_not_started"
            )
        return self.__graph

    async def startup(self) -> OrchestratorPostgresCheckpointRuntime:
        if self.__context is not None:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_already_started"
            )
        context = self.__factory(self.__database_url)
        entered = False

        async def initialize() -> tuple[Any, Any]:
            nonlocal entered
            saver = await context.__aenter__()
            entered = True
            await _setup_with_database_lock(saver)
            return saver, build_orchestrator_execution_graph(
                checkpointer=saver
            )

        try:
            saver, graph = await asyncio.wait_for(
                initialize(), timeout=self.__startup_timeout
            )
        except BaseException as exc:
            if entered:
                try:
                    await context.__aexit__(None, None, None)
                except Exception:
                    pass
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_startup_failed"
            ) from None
        self.__context = context
        self.__saver = saver
        self.__graph = graph
        return self

    async def shutdown(self) -> None:
        context = self.__context
        self.__graph = None
        self.__saver = None
        self.__context = None
        if context is None:
            return
        try:
            await context.__aexit__(None, None, None)
        except Exception:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_shutdown_failed"
            ) from None

    async def __aenter__(self) -> OrchestratorPostgresCheckpointRuntime:
        return await self.startup()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        return "OrchestratorPostgresCheckpointRuntime(started=%s)" % (
            self.__saver is not None
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("checkpoint_runtime_pickle_unsupported")


def build_production_checkpoint_runtime(
    settings: Settings,
) -> OrchestratorPostgresCheckpointRuntime:
    """Build an unstarted production runtime; caller owns startup/shutdown."""
    return OrchestratorPostgresCheckpointRuntime.from_settings(settings)


async def _setup_with_database_lock(saver: Any) -> None:
    """Serialize official migrations across processes with a Postgres lock."""
    conn = getattr(saver, "conn", None)
    if conn is None:  # explicit injected unit-test saver only
        await saver.setup()
        return
    acquired = False
    while not acquired:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                _MIGRATION_LOCK_KEYS,
            )
            row = await cursor.fetchone()
            if row is None:
                acquired = False
            else:
                value = (
                    next(iter(row.values()))
                    if isinstance(row, Mapping)
                    else row[0]
                )
                acquired = bool(value)
        if not acquired:
            # A blocking advisory-lock statement keeps a virtual transaction
            # open while waiting.  That can deadlock with the saver's
            # CREATE INDEX CONCURRENTLY migration.  Short try-lock statements
            # let each waiting transaction finish before the next poll.
            await asyncio.sleep(0.1)
    try:
        await saver.setup()
    finally:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT pg_advisory_unlock(%s, %s)", _MIGRATION_LOCK_KEYS
            )


__all__ = [
    "OrchestratorCheckpointRuntimeError",
    "OrchestratorPostgresCheckpointRuntime",
    "build_production_checkpoint_runtime",
]
