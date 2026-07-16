"""Production lifecycle for the durable Orchestrator Postgres checkpointer."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Callable

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
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
        run_lock_connection_factory: Callable[..., Any] = AsyncConnection.connect,
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
        self.__run_lock_connection_factory = run_lock_connection_factory
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

    @asynccontextmanager
    async def run_lock(self, run_id: str):
        """Acquire a non-blocking cross-process mutation lock for one run.

        A separate Postgres session owns the advisory lock so checkpoint saver
        traffic cannot accidentally release it. A competing backend fails
        compactly instead of planning or dispatching the same run twice.
        """
        from .orchestrator_execution_graph import execution_graph_config

        try:
            execution_graph_config(run_id)
        except Exception:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_run_lock_identity_invalid"
            ) from None
        lock_key = int.from_bytes(
            hashlib.sha256(run_id.encode("ascii")).digest()[:8],
            byteorder="big",
            signed=True,
        )
        connection = None
        try:
            connection = await _await_run_lock_operation(
                self.__run_lock_connection_factory(
                    self.__database_url, autocommit=True
                ),
                timeout_seconds=self.__startup_timeout,
                cancellation_result_cleanup=lambda created: (
                    _cleanup_run_lock_connection(
                        created,
                        lock_key=lock_key,
                        acquired=False,
                        timeout_seconds=self.__startup_timeout,
                        suppress_errors=True,
                    )
                ),
            )
            row = await _await_run_lock_operation(
                _try_advisory_lock(connection, lock_key),
                timeout_seconds=self.__startup_timeout,
            )
            value = (
                next(iter(row.values()))
                if isinstance(row, Mapping)
                else row[0] if row is not None else False
            )
            acquired = bool(value)
        except asyncio.CancelledError:
            if connection is not None:
                await _cleanup_run_lock_connection(
                    connection,
                    lock_key=lock_key,
                    acquired=False,
                    timeout_seconds=self.__startup_timeout,
                    suppress_errors=True,
                )
            raise
        except TimeoutError:
            if connection is not None:
                await _cleanup_run_lock_connection(
                    connection,
                    lock_key=lock_key,
                    acquired=False,
                    timeout_seconds=self.__startup_timeout,
                    suppress_errors=True,
                )
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_run_lock_timeout"
            ) from None
        except BaseException:
            if connection is not None:
                await _cleanup_run_lock_connection(
                    connection,
                    lock_key=lock_key,
                    acquired=False,
                    timeout_seconds=self.__startup_timeout,
                    suppress_errors=True,
                )
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_run_lock_failed"
            ) from None
        if not acquired:
            await _cleanup_run_lock_connection(
                connection,
                lock_key=lock_key,
                acquired=False,
                timeout_seconds=self.__startup_timeout,
                suppress_errors=True,
            )
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_run_lock_unavailable"
            )
        try:
            yield
        except BaseException:
            await _cleanup_run_lock_connection(
                connection,
                lock_key=lock_key,
                acquired=True,
                timeout_seconds=self.__startup_timeout,
                suppress_errors=True,
            )
            raise
        else:
            cleanup_error = await _cleanup_run_lock_connection(
                connection,
                lock_key=lock_key,
                acquired=True,
                timeout_seconds=self.__startup_timeout,
                suppress_errors=False,
            )
            if cleanup_error:
                raise OrchestratorCheckpointRuntimeError(
                    "checkpoint_run_lock_release_failed"
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


async def _await_run_lock_operation(
    awaitable: Any,
    *,
    timeout_seconds: float,
    cancellation_result_cleanup: Callable[[Any], Any] | None = None,
) -> Any:
    """Await one DB operation without abandoning it on caller cancellation."""
    operation = asyncio.create_task(
        asyncio.wait_for(awaitable, timeout=timeout_seconds)
    )
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
        try:
            result = operation.result()
        except BaseException:
            result = None
        if result is not None and cancellation_result_cleanup is not None:
            try:
                await cancellation_result_cleanup(result)
            except BaseException:
                pass
        raise


async def _try_advisory_lock(connection: Any, lock_key: int) -> Any:
    async with connection.cursor() as cursor:
        await cursor.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
        return await cursor.fetchone()


async def _unlock_advisory_lock(connection: Any, lock_key: int) -> None:
    async with connection.cursor() as cursor:
        await cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))


async def _cleanup_run_lock_connection(
    connection: Any,
    *,
    lock_key: int,
    acquired: bool,
    timeout_seconds: float,
    suppress_errors: bool,
) -> bool:
    """Bounded, cancellation-safe unlock/close; return whether cleanup failed."""

    async def cleanup() -> bool:
        failed = False
        if acquired:
            try:
                await asyncio.wait_for(
                    _unlock_advisory_lock(connection, lock_key),
                    timeout=timeout_seconds,
                )
            except BaseException:
                failed = True
        try:
            await asyncio.wait_for(
                connection.close(), timeout=timeout_seconds
            )
        except BaseException:
            failed = True
        return failed

    cleanup_task = asyncio.create_task(cleanup())
    cancelled: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    failed = cleanup_task.result()
    if cancelled is not None:
        raise cancelled
    return False if suppress_errors else failed


__all__ = [
    "OrchestratorCheckpointRuntimeError",
    "OrchestratorPostgresCheckpointRuntime",
    "build_production_checkpoint_runtime",
]
