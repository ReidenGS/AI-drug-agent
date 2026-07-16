"""Lifecycle tests for the production AsyncPostgresSaver runtime factory."""

from __future__ import annotations

import asyncio
import pickle

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr, ValidationError

from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
    OrchestratorPostgresCheckpointRuntime,
    build_production_checkpoint_runtime,
)
from app.settings import Settings


class _AdvisoryDatabase:
    def __init__(self, *, blocked_phase=None):
        self.blocked_phase = blocked_phase
        self.phase_entered = asyncio.Event()
        self.release_phase = asyncio.Event()
        self.locked = False
        self.connections = []

    async def connect(self, _dsn, *, autocommit):
        assert autocommit is True
        if self.blocked_phase == "connect":
            self.phase_entered.set()
            await self.release_phase.wait()
        connection = _AdvisoryConnection(self)
        self.connections.append(connection)
        return connection

    async def block(self, phase):
        if self.blocked_phase == phase:
            self.phase_entered.set()
            await self.release_phase.wait()


class _AdvisoryConnection:
    def __init__(self, database):
        self.database = database
        self.closed = False
        self.owns_lock = False

    def cursor(self):
        return _AdvisoryCursor(self)

    async def close(self):
        await self.database.block("close")
        if self.owns_lock:
            self.database.locked = False
            self.owns_lock = False
        self.closed = True


class _AdvisoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.row = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, sql, _params):
        if "pg_try_advisory_lock" in sql:
            await self.connection.database.block("query")
            acquired = not self.connection.database.locked
            if acquired:
                self.connection.database.locked = True
                self.connection.owns_lock = True
            self.row = (acquired,)
            return
        await self.connection.database.block("unlock")
        if self.connection.owns_lock:
            self.connection.database.locked = False
            self.connection.owns_lock = False
        self.row = (True,)

    async def fetchone(self):
        await self.connection.database.block("fetchone")
        return self.row


class _LifecycleSaver(InMemorySaver):
    def __init__(self):
        super().__init__()
        self.setup_count = 0

    async def setup(self):
        self.setup_count += 1


class _SaverContext:
    def __init__(self, saver, audit):
        self.saver = saver
        self.audit = audit

    async def __aenter__(self):
        self.audit.append("enter")
        return self.saver

    async def __aexit__(self, *_args):
        self.audit.append("exit")


@pytest.mark.asyncio
async def test_runtime_setup_graph_and_shutdown_are_explicit():
    saver = _LifecycleSaver()
    audit = []
    runtime = OrchestratorPostgresCheckpointRuntime(
        SecretStr("postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint"),
        saver_context_factory=lambda _dsn: _SaverContext(saver, audit),
    )
    assert "private" not in repr(runtime)
    with pytest.raises(
        TypeError, match="^checkpoint_runtime_pickle_unsupported$"
    ) as caught:
        pickle.dumps(runtime)
    assert "private" not in repr(caught.value)
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_not_started$",
    ):
        _ = runtime.graph
    await runtime.startup()
    assert runtime.saver is saver
    assert saver.setup_count == 1
    assert audit == ["enter"]
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_already_started$",
    ):
        await runtime.startup()
    await runtime.shutdown()
    assert audit == ["enter", "exit"]


@pytest.mark.asyncio
async def test_independent_runtimes_each_execute_database_setup():
    audit = []
    savers = [_LifecycleSaver(), _LifecycleSaver()]
    for saver in savers:
        runtime = OrchestratorPostgresCheckpointRuntime(
            "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
            saver_context_factory=lambda _dsn, saver=saver: _SaverContext(
                saver, audit
            ),
        )
        async with runtime:
            assert runtime.saver is saver
    assert [item.setup_count for item in savers] == [1, 1]
    assert audit == ["enter", "exit", "enter", "exit"]


def test_production_factory_requires_explicit_secret_postgres_url():
    missing = Settings(_env_file=None, langgraph_checkpoint_database_url=None)
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_database_url_required$",
    ):
        build_production_checkpoint_runtime(missing)
    configured = Settings(
        _env_file=None,
        langgraph_checkpoint_database_url=SecretStr(
            "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint"
        ),
    )
    runtime = build_production_checkpoint_runtime(configured)
    assert "private" not in repr(runtime)


@pytest.mark.asyncio
async def test_connection_or_migration_failure_is_compact_without_fallback():
    class _BrokenContext:
        async def __aenter__(self):
            raise RuntimeError("postgresql://user:secret@private-db/raw")

        async def __aexit__(self, *_args):
            return None

    runtime = OrchestratorPostgresCheckpointRuntime(
        "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
        saver_context_factory=lambda _dsn: _BrokenContext(),
    )
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ) as caught:
        await runtime.startup()
    assert "secret" not in repr(caught.value)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_checkpoint_startup_timeout_must_be_finite_positive(timeout):
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_startup_timeout_invalid$",
    ):
        OrchestratorPostgresCheckpointRuntime(
            "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
            startup_timeout_seconds=timeout,
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_worker_execution_timeout_when_configured_must_be_finite_positive(
    timeout,
):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            orchestrator_worker_timeout_seconds=timeout,
        )


def test_worker_execution_timeout_has_no_guessed_settings_default():
    assert Settings(
        _env_file=None
    ).orchestrator_worker_timeout_seconds is None


@pytest.mark.asyncio
async def test_setup_timeout_fails_compact_and_closes_entered_context():
    class _HangingSaver(InMemorySaver):
        async def setup(self):
            await asyncio.Event().wait()

    audit = []
    runtime = OrchestratorPostgresCheckpointRuntime(
        "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
        startup_timeout_seconds=0.02,
        saver_context_factory=lambda _dsn: _SaverContext(
            _HangingSaver(), audit
        ),
    )
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ):
        await runtime.startup()
    assert audit == ["enter", "exit"]
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_not_started$",
    ):
        _ = runtime.graph


@pytest.mark.asyncio
async def test_unavailable_migration_lock_times_out_and_closes_connection():
    class _Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, *_args):
            return None

        async def fetchone(self):
            return {"pg_try_advisory_lock": False}

    class _Connection:
        def cursor(self):
            return _Cursor()

    class _LockedSaver:
        conn = _Connection()

        async def setup(self):
            raise AssertionError("setup must not run without migration lock")

    audit = []
    runtime = OrchestratorPostgresCheckpointRuntime(
        "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
        startup_timeout_seconds=0.02,
        saver_context_factory=lambda _dsn: _SaverContext(
            _LockedSaver(), audit
        ),
    )
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ):
        await runtime.startup()
    assert audit == ["enter", "exit"]


@pytest.mark.asyncio
async def test_cancelled_startup_closes_entered_context_and_propagates_cancel():
    class _HangingSaver(InMemorySaver):
        async def setup(self):
            await asyncio.Event().wait()

    audit = []
    runtime = OrchestratorPostgresCheckpointRuntime(
        "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
        startup_timeout_seconds=10,
        saver_context_factory=lambda _dsn: _SaverContext(
            _HangingSaver(), audit
        ),
    )
    task = asyncio.create_task(runtime.startup())
    while audit != ["enter"]:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert audit == ["enter", "exit"]


def test_settings_exposes_documented_startup_timeout():
    settings = Settings(_env_file=None)
    assert settings.langgraph_checkpoint_startup_timeout_seconds == 30.0
    for invalid in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Settings(
                _env_file=None,
                langgraph_checkpoint_startup_timeout_seconds=invalid,
            )


def _lock_runtime(database, *, timeout=1):
    return OrchestratorPostgresCheckpointRuntime(
        "postgresql://checkpoint_user:private@checkpoint-db/adc_checkpoint",
        startup_timeout_seconds=timeout,
        run_lock_connection_factory=database.connect,
    )


async def _assert_subsequent_runtime_can_lock(database):
    database.blocked_phase = None
    second = _lock_runtime(database)
    async with second.run_lock("run_20260715_abcdef12"):
        assert database.locked is True
    assert database.locked is False
    assert database.connections[-1].closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connect", "query", "fetchone"])
async def test_run_lock_acquisition_cancellation_quiesces_and_closes(phase):
    database = _AdvisoryDatabase(blocked_phase=phase)
    context = _lock_runtime(database).run_lock("run_20260715_abcdef12")
    entering = asyncio.create_task(context.__aenter__())
    await asyncio.wait_for(database.phase_entered.wait(), timeout=1)

    entering.cancel()
    await asyncio.sleep(0)
    assert not entering.done()
    database.release_phase.set()
    with pytest.raises(asyncio.CancelledError):
        await entering

    assert database.locked is False
    assert database.connections
    assert all(item.closed for item in database.connections)
    await _assert_subsequent_runtime_can_lock(database)


@pytest.mark.asyncio
async def test_run_lock_body_cancellation_unlocks_closes_and_propagates():
    database = _AdvisoryDatabase()
    entered = asyncio.Event()

    async def holder():
        async with _lock_runtime(database).run_lock(
            "run_20260715_abcdef12"
        ):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(holder())
    await entered.wait()
    assert database.locked is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert database.locked is False
    assert database.connections[0].closed is True
    await _assert_subsequent_runtime_can_lock(database)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["unlock", "close"])
async def test_run_lock_release_cancellation_finishes_cleanup(phase):
    database = _AdvisoryDatabase(blocked_phase=phase)
    body_entered = asyncio.Event()
    leave_body = asyncio.Event()

    async def holder():
        async with _lock_runtime(database).run_lock(
            "run_20260715_abcdef12"
        ):
            body_entered.set()
            await leave_body.wait()

    task = asyncio.create_task(holder())
    await body_entered.wait()
    leave_body.set()
    await asyncio.wait_for(database.phase_entered.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    database.release_phase.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert database.locked is False
    assert database.connections[0].closed is True
    await _assert_subsequent_runtime_can_lock(database)


@pytest.mark.asyncio
async def test_run_lock_database_io_timeout_is_compact_and_closes_connection():
    database = _AdvisoryDatabase(blocked_phase="query")
    runtime = _lock_runtime(database, timeout=0.02)
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_run_lock_timeout$",
    ) as caught:
        async with runtime.run_lock("run_20260715_abcdef12"):
            pass
    assert "private" not in str(caught.value)
    assert "SELECT" not in repr(caught.value)
    assert database.connections[0].closed is True
    assert database.locked is False
