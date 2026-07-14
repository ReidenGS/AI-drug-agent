"""Lifecycle tests for the production AsyncPostgresSaver runtime factory."""

from __future__ import annotations

import asyncio
import pickle

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
    OrchestratorPostgresCheckpointRuntime,
    build_production_checkpoint_runtime,
)
from app.settings import Settings


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
