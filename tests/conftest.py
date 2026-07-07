"""Shared fixtures: tmp local storage + service factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_local import LocalStorage
from app.services.workflow_state_service import WorkflowStateService


@pytest.fixture(autouse=True)
def _default_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic even when a local .env enables live providers."""
    from app.settings import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("MCP_LIVE_TOOLS", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def local_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(root=str(tmp_path / "store"), prefix="adc_pilot")


@pytest.fixture
def registry_service(local_storage: LocalStorage) -> ArtifactRegistryService:
    return ArtifactRegistryService(storage=local_storage)


@pytest.fixture
def workflow_state_service(local_storage: LocalStorage) -> WorkflowStateService:
    return WorkflowStateService(storage=local_storage)
