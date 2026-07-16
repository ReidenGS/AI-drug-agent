"""Cross-process LocalStorage mutation regressions for shared worker volumes."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_local import LocalStorage
from app.services.storage_s3 import S3Storage
from app.services.storage_service import AtomicJsonStorage
from app.services.workflow_state_service import WorkflowStateService


class _ContendedLocalStorage(LocalStorage):
    """Makes the former unlocked read path deterministically race.

    Production ``atomic_update_json`` deliberately reads beneath this hook
    while holding its file lock. The hook is only a regression fixture; it
    adds no delay, cap, or branch to production behavior.
    """

    def __init__(self, root: str, prefix: str, barrier: Any, suffix: str) -> None:
        super().__init__(root, prefix)
        self._barrier = barrier
        self._suffix = suffix

    def read_json(self, key: str) -> dict:
        payload = super().read_json(key)
        if key.endswith(self._suffix):
            self._barrier.wait(timeout=5)
        return payload


def _update_registry(
    root: str,
    prefix: str,
    run_id: str,
    field: str,
    artifact_id: str,
    barrier: Any,
    queue: Any,
) -> None:
    storage = _ContendedLocalStorage(
        root, prefix, barrier, "registry/current.json"
    )
    try:
        result = ArtifactRegistryService(storage).update_active(
            run_id, **{field: artifact_id}
        )
        queue.put(("ok", result.version))
    except Exception as exc:  # pragma: no cover - surfaced in parent
        queue.put(("error", type(exc).__name__))


def _mark_workflow(
    root: str,
    prefix: str,
    run_id: str,
    step_key: str,
    barrier: Any,
    queue: Any,
) -> None:
    storage = _ContendedLocalStorage(
        root, prefix, barrier, "state/workflow_state.json"
    )
    try:
        result = WorkflowStateService(storage).mark(
            run_id, step_key, "completed"
        )
        queue.put(("ok", result["steps"][step_key]))
    except Exception as exc:  # pragma: no cover - surfaced in parent
        queue.put(("error", type(exc).__name__))


def _run_processes(target, args_by_process):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(args_by_process))
    queue = context.Queue()
    processes = [
        context.Process(target=target, args=(*args, barrier, queue))
        for args in args_by_process
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    results = [queue.get(timeout=2) for _ in processes]
    assert all(item[0] == "ok" for item in results)
    return results


def test_registry_updates_are_atomic_across_independent_processes(tmp_path: Path):
    root = str(tmp_path / "shared")
    prefix = "adc_pilot"
    run_id = "run_parallel_registry"
    storage = LocalStorage(root, prefix)
    service = ArtifactRegistryService(storage)
    service.init_registry(run_id)

    results = _run_processes(
        _update_registry,
        [
            (
                root,
                prefix,
                run_id,
                "structured_liability_summary_id",
                "structured_liability_summary_111111111111",
            ),
            (
                root,
                prefix,
                run_id,
                "prepared_structure_input_package_id",
                "prepared_structure_input_package_222222222222",
            ),
        ],
    )

    current = service.get(run_id)
    assert sorted(item[1] for item in results) == [2, 3]
    assert current.version == 3
    assert current.active_artifacts.structured_liability_summary_id == (
        "structured_liability_summary_111111111111"
    )
    assert current.active_artifacts.prepared_structure_input_package_id == (
        "prepared_structure_input_package_222222222222"
    )
    snapshots = [
        storage.read_json(
            storage.run_key(run_id, f"registry/snapshot_{version:04d}.json")
        )
        for version in (1, 2)
    ]
    assert [item["version"] for item in snapshots] == [1, 2]
    assert all(isinstance(item["active_artifacts"], dict) for item in snapshots)


def test_workflow_marks_are_atomic_across_independent_processes(tmp_path: Path):
    root = str(tmp_path / "shared")
    prefix = "adc_pilot"
    run_id = "run_parallel_workflow"
    storage = LocalStorage(root, prefix)
    service = WorkflowStateService(storage)
    service.init_run(run_id)

    results = _run_processes(
        _mark_workflow,
        [
            (root, prefix, run_id, "step_06"),
            (root, prefix, run_id, "step_07"),
        ],
    )

    state = service.get(run_id)
    assert results == [("ok", "completed"), ("ok", "completed")]
    assert state["steps"]["step_06"] == "completed"
    assert state["steps"]["step_07"] == "completed"
    assert storage.read_json(
        storage.run_key(run_id, "state/workflow_state.json")
    ) == state


def test_atomic_mutation_capability_is_local_only_and_not_claimed_for_s3(tmp_path: Path):
    assert isinstance(LocalStorage(str(tmp_path), "adc_pilot"), AtomicJsonStorage)
    assert not isinstance(
        S3Storage("test-bucket", "adc_pilot", "us-east-1"),
        AtomicJsonStorage,
    )
