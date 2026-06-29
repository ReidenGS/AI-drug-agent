"""Hydrate a pipeline snapshot into a fresh run (no LLM, no MCP, no steps).

Generic — works for any ``through_step`` snapshot. Example:

    python -m scripts.hydrate_pipeline_snapshot \
        --snapshot-dir ./.snapshots/my_step6_snapshot

Prints a compact summary only; never raw file/tool/prompt content.
"""

from __future__ import annotations

import argparse

from app.deps import get_storage
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.pipeline_snapshot_service import PipelineSnapshotService
from app.services.workflow_state_service import WorkflowStateService


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate a pipeline artifact snapshot.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--run-id", default=None, help="optional explicit hydrated run id")
    args = parser.parse_args()

    storage = get_storage()
    service = PipelineSnapshotService(
        storage,
        ArtifactRegistryService(storage),
        WorkflowStateService(storage),
    )
    hydrated_run_id = service.hydrate_pipeline_snapshot(
        snapshot_dir=args.snapshot_dir, new_run_id=args.run_id
    )

    registry = ArtifactRegistryService(storage).get(hydrated_run_id)
    workflow = WorkflowStateService(storage).get(hydrated_run_id)
    completed = sorted(k for k, v in (workflow.get("steps") or {}).items() if v == "completed")
    active = registry.active_artifacts.model_dump()

    print("=== pipeline snapshot hydrated ===")
    print("hydrated_run_id:", hydrated_run_id)
    print("completed_steps:", ", ".join(completed))
    print("active_artifact_count:", sum(1 for v in active.values() if v))


if __name__ == "__main__":
    main()
