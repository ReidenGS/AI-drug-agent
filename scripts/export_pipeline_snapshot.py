"""Export a reusable pipeline snapshot for any run, through any step.

Generic — not tied to Step 1-6. Example:

    python -m scripts.export_pipeline_snapshot \
        --run-id run_20260629_abcd1234 \
        --through-step step_06 \
        --output-dir ./.snapshots/my_step6_snapshot

Prints a compact summary only; never raw file/tool/prompt content.
"""

from __future__ import annotations

import argparse

from app.deps import get_storage
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.pipeline_snapshot_service import PipelineSnapshotService
from app.services.workflow_state_service import WorkflowStateService


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pipeline artifact snapshot.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--through-step", default="step_06")
    parser.add_argument("--output-dir", required=True)
    tool_grp = parser.add_mutually_exclusive_group()
    tool_grp.add_argument("--include-tool-outputs", dest="include_tool_outputs", action="store_true")
    tool_grp.add_argument("--no-tool-outputs", dest="include_tool_outputs", action="store_false")
    parser.set_defaults(include_tool_outputs=True)
    up_grp = parser.add_mutually_exclusive_group()
    up_grp.add_argument("--include-uploaded-files", dest="include_uploaded_files", action="store_true")
    up_grp.add_argument("--no-uploaded-files", dest="include_uploaded_files", action="store_false")
    parser.set_defaults(include_uploaded_files=True)
    args = parser.parse_args()

    storage = get_storage()
    service = PipelineSnapshotService(
        storage,
        ArtifactRegistryService(storage),
        WorkflowStateService(storage),
    )
    manifest = service.export_pipeline_snapshot(
        run_id=args.run_id,
        through_step=args.through_step,
        output_dir=args.output_dir,
        include_tool_outputs=args.include_tool_outputs,
        include_uploaded_files=args.include_uploaded_files,
    )

    print("=== pipeline snapshot exported ===")
    print("snapshot_id:", manifest.snapshot_id)
    print("source_run_id:", manifest.source_run_id)
    print("through_step:", manifest.through_step)
    print("completed_steps:", ", ".join(manifest.completed_steps))
    print("artifact_files:", len(manifest.artifact_files))
    print("uploaded_files:", len(manifest.uploaded_files))
    print("tool_output_files:", len(manifest.tool_output_files))
    print("output_dir:", args.output_dir)
    if manifest.notes:
        print("notes:", manifest.notes)


if __name__ == "__main__":
    main()
