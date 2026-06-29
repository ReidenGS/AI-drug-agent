"""Generic pipeline artifact snapshot / hydrate.

Export any run's completed-prefix artifacts (through an arbitrary
``through_step``) into a portable snapshot directory, then hydrate that
snapshot into a fresh run so downstream steps can continue WITHOUT re-running
the prior LLM / tool-heavy steps.

This layer is intentionally step-agnostic: artifacts are discovered from the
``ArtifactRegistryService`` active artifacts plus the run's files in storage —
nothing is hard-coded to Step 1-6. ``through_step`` is used only to validate
completion and to bound the exported range (tool outputs of later steps are
skipped).

Hydrate performs pure file I/O: it copies bytes back into the new run and
rewrites run-scoped storage keys / the top-level ``run_id`` so refs stay
intact. It calls NO LLM, NO MCP, and runs NO agent step.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from ..schemas.pipeline_snapshot import (
    ArtifactSnapshotEntry,
    PipelineSnapshotManifest,
    UploadedFileSnapshotEntry,
)
from ..utils.ids import new_artifact_id, new_run_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_MANIFEST_NAME = "snapshot_manifest.json"
_REGISTRY_PREFIX = "registry/"
_WORKFLOW_STATE_REL = "state/workflow_state.json"
_UPLOADED_FILES_PREFIX = "inputs/files/"
_TOOL_OUTPUTS_PREFIX = "tool_outputs/"
_STEP_DIR_RE = re.compile(r"step_(\d+)")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _step_num(step_key: str) -> Optional[int]:
    m = _STEP_DIR_RE.search(step_key or "")
    return int(m.group(1)) if m else None


class PipelineSnapshotService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    # ── export ────────────────────────────────────────────────────────────────

    def export_pipeline_snapshot(
        self,
        run_id: str,
        through_step: str,
        output_dir: str,
        include_tool_outputs: bool = True,
        include_uploaded_files: bool = True,
    ) -> PipelineSnapshotManifest:
        try:
            registry = self.registry.get(run_id)
        except Exception as exc:  # noqa: BLE001 — surface a clear domain error
            raise ValueError(f"run {run_id!r} not found or has no registry: {exc}") from exc
        try:
            workflow = self.workflow_state.get(run_id)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"run {run_id!r} has no workflow state: {exc}") from exc

        steps = workflow.get("steps") or {}
        if through_step not in steps:
            raise ValueError(
                f"through_step {through_step!r} is not a known workflow step"
            )
        if steps.get(through_step) != "completed":
            raise ValueError(
                f"through_step {through_step!r} is not completed "
                f"(status={steps.get(through_step)!r})"
            )
        through_num = _step_num(through_step)

        active = registry.active_artifacts.model_dump()
        if not any(active.values()):
            raise ValueError(
                f"run {run_id!r} has no active artifacts to snapshot"
            )

        completed_steps = sorted(k for k, v in steps.items() if v == "completed")

        # Discover all run files, then classify generically.
        run_prefix = self.storage.run_key(run_id)
        all_keys = self.storage.list_prefix(run_prefix)

        # Map uploaded-file storage_path -> metadata from the raw_request_record.
        uploaded_meta = self._uploaded_file_metadata(run_id)

        out_root = Path(output_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        artifact_files: list[ArtifactSnapshotEntry] = []
        uploaded_files: list[UploadedFileSnapshotEntry] = []
        tool_output_files: list[ArtifactSnapshotEntry] = []
        notes: dict = {}
        skipped_later_tool_outputs = 0

        for key in all_keys:
            rel = key[len(run_prefix):].lstrip("/")
            if not rel or rel.endswith(_MANIFEST_NAME):
                continue

            if rel.startswith(_UPLOADED_FILES_PREFIX):
                if not include_uploaded_files:
                    continue
                data = self.storage.read_bytes(key)
                snap_rel = f"files/{Path(rel).name}"
                self._write_snapshot_bytes(out_root, snap_rel, data)
                meta = uploaded_meta.get(key, {})
                uploaded_files.append(
                    UploadedFileSnapshotEntry(
                        file_id=str(meta.get("file_id") or Path(rel).stem),
                        original_filename=meta.get("original_filename"),
                        inferred_role=meta.get("role"),
                        content_type=meta.get("content_type"),
                        source_storage_ref=key,
                        run_relative_path=rel,
                        snapshot_relative_path=snap_rel,
                        sha256=_sha256(data),
                        size_bytes=len(data),
                    )
                )
                continue

            if rel.startswith(_TOOL_OUTPUTS_PREFIX):
                if not include_tool_outputs:
                    continue
                step_num = _step_num(rel)
                if through_num is not None and step_num is not None and step_num > through_num:
                    skipped_later_tool_outputs += 1
                    continue
                data = self.storage.read_bytes(key)
                snap_rel = f"tool_outputs/{rel[len(_TOOL_OUTPUTS_PREFIX):]}"
                self._write_snapshot_bytes(out_root, snap_rel, data)
                tool_output_files.append(
                    ArtifactSnapshotEntry(
                        artifact_name=Path(rel).stem,
                        artifact_type="tool_output",
                        artifact_id=self._maybe_artifact_id(data),
                        source_storage_key=key,
                        run_relative_path=rel,
                        snapshot_relative_path=snap_rel,
                        content_type="application/json",
                        sha256=_sha256(data),
                        size_bytes=len(data),
                    )
                )
                continue

            # Registry, workflow state, and every other JSON artifact restore
            # generically from their run-relative path.
            data = self.storage.read_bytes(key)
            snap_rel = f"artifacts/{rel}"
            self._write_snapshot_bytes(out_root, snap_rel, data)
            if rel.startswith(_REGISTRY_PREFIX):
                artifact_type = "registry"
            elif rel == _WORKFLOW_STATE_REL:
                artifact_type = "workflow_state"
            else:
                artifact_type = "artifact"
            artifact_files.append(
                ArtifactSnapshotEntry(
                    artifact_name=Path(rel).stem,
                    artifact_type=artifact_type,
                    artifact_id=self._maybe_artifact_id(data),
                    source_storage_key=key,
                    run_relative_path=rel,
                    snapshot_relative_path=snap_rel,
                    content_type="application/json",
                    sha256=_sha256(data),
                    size_bytes=len(data),
                )
            )

        later_completed = [
            s for s in completed_steps
            if (_step_num(s) or 0) > (through_num or 0)
        ]
        if later_completed:
            notes["completed_steps_beyond_through_step"] = later_completed
        if skipped_later_tool_outputs:
            notes["skipped_tool_outputs_beyond_through_step"] = skipped_later_tool_outputs
        if not include_tool_outputs:
            notes["tool_outputs_excluded"] = True
        if not include_uploaded_files:
            notes["uploaded_files_excluded"] = True

        manifest = PipelineSnapshotManifest(
            snapshot_id=new_artifact_id("pipeline_snapshot"),
            created_at=now_iso(),
            source_run_id=run_id,
            through_step=through_step,
            completed_steps=completed_steps,
            active_artifacts=active,
            artifact_files=artifact_files,
            uploaded_files=uploaded_files,
            tool_output_files=tool_output_files,
            workflow_state_summary={
                "current_step": workflow.get("current_step"),
                "completed_steps": completed_steps,
                "steps": steps,
            },
            registry_summary={
                "run_artifact_registry_id": registry.run_artifact_registry_id,
                "version": registry.version,
                "active_artifact_count": sum(1 for v in active.values() if v),
            },
            notes=notes or None,
        )
        (out_root / _MANIFEST_NAME).write_text(
            json.dumps(manifest.model_dump(), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    # ── hydrate ─────────────────────────────────────────────────────────────────

    def hydrate_pipeline_snapshot(
        self,
        snapshot_dir: str,
        new_run_id: Optional[str] = None,
    ) -> str:
        snap_root = Path(snapshot_dir).resolve()
        manifest_path = snap_root / _MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(f"no {_MANIFEST_NAME} in {snapshot_dir!r}")
        manifest = PipelineSnapshotManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

        target_run_id = new_run_id or _new_run_id()
        source_run_id = manifest.source_run_id

        # JSON artifacts (registry, workflow state, step outputs): rewrite
        # run-scoped storage keys + the top-level run_id so refs stay valid.
        for entry in [*manifest.artifact_files, *manifest.tool_output_files]:
            data = (snap_root / entry.snapshot_relative_path).read_bytes()
            data = self._rehydrate_json_bytes(data, source_run_id, target_run_id)
            dest_key = self.storage.run_key(target_run_id, *entry.run_relative_path.split("/"))
            self.storage.write_bytes(dest_key, data)

        # Uploaded files are opaque bytes — copied verbatim (never rewritten).
        for uf in manifest.uploaded_files:
            data = (snap_root / uf.snapshot_relative_path).read_bytes()
            dest_key = self.storage.run_key(target_run_id, *uf.run_relative_path.split("/"))
            self.storage.write_bytes(dest_key, data)

        return target_run_id

    # ── internals ─────────────────────────────────────────────────────────────

    def _uploaded_file_metadata(self, run_id: str) -> dict:
        """storage_path -> {file_id, original_filename, content_type, role}."""
        out: dict[str, dict] = {}
        raw_key = self.storage.run_key(run_id, "inputs", "raw_request_record.json")
        if not self.storage.exists(raw_key):
            return out
        try:
            raw = self.storage.read_json(raw_key)
        except Exception:  # noqa: BLE001
            return out
        for f in raw.get("uploaded_files") or []:
            if not isinstance(f, dict):
                continue
            sp = f.get("storage_path")
            if sp:
                out[sp] = {
                    "file_id": f.get("file_id"),
                    "original_filename": f.get("original_filename"),
                    "content_type": f.get("content_type"),
                    "role": f.get("role"),
                }
        return out

    @staticmethod
    def _maybe_artifact_id(data: bytes) -> Optional[str]:
        try:
            obj = json.loads(data.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if isinstance(obj, dict):
            value = obj.get("artifact_id")
            return value if isinstance(value, str) else None
        return None

    @staticmethod
    def _write_snapshot_bytes(out_root: Path, snap_rel: str, data: bytes) -> None:
        dest = out_root / snap_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    @staticmethod
    def _rehydrate_json_bytes(data: bytes, source_run_id: str, target_run_id: str) -> bytes:
        """Rewrite run-scoped storage keys + top-level run_id for a new run.

        Only ``runs/<source_run_id>/`` path substrings (storage keys, source
        refs, tool_output refs, uploaded storage_path) and the artifact's
        top-level ``run_id`` field are rewritten — biological payloads never
        contain ``runs/<run_id>/`` so this is safe and surgical.
        """
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        text = text.replace(f"runs/{source_run_id}/", f"runs/{target_run_id}/")
        try:
            obj = json.loads(text)
        except Exception:  # noqa: BLE001 — not JSON; keep the path-rewritten text
            return text.encode("utf-8")
        if isinstance(obj, dict) and obj.get("run_id") == source_run_id:
            obj["run_id"] = target_run_id
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False).encode("utf-8")


def _new_run_id() -> str:
    return new_run_id()
