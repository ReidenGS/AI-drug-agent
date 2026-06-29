"""Generic pipeline snapshot / hydrate tests.

Covers export, hydrate, downstream continuation (hydrated Step 1-6 → Step 7),
generic ``through_step`` handling (not hard-coded to Step 6), and privacy.
Hydrate must perform pure file I/O — no LLM / MCP / agent step.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.intake_service import IntakeService
from app.services.pipeline_snapshot_service import PipelineSnapshotService
from app.services.workflow_state_service import WorkflowStateService
from app.utils.ids import new_artifact_id


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


def _svc(local_storage, registry_service, workflow_state_service) -> PipelineSnapshotService:
    return PipelineSnapshotService(local_storage, registry_service, workflow_state_service)


def _seed_generic_run(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    through_num: int = 6,
    with_uploaded_file: bool = True,
    with_tool_output: bool = True,
) -> str:
    """Seed a synthetic run with a few artifacts + completed steps, WITHOUT
    running any agent/LLM/MCP. Generic — not tied to specific step semantics."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    uploaded = None
    run_id = intake.allocate_run_id()
    pdb_key = local_storage.run_key(run_id, "inputs", "files", "f_pdb.pdb")
    if with_uploaded_file:
        uploaded = [
            {
                "file_id": "f_pdb",
                "original_filename": "complex.pdb",
                "storage_path": pdb_key,
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 12,
            }
        ]
    rec = intake.submit(
        raw_user_query="Design HER2 ADC with vc-MMAE",
        user_provided_context={"target_or_antigen_text": "HER2"},
        uploaded_files=uploaded,
        run_id=run_id,
    )
    if with_uploaded_file:
        local_storage.write_bytes(pdb_key, b"HEADER PDB\n")

    # Synthetic step outputs (just enough structure to exercise snapshotting).
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, "run_id": run_id, "task_intent": {"primary_intent": "new_adc_design"}},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")

    if through_num >= 3:
        ir_id = new_artifact_id("input_readiness_status")
        local_storage.write_json(
            local_storage.run_key(run_id, "inputs/input_readiness_status.json"),
            {"artifact_id": ir_id, "run_id": run_id, "input_readiness_status": "ready"},
        )
        registry_service.update_active(run_id, input_readiness_status_id=ir_id)
        workflow_state_service.mark(run_id, "step_03", "completed")

    if through_num >= 6:
        cct_id = new_artifact_id("candidate_context_table")
        local_storage.write_json(
            local_storage.run_key(run_id, "candidate_context_table.json"),
            {"artifact_id": cct_id, "run_id": run_id, "candidate_records": []},
        )
        registry_service.update_active(run_id, candidate_context_table_id=cct_id)
        sls_id = new_artifact_id("structured_liability_summary")
        local_storage.write_json(
            local_storage.run_key(run_id, "structured_liability_summary.json"),
            {"artifact_id": sls_id, "run_id": run_id, "prefilter_status": "completed"},
        )
        registry_service.update_active(run_id, structured_liability_summary_id=sls_id)
        for n in (4, 5, 6):
            workflow_state_service.mark(run_id, f"step_{n:02d}", "completed")

    if with_tool_output and through_num >= 6:
        local_storage.write_json(
            local_storage.run_key(run_id, "tool_outputs", "step_06", "tc1.json"),
            {"tool_call_id": "tc1", "tool_name": "DrugProps_pains_filter", "output": {"alerts": []}},
        )
    return run_id


# ── A. export ─────────────────────────────────────────────────────────────────


def test_export_through_step_06(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    out = str(tmp_path / "snap")
    manifest = _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
        run_id, "step_06", out
    )
    assert manifest.source_run_id == run_id
    assert manifest.through_step == "step_06"
    assert "step_06" in manifest.completed_steps
    assert manifest.active_artifacts["structured_query_id"]
    assert manifest.active_artifacts["candidate_context_table_id"]
    # artifact_files include registry + workflow_state + the step outputs.
    names = {e.artifact_name for e in manifest.artifact_files}
    assert "structured_query" in names
    assert "candidate_context_table" in names
    types = {e.artifact_type for e in manifest.artifact_files}
    assert "registry" in types and "workflow_state" in types
    # Manifest + artifacts physically written.
    assert (tmp_path / "snap" / "snapshot_manifest.json").exists()


def test_export_requires_through_step_completed(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(
        local_storage, registry_service, workflow_state_service, through_num=3
    )
    # step_06 is still pending → clear error.
    with pytest.raises(ValueError, match="not completed"):
        _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
            run_id, "step_06", str(tmp_path / "snap")
        )


def test_export_requires_active_artifacts(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    # A run with step_06 marked completed but NO active artifacts registered.
    run_id = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).allocate_run_id()
    workflow_state_service.init_run(run_id)
    registry_service.init_registry(run_id)
    workflow_state_service.mark(run_id, "step_06", "completed")
    with pytest.raises(ValueError, match="no active artifacts"):
        _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
            run_id, "step_06", str(tmp_path / "snap")
        )


def test_export_unknown_run_raises(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    with pytest.raises(ValueError, match="not found|no workflow state"):
        _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
            "run_does_not_exist", "step_06", str(tmp_path / "snap")
        )


def test_export_excludes_tool_outputs_when_disabled(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    manifest = _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
        run_id, "step_06", str(tmp_path / "snap"), include_tool_outputs=False
    )
    assert manifest.tool_output_files == []
    assert manifest.notes and manifest.notes.get("tool_outputs_excluded") is True


def test_export_includes_tool_outputs_by_default(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    manifest = _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
        run_id, "step_06", str(tmp_path / "snap")
    )
    assert any(e.artifact_name == "tc1" for e in manifest.tool_output_files)


# ── B. hydrate ────────────────────────────────────────────────────────────────


def test_hydrate_restores_registry_workflow_and_artifacts(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    out = str(tmp_path / "snap")
    manifest = _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
        run_id, "step_06", out
    )
    svc = _svc(local_storage, registry_service, workflow_state_service)
    new_run = svc.hydrate_pipeline_snapshot(out)

    assert new_run != run_id
    # Registry active artifacts align with the manifest.
    reg = registry_service.get(new_run)
    assert reg.active_artifacts.structured_query_id == manifest.active_artifacts["structured_query_id"]
    assert reg.run_id == new_run
    # Workflow state restored to the completed steps.
    wf = workflow_state_service.get(new_run)
    assert wf["run_id"] == new_run
    completed = {k for k, v in wf["steps"].items() if v == "completed"}
    assert {"step_01", "step_02", "step_03", "step_04", "step_05", "step_06"} <= completed
    # Artifact JSON is readable on the new run.
    sq = local_storage.read_json(local_storage.run_key(new_run, "inputs/structured_query.json"))
    assert sq["run_id"] == new_run
    # Tool output ref restored and readable.
    to = local_storage.read_json(
        local_storage.run_key(new_run, "tool_outputs", "step_06", "tc1.json")
    )
    assert to["tool_call_id"] == "tc1"


def test_hydrate_rewrites_uploaded_storage_path_refs(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    out = str(tmp_path / "snap")
    svc = _svc(local_storage, registry_service, workflow_state_service)
    svc.export_pipeline_snapshot(run_id, "step_06", out)
    new_run = svc.hydrate_pipeline_snapshot(out)

    raw = local_storage.read_json(local_storage.run_key(new_run, "inputs/raw_request_record.json"))
    sp = raw["uploaded_files"][0]["storage_path"]
    # storage_path now points at the NEW run and the bytes are readable there.
    assert f"runs/{new_run}/" in sp
    assert run_id not in sp
    assert local_storage.read_bytes(sp) == b"HEADER PDB\n"


def test_hydrate_explicit_run_id(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    out = str(tmp_path / "snap")
    svc = _svc(local_storage, registry_service, workflow_state_service)
    svc.export_pipeline_snapshot(run_id, "step_06", out)
    new_run = svc.hydrate_pipeline_snapshot(out, new_run_id="run_custom_hydrated")
    assert new_run == "run_custom_hydrated"
    assert registry_service.get("run_custom_hydrated").run_id == "run_custom_hydrated"


def test_hydrate_service_takes_no_llm_or_mcp(
    local_storage, registry_service, workflow_state_service
):
    """Structural proof: the snapshot service neither accepts nor holds any
    LLM / MCP client — hydrate cannot call them."""
    svc = _svc(local_storage, registry_service, workflow_state_service)
    assert not hasattr(svc, "llm")
    assert not hasattr(svc, "mcp_client")


# ── D. generic through_step (not hard-coded to step 6) ──────────────────────────


def test_export_hydrate_generic_through_step_03(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(
        local_storage, registry_service, workflow_state_service, through_num=3,
        with_tool_output=False,
    )
    out = str(tmp_path / "snap3")
    svc = _svc(local_storage, registry_service, workflow_state_service)
    manifest = svc.export_pipeline_snapshot(run_id, "step_03", out)
    assert manifest.through_step == "step_03"
    assert "step_03" in manifest.completed_steps
    assert "step_06" not in manifest.completed_steps
    new_run = svc.hydrate_pipeline_snapshot(out)
    wf = workflow_state_service.get(new_run)
    completed = {k for k, v in wf["steps"].items() if v == "completed"}
    assert {"step_01", "step_02", "step_03"} <= completed
    assert "step_06" not in completed


def test_manifest_accepts_arbitrary_completed_steps(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    """No hard-coded Step 1-6 assumption: a run completed through step_08 with
    an arbitrary completed-step set exports and hydrates."""
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    # Push the run further: pretend step_07/08 also completed with an artifact.
    pkg_id = new_artifact_id("prepared_structure_input_package")
    local_storage.write_json(
        local_storage.run_key(run_id, "prepared_structure_input_package.json"),
        {"artifact_id": pkg_id, "run_id": run_id},
    )
    registry_service.update_active(run_id, prepared_structure_input_package_id=pkg_id)
    workflow_state_service.mark(run_id, "step_07", "completed")
    workflow_state_service.mark(run_id, "step_08", "completed")
    out = str(tmp_path / "snap8")
    svc = _svc(local_storage, registry_service, workflow_state_service)
    manifest = svc.export_pipeline_snapshot(run_id, "step_08", out)
    assert "step_08" in manifest.completed_steps
    assert manifest.active_artifacts["prepared_structure_input_package_id"] == pkg_id
    new_run = svc.hydrate_pipeline_snapshot(out)
    assert registry_service.get(new_run).active_artifacts.prepared_structure_input_package_id == pkg_id


# ── E. privacy ──────────────────────────────────────────────────────────────────


def test_manifest_contains_no_raw_payload_or_secrets(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    run_id = _seed_generic_run(local_storage, registry_service, workflow_state_service)
    out = str(tmp_path / "snap")
    manifest = _svc(local_storage, registry_service, workflow_state_service).export_pipeline_snapshot(
        run_id, "step_06", out
    )
    blob = (tmp_path / "snap" / "snapshot_manifest.json").read_text(encoding="utf-8").lower()
    # Manifest records paths/hashes/ids only — not raw content or secrets.
    assert "api_key" not in blob
    assert "bearer " not in blob
    assert "header pdb" not in blob  # uploaded file bytes never inlined
    assert "system instructions" not in blob
    # Sanity: it does record the metadata we expect.
    assert "structured_query" in blob
    assert "sha256" in blob


# ── C. downstream continuation: hydrated Step 1-6 → Step 7 ──────────────────────


def _inventory_or_skip():
    from app.services.tool_inventory_service import ToolInventoryService

    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _real_step1_6_run(local_storage, registry_service, workflow_state_service) -> str:
    """Build a real Step 1-6 run with Mock LLM + LocalMCPClient (no network)."""
    from app.agents.candidate_context_agent import CandidateContextAgent
    from app.agents.developability_agent import DevelopabilityAgent
    from app.agents.supervisor_agent import SupervisorAgent
    from app.llm.provider import MockLLMProvider
    from app.mcp.client import LocalMCPClient
    from app.services.input_readiness_service import InputReadinessService
    from app.services.structured_query_service import StructuredQueryService
    from app.services.workflow_setup_service import WorkflowSetupService

    inventory = _inventory_or_skip()
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    run_id = intake.allocate_run_id()
    pdb_key = local_storage.run_key(run_id, "inputs", "files", "file_pdb.pdb")
    intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "complex.pdb",
                "storage_path": pdb_key,
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 11,
            }
        ],
        run_id=run_id,
    )
    local_storage.write_bytes(pdb_key, b"HEADER PDB\n")

    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": []},
            "ChEMBL_search_molecules": lambda **kw: {"hits": []},
            "ChEMBL_search_substructure": lambda **kw: {"hits": []},
        }),
    ).run(run_id)
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    return run_id


def test_hydrated_step1_6_snapshot_continues_to_step7(
    local_storage, registry_service, workflow_state_service, tmp_path
):
    from app.agents.structure_and_design_agent import StructureAndDesignAgent
    from app.mcp.client import LocalMCPClient

    inventory = _inventory_or_skip()
    run_id = _real_step1_6_run(local_storage, registry_service, workflow_state_service)

    out = str(tmp_path / "snap_s6")
    svc = _svc(local_storage, registry_service, workflow_state_service)
    svc.export_pipeline_snapshot(run_id, "step_06", out)
    new_run = svc.hydrate_pipeline_snapshot(out)

    # Proof: immediately after hydrate (before any agent runs), the prior
    # steps are already marked complete and their artifacts exist — nothing
    # was re-run.
    wf = workflow_state_service.get(new_run)
    assert wf["steps"]["step_06"] == "completed"
    assert wf["steps"]["step_07"] == "pending"
    assert local_storage.exists(local_storage.run_key(new_run, "candidate_context_table.json"))
    # Uploaded PDB bytes are readable on the hydrated run.
    new_raw = local_storage.read_json(
        local_storage.run_key(new_run, "inputs/raw_request_record.json")
    )
    sp = new_raw["uploaded_files"][0]["storage_path"]
    assert local_storage.read_bytes(sp) == b"HEADER PDB\n"

    # Now continue to Step 7 on the hydrated run.
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    ).run_step_7(new_run)
    assert pkg is not None
    assert workflow_state_service.get(new_run)["steps"]["step_07"] == "completed"
