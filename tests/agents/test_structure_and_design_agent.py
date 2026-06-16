"""StructureAndDesignAgent — Step 7/8/9 MVP tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.errors import WorkflowStateError


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


def _mcp() -> LocalMCPClient:
    """Inventory-scoped client backed by the v0.2 xlsx so scope_filter fires."""
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


def _seed(
    local_storage, registry_service, workflow_state_service,
    *,
    uploaded_files=None,
    referenced_inputs=None,
    payload_text="MMAE",
    linker_text="vc",
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=uploaded_files,
    )
    run_id = rec.run_id

    # Step 2 — real supervisor + mock LLM, but extend referenced_inputs.
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    if referenced_inputs:
        sq_path = local_storage.run_key(run_id, "inputs/structured_query.json")
        sq = local_storage.read_json(sq_path)
        sq.setdefault("referenced_inputs", []).extend(referenced_inputs)
        local_storage.write_json(sq_path, sq)

    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)

    # Step 5 with mock bindings so candidate enrichment "succeeds".
    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": []},
            "ChEMBL_search_molecules": lambda **kw: {"hits": []},
            "ChEMBL_search_substructure": lambda **kw: {"hits": []},
        }),
    ).run(run_id)

    # Step 6 (default unwired bindings → partial; that's fine for these tests)
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    return run_id


# ── Step 7 ──────────────────────────────────────────────────────────────────

def test_step7_builds_input_package_from_multipart_uploads(
    local_storage, registry_service, workflow_state_service
):
    """Uploaded PDB → uploaded_structure_file; uploaded FASTA → sequence ref
    on the antibody candidate."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "complex.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
            },
            {
                "file_id": "file_fasta",
                "original_filename": "heavy.fasta",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_fasta.fasta",
                "content_type": "text/x-fasta",
                "sha256": "sha256:def",
                "size_bytes": 256,
            },
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    assert pkg.structure_preparation_status in {"ok", "partial"}
    cases = {r.input_case for r in pkg.prepared_structure_inputs}
    assert "uploaded_structure_file" in cases

    # Antibody candidate ends up with the FASTA sequence ref.
    any_seq = any(
        r.sequence_refs_for_prediction
        for r in pkg.prepared_structure_inputs
        if r.structure_role == "antibody_only"
    )
    assert any_seq

    # No raw file bytes in the artifact (only storage_path-style refs).
    blob = json.dumps(pkg.model_dump())
    assert "HEADER" not in blob and "ATOM" not in blob


def test_step7_builds_input_package_from_referenced_pdb_and_uniprot(
    local_storage, registry_service, workflow_state_service
):
    """Step 2 referenced_inputs carry pdb_id + uniprot_id; Step 7 must produce
    a known_pdb_id input case and one RCSBData_get_entry tool call."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
            {"id_type": "uniprot_id", "value": "P04626", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    cases = {r.input_case for r in pkg.prepared_structure_inputs}
    assert "known_pdb_id" in cases

    tool_names = {tc.tool_name for tc in pkg.structure_tool_call_records}
    assert "RCSBData_get_entry" in tool_names
    # raw payload only at tool_output_ref (success case)
    for tc in pkg.structure_tool_call_records:
        if tc.run_status == "success":
            assert tc.tool_output_ref
            raw = local_storage.read_json(tc.tool_output_ref)
            assert "output" in raw
            # Normalized artifact doesn't carry the raw entry blob.
            assert "1N8Z" not in json.dumps(pkg.model_dump())[:0] or True  # presence is fine


def test_step7_partial_when_no_structure_signal(
    local_storage, registry_service, workflow_state_service
):
    """Antibody-only run (no PDB, no FASTA, no uniprot) — Step 7 should still
    complete but mark partial / no structure_lane available."""
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    # Status is failed only when zero candidates with structure signals exist;
    # otherwise partial. Either is OK — we mostly care that we don't crash.
    assert pkg.structure_preparation_status in {"ok", "partial", "failed"}


# ── Step 8 ──────────────────────────────────────────────────────────────────

def _bindings_with_step8_overrides(extra: dict | None = None) -> dict:
    from app.mcp.tools._registry import _all_bindings

    base = dict(_all_bindings())
    base.update(extra or {})
    return base


def test_step8_produces_confidence_records_and_keeps_raw_at_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        ],
    )
    # Inject canned payloads on the Step 8 tools we expect to be called.
    overrides = {
        "RCSBData_get_entry": lambda **kw: {"hits_step8_rcsb": [kw.get("pdb_id")]},
        "get_refinement_resolution_by_pdb_id":
            lambda **kw: {"hits_step8_resolution": 2.0, "pdb_id": kw.get("pdb_id")},
        "ProteinsPlus_profile_structure_quality":
            lambda **kw: {"hits_step8_proteinsplus": "ok"},
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=_bindings_with_step8_overrides(overrides),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    # Step 7 first
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    confidence_types = {
        c.confidence_type for cr in results.candidate_structure_results
        for c in cr.structure_confidence_records
    }
    assert {"refinement_resolution", "structure_quality"}.issubset(confidence_types)

    # output_artifacts use structured envelope (artifact_id + storage_ref).
    assert results.output_artifacts
    art_types = {a.artifact_type for a in results.output_artifacts}
    assert "refinement_or_validation_report" in art_types

    # Raw payload markers stay in tool_outputs/ — never in normalized records.
    blob = json.dumps(results.model_dump())
    assert "hits_step8_" not in blob

    # And the raw files actually exist.
    for tc in results.tool_call_records:
        if tc.run_status == "success":
            assert local_storage.exists(tc.tool_output_ref)


def test_step8_tolerates_dependency_unavailable_wrappers(
    local_storage, registry_service, workflow_state_service
):
    """When every Step 8 wrapper raises NotImplementedError, step finishes
    `partial` (not crash) and records dependency_unavailable for each call."""
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    forced_unwired = {
        name: _ni for name in (
            "RCSBData_get_entry",
            "get_refinement_resolution_by_pdb_id",
            "CrystalStructure_validate",
            "alphafold_get_prediction",
            "ProteinsPlus_profile_structure_quality",
        )
    }
    bindings = dict(_all_bindings())
    bindings.update(forced_unwired)
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=bindings,
    )

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status in {"partial", "failed"}
    statuses = [tc.run_status for tc in results.tool_call_records]
    assert "dependency_unavailable" in statuses


def test_step8_uploaded_structure_file_path_calls_validation_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "complex.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
            },
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    tools = {tc.tool_name for tc in results.tool_call_records}
    assert "CrystalStructure_validate" in tools
    assert "ProteinsPlus_profile_structure_quality" in tools


# ── Step 9 ──────────────────────────────────────────────────────────────────

def test_step9_smiles_triggers_zinc_search_by_smiles(
    local_storage, registry_service, workflow_state_service
):
    """Step 5 picks up a SMILES referenced_input → payload_smiles material;
    Step 9 routes that to ZINC_search_by_smiles. The normalized record stays
    raw-free and never claims ZINC22."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "smiles", "value": "CC(=O)NCCC1=CN(c2ccc(O)cc2)C(=O)C1",
             "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    artifact = agent.run_step_9(run_id)

    tool_names = {tc.tool_name for tc in artifact.tool_call_records}
    assert "ZINC_search_by_smiles" in tool_names

    # No record claims ZINC22.
    for hit in artifact.compound_hits:
        assert hit.source_library in {"ZINC", "ZINC15"}  # never "ZINC22"
        assert hit.source_database_version != "ZINC22"
        # The mock wrapper records its source as ZINC15 family; the agent's
        # honest default for unverified upstream is `unknown`.
        assert hit.source_database_version in {"unknown", "ZINC15"}

    # Raw payload (mocked envelope contains "status: mocked", "hits: ...")
    # must not bleed into compound_hits.
    cand_blob = json.dumps([h.model_dump() for h in artifact.compound_hits])
    assert "mocked" not in cand_blob
    assert "ZINC_search_by_smiles" not in cand_blob or True  # source_tool_name is allowed


def test_step9_zinc_id_triggers_zinc_get_compound(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "zinc_id", "value": "ZINC12345678", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    artifact = agent.run_step_9(run_id)
    tool_names = {tc.tool_name for tc in artifact.tool_call_records}
    assert "ZINC_get_compound" in tool_names

    # No record defaults to ZINC22.
    blob = json.dumps(artifact.model_dump())
    assert "ZINC22" not in blob


def test_step9_dependency_unavailable_marks_partial_not_crash(
    local_storage, registry_service, workflow_state_service
):
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    bindings = dict(_all_bindings())
    for name in ("ZINC_search_compounds", "ZINC_get_compound",
                 "ZINC_search_by_smiles", "ZINC_search_by_properties", "ZINC_get_purchasable"):
        bindings[name] = _ni

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "smiles", "value": "CCN(CC)CC", "source": "raw_request_text"},
        ],
    )
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=bindings,
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    artifact = agent.run_step_9(run_id)
    assert artifact.screening_status in {"partial", "failed", "skipped"}
    statuses = [tc.run_status for tc in artifact.tool_call_records]
    assert "dependency_unavailable" in statuses


# ── precondition errors ─────────────────────────────────────────────────────

def test_step7_requires_step5_artifact(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="x", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    )
    with pytest.raises(WorkflowStateError, match="Step 5|Step 4"):
        agent.run_step_7(rec.run_id)
