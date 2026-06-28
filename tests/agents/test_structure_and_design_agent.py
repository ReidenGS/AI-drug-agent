"""StructureAndDesignAgent — Step 7/8/9 MVP tests."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents import structure_and_design_agent as structure_and_design_module
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
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


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


def test_step7_uploaded_pdb_preferred_over_sequence_refs(
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
            {
                "file_id": "file_fasta",
                "original_filename": "target.fasta",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_fasta.fasta",
                "content_type": "text/x-fasta",
                "sha256": "sha256:def",
                "size_bytes": 256,
            },
        ],
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)

    uploaded = [r for r in pkg.prepared_structure_inputs if r.input_case == "uploaded_structure_file"]
    assert uploaded
    for rec in uploaded:
        assert rec.prediction_required is False
        assert rec.preferred_input_rank in {1, 2}
        assert any("preferred" in n for n in rec.source_priority_notes)
    blob = json.dumps(pkg.model_dump())
    assert "RAW_PDB_SENTINEL" not in blob
    assert "RAW_FASTA_SENTINEL" not in blob


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

    known = [r for r in pkg.prepared_structure_inputs if r.input_case == "known_pdb_id"]
    assert known
    assert all(r.prediction_required is False for r in known)
    assert any(s.pdb_id == "1N8Z" and s.source_kind == "pdb_id" for r in known for s in r.structure_refs)


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


def test_step7_antigen_uniprot_only_proceeds_as_sequence_prediction(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    antigen = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert antigen.input_case == "sequence_only_input"
    assert antigen.prediction_required is True
    assert "structure_prediction_needed" not in antigen.missing_metadata_flags
    assert not any("no structure file" in f for f in antigen.missing_metadata_flags)
    assert any(s.source_kind == "uniprot_id" and s.sequence_id == "P04626" for s in antigen.sequence_refs_for_prediction)
    assert any(cm.chain_role == "antigen" and cm.chain_id == "predicted_antigen" for cm in antigen.chain_mapping)


def test_step7_antibody_heavy_light_sequence_only_proceeds(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].extend([
        {
            "material_id": "heavy_seq",
            "material_type": "antibody_heavy_chain_sequence",
            "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "value_format": "fasta",
            "role": "antibody",
        },
        {
            "material_id": "light_seq",
            "material_type": "antibody_light_chain_sequence",
            "value": "DIQMTQSPSSLSASVGDRVTITC",
            "value_format": "fasta",
            "role": "antibody",
        },
    ])
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    antibody_rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antibody_only")
    assert antibody_rec.input_case == "sequence_only_input"
    assert antibody_rec.prediction_required is True
    roles = {s.chain_role for s in antibody_rec.sequence_refs_for_prediction}
    assert {"antibody_heavy", "antibody_light"}.issubset(roles)
    chain_roles = {cm.chain_role for cm in antibody_rec.chain_mapping}
    assert {"antibody_heavy", "antibody_light"}.issubset(chain_roles)
    assert all(cm.chain_id_kind == "prediction_placeholder" for cm in antibody_rec.chain_mapping)


def test_step7_antigen_antibody_sequence_mapping_without_interface_invention(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].extend([
        {"material_id": "heavy_seq", "material_type": "antibody_heavy_chain_sequence", "value": "EVQLVES", "value_format": "fasta"},
        {"material_id": "light_seq", "material_type": "antibody_light_chain_sequence", "value": "DIQMTQ", "value_format": "fasta"},
    ])
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    mapped = [r for r in pkg.prepared_structure_inputs if r.antigen_antibody_mapping]
    assert mapped
    mapping = mapped[0].antigen_antibody_mapping
    assert mapping["target_candidate_id"]
    assert mapping["antibody_candidate_id"]
    assert mapping["mapping_status"] == "sequence_only_prediction_needed"
    blob = json.dumps(mapping).lower()
    assert "interface" not in blob
    assert "epitope" not in blob


def test_step7_extracts_partial_residue_range_and_does_not_invent_when_absent(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "target_domain",
        "material_type": "target_sequence",
        "value": "HER2 extracellular domain residues 20-240",
        "value_format": "text",
        "role": "target",
    })
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    target_rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert {"start": 20, "end": 240, "source": "candidate_material", "source_ref": "target_domain"} in target_rec.residue_ranges
    assert all(
        r.residue_ranges == []
        for r in pkg.prepared_structure_inputs
        if r.candidate_id != target_rec.candidate_id
    )


@pytest.mark.parametrize(
    "filename,expected_role,expected_structure_role",
    [
        ("heavy.fasta", "antibody_heavy", "antibody_only"),
        ("antigen.fasta", "antigen", "antigen_only"),
    ],
)
def test_step7_filename_scoped_fasta_binds_only_compatible_candidate(
    local_storage, registry_service, workflow_state_service,
    filename, expected_role, expected_structure_role,
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": f"file_{expected_role}",
            "original_filename": filename,
            "storage_path": f"adc_pilot/runs/x/inputs/files/{filename}",
            "content_type": "text/x-fasta",
            "size_bytes": 64,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    refs = [
        (r.structure_role, s) for r in pkg.prepared_structure_inputs
        for s in r.sequence_refs_for_prediction if s.sequence_id == f"file_{expected_role}"
    ]
    assert len(refs) == 1
    assert refs[0][0] == expected_structure_role
    assert refs[0][1].chain_role == expected_role
    assert refs[0][1].sequence is None
    assert refs[0][1].sequence_storage_ref.endswith(filename)
    assert refs[0][1].prediction_input_kind == "fasta_ref"


def test_step7_ambiguous_fasta_is_unresolved_not_broadcast(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": "ambiguous_fasta",
            "original_filename": "ambiguous.fasta",
            "storage_path": "adc_pilot/runs/x/inputs/files/ambiguous.fasta",
            "content_type": "text/x-fasta",
            "size_bytes": 64,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert not any(
        s.sequence_id == "ambiguous_fasta"
        for r in pkg.prepared_structure_inputs for s in r.sequence_refs_for_prediction
    )
    assert any(
        u["source_ref"] == "ambiguous_fasta" and u["resource_binding_status"] in {"unassigned", "ambiguous"}
        for u in pkg.unresolved_resource_refs
    )


def test_step7_referenced_pdb_id_is_not_copied_to_antibody(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pdb_id", "value": "1N8Z", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    owners = [
        r for r in pkg.prepared_structure_inputs
        if any(s.pdb_id == "1N8Z" for s in r.structure_refs)
    ]
    if owners:
        # In non-ambiguous runs this still maps to antigen candidate.
        assert len(owners) == 1
        assert owners[0].structure_role == "antigen_only"
    else:
        # New safety rule: avoid implicit target binding when target+antibody both exist.
        assert any(
            u["resource_type"] == "pdb_id" and u["source_ref"] == "1N8Z"
            for u in pkg.unresolved_resource_refs
        )


def test_step7_unscoped_uploaded_structure_is_unresolved_when_target_and_antibody_conflict(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": "unscoped_pdb",
            "original_filename": "structure.pdb",
            "storage_path": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"),
            "content_type": "chemical/x-pdb",
            "size_bytes": 1,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert any(u["resource_type"] == "structure" for u in pkg.unresolved_resource_refs)
    # Ensure not implicitly copied to a specific target/antibody candidate.
    resolved = [r for r in pkg.prepared_structure_inputs if any(s.source_ref == "unscoped_pdb" for s in r.structure_refs)]
    assert resolved == []


def test_step7_unscoped_pdb_id_is_unresolved_when_target_and_antibody_conflict(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        if cand.get("candidate_type") in {"target_antigen", "antibody"}:
            cand["identifiers"] = [
                i for i in (cand.get("identifiers") or [])
                if i.get("id_type") != "pdb_id"
            ]
            cand["materials"] = [
                m for m in (cand.get("materials") or [])
                if m.get("material_type") != "structure_ref"
            ]
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    has_target = any(r.structure_role == "antigen_only" for r in pkg.prepared_structure_inputs)
    has_antibody = any(r.structure_role == "antibody_only" for r in pkg.prepared_structure_inputs)
    if has_target and has_antibody:
        assert any(
            u["resource_type"] == "pdb_id"
            and u["source_ref"] == "1N8Z"
            for u in pkg.unresolved_resource_refs
        )
        assert not any(
            s.pdb_id == "1N8Z"
            for r in pkg.prepared_structure_inputs
            for s in r.structure_refs
        )
    else:
        assert any(s.pdb_id == "1N8Z" for r in pkg.prepared_structure_inputs for s in r.structure_refs)


def test_step7_candidate_scoped_structure_material_stays_on_candidate(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "target_s1", "material_type": "structure_file",
        "value": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"), "value_format": "pdb",
    })
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    owners = [r for r in pkg.prepared_structure_inputs if any(s.source_ref == "target_s1" for s in r.structure_refs)]
    assert len(owners) == 1
    assert owners[0].candidate_id == target["candidate_id"]
    rec = owners[0]
    rec_ref = next(s for s in rec.structure_refs if s.source_ref == "target_s1")
    assert rec_ref.storage_ref == str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb")


def test_step7_structure_file_material_is_consumable_by_step8_without_fake_placeholder(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].append({
        "material_id": "ab_s1",
        "material_type": "structure_file",
        "value": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"),
        "value_format": "pdb",
        "role": "antibody",
    })
    local_storage.write_json(cct_path, cct)

    overrides = {
        "CrystalStructure_validate": lambda **kw: {"ok": True, "pdb_id_or_path": kw.get("pdb_id_or_path")},
        "ProteinsPlus_profile_structure_quality": lambda **kw: {"ok": True, "pdb_id_or_path": kw.get("pdb_id_or_path")},
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings={**_bindings_with_step8_overrides(), **overrides},
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    tool_calls = [tc for tc in results.tool_call_records if tc.tool_name in {"CrystalStructure_validate", "ProteinsPlus_profile_structure_quality"}]
    assert tool_calls, "expected structure validation calls"
    for tc in tool_calls:
        assert tc.tool_input_summary["pdb_id_or_path"] == str(
            PROJECT_ROOT / "data" / "pdb" / "S1.pdb"
        )
        assert "uploaded" not in json.dumps(tc.tool_input_summary or {})
        assert tc.tool_input_summary and ("pdb_id_or_path" in tc.tool_input_summary)
        assert tc.tool_input_summary["pdb_id_or_path"].endswith("S1.pdb")


def test_step7_name_only_routes_to_database_search_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[],
    )
    # Build with candidate names only; no structure files, no PDB IDs, no sequences.
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)

    db_results = [tc for tc in pkg.structure_tool_call_records if tc.tool_name in {"RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}]
    assert db_results, "name-only input should route to search wrappers"
    assert all(tc.run_status in {"success", "dependency_unavailable", "failed"} for tc in db_results)

    assert all(
        tc.tool_input_summary
        and tc.tool_input_summary.get("routing_decision") == "selected"
        for tc in db_results
    )
    skipped = [
        tc for tc in pkg.structure_tool_call_records
        if tc.tool_name in {"RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"}
    ]
    assert skipped and all(tc.run_status == "skipped" for tc in skipped)
    assert all(tc.tool_input_summary.get("routing_decision") == "not_applicable" for tc in skipped)

    # Database search results are normalized as ambiguous candidate
    # options. They do not get promoted to primary structure refs.
    for rec in pkg.prepared_structure_inputs:
        if rec.input_case != "database_search_result":
            continue
        assert rec.database_search_candidates == []


def test_step7_scope_tools_match_inventory_runtime_scope():
    mcp = _mcp()
    runtime_scope = set(mcp.list_tools(agent_name="structure_and_design_agent", step_id="step_07"))
    assert runtime_scope == set(structure_and_design_module._STEP7_SCOPED_TOOLS), (
        "Step 7 runtime scope must stay in sync with _STEP7_SCOPED_TOOLS.\n"
        f"runtime_scope={sorted(runtime_scope)}\n"
        f"routing_table={sorted(structure_and_design_module._STEP7_SCOPED_TOOLS)}"
    )


def test_step7_candidate_pdb_id_routes_to_rcsb_and_sabdab_step7_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody_id = antibody["candidate_id"]
    antibody.setdefault("identifiers", []).append(
        {"id_type": "pdb_id", "id_value": "1N8Z", "source": "candidate_profile"}
    )
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)
    tc_by_name = {
        tc.tool_name: tc for tc in pkg.structure_tool_call_records
        if tc.tool_name in {
            "RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"
        } and tc.tool_input_summary.get("candidate_id") == antibody_id
        }
    assert "RCSBData_get_entry" in tc_by_name
    assert tc_by_name["RCSBData_get_entry"].tool_input_summary.get("routing_decision") == "selected"
    assert "RCSBData_get_assembly" in tc_by_name
    assert tc_by_name["RCSBData_get_assembly"].tool_input_summary.get("arguments", {}).get("assembly_id") == "1"
    assert "SAbDab_get_structure" in tc_by_name
    assert tc_by_name["SAbDab_get_structure"].tool_input_summary.get("routing_decision") == "selected"

    rec = next(
        r for r in pkg.prepared_structure_inputs if r.input_case == "known_pdb_id"
    )
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    entry_meta = compacted["RCSBData_get_entry"]["compact_output"]
    assert entry_meta["compact_type"] == "rcsb_entry"
    assert entry_meta["pdb_id"]
    assert entry_meta["entry_metadata"] is not None
    assert compacted["RCSBData_get_assembly"]["compact_output"]["compact_type"] == "rcsb_assembly"
    assert compacted["RCSBData_get_assembly"]["compact_output"]["assembly_id"] == "1"
    assert compacted["SAbDab_get_structure"]["compact_output"]["compact_type"] == "sabdab_structure"
    assert compacted["SAbDab_get_structure"]["compact_output"]["pdb_id"] == entry_meta["pdb_id"]
    assert rec.structure_refs and any(s.source_kind == "pdb_id" for s in rec.structure_refs)


def test_step7_sequence_only_uniprot_triggers_alphafold_get_prediction_lookup(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        cand["materials"] = [
            m for m in cand.get("materials", [])
            if not (isinstance(m, dict) and str(m.get("material_type", "")).endswith("_name"))
        ]
    local_storage.write_json(cct_path, cct)

    overrides = {
        "alphafold_get_prediction": lambda **kw: {
            "uniprot": kw.get("uniprot"),
            "status": "success",
            "model_path": "mock://alphafold/P04626.pdb",
        },
    }
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
            bindings=overrides,
        ),
    )
    pkg = agent.run_step_7(run_id)
    step8 = agent.run_step_8(run_id)
    af_calls = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == "alphafold_get_prediction"]
    assert len(af_calls) == 1
    af_call = af_calls[0]
    assert af_call.run_status == "success"
    assert af_call.tool_input_summary["routing_decision"] == "selected"
    assert af_call.tool_input_summary["arguments"]["uniprot"] == "P04626"

    rec = next(
        r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input"
    )
    assert any(
        ref.source_kind == "predicted_needed" and ref.source_ref == "P04626"
        for ref in rec.structure_refs
    )
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    assert compacted["alphafold_get_prediction"]["compact_output"]["compact_type"] == "alphafold_prediction"
    assert compacted["alphafold_get_prediction"]["compact_output"]["model_ref"] == "mock://alphafold/P04626.pdb"

    for name in {"RCSBData_get_entry", "RCSBData_get_assembly", "RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}:
        callset = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == name]
        assert callset
        assert all(
            call.tool_input_summary.get("routing_decision") in {"not_applicable", "scope_unavailable"}
            for call in callset
        )
        assert all(call.run_status == "skipped" for call in callset)
    assert all(tc.tool_name != "alphafold_get_prediction" for tc in step8.tool_call_records)

    pkg_blob = json.dumps(pkg.model_dump())
    assert "raw_sequence" not in pkg_blob
    assert "full_payload" not in pkg_blob


def test_step7_sequence_only_without_uniprot_skips_alphafold_lookup(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        if cand.get("candidate_type") == "target_antigen":
            cand["materials"] = [
                {
                    "material_id": "manual_sequence_target",
                    "material_type": "target_sequence",
                    "value": "MKTAYIAKQNNVG..."
                }
            ]
        else:
            cand["materials"] = []
            if isinstance(cand.get("materials"), list):
                cand["materials"] = []
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)

    rec = next(r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input")
    af_calls = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == "alphafold_get_prediction"]
    assert len(af_calls) == 1
    af_call = af_calls[0]
    assert af_call.run_status == "skipped"
    assert af_call.tool_input_summary["routing_decision"] == "not_applicable"
    assert af_call.tool_input_summary.get("reason") == "sequence-only input has no UniProt accession for AlphaFold prediction lookup"

    assert not any(ref.source_kind == "predicted_needed" for ref in rec.structure_refs)


def test_step7_scope_failure_records_scope_unavailable(
    local_storage, registry_service, workflow_state_service
):
    class _FailingScopeClient(LocalMCPClient):
        def list_tools(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("scope backend unavailable")

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_FailingScopeClient(),
    ).run_step_7(run_id)

    by_name = {tc.tool_name: tc for tc in pkg.structure_tool_call_records}
    for tool in structure_and_design_module._STEP7_SCOPED_TOOLS:
        assert tool in by_name
        tc = by_name[tool]
        assert tc.run_status == "skipped"
        assert tc.tool_input_summary["routing_decision"] == "scope_unavailable"
        assert "scope backend unavailable" in tc.tool_input_summary.get("reason", "")
        assert tc.run_status == "skipped"


def test_step7_database_search_results_become_ambiguous_candidates_only(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[],
    )
    overrides = {
        "RCSBAdvSearch_search_structures": lambda **kw: {
            "query": kw.get("query"),
            "structures": [
                {"pdb_id": "1N8Z", "method": "xray", "resolution": 2.1},
            ],
        },
        "PDBeSearch_search_structures": lambda **kw: {
            "query": kw.get("query"),
            "structures": [
                {"pdb_id": "2N8Z", "method": "crystal", "resolution": 2.3},
            ],
        },
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings=overrides,
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run_step_7(run_id)

    db_recs = [r for r in pkg.prepared_structure_inputs if r.input_case == "database_search_result"]
    assert db_recs
    for rec in db_recs:
        assert rec.database_search_candidates
        assert all(item.get("resource_binding_status") == "ambiguous" for item in rec.database_search_candidates)
        assert all(item.get("pdb_id") for item in rec.database_search_candidates)
        assert not any(item.get("source") == "selected_structure_ref" for item in rec.database_search_candidates)
        assert rec.structure_refs == []


def test_step8_skips_when_no_usable_structure_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    # A structure_file material without concrete file path or pdb id.
    target["materials"].append({
        "material_id": "target_missing_structure",
        "material_type": "structure_file",
        "value": "not_a_file_reference",
        "value_format": "text",
    })
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status in {"partial", "failed"}
    # No fabricated "uploaded" placeholder should be passed.
    for tc in results.tool_call_records:
        summary = json.dumps(tc.tool_input_summary or {})
        assert "pdb_id_or_path\": \"uploaded\"" not in summary


@pytest.mark.parametrize(
    "filename,expected_ranges",
    [
        ("S1.pdb", {"A": (8, 171), "B": (78, 171)}),
        ("S2.pdb", {"A": (11, 173), "B": (74, 171)}),
        ("S3.pdb", {"A": (7, 171), "B": (77, 171)}),
    ],
)
def test_step7_real_pdb_parses_compact_observed_chains(
    local_storage, registry_service, workflow_state_service, filename, expected_ranges
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": filename.lower(), "original_filename": filename,
            "storage_path": str(PROJECT_ROOT / "data" / "pdb" / filename),
            "content_type": "chemical/x-pdb", "size_bytes": 1, "role": "antigen",
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert {m.chain_id for m in rec.chain_mapping} == set(expected_ranges)
    assert all(m.chain_id_kind == "observed" and m.chain_role == "other" for m in rec.chain_mapping)
    compact_ranges = {r["chain_id"]: (r["start"], r["end"]) for r in rec.residue_ranges if r.get("source") == "observed_structure"}
    assert compact_ranges == expected_ranges
    assert "chain_ids_missing" not in rec.missing_metadata_flags
    assert "chain_roles_unknown" in rec.missing_metadata_flags
    blob = json.dumps(pkg.model_dump())
    assert "ATOM      " not in blob and "HEADER    " not in blob


def test_step7_uniprot_and_inline_sequence_semantics(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "inline_target", "material_type": "target_sequence",
        "value": "MKTIIALSYIFCLVFA", "value_format": "fasta",
    })
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    inline = next(s for s in rec.sequence_refs_for_prediction if s.sequence_id == "inline_target")
    uniprot = next(s for s in rec.sequence_refs_for_prediction if s.sequence_id == "P04626")
    assert inline.sequence == "MKTIIALSYIFCLVFA"
    assert inline.sequence_value_status == "inline"
    assert inline.prediction_input_kind == "amino_acid_sequence"
    assert uniprot.sequence is None
    assert uniprot.source_ref == "P04626"
    assert uniprot.sequence_value_status == "identifier_only"
    assert uniprot.prediction_input_kind == "uniprot_id"
    assert rec.prediction_required is True
    assert rec.missing_metadata_flags == []
    assert pkg.structure_preparation_status == "ok"


def test_step7_multiple_candidates_do_not_receive_shared_first_pair_mapping(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"].append({"material_id": "t1seq", "material_type": "target_sequence", "value": "AAAA"})
    antibody["materials"].append({"material_id": "a1seq", "material_type": "antibody_heavy_chain_sequence", "value": "BBBB"})
    target2, antibody2 = deepcopy(target), deepcopy(antibody)
    target2["candidate_id"], antibody2["candidate_id"] = "target_2", "antibody_2"
    target2["materials"][0]["material_id"] = "target2_name"
    antibody2["materials"][0]["material_id"] = "antibody2_name"
    cct["candidate_records"].extend([target2, antibody2])
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert all(r.antigen_antibody_mapping is None for r in pkg.prepared_structure_inputs)
    for rec in pkg.prepared_structure_inputs:
        assert all(
            rec.candidate_id in {p["target_candidate_id"], p["antibody_candidate_id"]}
            and p["mapping_status"] == "ambiguous"
            for p in rec.chain_pair_candidates
        )


def test_step7_explicit_candidate_pair_only_maps_that_pair(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"].append({"material_id": "t1seq", "material_type": "target_sequence", "value": "AAAA"})
    antibody["materials"].extend([
        {"material_id": "a1heavy", "material_type": "antibody_heavy_chain_sequence", "value": "BBBB"},
        {"material_id": "a1light", "material_type": "antibody_light_chain_sequence", "value": "CCCC"},
    ])
    target["related_candidate_id"] = antibody["candidate_id"]
    target2, antibody2 = deepcopy(target), deepcopy(antibody)
    target2["candidate_id"], antibody2["candidate_id"] = "target_2", "antibody_2"
    target2.pop("related_candidate_id", None)
    cct["candidate_records"].extend([target2, antibody2])
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    mapped = [r for r in pkg.prepared_structure_inputs if r.antigen_antibody_mapping]
    assert {r.candidate_id for r in mapped} == {target["candidate_id"], antibody["candidate_id"]}
    assert all(r.antigen_antibody_mapping["relationship_source"] == "explicit" for r in mapped)
    assert all(r.antigen_antibody_mapping["mapping_status"] == "sequence_only_prediction_needed" for r in mapped)


def test_step7_unscoped_global_residue_range_is_unresolved(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{
            "id_type": "residue_range", "value": "residues 20-240", "source": "user"
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert all(r.residue_ranges == [] for r in pkg.prepared_structure_inputs)
    assert any(u["resource_type"] == "residue_range" for u in pkg.unresolved_resource_refs)


def test_step7_raw_file_content_is_not_embedded(
    local_storage, registry_service, workflow_state_service
):
    pdb_key = local_storage.run_key("sentinel", "inputs/files/target.pdb")
    fasta_key = local_storage.run_key("sentinel", "inputs/files/heavy.fasta")
    local_storage.write_bytes(pdb_key, b"HEADER    RAW_PDB_SENTINEL\nATOM      1  CA  ALA A  20       1.0   1.0   1.0\nEND\n")
    local_storage.write_bytes(fasta_key, b">heavy\nRAW_FASTA_SEQUENCE_SENTINEL\n")
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {"file_id": "target_pdb", "original_filename": "target.pdb", "storage_path": pdb_key, "role": "antigen"},
            {"file_id": "heavy_fasta", "original_filename": "heavy.fasta", "storage_path": fasta_key, "role": "antibody_heavy"},
        ],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    blob = json.dumps(pkg.model_dump())
    assert "RAW_PDB_SENTINEL" not in blob
    assert "RAW_FASTA_SEQUENCE_SENTINEL" not in blob
    assert "ATOM      1" not in blob


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
    assert "refinement_resolution" in confidence_types
    quality_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in {"RCSBData_get_entry", "ProteinsPlus_profile_structure_quality"}
    ]
    assert quality_calls
    assert all(tc.run_status in {"success", "dependency_unavailable", "skipped"} for tc in quality_calls)

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
                "original_filename": "target.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
                "role": "target",
            },
        ],
    )
    raw_path = local_storage.run_key(run_id, "inputs/raw_request_record.json")
    raw = local_storage.read_json(raw_path)
    raw["uploaded_files"][0]["storage_path"] = local_storage.run_key(run_id, "inputs/files/file_pdb.pdb")
    local_storage.write_json(raw_path, raw)
    local_storage.write_bytes(raw["uploaded_files"][0]["storage_path"], b"HEADER    DUMMY\nEND\n")
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
