"""Step 5 batch-6 regression — material/context organization.

The professor clarified Step 5 must organize available reference inputs,
materials, fragments, known components, identifiers, and data gaps —
**not** generate complete ADC candidates or modeled pose candidates.

This suite covers the six professor benchmark cases:

A. HER2 benchmark comparing T-DM1 vs T-DXd / Enhertu.
B. New TROP2 ADC with MMAE (partial context, gaps explicit).
C. ChEMBL / ZINC compound IDs as possible HER2 payload candidates.
D. vc-MMAE linker_payload decomposition.
E. PDB / structure file → reference material, NEVER generated pose.
F. Downstream query hint ordering — antibody only when explicit.

Plus scope guarantees:
G. Step 5 catalog does not expose Step 6 / 13 / 14 tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService


def _run_through_step_4(
    local_storage, registry_service, workflow_state_service,
    *,
    raw_user_query: str,
    user_provided_context: dict,
    uploaded_files: list[dict] | None = None,
) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query=raw_user_query,
        user_provided_context=user_provided_context,
        uploaded_files=uploaded_files,
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(rec.run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)
    return rec.run_id


def _agent(local_storage, registry_service, workflow_state_service) -> CandidateContextAgent:
    return CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )


# ── A. HER2 benchmark with T-DM1 vs T-DXd / Enhertu ────────────────────────


def test_A_her2_benchmark_tdm1_vs_tdxd(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query=(
            "We want to evaluate the HER2 ADC benchmark comparing T-DM1 "
            "vs T-DXd (Enhertu) for breast cancer."
        ),
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)

    by_label = {r.candidate_label: r for r in table.candidate_records}
    # Both reference ADCs surface.
    assert "T-DM1" in by_label
    assert "T-DXd" in by_label or "Enhertu" in by_label

    for ref_label in ("T-DM1", "T-DXd", "Enhertu"):
        if ref_label in by_label:
            rec = by_label[ref_label]
            assert rec.candidate_role == "reference_benchmark"
            assert rec.is_generated_candidate is False
            assert rec.candidate_type == "adc_construct"
            assert rec.context_status == "complete_reference"
            # Components surface as materials with role tags.
            roles = {m.role for m in rec.materials if m.role}
            assert "antibody" in roles
            # T-DM1 has explicit payload; T-DXd has linker_payload + payload.
            assert "payload" in roles or "linker_payload" in roles


# ── B. TROP2 ADC with MMAE (partial context, no invented antibody) ─────────


def test_B_trop2_adc_with_mmae_partial_context(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Design a new TROP2 ADC with MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "TROP2",
            "payload_linker_text": "MMAE",
        },
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)

    # Target preserved.
    target_records = [
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    ]
    assert target_records
    target_rec = target_records[0]
    assert target_rec.candidate_label == "TROP2"
    assert target_rec.is_generated_candidate is False
    assert target_rec.candidate_role == "partial_context"
    # Explicit data gaps for missing antibody / linker.
    assert "antibody" in target_rec.missing_material_roles
    assert "linker" in target_rec.missing_material_roles
    assert any("complete ADC" in g.lower() or "complete adc" in g.lower()
               for g in target_rec.data_gaps)

    # No antibody candidate was invented.
    assert not any(
        r.candidate_type == "antibody" for r in table.candidate_records
    )
    # No reference benchmark was created.
    assert not any(
        r.candidate_role == "reference_benchmark" for r in table.candidate_records
    )

    # Payload preserved as a compound_component with MMAE material.
    payload_records = [
        r for r in table.candidate_records
        if any(m.role == "payload" for m in r.materials)
    ]
    assert payload_records
    assert payload_records[0].is_generated_candidate is False


# ── C. ChEMBL / ZINC compounds as possible HER2 payload candidates ─────────


def test_C_chembl_zinc_as_compound_materials_only(
    local_storage, registry_service, workflow_state_service
):
    """Without explicit 'payload candidates' phrasing, ChEMBL/ZINC IDs
    stay generic compound material records."""
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Look at CHEMBL1201585 and ZINC98765 for HER2 work",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    compound = next(
        r for r in table.candidate_records
        if r.candidate_type == "compound_component"
    )
    assert compound.is_generated_candidate is False
    assert compound.candidate_role == "material_only"
    assert compound.context_status == "material_pool"
    id_types = {i.id_type for i in compound.identifiers}
    assert "chembl_id" in id_types
    assert "zinc_id" in id_types
    # ZINC stays zinc_id — never zinc22 anywhere.
    blob = json.dumps(table.model_dump()).upper()
    assert "ZINC22" not in blob


def test_C_explicit_payload_phrasing_promotes_role(
    local_storage, registry_service, workflow_state_service
):
    """When the user explicitly says 'as payload candidates', role
    promotes to payload_candidate."""
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query=(
            "Screen CHEMBL1201585 and ZINC98765 as payload candidates "
            "for a HER2 ADC."
        ),
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    compound = next(
        r for r in table.candidate_records
        if r.candidate_type == "compound_component"
    )
    # Role promoted on the materials.
    roles = {m.role for m in compound.materials}
    assert "payload_candidate" in roles
    assert compound.candidate_role == "user_provided_candidate"


# ── D. vc-MMAE linker_payload decomposition ────────────────────────────────


def test_D_vc_mmae_decomposition_preserved_in_step5(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Design HER2 ADC with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "payload_linker_text": "vc-MMAE",
        },
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    # A linker_payload material surfaces for vc-MMAE.
    lp_records = [
        r for r in table.candidate_records
        if any(m.role == "linker_payload" for m in r.materials)
    ]
    assert lp_records
    rec = lp_records[0]
    # Decomposition into linker (inferred) + payload (explicit) is preserved.
    by_role = {
        m.role: m for m in rec.materials if m.role in {"linker", "payload", "linker_payload"}
    }
    assert "linker_payload" in by_role
    assert "payload" in by_role
    # MMAE payload explicit (user wrote MMAE inside the alias).
    assert by_role["payload"].role_status == "explicit"
    # Linker remains inferred unless user named it.
    if "linker" in by_role:
        assert by_role["linker"].role_status == "inferred"


# ── E. PDB structure file → reference material, never generated pose ──────


def test_E_uploaded_pdb_is_reference_material_not_generated_pose(
    local_storage, registry_service, workflow_state_service, tmp_path: Path
):
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("HEADER fake\nEND\n")
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="HER2 ADC with attached structure",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[
            {
                "file_id": "f1",
                "original_filename": "complex.pdb",
                "storage_path": str(pdb_path),
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": pdb_path.stat().st_size,
            }
        ],
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    # Structure material lives under the target candidate as a reference.
    target = next(
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    )
    structure_mats = [m for m in target.materials if m.material_type in {"structure_file", "structure_ref"}]
    assert structure_mats
    _structure_role_family = {
        "structure_reference",
        "receptor_structure_reference",
        "antigen_structure_reference",
        "fab_structure_reference",
        "fc_region_reference",
        "antibody_arm_reference",
        "glycan_or_glycosylation_reference",
        "n297_site_reference",
        "linker_attachment_site",
        "experimental_fragment_reference",
    }
    for m in structure_mats:
        assert m.role in _structure_role_family, m.role
        assert m.role_status == "explicit"
    # No "pose" / "pose_ensemble" / "generated_pose" anywhere in Step 5 output.
    blob = json.dumps(table.model_dump()).lower()
    for forbidden in (
        "pose_ensemble",
        "generated_pose",
        "modeled_pose",
        "pose_candidate",
    ):
        assert forbidden not in blob, f"Step 5 must not emit {forbidden!r}"
    # And no record is marked as a generated candidate.
    assert not any(r.is_generated_candidate for r in table.candidate_records)


# ── F. downstream query hints — antibody only when explicitly provided ────


def test_F_downstream_hints_omit_antibody_when_not_provided(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Design a TROP2 ADC with MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "TROP2",
            "payload_linker_text": "MMAE",
        },
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    hints = table.downstream_query_hints
    roles_in_hints = [h["role"] for h in hints]
    # antibody must NOT be in hints when not explicitly provided.
    assert "antibody" not in roles_in_hints, hints
    # payload + target are present.
    assert "payload" in roles_in_hints
    assert "target" in roles_in_hints
    # Priority ordering: payload appears before target.
    assert roles_in_hints.index("payload") < roles_in_hints.index("target")


def test_F_downstream_hints_include_antibody_when_explicit(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Design HER2 ADC with Trastuzumab + vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    roles_in_hints = [h["role"] for h in table.downstream_query_hints]
    assert "antibody" in roles_in_hints


def test_F_downstream_hints_prioritize_complete_adc_first(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Compare T-DM1 vs T-DXd for HER2 ADC benchmarks",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    roles = [h["role"] for h in table.downstream_query_hints]
    # complete_adc comes first.
    assert roles[0] == "complete_adc"


# ── G. Step 5 catalog scope guarantees ─────────────────────────────────────


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XLSX = (
    _PROJECT_ROOT.parent / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
)


@pytest.fixture
def inventory_client():
    import os
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(_DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


def test_G_step5_catalog_is_scoped(inventory_client):
    tools = inventory_client.list_tools(
        agent_name="candidate_context_agent", step_id="step_05"
    )
    names = set(tools)
    # Step 5 catalog is small and curated, not the full 99-tool surface.
    assert 0 < len(names) < 30
    # Off-limit Step 6 / 13 / 14 tools must not leak.
    for off_limit in (
        "DrugProps_pains_filter",
        "SwissADME_calculate_adme",
        "ChEMBL_search_activities",
        "EuropePMC_search_articles",
        "PubChem_get_associated_patents_by_CID",
        "FDA_OrangeBook_get_patent_info",
        "drugbank_get_drug_references_by_drug_name_or_id",
    ):
        assert off_limit not in names, f"{off_limit} leaked into Step 5"


def test_G_step5_zinc_is_intentionally_disabled_for_live(inventory_client):
    """ZINC stays in the inventory at Step 5 scope (architecture
    permits it) but `_live=True` must always raise NotImplementedError.
    Step 5 must never label ZINC as ZINC22."""
    tools = set(inventory_client.list_tools(
        agent_name="candidate_context_agent", step_id="step_05"
    ))
    assert "ZINC_search_by_smiles" in tools  # ZINC is Step 5 scoped.
    # And asking for it live must surface dependency_unavailable.
    res = inventory_client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ZINC_search_by_smiles",
        smiles="CCO", _live=True,
    )
    assert res["run_status"] == "dependency_unavailable"
    assert res["executor"] == "deferred"


def test_G_step5_agent_cannot_call_step6_tool(inventory_client):
    """Even if the agent attempted to invoke a Step 6 tool, scope_filter
    would block it. Direct call yields skipped / tool_not_in_agent_scope."""
    res = inventory_client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="DrugProps_pains_filter",
        smiles="CCO",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"


# ── raw-payload isolation still holds ──────────────────────────────────────


def test_raw_tool_payload_does_not_leak_into_candidate_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _run_through_step_4(
        local_storage, registry_service, workflow_state_service,
        raw_user_query="Design HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    cand_blob = json.dumps(persisted["candidate_records"])
    # Mock SAbDab/ChEMBL/etc. envelopes use the "status: mocked" shape;
    # raw payload markers must NOT appear inside normalized records.
    for raw_marker in ("\"hits\":", "\"results\":", "\"executor\":"):
        assert raw_marker not in cand_blob, (
            f"raw upstream payload leaked into candidate_records: {raw_marker}"
        )
