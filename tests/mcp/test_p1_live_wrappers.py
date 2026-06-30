"""P1 migration round: 10 wrappers now wired through ToolUniverseAdapter.

Covers Step 5 (ChEMBL family), Step 6 (DrugProps_lipinski_filter),
Step 7 (RCSBData_get_assembly + PDBeSearch_search_structures),
Step 13 (openalex/SemanticScholar/PubTator3).

Each tool gets at minimum:
- `_live=False` mock unchanged
- `_live=True` routes through fake adapter with the right TU argument mapping
- invalid args raise locally
- TU `{status: error}` is normalized to `upstream_error`
"""

from __future__ import annotations

import pytest

from app.mcp.tools.chembl import (
    ChEMBL_get_assay_activities,
    ChEMBL_get_drug,
    ChEMBL_get_drug_mechanisms,
    ChEMBL_get_molecule,
    ChEMBL_get_molecule_targets,
    ChEMBL_get_target_activities,
    ChEMBL_get_target_assays,
    ChEMBL_search_assays,
    ChEMBL_search_binding_sites,
    ChEMBL_search_compound_structural_alerts,
    ChEMBL_search_drugs,
    ChEMBL_search_molecules,
    ChEMBL_search_targets,
)
from app.mcp.tools.developability_compounds import DrugProps_lipinski_filter
from app.mcp.tools.evidence import (
    PubTator3_LiteratureSearch,
    PubTator3_get_annotations,
    SemanticScholar_search_papers,
    openalex_search_works,
)
from app.mcp.tools.pdbe_pisa import (
    PDBePISA_get_interfaces,
    PDBePISA_get_monomer_analysis,
    PDBe_KB_get_interface_residues,
)
from app.mcp.tools.proteins_plus import (
    BINDINGS as PROTEINSPLUS_BINDINGS,
    ProteinsPlus_profile_structure_quality,
)
from app.mcp.tools.rcsb_pdbe import PDBeSearch_search_structures, RCSBData_get_assembly


# ── ChEMBL_search_molecules ────────────────────────────────────────────────

def test_chembl_search_molecules_mock_unchanged():
    out = ChEMBL_search_molecules(query="aspirin")
    assert out["status"] == "mocked"
    assert out["query"] == "aspirin"
    assert out["molecules"] == []


def test_chembl_search_molecules_live_routes(install_universe):
    fake = install_universe(
        tools={
            "ChEMBL_search_molecules": lambda args: {
                "molecules": [{"molecule_chembl_id": "CHEMBL25"}]
            }
        }
    )
    out = ChEMBL_search_molecules(query="aspirin", limit=10, _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "aspirin", "limit": 10}


def test_chembl_search_molecules_live_empty_query_omits_arg(install_universe):
    fake = install_universe(tools={"ChEMBL_search_molecules": lambda args: {"molecules": []}})
    ChEMBL_search_molecules(_live=True)
    assert "query" not in fake.calls[0]["arguments"]
    assert fake.calls[0]["arguments"]["limit"] == 20


def test_chembl_search_molecules_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_molecules": lambda args: {
                "status": "error",
                "error": "timeout",
            }
        }
    )
    out = ChEMBL_search_molecules(query="x", _live=True)
    assert out["status"] == "upstream_error"


# ── ChEMBL_get_molecule ────────────────────────────────────────────────────

def test_chembl_get_molecule_mock_unchanged():
    out = ChEMBL_get_molecule(chembl_id="CHEMBL25")
    assert out["status"] == "mocked"
    assert out["chembl_id"] == "CHEMBL25"


def test_chembl_get_molecule_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_molecule()


def test_chembl_get_molecule_live_routes(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_molecule": lambda args: {"molecule_chembl_id": args["chembl_id"]}}
    )
    out = ChEMBL_get_molecule(chembl_id="CHEMBL25", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"chembl_id": "CHEMBL25"}


def test_chembl_get_molecule_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_molecule": lambda args: {"status": "error", "error": "not found"}}
    )
    out = ChEMBL_get_molecule(chembl_id="CHEMBL_BAD", _live=True)
    assert out["status"] == "upstream_error"


# ── ChEMBL_search_drugs ────────────────────────────────────────────────────

def test_chembl_search_drugs_mock_unchanged():
    out = ChEMBL_search_drugs(query="sotorasib")
    assert out["status"] == "mocked"
    assert out["drugs"] == []


def test_chembl_search_drugs_live_routes(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_drugs": lambda args: {"drugs": [{"name": "Sotorasib"}]}}
    )
    out = ChEMBL_search_drugs(query="sotorasib", limit=5, _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "sotorasib", "limit": 5}


def test_chembl_search_drugs_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_search_drugs": lambda args: {"status": "error", "error": "slow"}}
    )
    out = ChEMBL_search_drugs(query="x", _live=True)
    assert out["status"] == "upstream_error"


# ── ChEMBL_get_drug ────────────────────────────────────────────────────────

def test_chembl_get_drug_mock_unchanged():
    out = ChEMBL_get_drug(drug_chembl_id="CHEMBL1201581")
    assert out["status"] == "mocked"


def test_chembl_get_drug_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_drug()


def test_chembl_get_drug_live_routes(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_drug": lambda args: {"drug": {"name": "Adalimumab"}}}
    )
    out = ChEMBL_get_drug(drug_chembl_id="CHEMBL1201581", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"drug_chembl_id": "CHEMBL1201581"}


def test_chembl_get_drug_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_drug": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_drug(drug_chembl_id="CHEMBL_BAD", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 6 batch 2: remaining ChEMBL developability wrappers ───────────────

def test_chembl_structural_alerts_mock_unchanged():
    out = ChEMBL_search_compound_structural_alerts(
        molecule_chembl_id="CHEMBL25", alert_name_contains="toxic"
    )
    assert out["status"] == "mocked"
    assert out["molecule_chembl_id"] == "CHEMBL25"
    assert out["alert_name_contains"] == "toxic"
    assert out["structural_alerts"] == []


def test_chembl_structural_alerts_live_maps_filters_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_compound_structural_alerts": lambda args: {"alerts": []}}
    )
    ChEMBL_search_compound_structural_alerts(
        molecule_chembl_id="CHEMBL25",
        alert_name_contains="quinone",
        limit=9999,
        offset=-4,
        _live=True,
    )
    assert fake.calls[0]["arguments"] == {
        "molecule_chembl_id": "CHEMBL25",
        "alert_name__contains": "quinone",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_structural_alerts_invalid_limit():
    with pytest.raises(ValueError):
        ChEMBL_search_compound_structural_alerts(limit="bad", _live=True)  # type: ignore[arg-type]


def test_chembl_structural_alerts_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_compound_structural_alerts": lambda args: {
                "status": "error",
                "error": "chembl down",
            }
        }
    )
    out = ChEMBL_search_compound_structural_alerts(molecule_chembl_id="X", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_molecule_targets_mock_unchanged():
    out = ChEMBL_get_molecule_targets(molecule_chembl_id="CHEMBL25")
    assert out["status"] == "mocked"
    assert out["molecule_chembl_id"] == "CHEMBL25"
    assert out["targets"] == []


def test_chembl_molecule_targets_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_molecule_targets()


def test_chembl_molecule_targets_live_maps_exact_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_molecule_targets": lambda args: {"targets": [{"id": "CHEMBL203"}]}}
    )
    out = ChEMBL_get_molecule_targets(molecule_chembl_id="CHEMBL25", limit=5000, _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "molecule_chembl_id__exact": "CHEMBL25",
        "limit": 1000,
    }


def test_chembl_molecule_targets_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_molecule_targets": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_molecule_targets(molecule_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_search_targets_mock_unchanged():
    out = ChEMBL_search_targets(pref_name_contains="kinase", organism="Homo sapiens")
    assert out["status"] == "mocked"
    assert out["pref_name_contains"] == "kinase"
    assert out["targets"] == []


def test_chembl_search_targets_rejects_unknown_projection_field():
    with pytest.raises(ValueError):
        ChEMBL_search_targets(fields=["not_a_field"])


def test_chembl_search_targets_live_maps_projection_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_targets": lambda args: {"targets": [{"target_chembl_id": "CHEMBL203"}]}}
    )
    ChEMBL_search_targets(
        pref_name_contains="kinase",
        organism="Homo sapiens",
        target_type="SINGLE PROTEIN",
        fields=["target_chembl_id", "pref_name"],
        limit=0,
        offset=-10,
        _live=True,
    )
    assert fake.calls[0]["arguments"] == {
        "pref_name__contains": "kinase",
        "organism": "Homo sapiens",
        "target_type": "SINGLE PROTEIN",
        "fields": ["target_chembl_id", "pref_name"],
        "limit": 1,
        "offset": 0,
    }


def test_chembl_search_targets_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_search_targets": lambda args: {"status": "error", "error": "timeout"}}
    )
    out = ChEMBL_search_targets(pref_name_contains="x", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_target_activities_mock_unchanged():
    out = ChEMBL_get_target_activities(target_chembl_id="CHEMBL2074")
    assert out["status"] == "mocked"
    assert out["target_chembl_id"] == "CHEMBL2074"
    assert out["activities"] == []


def test_chembl_target_activities_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_target_activities()


def test_chembl_target_activities_live_maps_exact_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_target_activities": lambda args: {"activities": [{"activity_id": 1}]}}
    )
    ChEMBL_get_target_activities(
        target_chembl_id="CHEMBL2074", limit=5000, offset=-2, _live=True
    )
    assert fake.calls[0]["arguments"] == {
        "target_chembl_id__exact": "CHEMBL2074",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_target_activities_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_target_activities": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_target_activities(target_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_drug_mechanisms_mock_unchanged():
    out = ChEMBL_get_drug_mechanisms(drug_name="trastuzumab")
    assert out["status"] == "mocked"
    assert out["drug_name"] == "trastuzumab"
    assert out["mechanisms"] == []


def test_chembl_drug_mechanisms_requires_id_or_name():
    with pytest.raises(ValueError):
        ChEMBL_get_drug_mechanisms()


def test_chembl_drug_mechanisms_live_routes_id_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_drug_mechanisms": lambda args: {"mechanisms": [{"action": "inhibitor"}]}}
    )
    ChEMBL_get_drug_mechanisms(drug_chembl_id="CHEMBL1201581", limit=5000, _live=True)
    assert fake.calls[0]["arguments"] == {
        "drug_chembl_id": "CHEMBL1201581",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_drug_mechanisms_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_drug_mechanisms": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_drug_mechanisms(drug_name="missing", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_search_assays_mock_unchanged():
    out = ChEMBL_search_assays(target_chembl_id="CHEMBL2074", assay_type="B")
    assert out["status"] == "mocked"
    assert out["target_chembl_id"] == "CHEMBL2074"
    assert out["assays"] == []


def test_chembl_search_assays_rejects_unknown_projection_field():
    with pytest.raises(ValueError):
        ChEMBL_search_assays(fields=["not_a_field"])


def test_chembl_search_assays_live_maps_fields_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_assays": lambda args: {"assays": [{"assay_chembl_id": "CHEMBL1"}]}}
    )
    ChEMBL_search_assays(
        target_chembl_id="CHEMBL2074",
        assay_type="B",
        fields=["assay_chembl_id", "assay_type"],
        limit=5000,
        offset=-1,
        _live=True,
    )
    assert fake.calls[0]["arguments"] == {
        "assay_type": "B",
        "target_chembl_id": "CHEMBL2074",
        "fields": ["assay_chembl_id", "assay_type"],
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_search_assays_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_search_assays": lambda args: {"status": "error", "error": "bad filter"}}
    )
    out = ChEMBL_search_assays(target_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_target_assays_mock_unchanged():
    out = ChEMBL_get_target_assays(target_chembl_id="CHEMBL2074")
    assert out["status"] == "mocked"
    assert out["target_chembl_id"] == "CHEMBL2074"
    assert out["assays"] == []


def test_chembl_target_assays_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_target_assays()


def test_chembl_target_assays_live_maps_exact_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_target_assays": lambda args: {"assays": [{"assay_chembl_id": "CHEMBL1"}]}}
    )
    ChEMBL_get_target_assays(target_chembl_id="CHEMBL2074", limit=5000, offset=-1, _live=True)
    assert fake.calls[0]["arguments"] == {
        "target_chembl_id__exact": "CHEMBL2074",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_target_assays_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_target_assays": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_target_assays(target_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_assay_activities_mock_unchanged():
    out = ChEMBL_get_assay_activities(assay_chembl_id="CHEMBL615117")
    assert out["status"] == "mocked"
    assert out["assay_chembl_id"] == "CHEMBL615117"
    assert out["activities"] == []


def test_chembl_assay_activities_requires_id():
    with pytest.raises(ValueError):
        ChEMBL_get_assay_activities()


def test_chembl_assay_activities_live_maps_exact_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_get_assay_activities": lambda args: {"activities": [{"activity_id": 1}]}}
    )
    ChEMBL_get_assay_activities(
        assay_chembl_id="CHEMBL615117", limit=5000, offset=-5, _live=True
    )
    assert fake.calls[0]["arguments"] == {
        "assay_chembl_id__exact": "CHEMBL615117",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_assay_activities_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_get_assay_activities": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_get_assay_activities(assay_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_chembl_binding_sites_mock_unchanged():
    out = ChEMBL_search_binding_sites(
        target_chembl_id="CHEMBL2074", site_name_contains="ATP"
    )
    assert out["status"] == "mocked"
    assert out["target_chembl_id"] == "CHEMBL2074"
    assert out["site_name_contains"] == "ATP"
    assert out["binding_sites"] == []


def test_chembl_binding_sites_live_maps_filters_and_clamps(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_binding_sites": lambda args: {"binding_sites": [{"site_id": 1}]}}
    )
    out = ChEMBL_search_binding_sites(
        target_chembl_id="CHEMBL2074",
        site_name_contains="ATP",
        limit=5000,
        offset=-8,
        _live=True,
    )
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "target_chembl_id": "CHEMBL2074",
        "site_name__contains": "ATP",
        "limit": 1000,
        "offset": 0,
    }


def test_chembl_binding_sites_invalid_limit():
    with pytest.raises(ValueError):
        ChEMBL_search_binding_sites(limit="bad", _live=True)  # type: ignore[arg-type]


def test_chembl_binding_sites_live_upstream_error(install_universe):
    install_universe(
        tools={"ChEMBL_search_binding_sites": lambda args: {"status": "error", "error": "404"}}
    )
    out = ChEMBL_search_binding_sites(target_chembl_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


# ── DrugProps_lipinski_filter ──────────────────────────────────────────────

def test_drugprops_lipinski_mock_unchanged():
    out = DrugProps_lipinski_filter(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["passes_lipinski"] is None


def test_drugprops_lipinski_requires_smiles():
    with pytest.raises(ValueError):
        DrugProps_lipinski_filter()


def test_drugprops_lipinski_live_routes(install_universe):
    fake = install_universe(
        tools={
            "DrugProps_lipinski_filter": lambda args: {"passes_lipinski": True}
        }
    )
    out = DrugProps_lipinski_filter(smiles="CC(=O)Oc1ccccc1C(=O)O", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}


def test_drugprops_lipinski_live_upstream_error_propagates_rdkit_message(install_universe):
    install_universe(
        tools={
            "DrugProps_lipinski_filter": lambda args: {
                "status": "error",
                "error": "RDKit is required for drug property calculations.",
            }
        }
    )
    out = DrugProps_lipinski_filter(smiles="CCO", _live=True)
    assert out["status"] == "upstream_error"
    assert "rdkit" in out["error_message"].lower()


# ── RCSBData_get_assembly ──────────────────────────────────────────────────

def test_rcsb_assembly_mock_unchanged():
    out = RCSBData_get_assembly(pdb_id="4HHB")
    assert out["status"] == "mocked"
    assert out["pdb_id"] == "4hhb"
    assert out["assembly_id"] == "1"


def test_rcsb_assembly_requires_pdb_id():
    with pytest.raises(ValueError):
        RCSBData_get_assembly()


def test_rcsb_assembly_live_routes(install_universe):
    fake = install_universe(
        tools={"RCSBData_get_assembly": lambda args: {"assembly": {"id": args["pdb_id"]}}}
    )
    out = RCSBData_get_assembly(pdb_id="4HHB", assembly_id="2", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"pdb_id": "4hhb", "assembly_id": "2"}


def test_rcsb_assembly_live_upstream_error(install_universe):
    install_universe(
        tools={"RCSBData_get_assembly": lambda args: {"status": "error", "error": "404"}}
    )
    out = RCSBData_get_assembly(pdb_id="0000", _live=True)
    assert out["status"] == "upstream_error"


# ── PDBeSearch_search_structures ───────────────────────────────────────────

def test_pdbe_search_mock_unchanged():
    out = PDBeSearch_search_structures(query="kinase")
    assert out["status"] == "mocked"
    assert out["structures"] == []


def test_pdbe_search_requires_query():
    with pytest.raises(ValueError):
        PDBeSearch_search_structures()


def test_pdbe_search_live_routes(install_universe):
    fake = install_universe(
        tools={"PDBeSearch_search_structures": lambda args: {"results": [{"pdb_id": "1abc"}]}}
    )
    out = PDBeSearch_search_structures(query="kinase", limit=20, _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "kinase", "limit": 20}


def test_pdbe_search_live_clamps_limit(install_universe):
    fake = install_universe(tools={"PDBeSearch_search_structures": lambda args: {"results": []}})
    PDBeSearch_search_structures(query="x", limit=999, _live=True)
    assert fake.calls[0]["arguments"]["limit"] == 50


def test_pdbe_search_live_upstream_error(install_universe):
    install_universe(
        tools={"PDBeSearch_search_structures": lambda args: {"status": "error", "error": "x"}}
    )
    out = PDBeSearch_search_structures(query="x", _live=True)
    assert out["status"] == "upstream_error"


# ── openalex_search_works ──────────────────────────────────────────────────

def test_openalex_mock_unchanged():
    out = openalex_search_works("HER2 ADC")
    assert out["status"] == "mocked"
    assert out["results"] == []


def test_openalex_requires_query():
    with pytest.raises(ValueError):
        openalex_search_works("")


def test_openalex_live_routes(install_universe):
    fake = install_universe(
        tools={"openalex_search_works": lambda args: {"results": [{"id": "W1"}]}}
    )
    out = openalex_search_works("HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"query": "HER2 ADC"}


def test_openalex_live_upstream_error(install_universe):
    install_universe(
        tools={"openalex_search_works": lambda args: {"status": "error", "error": "rate"}}
    )
    out = openalex_search_works("x", _live=True)
    assert out["status"] == "upstream_error"


# ── SemanticScholar_search_papers ──────────────────────────────────────────

def test_semantic_scholar_mock_unchanged():
    out = SemanticScholar_search_papers("HER2 ADC")
    assert out["status"] == "mocked"


def test_semantic_scholar_requires_query():
    with pytest.raises(ValueError):
        SemanticScholar_search_papers("")


def test_semantic_scholar_live_routes(install_universe):
    fake = install_universe(
        tools={"SemanticScholar_search_papers": lambda args: {"papers": [{"id": "p1"}]}}
    )
    out = SemanticScholar_search_papers("HER2 ADC", limit=3, _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"query": "HER2 ADC", "limit": 3}


def test_semantic_scholar_live_clamps_limit(install_universe):
    fake = install_universe(tools={"SemanticScholar_search_papers": lambda args: {"papers": []}})
    SemanticScholar_search_papers("x", limit=9999, _live=True)
    assert fake.calls[0]["arguments"]["limit"] == 100


def test_semantic_scholar_live_upstream_error(install_universe):
    install_universe(
        tools={"SemanticScholar_search_papers": lambda args: {"status": "error", "error": "rate"}}
    )
    out = SemanticScholar_search_papers("x", _live=True)
    assert out["status"] == "upstream_error"


# ── PubTator3_LiteratureSearch ─────────────────────────────────────────────

def test_pubtator_search_mock_unchanged():
    out = PubTator3_LiteratureSearch("HER2")
    assert out["status"] == "mocked"


def test_pubtator_search_requires_query():
    with pytest.raises(ValueError):
        PubTator3_LiteratureSearch("")


def test_pubtator_search_live_routes(install_universe):
    fake = install_universe(
        tools={"PubTator3_LiteratureSearch": lambda args: {"results": [{"pmid": "1"}]}}
    )
    out = PubTator3_LiteratureSearch("HER2", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"query": "HER2"}


def test_pubtator_search_live_upstream_error(install_universe):
    install_universe(
        tools={"PubTator3_LiteratureSearch": lambda args: {"status": "error", "error": "5xx"}}
    )
    out = PubTator3_LiteratureSearch("x", _live=True)
    assert out["status"] == "upstream_error"


# ── PubTator3_get_annotations (legacy pmid → TU pmids mapping) ─────────────

def test_pubtator_annotations_mock_unchanged():
    out = PubTator3_get_annotations("33205991")
    assert out["status"] == "mocked"


def test_pubtator_annotations_requires_pmid():
    with pytest.raises(ValueError):
        PubTator3_get_annotations("")


def test_pubtator_annotations_live_maps_pmid_to_pmids(install_universe):
    """Legacy wrapper takes singular `pmid`; TU expects `pmids`. The
    wrapper must rename, not pass `pmid`."""
    fake = install_universe(
        tools={"PubTator3_get_annotations": lambda args: {"annotations": [{"id": "G"}]}}
    )
    out = PubTator3_get_annotations("33205991", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"pmids": "33205991"}
    assert "pmid" not in fake.calls[0]["arguments"]


def test_pubtator_annotations_live_forwards_comma_separated(install_universe):
    fake = install_universe(tools={"PubTator3_get_annotations": lambda args: {"annotations": []}})
    PubTator3_get_annotations("33205991,34234088", _live=True)
    assert fake.calls[0]["arguments"]["pmids"] == "33205991,34234088"


def test_pubtator_annotations_live_upstream_error(install_universe):
    install_universe(
        tools={"PubTator3_get_annotations": lambda args: {"status": "error", "error": "x"}}
    )
    out = PubTator3_get_annotations("33205991", _live=True)
    assert out["status"] == "upstream_error"


# ── SAbDab_get_structure (Step 7 carve-out: completes Step 7 6/6) ──────────

from app.mcp.tools.sabdab import SAbDab_get_structure


def test_sabdab_get_structure_mock_unchanged():
    out = SAbDab_get_structure(pdb_id="6W41")
    assert out["status"] == "mocked"
    assert out["pdb_id"] == "6w41"
    assert out["structure"] is None


def test_sabdab_get_structure_requires_pdb_id():
    with pytest.raises(ValueError):
        SAbDab_get_structure()


def test_sabdab_get_structure_live_routes(install_universe):
    fake = install_universe(
        tools={
            "SAbDab_get_structure": lambda args: {
                "pdb_id": args["pdb_id"],
                "chains": [{"chain_id": "H", "type": "heavy"}],
            }
        }
    )
    out = SAbDab_get_structure(pdb_id="6W41", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    # Wrapper sends only canonical `pdb_id`, never `pdb_code`.
    assert fake.calls[0]["arguments"] == {"pdb_id": "6w41"}
    assert "pdb_code" not in fake.calls[0]["arguments"]


def test_sabdab_get_structure_live_upstream_error(install_universe):
    install_universe(
        tools={
            "SAbDab_get_structure": lambda args: {
                "status": "error",
                "error": "PDB id not in SAbDab",
            }
        }
    )
    out = SAbDab_get_structure(pdb_id="0000", _live=True)
    assert out["status"] == "upstream_error"
    assert "sabdab" in out["error_message"].lower()


# ── RCSBAdvSearch_search_structures (Step 7 6/6 close-out) ────────────────

from app.mcp.tools.rcsb_pdbe import RCSBAdvSearch_search_structures


def test_rcsb_advsearch_mock_unchanged():
    out = RCSBAdvSearch_search_structures(query="kinase")
    assert out["status"] == "mocked"
    assert out["query"] == "kinase"
    assert out["structures"] == []


def test_rcsb_advsearch_live_forwards_only_provided_args(install_universe):
    """All TU args are optional; only non-empty values must be sent."""
    fake = install_universe(
        tools={"RCSBAdvSearch_search_structures": lambda args: {"results": [{"pdb_id": "1abc"}]}}
    )
    out = RCSBAdvSearch_search_structures(
        query="kinase", organism="Homo sapiens", max_resolution=2.5, rows=15, _live=True
    )
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "query": "kinase", "organism": "Homo sapiens",
        "max_resolution": 2.5, "rows": 15,
    }
    # No spurious empty filters.
    for omitted in ("experimental_method", "polymer_description",
                     "min_deposition_date", "sort_by"):
        assert omitted not in fake.calls[0]["arguments"]


def test_rcsb_advsearch_live_clamps_rows(install_universe):
    fake = install_universe(
        tools={"RCSBAdvSearch_search_structures": lambda args: {"results": []}}
    )
    RCSBAdvSearch_search_structures(query="x", rows=999, _live=True)
    assert fake.calls[0]["arguments"]["rows"] == 50


def test_rcsb_advsearch_live_no_args_sends_empty_dict(install_universe):
    """Caller can pass no filters at all — TU treats it as 'list everything'."""
    fake = install_universe(
        tools={"RCSBAdvSearch_search_structures": lambda args: {"results": []}}
    )
    RCSBAdvSearch_search_structures(_live=True)
    assert fake.calls[0]["arguments"] == {}


def test_rcsb_advsearch_live_upstream_error(install_universe):
    install_universe(
        tools={"RCSBAdvSearch_search_structures": lambda args: {"status": "error", "error": "5xx"}}
    )
    out = RCSBAdvSearch_search_structures(query="x", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 13 close-out — LiteratureSearchTool / MultiAgentLiteratureSearch
# both now route through ToolUniverseAdapter. The TU implementations are
# ComposeTools that may consume an LLM key from the environment at TU's
# discretion — the wrapper never forwards or logs the key.

from app.mcp.tools.evidence import (  # noqa: E402
    LiteratureSearchTool,
    MultiAgentLiteratureSearch,
)


def test_literature_search_mock_unchanged():
    out = LiteratureSearchTool(query="HER2 ADC")
    assert out["status"] == "mocked"
    assert out["query"] == "HER2 ADC"


def test_literature_search_accepts_research_topic_alias_in_mock():
    out = LiteratureSearchTool(research_topic="HER2 ADC")
    assert out["status"] == "mocked"
    assert out["query"] == "HER2 ADC"


def test_literature_search_requires_query():
    with pytest.raises(ValueError):
        LiteratureSearchTool()


def test_literature_search_live_routes_through_tu_with_research_topic(install_universe):
    fake = install_universe(
        tools={
            "LiteratureSearchTool": lambda args: {
                "summary": "...",
                "papers": [{"pmid": "1"}],
            }
        }
    )
    out = LiteratureSearchTool(query="HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    # Wrapper must forward only `research_topic`, never the legacy `query`.
    assert fake.calls[0]["arguments"] == {"research_topic": "HER2 ADC"}


def test_literature_search_live_accepts_research_topic_alias(install_universe):
    fake = install_universe(
        tools={"LiteratureSearchTool": lambda args: {"summary": ".", "papers": []}}
    )
    LiteratureSearchTool(research_topic="EGFR", _live=True)
    assert fake.calls[0]["arguments"] == {"research_topic": "EGFR"}


def test_literature_search_live_upstream_error(install_universe):
    install_universe(
        tools={
            "LiteratureSearchTool": lambda args: {
                "status": "error",
                "error": "LLM key not configured",
            }
        }
    )
    out = LiteratureSearchTool(query="HER2", _live=True)
    assert out["status"] == "upstream_error"


def test_multi_agent_literature_mock_unchanged():
    out = MultiAgentLiteratureSearch(query="HER2 ADC")
    assert out["status"] == "mocked"
    assert out["total_papers"] == 0


def test_multi_agent_literature_requires_query():
    with pytest.raises(ValueError):
        MultiAgentLiteratureSearch(query="")


def test_multi_agent_literature_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "MultiAgentLiteratureSearch": lambda args: {
                "search_plans": [{"backend": "EuropePMC"}],
                "total_papers": 5,
            }
        }
    )
    out = MultiAgentLiteratureSearch(query="HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "query": "HER2 ADC",
        "max_iterations": 1,  # default
        "quality_threshold": 0.5,
    }


def test_multi_agent_literature_live_clamps_max_iterations_to_1(install_universe):
    """Caller cannot bypass the safety clamp."""
    fake = install_universe(
        tools={"MultiAgentLiteratureSearch": lambda args: {"total_papers": 0}}
    )
    MultiAgentLiteratureSearch(query="x", max_iterations=99, _live=True)
    assert fake.calls[0]["arguments"]["max_iterations"] == 1


def test_multi_agent_literature_live_forwards_quality_threshold(install_universe):
    fake = install_universe(
        tools={"MultiAgentLiteratureSearch": lambda args: {"total_papers": 0}}
    )
    MultiAgentLiteratureSearch(query="x", quality_threshold=0.8, _live=True)
    assert fake.calls[0]["arguments"]["quality_threshold"] == 0.8


def test_multi_agent_literature_live_upstream_error(install_universe):
    install_universe(
        tools={
            "MultiAgentLiteratureSearch": lambda args: {
                "status": "error",
                "error": "agent loop failed",
            }
        }
    )
    out = MultiAgentLiteratureSearch(query="HER2", _live=True)
    assert out["status"] == "upstream_error"


# ── ChEMBL_search_documents (Step 13 mechanical wire-up) ──────────────────

from app.mcp.tools.chembl import ChEMBL_search_documents  # noqa: E402


def test_chembl_search_documents_mock_unchanged():
    out = ChEMBL_search_documents(title_contains="HER2")
    assert out["status"] == "mocked"
    assert out["title_contains"] == "HER2"
    assert out["documents"] == []


def test_chembl_search_documents_live_routes(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_documents": lambda args: {"documents": [{"id": "D1"}]}}
    )
    out = ChEMBL_search_documents(title_contains="HER2", limit=5, offset=10, _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    # Wrapper translates `title_contains` → TU's `title__contains`.
    assert fake.calls[0]["arguments"] == {
        "title__contains": "HER2", "limit": 5, "offset": 10,
    }


def test_chembl_search_documents_live_omits_unset_filters(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_documents": lambda args: {"documents": []}}
    )
    ChEMBL_search_documents(_live=True)
    # No filter keys, only the defaults.
    assert fake.calls[0]["arguments"] == {"limit": 20, "offset": 0}
    assert "document_id" not in fake.calls[0]["arguments"]
    assert "title__contains" not in fake.calls[0]["arguments"]


def test_chembl_search_documents_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_documents": lambda args: {
                "status": "error",
                "error": "ChEMBL down",
            }
        }
    )
    out = ChEMBL_search_documents(title_contains="x", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 5 close-out — non-ZINC migration ────────────────────────────────

from app.mcp.tools.sabdab import (  # noqa: E402
    SAbDab_search_structures,
    TheraSAbDab_search_by_target,
    TheraSAbDab_search_therapeutics,
    iedb_search_bcr_sequences,
)
from app.mcp.tools.zinc import (  # noqa: E402
    ZINC_get_compound,
    ZINC_get_purchasable,
    ZINC_search_by_properties,
    ZINC_search_by_smiles,
    ZINC_search_compounds,
)


# SAbDab_search_structures

def test_sabdab_search_structures_mock_unchanged():
    out = SAbDab_search_structures(query="HER2")
    assert out["status"] == "mocked"
    assert out["query"] == "HER2"
    assert out["structures"] == []


def test_sabdab_search_structures_live_routes(install_universe):
    fake = install_universe(
        tools={"SAbDab_search_structures": lambda args: {"results": [{"pdb_id": "6w41"}]}}
    )
    out = SAbDab_search_structures(query="HER2", limit=20, _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "HER2", "limit": 20}
    # Wrapper exposes only canonical `query`, never sends `antigen` alias.
    assert "antigen" not in fake.calls[0]["arguments"]


def test_sabdab_search_structures_live_no_query_sends_only_limit(install_universe):
    fake = install_universe(tools={"SAbDab_search_structures": lambda args: {"results": []}})
    SAbDab_search_structures(_live=True)
    assert fake.calls[0]["arguments"] == {"limit": 50}


def test_sabdab_search_structures_live_clamps_limit(install_universe):
    fake = install_universe(tools={"SAbDab_search_structures": lambda args: {"results": []}})
    SAbDab_search_structures(query="x", limit=999, _live=True)
    assert fake.calls[0]["arguments"]["limit"] == 200


def test_sabdab_search_structures_live_upstream_error(install_universe):
    install_universe(
        tools={"SAbDab_search_structures": lambda args: {"status": "error", "error": "down"}}
    )
    out = SAbDab_search_structures(query="x", _live=True)
    assert out["status"] == "upstream_error"


# TheraSAbDab_search_by_target

def test_therasabdab_by_target_mock_unchanged():
    out = TheraSAbDab_search_by_target(target="HER2")
    assert out["status"] == "mocked"
    assert out["target"] == "HER2"
    assert out["therapeutics"] == []


def test_therasabdab_by_target_requires_target():
    with pytest.raises(ValueError):
        TheraSAbDab_search_by_target()


def test_therasabdab_by_target_live_routes(install_universe):
    fake = install_universe(
        tools={
            "TheraSAbDab_search_by_target": lambda args: {
                "therapeutics": [{"name": "Trastuzumab"}]
            }
        }
    )
    out = TheraSAbDab_search_by_target(target="HER2", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"target": "HER2"}


def test_therasabdab_by_target_live_upstream_error(install_universe):
    install_universe(
        tools={
            "TheraSAbDab_search_by_target": lambda args: {
                "status": "error", "error": "5xx"
            }
        }
    )
    out = TheraSAbDab_search_by_target(target="HER2", _live=True)
    assert out["status"] == "upstream_error"


# TheraSAbDab_search_therapeutics

def test_therasabdab_therapeutics_mock_unchanged():
    out = TheraSAbDab_search_therapeutics(query="trastuzumab")
    assert out["status"] == "mocked"
    assert out["query"] == "trastuzumab"


def test_therasabdab_therapeutics_requires_query():
    with pytest.raises(ValueError):
        TheraSAbDab_search_therapeutics()


def test_therasabdab_therapeutics_live_routes(install_universe):
    fake = install_universe(
        tools={
            "TheraSAbDab_search_therapeutics": lambda args: {
                "therapeutics": [{"name": "Trastuzumab", "format": "IgG1"}]
            }
        }
    )
    out = TheraSAbDab_search_therapeutics(query="trastuzumab", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "trastuzumab"}


def test_therasabdab_therapeutics_live_upstream_error(install_universe):
    install_universe(
        tools={
            "TheraSAbDab_search_therapeutics": lambda args: {
                "status": "error", "error": "down"
            }
        }
    )
    out = TheraSAbDab_search_therapeutics(query="x", _live=True)
    assert out["status"] == "upstream_error"


# iedb_search_bcr_sequences

def test_iedb_bcr_non_live_reports_not_live():
    out = iedb_search_bcr_sequences()
    assert out["status"] == "not_live"
    assert "live" in out["reason"].lower()
    assert out["sequences"] == []


def test_iedb_bcr_live_routes_default(install_universe):
    fake = install_universe(
        tools={"iedb_search_bcr_sequences": lambda args: {"results": [{"seq": "QVQL"}]}}
    )
    out = iedb_search_bcr_sequences(_live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"limit": 10, "offset": 0}
    assert "filters" not in fake.calls[0]["arguments"]


def test_iedb_bcr_live_forwards_filters(install_universe):
    fake = install_universe(tools={"iedb_search_bcr_sequences": lambda args: {"results": []}})
    iedb_search_bcr_sequences(
        filters={"receptor_group_id": "eq.123"}, limit=25, _live=True
    )
    assert fake.calls[0]["arguments"] == {
        "limit": 25, "offset": 0, "filters": {"receptor_group_id": "eq.123"},
    }


def test_iedb_bcr_live_clamps_limit(install_universe):
    fake = install_universe(tools={"iedb_search_bcr_sequences": lambda args: {"results": []}})
    iedb_search_bcr_sequences(limit=999, _live=True)
    assert fake.calls[0]["arguments"]["limit"] == 200


def test_iedb_bcr_live_upstream_error(install_universe):
    install_universe(
        tools={"iedb_search_bcr_sequences": lambda args: {"status": "error", "error": "down"}}
    )
    out = iedb_search_bcr_sequences(_live=True)
    assert out["status"] == "upstream_error"


# ── ZINC stays intentionally_disabled — no adapter routing ────────────────

@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ZINC_search_compounds, {"query": "HER2"}),
        (ZINC_get_compound, {"zinc_id": "ZINC0000001"}),
        (ZINC_search_by_smiles, {"smiles": "CCO"}),
        (ZINC_search_by_properties, {"properties": {"mw_lt": 500}}),
        (ZINC_get_purchasable, {"zinc_id": "ZINC0000001"}),
    ],
)
def test_zinc_live_is_intentionally_disabled_and_does_not_hit_tu(
    fn, kwargs, install_universe
):
    """ZINC `_live=True` must raise NotImplementedError BEFORE any adapter
    call — verified by checking the fake universe records zero calls."""
    fake = install_universe(tools={})  # any TU call would be visible here
    with pytest.raises(NotImplementedError) as exc:
        fn(_live=True, **kwargs)
    msg = str(exc.value).lower()
    assert "zinc" in msg
    assert "captcha" in msg or "disabled" in msg
    assert fake.calls == []


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ZINC_search_compounds, {"query": "HER2"}),
        (ZINC_get_compound, {"zinc_id": "ZINC0000001"}),
    ],
)
def test_zinc_mock_envelope_does_not_claim_zinc22(fn, kwargs):
    """Per hard rule: never mark ZINC as ZINC22 / live_ready in defaults."""
    out = fn(**kwargs)
    blob = str(out).lower()
    assert "zinc22" not in blob
    assert out.get("status") != "live_ready"


# ── Step 5 mechanical close-out: ChEMBL similarity + substructure ──────────

from app.mcp.tools.chembl import (  # noqa: E402
    ChEMBL_search_similarity,
    ChEMBL_search_substructure,
)


# ChEMBL_search_similarity

def test_chembl_similarity_mock_unchanged():
    out = ChEMBL_search_similarity(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["smiles"] == "CCO"
    assert out["threshold"] == 80
    assert out["molecules"] == []


def test_chembl_similarity_requires_smiles():
    with pytest.raises(ValueError):
        ChEMBL_search_similarity()


def test_chembl_similarity_live_routes(install_universe):
    fake = install_universe(
        tools={
            "ChEMBL_search_similarity": lambda args: {
                "molecules": [{"chembl_id": "CHEMBL25", "similarity": 0.92}]
            }
        }
    )
    out = ChEMBL_search_similarity(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        threshold=70,
        limit=25,
        offset=5,
        _live=True,
    )
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "threshold": 70,
        "limit": 25,
        "offset": 5,
    }


def test_chembl_similarity_live_clamps_threshold_and_limit(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_similarity": lambda args: {"molecules": []}}
    )
    ChEMBL_search_similarity(
        smiles="CCO", threshold=999, limit=99999, offset=-3, _live=True
    )
    args = fake.calls[0]["arguments"]
    assert args["threshold"] == 100  # clamp to 100
    assert args["limit"] == 1000     # clamp to 1000
    assert args["offset"] == 0       # clamp non-negative


def test_chembl_similarity_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_similarity": lambda args: {
                "status": "error",
                "error": "threshold out of range",
            }
        }
    )
    out = ChEMBL_search_similarity(smiles="CCO", _live=True)
    assert out["status"] == "upstream_error"


# ChEMBL_search_substructure

def test_chembl_substructure_mock_unchanged():
    out = ChEMBL_search_substructure(smiles="c1ccccc1")
    assert out["status"] == "mocked"
    assert out["smiles"] == "c1ccccc1"
    assert out["molecules"] == []


def test_chembl_substructure_requires_smiles():
    with pytest.raises(ValueError):
        ChEMBL_search_substructure()


def test_chembl_substructure_live_routes(install_universe):
    fake = install_universe(
        tools={
            "ChEMBL_search_substructure": lambda args: {
                "molecules": [{"chembl_id": "CHEMBL113"}]
            }
        }
    )
    out = ChEMBL_search_substructure(
        smiles="c1ccccc1", limit=10, offset=2, _live=True
    )
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "smiles": "c1ccccc1", "limit": 10, "offset": 2,
    }


def test_chembl_substructure_live_clamps_limit(install_universe):
    fake = install_universe(
        tools={"ChEMBL_search_substructure": lambda args: {"molecules": []}}
    )
    ChEMBL_search_substructure(smiles="c1ccccc1", limit=99999, _live=True)
    assert fake.calls[0]["arguments"]["limit"] == 1000


def test_chembl_substructure_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_substructure": lambda args: {
                "status": "error",
                "error": "invalid SMILES",
            }
        }
    )
    out = ChEMBL_search_substructure(smiles="bad-smiles", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 6 batch 1: PROSITE / DrugProps_pains / BindingDB / EBI epitopes/antigen

from app.mcp.tools.sequence_features import (  # noqa: E402
    GlyGen_get_glycoprotein,
    GlyGen_get_site,
    IEDB_predict_mhci_binding,
    PROSITE_scan_sequence,
    iPTMnet_get_ptm_sites,
)
from app.mcp.tools.developability_compounds import (  # noqa: E402
    BindingDB_get_targets_by_compound,
    DrugProps_pains_filter,
)
from app.mcp.tools.ebi_proteins import (  # noqa: E402
    EBIProteins_get_antigen,
    EBIProteins_get_epitopes,
)


# PROSITE_scan_sequence

def test_prosite_mock_unchanged():
    out = PROSITE_scan_sequence(sequence="MKTAYIAKQR")
    assert out["status"] == "mocked"
    assert out["sequence_length"] == 10
    assert out["matches"] == []


def test_prosite_requires_sequence():
    with pytest.raises(ValueError):
        PROSITE_scan_sequence()


def test_prosite_live_default_omits_skip_frequent(install_universe):
    fake = install_universe(
        tools={"PROSITE_scan_sequence": lambda args: {"matches": [{"signature": "PS00001"}]}}
    )
    out = PROSITE_scan_sequence(sequence="MKTAYIAKQR", _live=True)
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {"sequence": "MKTAYIAKQR"}
    assert "skip_frequent" not in fake.calls[0]["arguments"]


def test_prosite_live_forwards_skip_frequent_when_set(install_universe):
    fake = install_universe(tools={"PROSITE_scan_sequence": lambda args: {"matches": []}})
    PROSITE_scan_sequence(sequence="MKT", skip_frequent=False, _live=True)
    assert fake.calls[0]["arguments"]["skip_frequent"] is False


def test_prosite_live_upstream_error(install_universe):
    install_universe(
        tools={"PROSITE_scan_sequence": lambda args: {"status": "error", "error": "scan down"}}
    )
    out = PROSITE_scan_sequence(sequence="MKT", _live=True)
    assert out["status"] == "upstream_error"


# GlyGen_get_glycoprotein / GlyGen_get_site

def test_glygen_glycoprotein_mock_unchanged():
    out = GlyGen_get_glycoprotein(uniprot_ac="P14210")
    assert out["status"] == "mocked"
    assert out["uniprot_ac"] == "P14210"
    assert out["glycosylation_sites"] == []


def test_glygen_glycoprotein_requires_uniprot_ac():
    with pytest.raises(ValueError):
        GlyGen_get_glycoprotein()


def test_glygen_glycoprotein_live_routes(install_universe):
    fake = install_universe(
        tools={"GlyGen_get_glycoprotein": lambda args: {"protein": {"uniprot_ac": args["uniprot_ac"]}}}
    )
    out = GlyGen_get_glycoprotein(uniprot_ac="P14210", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"uniprot_ac": "P14210"}


def test_glygen_glycoprotein_live_upstream_error(install_universe):
    install_universe(
        tools={"GlyGen_get_glycoprotein": lambda args: {"status": "error", "error": "404"}}
    )
    out = GlyGen_get_glycoprotein(uniprot_ac="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_glygen_site_mock_unchanged():
    out = GlyGen_get_site(site_id="P02724-1.52.52")
    assert out["status"] == "mocked"
    assert out["site_id"] == "P02724-1.52.52"
    assert out["site"] is None


def test_glygen_site_requires_site_id():
    with pytest.raises(ValueError):
        GlyGen_get_site()


def test_glygen_site_live_routes(install_universe):
    fake = install_universe(
        tools={"GlyGen_get_site": lambda args: {"site": {"site_id": args["site_id"]}}}
    )
    out = GlyGen_get_site(site_id="P02724-1.52.52", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"site_id": "P02724-1.52.52"}


def test_glygen_site_live_upstream_error(install_universe):
    install_universe(
        tools={"GlyGen_get_site": lambda args: {"status": "error", "error": "down"}}
    )
    out = GlyGen_get_site(site_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


# iPTMnet_get_ptm_sites

def test_iptmnet_ptm_sites_mock_unchanged():
    out = iPTMnet_get_ptm_sites(uniprot_id="P04637", ptm_type="Phosphorylation")
    assert out["status"] == "mocked"
    assert out["uniprot_id"] == "P04637"
    assert out["ptm_type"] == "Phosphorylation"
    assert out["ptm_sites"] == []


def test_iptmnet_ptm_sites_requires_uniprot_id():
    with pytest.raises(ValueError):
        iPTMnet_get_ptm_sites()


def test_iptmnet_ptm_sites_live_maps_operation_and_optional_type(install_universe):
    fake = install_universe(
        tools={"iPTMnet_get_ptm_sites": lambda args: {"ptm_sites": [{"position": 15}]}}
    )
    out = iPTMnet_get_ptm_sites(
        uniprot_id="P04637", ptm_type="Phosphorylation", _live=True
    )
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "operation": "get_ptm_sites",
        "uniprot_id": "P04637",
        "ptm_type": "Phosphorylation",
    }


def test_iptmnet_ptm_sites_live_omits_empty_ptm_type(install_universe):
    fake = install_universe(
        tools={"iPTMnet_get_ptm_sites": lambda args: {"ptm_sites": []}}
    )
    iPTMnet_get_ptm_sites(uniprot_id="P00533", ptm_type="", _live=True)
    assert fake.calls[0]["arguments"] == {
        "operation": "get_ptm_sites",
        "uniprot_id": "P00533",
    }


def test_iptmnet_ptm_sites_live_upstream_error(install_universe):
    install_universe(
        tools={"iPTMnet_get_ptm_sites": lambda args: {"status": "error", "error": "404"}}
    )
    out = iPTMnet_get_ptm_sites(uniprot_id="BAD", _live=True)
    assert out["status"] == "upstream_error"


# IEDB_predict_mhci_binding

def test_iedb_mhci_mock_unchanged():
    out = IEDB_predict_mhci_binding(sequence="GILGFVFTL")
    assert out["status"] == "mocked"
    assert out["sequence_length"] == 9
    assert out["allele"] == "HLA-A*02:01"
    assert out["length"] == 9
    assert out["predictions"] == []


def test_iedb_mhci_requires_sequence():
    with pytest.raises(ValueError):
        IEDB_predict_mhci_binding()


def test_iedb_mhci_rejects_invalid_length():
    with pytest.raises(ValueError):
        IEDB_predict_mhci_binding(sequence="GILGFVFTL", length=7)


def test_iedb_mhci_live_routes_and_maps_defaults(install_universe):
    fake = install_universe(
        tools={"IEDB_predict_mhci_binding": lambda args: {"predictions": [{"peptide": args["sequence"]}]}}
    )
    out = IEDB_predict_mhci_binding(sequence="GILGFVFTL", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "sequence": "GILGFVFTL",
        "allele": "HLA-A*02:01",
        "method": "netmhcpan_el",
        "length": 9,
    }


def test_iedb_mhci_live_routes_custom_args(install_universe):
    fake = install_universe(
        tools={"IEDB_predict_mhci_binding": lambda args: {"predictions": []}}
    )
    IEDB_predict_mhci_binding(
        sequence="TYQRTRALV",
        allele="H-2-Kd",
        method="netmhcpan_ba",
        length=10,
        _live=True,
    )
    assert fake.calls[0]["arguments"] == {
        "sequence": "TYQRTRALV",
        "allele": "H-2-Kd",
        "method": "netmhcpan_ba",
        "length": 10,
    }


def test_iedb_mhci_live_upstream_error(install_universe):
    install_universe(
        tools={"IEDB_predict_mhci_binding": lambda args: {"status": "error", "error": "IEDB down"}}
    )
    out = IEDB_predict_mhci_binding(sequence="GILGFVFTL", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 6 structure/interface quality: PDBePISA / PDBe-KB ────────────────

def test_pdbe_pisa_interfaces_mock_unchanged():
    out = PDBePISA_get_interfaces(pdb_id="4HHB")
    assert out["status"] == "mocked"
    assert out["pdb_id"] == "4hhb"
    assert out["interfaces"] == []


def test_pdbe_pisa_interfaces_requires_pdb_id():
    with pytest.raises(ValueError):
        PDBePISA_get_interfaces()


def test_pdbe_pisa_interfaces_live_routes_and_normalizes_pdb_id(install_universe):
    fake = install_universe(
        tools={"PDBePISA_get_interfaces": lambda args: {"interfaces": [{"id": 1}]}}
    )
    out = PDBePISA_get_interfaces(pdb_id="4HHB", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"pdb_id": "4hhb"}


def test_pdbe_pisa_interfaces_live_upstream_error(install_universe):
    install_universe(
        tools={"PDBePISA_get_interfaces": lambda args: {"status": "error", "error": "404"}}
    )
    out = PDBePISA_get_interfaces(pdb_id="0000", _live=True)
    assert out["status"] == "upstream_error"


def test_pdbe_pisa_monomer_mock_unchanged():
    out = PDBePISA_get_monomer_analysis(pdb_id="1CBS")
    assert out["status"] == "mocked"
    assert out["pdb_id"] == "1cbs"
    assert out["monomers"] == []


def test_pdbe_pisa_monomer_requires_pdb_id():
    with pytest.raises(ValueError):
        PDBePISA_get_monomer_analysis()


def test_pdbe_pisa_monomer_live_routes_and_normalizes_pdb_id(install_universe):
    fake = install_universe(
        tools={"PDBePISA_get_monomer_analysis": lambda args: {"monomers": [{"chain": "A"}]}}
    )
    out = PDBePISA_get_monomer_analysis(pdb_id="1CBS", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"pdb_id": "1cbs"}


def test_pdbe_pisa_monomer_live_upstream_error(install_universe):
    install_universe(
        tools={"PDBePISA_get_monomer_analysis": lambda args: {"status": "error", "error": "down"}}
    )
    out = PDBePISA_get_monomer_analysis(pdb_id="0000", _live=True)
    assert out["status"] == "upstream_error"


def test_pdbe_kb_interface_residues_mock_unchanged():
    out = PDBe_KB_get_interface_residues(uniprot_accession="P04637")
    assert out["status"] == "mocked"
    assert out["uniprot_accession"] == "P04637"
    assert out["interface_residues"] == []


def test_pdbe_kb_interface_residues_requires_accession():
    with pytest.raises(ValueError):
        PDBe_KB_get_interface_residues()


def test_pdbe_kb_interface_residues_live_routes(install_universe):
    fake = install_universe(
        tools={"PDBe_KB_get_interface_residues": lambda args: {"residues": [{"position": 10}]}}
    )
    out = PDBe_KB_get_interface_residues(uniprot_accession="P04637", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"uniprot_accession": "P04637"}


def test_pdbe_kb_interface_residues_live_upstream_error(install_universe):
    install_universe(
        tools={"PDBe_KB_get_interface_residues": lambda args: {"status": "error", "error": "404"}}
    )
    out = PDBe_KB_get_interface_residues(uniprot_accession="BAD", _live=True)
    assert out["status"] == "upstream_error"


def test_proteinsplus_profile_mock_unchanged_and_live_deferred():
    out = ProteinsPlus_profile_structure_quality("1KZK")
    assert out["status"] == "mocked"
    assert out["input"] == "1KZK"
    with pytest.raises(NotImplementedError):
        ProteinsPlus_profile_structure_quality("1KZK", _live=True)


def test_proteinsplus_binding_site_tools_deferred():
    bindings = dict(PROTEINSPLUS_BINDINGS)
    with pytest.raises(NotImplementedError):
        bindings["ProteinsPlus_predict_binding_sites"]("1KZK")
    with pytest.raises(NotImplementedError):
        bindings["ProteinsPlus_predict_binding_sites_v3"]("1KZK")


# DrugProps_pains_filter

def test_drugprops_pains_mock_unchanged():
    out = DrugProps_pains_filter(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["smiles"] == "CCO"
    assert out["alerts"] == []
    assert out["passes"] is None


def test_drugprops_pains_requires_smiles():
    with pytest.raises(ValueError):
        DrugProps_pains_filter()


def test_drugprops_pains_live_routes(install_universe):
    fake = install_universe(
        tools={
            "DrugProps_pains_filter": lambda args: {"alerts": [], "passes": True}
        }
    )
    out = DrugProps_pains_filter(smiles="CC(=O)Oc1ccccc1C(=O)O", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}


def test_drugprops_pains_live_surfaces_rdkit_missing(install_universe):
    install_universe(
        tools={
            "DrugProps_pains_filter": lambda args: {
                "status": "error",
                "error": "RDKit is required for drug property calculations.",
            }
        }
    )
    out = DrugProps_pains_filter(smiles="CCO", _live=True)
    assert out["status"] == "upstream_error"
    assert "rdkit" in out["error_message"].lower()


# BindingDB_get_targets_by_compound

def test_bindingdb_mock_unchanged():
    out = BindingDB_get_targets_by_compound(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["smiles"] == "CCO"
    assert out["similarity_cutoff"] == 0.85
    assert out["targets"] == []


def test_bindingdb_requires_smiles():
    with pytest.raises(ValueError):
        BindingDB_get_targets_by_compound()


def test_bindingdb_live_routes_with_default_cutoff(install_universe):
    fake = install_universe(
        tools={
            "BindingDB_get_targets_by_compound": lambda args: {
                "targets": [{"target": "EGFR"}]
            }
        }
    )
    out = BindingDB_get_targets_by_compound(smiles="CCO", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"smiles": "CCO", "similarity_cutoff": 0.85}


def test_bindingdb_live_clamps_similarity_cutoff(install_universe):
    fake = install_universe(
        tools={"BindingDB_get_targets_by_compound": lambda args: {"targets": []}}
    )
    BindingDB_get_targets_by_compound(smiles="CCO", similarity_cutoff=1.5, _live=True)
    assert fake.calls[0]["arguments"]["similarity_cutoff"] == 1.0

    fake2 = install_universe(
        tools={"BindingDB_get_targets_by_compound": lambda args: {"targets": []}}
    )
    BindingDB_get_targets_by_compound(smiles="CCO", similarity_cutoff=-0.3, _live=True)
    assert fake2.calls[0]["arguments"]["similarity_cutoff"] == 0.0


def test_bindingdb_live_upstream_error(install_universe):
    install_universe(
        tools={
            "BindingDB_get_targets_by_compound": lambda args: {
                "status": "error",
                "error": "no similar compounds",
            }
        }
    )
    out = BindingDB_get_targets_by_compound(smiles="X", _live=True)
    assert out["status"] == "upstream_error"


# EBIProteins_get_epitopes

def test_ebi_epitopes_mock_unchanged():
    out = EBIProteins_get_epitopes(accession="P04637")
    assert out["status"] == "mocked"
    assert out["accession"] == "P04637"
    assert out["epitopes"] == []


def test_ebi_epitopes_requires_accession():
    with pytest.raises(ValueError):
        EBIProteins_get_epitopes(accession="")


def test_ebi_epitopes_live_routes(install_universe):
    fake = install_universe(
        tools={
            "EBIProteins_get_epitopes": lambda args: {
                "epitopes": [{"begin": "10", "end": "25"}]
            }
        }
    )
    out = EBIProteins_get_epitopes(accession="P04637", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"accession": "P04637"}


def test_ebi_epitopes_live_upstream_error(install_universe):
    install_universe(
        tools={
            "EBIProteins_get_epitopes": lambda args: {"status": "error", "error": "404"}
        }
    )
    out = EBIProteins_get_epitopes(accession="P00000", _live=True)
    assert out["status"] == "upstream_error"


# EBIProteins_get_antigen

def test_ebi_antigen_mock_unchanged():
    out = EBIProteins_get_antigen(accession="P00533")
    assert out["status"] == "mocked"
    assert out["antigens"] == []


def test_ebi_antigen_requires_accession():
    with pytest.raises(ValueError):
        EBIProteins_get_antigen(accession="")


def test_ebi_antigen_live_routes(install_universe):
    fake = install_universe(
        tools={
            "EBIProteins_get_antigen": lambda args: {
                "antigenic_regions": [{"begin": "100", "end": "120"}]
            }
        }
    )
    out = EBIProteins_get_antigen(accession="P00533", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"accession": "P00533"}


def test_ebi_antigen_live_upstream_error(install_universe):
    install_universe(
        tools={
            "EBIProteins_get_antigen": lambda args: {"status": "error", "error": "down"}
        }
    )
    out = EBIProteins_get_antigen(accession="P00533", _live=True)
    assert out["status"] == "upstream_error"


# ── Step 6 ADME batch: SwissADME (migrated) + ADMETAI (deferred) ───────────

from app.mcp.tools.developability_compounds import (  # noqa: E402
    BINDINGS as DEVELOPABILITY_BINDINGS,
    SwissADME_calculate_adme,
    SwissADME_check_druglikeness,
)


# SwissADME_calculate_adme

def test_swissadme_calculate_mock_unchanged():
    out = SwissADME_calculate_adme(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["smiles"] == "CCO"
    assert out["molecule_name"] is None
    assert out["adme"] is None


def test_swissadme_calculate_requires_smiles():
    with pytest.raises(ValueError):
        SwissADME_calculate_adme()


def test_swissadme_calculate_live_routes(install_universe):
    fake = install_universe(
        tools={
            "SwissADME_calculate_adme": lambda args: {
                "results": [{"smiles": args["smiles"], "MW": 46.07}]
            }
        }
    )
    out = SwissADME_calculate_adme(smiles="CCO", _live=True)
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {
        "operation": "calculate_adme",
        "smiles": "CCO",
    }


def test_swissadme_calculate_live_forwards_molecule_name(install_universe):
    fake = install_universe(
        tools={"SwissADME_calculate_adme": lambda args: {"results": []}}
    )
    SwissADME_calculate_adme(smiles="CCO", molecule_name="ethanol", _live=True)
    assert fake.calls[0]["arguments"] == {
        "operation": "calculate_adme",
        "smiles": "CCO",
        "molecule_name": "ethanol",
    }


def test_swissadme_calculate_live_upstream_error(install_universe):
    install_universe(
        tools={
            "SwissADME_calculate_adme": lambda args: {
                "status": "error",
                "error": "SwissADME service unavailable",
            }
        }
    )
    out = SwissADME_calculate_adme(smiles="CCO", _live=True)
    assert out["status"] == "upstream_error"


# SwissADME_check_druglikeness

def test_swissadme_druglikeness_mock_unchanged():
    out = SwissADME_check_druglikeness(smiles="CCO")
    assert out["status"] == "mocked"
    assert out["smiles"] == "CCO"
    assert out["rules"] is None
    assert out["results"] == {}


def test_swissadme_druglikeness_requires_smiles():
    with pytest.raises(ValueError):
        SwissADME_check_druglikeness()


def test_swissadme_druglikeness_rejects_invalid_rules():
    with pytest.raises(ValueError):
        SwissADME_check_druglikeness(smiles="CCO", rules=["lipinski", "bogus"])


def test_swissadme_druglikeness_live_default_omits_rules(install_universe):
    fake = install_universe(
        tools={"SwissADME_check_druglikeness": lambda args: {"results": {}}}
    )
    out = SwissADME_check_druglikeness(smiles="CCO", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "operation": "check_druglikeness",
        "smiles": "CCO",
    }
    assert "rules" not in fake.calls[0]["arguments"]


def test_swissadme_druglikeness_live_forwards_rules_lowercased(install_universe):
    fake = install_universe(
        tools={"SwissADME_check_druglikeness": lambda args: {"results": {}}}
    )
    SwissADME_check_druglikeness(
        smiles="CCO", rules=["Lipinski", "VEBER"], _live=True
    )
    assert fake.calls[0]["arguments"] == {
        "operation": "check_druglikeness",
        "smiles": "CCO",
        "rules": ["lipinski", "veber"],
    }


def test_swissadme_druglikeness_live_upstream_error(install_universe):
    install_universe(
        tools={
            "SwissADME_check_druglikeness": lambda args: {
                "status": "error",
                "error": "timeout",
            }
        }
    )
    out = SwissADME_check_druglikeness(smiles="CCO", _live=True)
    assert out["status"] == "upstream_error"


# ADMETAI_* — deferred, no adapter routing

_ADMETAI_TOOLS = (
    "ADMETAI_predict_toxicity",
    "ADMETAI_predict_physicochemical_properties",
    "ADMETAI_predict_solubility_lipophilicity_hydration",
    "ADMETAI_predict_CYP_interactions",
    "ADMETAI_predict_bioavailability",
    "ADMETAI_predict_clearance_distribution",
    "ADMETAI_predict_stress_response",
    "ADMETAI_predict_nuclear_receptor_activity",
)


def test_admetai_mock_mode_returns_deterministic_envelope():
    """Without ``_live=True`` every ADMETAI wrapper returns a compact
    mock envelope. No ToolUniverse / network / model load."""
    bindings = dict(DEVELOPABILITY_BINDINGS)
    for name in _ADMETAI_TOOLS:
        assert name in bindings, f"missing binding for {name}"
        out = bindings[name](smiles="CCO")
        assert out["status"] == "mocked"
        assert out["source"] == name
        assert out["smiles"] == "CCO"
        assert out["predictions"] is None


def test_admetai_requires_non_empty_smiles():
    bindings = dict(DEVELOPABILITY_BINDINGS)
    for name in _ADMETAI_TOOLS:
        with pytest.raises(ValueError):
            bindings[name](smiles="")
        with pytest.raises(ValueError):
            bindings[name](smiles="", _live=True)


def test_admetai_live_routes_through_tooluniverse_adapter(install_universe):
    """_live=True dispatches each ADMETAI tool through TU with
    ``{"smiles": <value>}`` — exactly the official spec's required
    parameter — and returns the adapter envelope."""
    canned = {
        name: lambda args, _n=name: {"smiles_in": args.get("smiles"), "tool": _n}
        for name in _ADMETAI_TOOLS
    }
    fake = install_universe(tools=canned)
    bindings = dict(DEVELOPABILITY_BINDINGS)
    for name in _ADMETAI_TOOLS:
        env = bindings[name](smiles="CCO", _live=True)
        assert env["status"] == "ok"
        assert env["executor"] == "tooluniverse"
        assert env["source"] == name
        assert env["payload"] == {"smiles_in": "CCO", "tool": name}
    assert {c["name"] for c in fake.calls} == set(_ADMETAI_TOOLS)
    for c in fake.calls:
        assert c["arguments"] == {"smiles": "CCO"}


def test_admetai_live_upstream_error_envelope_when_universe_returns_error(
    install_universe,
):
    """If ToolUniverse signals an error (e.g. ``admet_ai`` package
    missing in the runtime venv), the wrapper does NOT raise — it
    surfaces ``status="upstream_error"`` with TU's error message."""
    install_universe(tools={
        "ADMETAI_predict_toxicity": lambda args: {
            "status": "error",
            "error": "ADMETModel requires 'admet-ai' package",
        }
    })
    bindings = dict(DEVELOPABILITY_BINDINGS)
    env = bindings["ADMETAI_predict_toxicity"](smiles="CCO", _live=True)
    assert env["status"] == "upstream_error"
    assert env["executor"] == "tooluniverse"
    assert "admet-ai" in (env.get("error_message") or "")


# ── Step 9 variant batch: AlphaMissense migrated; DynaMut2 / ESM deferred ──

from app.mcp.tools.variant import (  # noqa: E402
    BINDINGS as VARIANT_BINDINGS,
    AlphaMissense_get_variant_score,
)


def test_alphamissense_mock_unchanged():
    out = AlphaMissense_get_variant_score(uniprot_id="P00533", variant="V600E")
    assert out["status"] == "mocked"
    assert out["uniprot_id"] == "P00533"
    assert out["variant"] == "V600E"
    assert out["score"] is None
    assert out["classification"] is None


def test_alphamissense_requires_uniprot_id():
    with pytest.raises(ValueError):
        AlphaMissense_get_variant_score(variant="V600E")


def test_alphamissense_requires_variant():
    with pytest.raises(ValueError):
        AlphaMissense_get_variant_score(uniprot_id="P00533")


def test_alphamissense_live_routes(install_universe):
    fake = install_universe(
        tools={
            "AlphaMissense_get_variant_score": lambda args: {
                "uniprot_id": args["uniprot_id"],
                "variant": args["variant"],
                "score": 0.81,
                "classification": "likely_pathogenic",
            }
        }
    )
    out = AlphaMissense_get_variant_score(
        uniprot_id="P00533", variant="L858R", _live=True
    )
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {
        "uniprot_id": "P00533",
        "variant": "L858R",
    }


def test_alphamissense_live_upstream_error(install_universe):
    install_universe(
        tools={
            "AlphaMissense_get_variant_score": lambda args: {
                "status": "error",
                "error": "variant out of range",
            }
        }
    )
    out = AlphaMissense_get_variant_score(
        uniprot_id="P00533", variant="Z9999A", _live=True
    )
    assert out["status"] == "upstream_error"


_VARIANT_DEFERRED = (
    "DynaMut2_predict_stability",
    "ESM_generate_protein_sequence",
    "ESM_score_variant_sae_batch",
)


def test_variant_deferred_tools_raise_not_implemented():
    bindings = dict(VARIANT_BINDINGS)
    for name in _VARIANT_DEFERRED:
        assert name in bindings, f"missing binding for deferred {name}"
        with pytest.raises(NotImplementedError):
            bindings[name]()
        with pytest.raises(NotImplementedError):
            bindings[name](_live=True)


def test_variant_deferred_does_not_touch_universe(install_universe):
    fake = install_universe(
        tools={name: lambda args: {"ok": True} for name in _VARIANT_DEFERRED}
    )
    bindings = dict(VARIANT_BINDINGS)
    for name in _VARIANT_DEFERRED:
        with pytest.raises(NotImplementedError):
            bindings[name](_live=True)
    assert fake.calls == []
