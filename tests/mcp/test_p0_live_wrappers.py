"""P0 wrappers (post-audit): mock behavior + TU-routed live behavior.

All P0 wrappers (`ChEMBL_search_activities`, `EBIProteins_get_features`,
`get_refinement_resolution_by_pdb_id`, `CrystalStructure_validate`) lost
their hand-written httpx / local-parser implementations after the
ToolUniverse runtime integration audit. `_live=True` now goes directly
through the adapter. These tests use the shared `install_universe`
fixture from `tests/mcp/conftest.py`.
"""

from __future__ import annotations

import pytest

from app.mcp.tools.chembl import ChEMBL_search_activities
from app.mcp.tools.ebi_proteins import EBIProteins_get_features
from app.mcp.tools.rcsb_pdbe import (
    CrystalStructure_validate,
    get_refinement_resolution_by_pdb_id,
)


# ── ChEMBL_search_activities ───────────────────────────────────────────────

def test_chembl_activities_mock_unchanged():
    out = ChEMBL_search_activities(target_chembl_id="CHEMBL1824")
    assert out["status"] == "mocked"
    assert out["target_chembl_id"] == "CHEMBL1824"
    assert out["activities"] == []


def test_chembl_activities_requires_one_arg():
    with pytest.raises(ValueError):
        ChEMBL_search_activities()


def test_chembl_activities_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "ChEMBL_search_activities": lambda args: {
                "activities": [
                    {"standard_type": "IC50", "standard_value": "10.0", "standard_units": "nM"}
                ]
            }
        }
    )
    out = ChEMBL_search_activities(target_chembl_id="CHEMBL1824", limit=5, _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"target_chembl_id": "CHEMBL1824", "limit": 5}


def test_chembl_activities_live_empty(install_universe):
    install_universe(tools={"ChEMBL_search_activities": lambda args: {"activities": []}})
    out = ChEMBL_search_activities(molecule_chembl_id="CHEMBL999", _live=True)
    assert out["status"] == "empty"


def test_chembl_activities_live_upstream_error(install_universe):
    install_universe(
        tools={
            "ChEMBL_search_activities": lambda args: {
                "status": "error",
                "error": "timeout",
            }
        }
    )
    out = ChEMBL_search_activities(target_chembl_id="CHEMBL1824", _live=True)
    assert out["status"] == "upstream_error"


# ── EBIProteins_get_features ───────────────────────────────────────────────

def test_ebi_proteins_features_mock_unchanged():
    out = EBIProteins_get_features("P00533")
    assert out["status"] == "mocked"


def test_ebi_proteins_features_requires_accession():
    with pytest.raises(ValueError):
        EBIProteins_get_features("")


def test_ebi_proteins_features_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "EBIProteins_get_features": lambda args: {
                "features": [
                    {"type": "DOMAIN", "begin": "712", "end": "979"},
                    {"type": "BINDING", "begin": "745", "end": "745"},
                ]
            }
        }
    )
    out = EBIProteins_get_features("P00533", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"accession": "P00533"}


def test_ebi_proteins_features_live_passes_types_filter(install_universe):
    fake = install_universe(tools={"EBIProteins_get_features": lambda args: {"features": []}})
    EBIProteins_get_features("P00533", types="DOMAIN,BINDING", _live=True)
    assert fake.calls[0]["arguments"]["types"] == "DOMAIN,BINDING"


def test_ebi_proteins_features_live_upstream_error(install_universe):
    install_universe(
        tools={"EBIProteins_get_features": lambda args: {"status": "error", "error": "slow"}}
    )
    out = EBIProteins_get_features("P00533", _live=True)
    assert out["status"] == "upstream_error"


# ── get_refinement_resolution_by_pdb_id ────────────────────────────────────

def test_resolution_lookup_mock_unchanged():
    out = get_refinement_resolution_by_pdb_id("1ABC")
    assert out["status"] == "mocked"
    assert out["resolution_angstrom"] == 2.0


def test_resolution_lookup_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "get_refinement_resolution_by_pdb_id": lambda args: {
                "resolution_angstrom": 1.95,
                "pdb_id": args["pdb_id"],
            }
        }
    )
    out = get_refinement_resolution_by_pdb_id("1ABC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"pdb_id": "1abc"}


# ── CrystalStructure_validate (TU semantics: cell parameters) ──────────────

def test_crystal_structure_mock_unchanged():
    out = CrystalStructure_validate()
    assert out["status"] == "mocked"
    assert out["validation_pass"] is True


def test_crystal_structure_live_requires_some_arg(install_universe):
    """Live mode needs either pdb_id_or_path (legacy compat) or cell params."""
    install_universe(tools={"CrystalStructure_validate": lambda args: {"validation_pass": True}})
    with pytest.raises(ValueError):
        CrystalStructure_validate(_live=True)


def test_crystal_structure_live_forwards_legacy_pdb_arg(install_universe):
    """Step 8 still passes `pdb_id_or_path`; we forward it honestly so TU
    can decide. This is a semantic mismatch with TU's cell-param schema —
    expected to surface as `upstream_error` in real life — but the wrapper
    must not silently drop the input."""
    fake = install_universe(
        tools={"CrystalStructure_validate": lambda args: {
            "status": "error",
            "error": "pdb_id_or_path is not a valid cell parameter",
        }}
    )
    out = CrystalStructure_validate(pdb_id_or_path="1N8Z", _live=True)
    assert fake.calls[0]["arguments"] == {"pdb_id_or_path": "1N8Z"}
    assert out["status"] == "upstream_error"


def test_crystal_structure_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "CrystalStructure_validate": lambda args: {
                "validation_pass": True,
                "issues": [],
                "input_cell": args,
            }
        }
    )
    out = CrystalStructure_validate(a=78.0, b=78.0, c=37.0, alpha=90, beta=90, gamma=120, _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {
        "a": 78.0, "b": 78.0, "c": 37.0, "alpha": 90, "beta": 90, "gamma": 120,
    }


def test_crystal_structure_live_upstream_error(install_universe):
    install_universe(
        tools={
            "CrystalStructure_validate": lambda args: {
                "status": "error",
                "error": "non-positive cell length",
            }
        }
    )
    out = CrystalStructure_validate(a=-1.0, _live=True)
    assert out["status"] == "upstream_error"
