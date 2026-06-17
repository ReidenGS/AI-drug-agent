"""PDBe PISA / KB interface wrappers (Steps 6, 8)."""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def _pdb_id(value: str, tool_name: str) -> str:
    if not value or not isinstance(value, str):
        raise ValueError(f"{tool_name} requires a non-empty pdb_id")
    return value.strip().lower()


def PDBePISA_get_interfaces(pdb_id: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Analyze interfaces in a PDB structure using PDBePISA."""
    pdb = _pdb_id(pdb_id, "PDBePISA_get_interfaces")
    if not _live:
        return {
            "status": "mocked",
            "source": "PDBePISA_get_interfaces",
            "pdb_id": pdb,
            "interfaces": [],
        }
    return _tu("PDBePISA_get_interfaces", {"pdb_id": pdb})


def PDBePISA_get_monomer_analysis(pdb_id: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Analyze per-chain solvent/interface participation using PDBePISA."""
    pdb = _pdb_id(pdb_id, "PDBePISA_get_monomer_analysis")
    if not _live:
        return {
            "status": "mocked",
            "source": "PDBePISA_get_monomer_analysis",
            "pdb_id": pdb,
            "monomers": [],
        }
    return _tu("PDBePISA_get_monomer_analysis", {"pdb_id": pdb})


def PDBe_KB_get_interface_residues(
    uniprot_accession: str = "", *, _live: bool = False
) -> dict[str, Any]:
    """Get interface residues for a UniProt accession from PDBe-KB."""
    if not uniprot_accession or not isinstance(uniprot_accession, str):
        raise ValueError("PDBe_KB_get_interface_residues requires a non-empty uniprot_accession")
    if not _live:
        return {
            "status": "mocked",
            "source": "PDBe_KB_get_interface_residues",
            "uniprot_accession": uniprot_accession,
            "interface_residues": [],
        }
    return _tu(
        "PDBe_KB_get_interface_residues",
        {"uniprot_accession": uniprot_accession},
    )


BINDINGS = [
    ("PDBePISA_get_interfaces", PDBePISA_get_interfaces),
    ("PDBePISA_get_monomer_analysis", PDBePISA_get_monomer_analysis),
    ("PDBe_KB_get_interface_residues", PDBe_KB_get_interface_residues),
]
