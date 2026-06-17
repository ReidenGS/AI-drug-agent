"""PROSITE / GlyGen / iPTMnet / IEDB MHC-I wrappers (Step 6).

Thin MCP binding layer. `_live=False` (default) returns deterministic
mock envelopes. `_live=True` for the wired subset routes through
`ToolUniverseAdapter`. Other wrappers stay `_ni` until promoted.

Audit doc: `项目文件/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def PROSITE_scan_sequence(
    sequence: str = "", *, skip_frequent: bool | None = None, _live: bool = False
) -> dict[str, Any]:
    """Scan a protein sequence against PROSITE patterns and profiles.

    TU required: `sequence` (single-letter amino acid code).
    Optional: `skip_frequent` — when None (wrapper default), the flag is
    omitted so TU's own default (skip frequent low-info patterns) applies.
    """
    if not sequence or not isinstance(sequence, str):
        raise ValueError("PROSITE_scan_sequence requires a non-empty sequence")
    if not _live:
        return {
            "status": "mocked",
            "source": "PROSITE_scan_sequence",
            "sequence_length": len(sequence),
            "matches": [],
        }
    args: dict[str, Any] = {"sequence": sequence}
    if skip_frequent is not None:
        args["skip_frequent"] = bool(skip_frequent)
    return _tu("PROSITE_scan_sequence", args)


def GlyGen_get_glycoprotein(uniprot_ac: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Get GlyGen glycoprotein details by UniProt accession."""
    if not uniprot_ac or not isinstance(uniprot_ac, str):
        raise ValueError("GlyGen_get_glycoprotein requires a non-empty uniprot_ac")
    if not _live:
        return {
            "status": "mocked",
            "source": "GlyGen_get_glycoprotein",
            "uniprot_ac": uniprot_ac,
            "glycoprotein": None,
            "glycosylation_sites": [],
        }
    return _tu("GlyGen_get_glycoprotein", {"uniprot_ac": uniprot_ac})


def GlyGen_get_site(site_id: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Get GlyGen details for one glycosylation site identifier."""
    if not site_id or not isinstance(site_id, str):
        raise ValueError("GlyGen_get_site requires a non-empty site_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "GlyGen_get_site",
            "site_id": site_id,
            "site": None,
        }
    return _tu("GlyGen_get_site", {"site_id": site_id})


def iPTMnet_get_ptm_sites(
    uniprot_id: str = "",
    *,
    ptm_type: str | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """Get PTM sites from iPTMnet by UniProt accession."""
    if not uniprot_id or not isinstance(uniprot_id, str):
        raise ValueError("iPTMnet_get_ptm_sites requires a non-empty uniprot_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "iPTMnet_get_ptm_sites",
            "uniprot_id": uniprot_id,
            "ptm_type": ptm_type,
            "ptm_sites": [],
        }
    args: dict[str, Any] = {"operation": "get_ptm_sites", "uniprot_id": uniprot_id}
    if ptm_type:
        args["ptm_type"] = ptm_type
    return _tu("iPTMnet_get_ptm_sites", args)


def IEDB_predict_mhci_binding(
    sequence: str = "",
    *,
    allele: str = "HLA-A*02:01",
    method: str = "netmhcpan_el",
    length: int = 9,
    _live: bool = False,
) -> dict[str, Any]:
    """Predict MHC-I peptide binding through the IEDB prediction tool."""
    if not sequence or not isinstance(sequence, str):
        raise ValueError("IEDB_predict_mhci_binding requires a non-empty sequence")
    peptide_length = int(length)
    if peptide_length < 8 or peptide_length > 14:
        raise ValueError("IEDB_predict_mhci_binding length must be between 8 and 14")
    if not allele or not isinstance(allele, str):
        raise ValueError("IEDB_predict_mhci_binding requires a non-empty allele")
    if not method or not isinstance(method, str):
        raise ValueError("IEDB_predict_mhci_binding requires a non-empty method")
    if not _live:
        return {
            "status": "mocked",
            "source": "IEDB_predict_mhci_binding",
            "sequence_length": len(sequence),
            "allele": allele,
            "method": method,
            "length": peptide_length,
            "predictions": [],
        }
    return _tu(
        "IEDB_predict_mhci_binding",
        {
            "sequence": sequence,
            "allele": allele,
            "method": method,
            "length": peptide_length,
        },
    )


BINDINGS = [
    ("PROSITE_scan_sequence", PROSITE_scan_sequence),
    ("GlyGen_get_glycoprotein", GlyGen_get_glycoprotein),
    ("GlyGen_get_site", GlyGen_get_site),
    ("iPTMnet_get_ptm_sites", iPTMnet_get_ptm_sites),
    ("IEDB_predict_mhci_binding", IEDB_predict_mhci_binding),
]
