"""EBI Proteins API wrappers (Step 6).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` for the wired subset routes through
`ToolUniverseAdapter`. No manual httpx implementation here.

Audit doc: `项目文件/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def EBIProteins_get_features(
    accession: str = "", *, types: str = "", _live: bool = False
) -> dict[str, Any]:
    if not accession or not isinstance(accession, str):
        raise ValueError("EBIProteins_get_features requires a non-empty accession")
    if not _live:
        return {
            "status": "mocked",
            "source": "EBIProteins_get_features",
            "accession": accession,
            "features": [],
        }
    args: dict[str, Any] = {"accession": accession}
    if types:
        args["types"] = types
    return _tu("EBIProteins_get_features", args)


def EBIProteins_get_epitopes(accession: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Get experimentally-determined immune epitope regions for a protein.

    TU required: `accession` (UniProt). Mock returns empty envelope.
    """
    if not accession or not isinstance(accession, str):
        raise ValueError("EBIProteins_get_epitopes requires a non-empty accession")
    if not _live:
        return {
            "status": "mocked",
            "source": "EBIProteins_get_epitopes",
            "accession": accession,
            "epitopes": [],
        }
    return _tu("EBIProteins_get_epitopes", {"accession": accession})


def EBIProteins_get_antigen(accession: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Get predicted antigenic regions for a protein.

    TU required: `accession` (UniProt). Mock returns empty envelope.
    """
    if not accession or not isinstance(accession, str):
        raise ValueError("EBIProteins_get_antigen requires a non-empty accession")
    if not _live:
        return {
            "status": "mocked",
            "source": "EBIProteins_get_antigen",
            "accession": accession,
            "antigens": [],
        }
    return _tu("EBIProteins_get_antigen", {"accession": accession})


BINDINGS = [
    ("EBIProteins_get_epitopes", EBIProteins_get_epitopes),
    ("EBIProteins_get_antigen", EBIProteins_get_antigen),
    ("EBIProteins_get_features", EBIProteins_get_features),
]
