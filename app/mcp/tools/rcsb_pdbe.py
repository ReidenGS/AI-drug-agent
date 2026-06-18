"""RCSB / PDBe structure search & data wrappers (Steps 5, 7, 8).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` routes through `ToolUniverseAdapter` for the
wired subset (`RCSBData_get_entry`, `get_refinement_resolution_by_pdb_id`,
`CrystalStructure_validate`). The remaining wrappers stay `_ni` until
promoted in a future audit round.

`CrystalStructure_validate` previously held a hand-written PDB header
parser; the official ToolUniverse implementation is a cell-parameter
validator. Per the integration audit's "no two execution sources" rule,
the manual parser was removed and `_live=True` is routed to TU. The
wrapper signature now matches TU's expected input.

Audit doc: `\u9879\u76ee\u6587\u4ef6/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def RCSBData_get_entry(pdb_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not pdb_id or not isinstance(pdb_id, str):
        raise ValueError("RCSBData_get_entry requires a non-empty PDB id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {"pdb_id": pdb_id, "status": "mocked"}
    return _tu("RCSBData_get_entry", {"pdb_id": pdb_id})


def get_refinement_resolution_by_pdb_id(pdb_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not pdb_id:
        raise ValueError("get_refinement_resolution_by_pdb_id requires a non-empty PDB id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {"pdb_id": pdb_id, "status": "mocked", "resolution_angstrom": 2.0}
    return _tu("get_refinement_resolution_by_pdb_id", {"pdb_id": pdb_id})


def CrystalStructure_validate(
    pdb_id_or_path: str = "",
    *,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    operation: str | None = None,
    Z: int | None = None,
    mw: float | None = None,
    reported_density: float | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Crystal structure validation wrapper.

    NOTE: ToolUniverse's `CrystalStructure_validate` is a unit-cell
    parameter validator (a, b, c, alpha, beta, gamma) — the historic Step
    8 plan in this codebase passes `pdb_id_or_path` instead, which is a
    semantic mismatch. We keep the legacy `pdb_id_or_path` argument so
    Step 8 mock-mode dispatch still works, and forward whatever the
    caller actually provided to TU; when only `pdb_id_or_path` is given,
    the TU call will return an `upstream_error` honestly rather than
    masking the mismatch.
    """
    if not _live:
        return {
            "input": pdb_id_or_path
            or {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
            "status": "mocked",
            "validation_pass": True,
            "issues": [],
        }
    args: dict[str, Any] = {}
    if pdb_id_or_path:
        args["pdb_id_or_path"] = pdb_id_or_path
    for name, value in (
        ("a", a), ("b", b), ("c", c), ("alpha", alpha), ("beta", beta), ("gamma", gamma),
        ("Z", Z), ("mw", mw), ("reported_density", reported_density),
    ):
        if value is not None:
            args[name] = value
    if operation:
        args["operation"] = operation
    if not args:
        raise ValueError(
            "CrystalStructure_validate live mode requires either cell parameters "
            "(a/b/c/alpha/beta/gamma + Z/mw) or a `pdb_id_or_path`."
        )
    return _tu("CrystalStructure_validate", args)


def RCSBData_get_assembly(
    pdb_id: str = "", *, assembly_id: str = "1", _live: bool = False
) -> dict[str, Any]:
    """Fetch a PDB assembly record from the RCSB Data API."""
    if not pdb_id:
        raise ValueError("RCSBData_get_assembly requires a non-empty pdb_id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {
            "status": "mocked",
            "source": "RCSBData_get_assembly",
            "pdb_id": pdb_id,
            "assembly_id": assembly_id,
            "assembly": None,
        }
    return _tu(
        "RCSBData_get_assembly",
        {"pdb_id": pdb_id, "assembly_id": assembly_id},
    )


def RCSBAdvSearch_search_structures(
    query: str = "",
    *,
    organism: str = "",
    experimental_method: str = "",
    polymer_description: str = "",
    max_resolution: float | None = None,
    min_deposition_date: str = "",
    rows: int | None = None,
    sort_by: str = "",
    _live: bool = False,
) -> dict[str, Any]:
    """RCSB advanced structure search.

    All TU parameters are optional; the wrapper forwards only the values
    the caller actually provided (empty strings / None are dropped) so TU
    builds the right Solr query without spurious filters.
    """
    if not _live:
        return {
            "status": "mocked",
            "source": "RCSBAdvSearch_search_structures",
            "query": query,
            "structures": [],
        }
    args: dict[str, Any] = {}
    if query:
        args["query"] = query
    if organism:
        args["organism"] = organism
    if experimental_method:
        args["experimental_method"] = experimental_method
    if polymer_description:
        args["polymer_description"] = polymer_description
    if max_resolution is not None:
        args["max_resolution"] = max_resolution
    if min_deposition_date:
        args["min_deposition_date"] = min_deposition_date
    if rows is not None:
        args["rows"] = max(1, min(int(rows), 50))
    if sort_by:
        args["sort_by"] = sort_by
    return _tu("RCSBAdvSearch_search_structures", args)


def PDBeSearch_search_structures(
    query: str = "", *, limit: int = 10, _live: bool = False
) -> dict[str, Any]:
    """Search PDBe structures via the PDBe Search API (Solr-backed)."""
    if not query:
        raise ValueError("PDBeSearch_search_structures requires a non-empty query")
    if not _live:
        return {
            "status": "mocked",
            "source": "PDBeSearch_search_structures",
            "query": query,
            "structures": [],
        }
    return _tu(
        "PDBeSearch_search_structures",
        {"query": query, "limit": max(1, min(int(limit), 50))},
    )


BINDINGS = [
    ("RCSBData_get_entry", RCSBData_get_entry),
    ("RCSBData_get_assembly", RCSBData_get_assembly),
    ("RCSBAdvSearch_search_structures", RCSBAdvSearch_search_structures),
    ("PDBeSearch_search_structures", PDBeSearch_search_structures),
    ("get_refinement_resolution_by_pdb_id", get_refinement_resolution_by_pdb_id),
    ("CrystalStructure_validate", CrystalStructure_validate),
]
