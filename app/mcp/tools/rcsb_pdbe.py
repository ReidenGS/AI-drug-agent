"""RCSB / PDBe structure search & data wrappers (Steps 5, 7, 8).

Only `RCSBData_get_entry` has a real upstream call path (public REST,
unauthenticated). Everything else stays NotImplementedError pending the next
iteration.

Upstream: https://data.rcsb.org/rest/v1/core/entry/{pdb_id}
"""

from __future__ import annotations

from typing import Any


def RCSBData_get_entry(pdb_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not pdb_id or not isinstance(pdb_id, str):
        raise ValueError("RCSBData_get_entry requires a non-empty PDB id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {"pdb_id": pdb_id, "status": "mocked"}
    import httpx

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return {"pdb_id": pdb_id, "status": "live", "entry": resp.json()}


def get_refinement_resolution_by_pdb_id(pdb_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not pdb_id:
        raise ValueError("get_refinement_resolution_by_pdb_id requires a non-empty PDB id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {"pdb_id": pdb_id, "status": "mocked", "resolution_angstrom": 2.0}
    import httpx

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        entry = resp.json()
        return {
            "pdb_id": pdb_id,
            "status": "live",
            "resolution_angstrom": (entry.get("refine") or [{}])[0].get("ls_d_res_high"),
        }


def CrystalStructure_validate(pdb_id_or_path: str, *, _live: bool = False) -> dict[str, Any]:
    if not pdb_id_or_path:
        raise ValueError("CrystalStructure_validate requires input")
    if not _live:
        return {
            "input": pdb_id_or_path,
            "status": "mocked",
            "validation_pass": True,
            "issues": [],
        }
    raise NotImplementedError("CrystalStructure_validate live mode not wired")


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("RCSBData_get_entry", RCSBData_get_entry),
    ("RCSBData_get_assembly", _ni),
    ("RCSBAdvSearch_search_structures", _ni),
    ("PDBeSearch_search_structures", _ni),
    ("get_refinement_resolution_by_pdb_id", get_refinement_resolution_by_pdb_id),
    ("CrystalStructure_validate", CrystalStructure_validate),
]
