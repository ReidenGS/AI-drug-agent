"""ProteinsPlus wrappers (Step 6/8).

`ProteinsPlus_profile_structure_quality` is mockable so the Step 8 graph runs
without network. The other wrappers stay NotImplementedError until wired.
"""

from __future__ import annotations

from typing import Any


def ProteinsPlus_profile_structure_quality(
    pdb_id_or_path: str = "",
    *,
    pdb_id: str = "",
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    pdb_id_or_path = pdb_id_or_path or pdb_id
    if not pdb_id_or_path:
        raise ValueError("ProteinsPlus_profile_structure_quality requires input")
    if not _live:
        return {
            "input": pdb_id_or_path,
            "status": "mocked",
            "quality_score": 0.75,
            "clashes": 0,
            "ramachandran_outliers": 0.0,
        }
    raise NotImplementedError("ProteinsPlus live mode not wired")


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("ProteinsPlus_profile_structure_quality", ProteinsPlus_profile_structure_quality),
    ("ProteinsPlus_predict_binding_sites", _ni),
    ("ProteinsPlus_predict_binding_sites_v3", _ni),
]
