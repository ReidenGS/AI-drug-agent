"""ChEMBL wrappers (Steps 5, 6, 13).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` for the wired subset routes through
`ToolUniverseAdapter`. Wrappers still on `_ni` raise `NotImplementedError`
on `_live=True` and the MCPClient surfaces it as `dependency_unavailable`.

Argument mappings (legacy wrapper → TU schema):
    ChEMBL_search_molecules(query, limit) → {query, limit}
    ChEMBL_get_molecule(chembl_id)        → {chembl_id}
    ChEMBL_search_drugs(query, limit)     → {query, limit}
    ChEMBL_get_drug(drug_chembl_id)       → {drug_chembl_id}
    ChEMBL_search_activities(...)         → {target_chembl_id?, molecule_chembl_id?, limit}
    ChEMBL_search_documents(...)          → {document_id?, title__contains?, limit, offset}
    ChEMBL_search_similarity(smiles, threshold, limit, offset) → {smiles, threshold, limit, offset}
    ChEMBL_search_substructure(smiles, limit, offset)          → {smiles, limit, offset}
    ChEMBL_search_compound_structural_alerts(...) → {molecule_chembl_id?, alert_name__contains?, limit, offset}
    ChEMBL_get_molecule_targets(molecule_chembl_id, limit) → {molecule_chembl_id__exact, limit}
    ChEMBL_search_targets(...)             → {target_chembl_id?, pref_name__contains?, organism?, target_type?, fields?, limit, offset}
    ChEMBL_get_target_activities(target_chembl_id, limit, offset) → {target_chembl_id__exact, limit, offset}
    ChEMBL_get_drug_mechanisms(...)        → {drug_chembl_id?/drug_name?, limit, offset}
    ChEMBL_search_assays(...)              → {assay_chembl_id?, assay_type?, target_chembl_id?, fields?, limit, offset}
    ChEMBL_get_target_assays(target_chembl_id, limit, offset) → {target_chembl_id__exact, limit, offset}
    ChEMBL_get_assay_activities(assay_chembl_id, limit, offset) → {assay_chembl_id__exact, limit, offset}
    ChEMBL_search_binding_sites(...)       → {target_chembl_id?, site_name__contains?, limit, offset}

Audit doc: `\u9879\u76ee\u6587\u4ef6/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def _limit(value: int, *, maximum: int = 1000) -> int:
    return max(1, min(int(value), maximum))


def _offset(value: int) -> int:
    return max(0, int(value))


def _fields(value: list[str] | tuple[str, ...] | None, *, allowed: set[str]) -> list[str] | None:
    if value is None:
        return None
    fields = list(value)
    invalid = [f for f in fields if f not in allowed]
    if invalid:
        raise ValueError(f"Unsupported ChEMBL fields: {', '.join(invalid)}")
    return fields


def ChEMBL_search_molecules(
    query: str = "", *, limit: int = 20, _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL molecules by free-text name/alias.

    Mock mode returns an empty envelope; live mode forwards `query` +
    `limit` to TU's `ChEMBL_search_molecules` (TU treats `query` as an
    alias for `pref_name__contains`).
    """
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_molecules",
            "query": query,
            "molecules": [],
        }
    args: dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
    if query:
        args["query"] = query
    return _tu("ChEMBL_search_molecules", args)


def ChEMBL_get_molecule(chembl_id: str = "", *, _live: bool = False, **_extra: Any) -> dict[str, Any]:
    """Fetch a single ChEMBL molecule by ID (e.g. CHEMBL25).

    Mock mode returns an empty envelope; live mode forwards `chembl_id`
    to TU. The TU schema lists `chembl_id` and its alias
    `molecule_chembl_id`; the wrapper exposes the canonical `chembl_id`.
    """
    if not chembl_id:
        raise ValueError("ChEMBL_get_molecule requires a non-empty chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_molecule",
            "chembl_id": chembl_id,
            "molecule": None,
        }
    return _tu("ChEMBL_get_molecule", {"chembl_id": chembl_id})


def ChEMBL_search_drugs(
    query: str = "", *, limit: int = 20, _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL drug records by free-text name."""
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_drugs",
            "query": query,
            "drugs": [],
        }
    args: dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
    if query:
        args["query"] = query
    return _tu("ChEMBL_search_drugs", args)


def ChEMBL_get_drug(drug_chembl_id: str = "", *, _live: bool = False, **_extra: Any) -> dict[str, Any]:
    """Fetch a single ChEMBL drug record by ID (e.g. CHEMBL1201581).

    TU requires `drug_chembl_id`; the wrapper enforces it before routing.
    """
    if not drug_chembl_id:
        raise ValueError("ChEMBL_get_drug requires a non-empty drug_chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_drug",
            "drug_chembl_id": drug_chembl_id,
            "drug": None,
        }
    return _tu("ChEMBL_get_drug", {"drug_chembl_id": drug_chembl_id})


def ChEMBL_search_activities(
    target_chembl_id: str = "",
    molecule_chembl_id: str = "",
    *,
    limit: int = 25,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    if not target_chembl_id and not molecule_chembl_id:
        raise ValueError(
            "ChEMBL_search_activities requires target_chembl_id or molecule_chembl_id"
        )
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_activities",
            "target_chembl_id": target_chembl_id,
            "molecule_chembl_id": molecule_chembl_id,
            "activities": [],
        }
    args: dict[str, Any] = {"limit": max(1, min(int(limit), 200))}
    if target_chembl_id:
        args["target_chembl_id"] = target_chembl_id
    if molecule_chembl_id:
        args["molecule_chembl_id"] = molecule_chembl_id
    return _tu("ChEMBL_search_activities", args)


def ChEMBL_search_similarity(
    smiles: str = "",
    *,
    threshold: int = 80,
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Tanimoto similarity search by SMILES.

    TU requires both `smiles` and `threshold`. Wrapper enforces a
    non-empty SMILES locally and clamps `threshold` to [0, 100] and
    `limit` to [1, 1000] before forwarding.
    """
    if not smiles:
        raise ValueError("ChEMBL_search_similarity requires a non-empty smiles")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_similarity",
            "smiles": smiles,
            "threshold": int(threshold),
            "molecules": [],
        }
    return _tu(
        "ChEMBL_search_similarity",
        {
            "smiles": smiles,
            "threshold": max(0, min(int(threshold), 100)),
            "limit": max(1, min(int(limit), 1000)),
            "offset": max(0, int(offset)),
        },
    )


def ChEMBL_search_substructure(
    smiles: str = "",
    *,
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Substructure search by SMILES."""
    if not smiles:
        raise ValueError("ChEMBL_search_substructure requires a non-empty smiles")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_substructure",
            "smiles": smiles,
            "molecules": [],
        }
    return _tu(
        "ChEMBL_search_substructure",
        {
            "smiles": smiles,
            "limit": max(1, min(int(limit), 1000)),
            "offset": max(0, int(offset)),
        },
    )


def ChEMBL_search_documents(
    document_id: str = "",
    *,
    title_contains: str = "",
    title__contains: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL documents (publications).

    All TU filters are optional; only non-empty values are forwarded.
    Legacy `title_contains` maps to TU's `title__contains` (double
    underscore — TU's standard Django-style lookup suffix).
    """
    from ._arg_compat import pick

    title_filter = pick(title__contains, title_contains, name="title__contains") or ""
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_documents",
            "document_id": document_id,
            "title_contains": title_filter,
            "documents": [],
        }
    args: dict[str, Any] = {
        "limit": max(1, min(int(limit), 1000)),
        "offset": max(0, int(offset)),
    }
    if document_id:
        args["document_id"] = document_id
    if title_filter:
        args["title__contains"] = title_filter
    return _tu("ChEMBL_search_documents", args)


def ChEMBL_search_compound_structural_alerts(
    molecule_chembl_id: str = "",
    *,
    alert_name_contains: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL compound structural alerts.

    TU exposes `alert_name__contains`; the wrapper keeps the legacy
    Python-friendly `alert_name_contains` spelling.
    """
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_compound_structural_alerts",
            "molecule_chembl_id": molecule_chembl_id,
            "alert_name_contains": alert_name_contains,
            "structural_alerts": [],
        }
    args: dict[str, Any] = {"limit": _limit(limit), "offset": _offset(offset)}
    if molecule_chembl_id:
        args["molecule_chembl_id"] = molecule_chembl_id
    if alert_name_contains:
        args["alert_name__contains"] = alert_name_contains
    return _tu("ChEMBL_search_compound_structural_alerts", args)


def ChEMBL_get_molecule_targets(
    molecule_chembl_id: str = "",
    *,
    molecule_chembl_id__exact: str = "",
    limit: int = 500,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Get unique targets associated with a ChEMBL molecule ID."""
    from ._arg_compat import pick

    molecule_chembl_id = pick(
        molecule_chembl_id__exact, molecule_chembl_id,
        name="molecule_chembl_id",
    ) or ""
    if not molecule_chembl_id:
        raise ValueError("ChEMBL_get_molecule_targets requires a non-empty molecule_chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_molecule_targets",
            "molecule_chembl_id": molecule_chembl_id,
            "targets": [],
        }
    return _tu(
        "ChEMBL_get_molecule_targets",
        {"molecule_chembl_id__exact": molecule_chembl_id, "limit": _limit(limit)},
    )


_TARGET_FIELDS = {
    "target_chembl_id",
    "pref_name",
    "organism",
    "target_type",
    "target_components",
}


def ChEMBL_search_targets(
    target_chembl_id: str = "",
    *,
    pref_name_contains: str = "",
    organism: str = "",
    target_type: str = "",
    fields: list[str] | tuple[str, ...] | None = None,
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL targets by ID, name, organism, or target type."""
    selected_fields = _fields(fields, allowed=_TARGET_FIELDS)
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_targets",
            "target_chembl_id": target_chembl_id,
            "pref_name_contains": pref_name_contains,
            "organism": organism,
            "target_type": target_type,
            "targets": [],
        }
    args: dict[str, Any] = {"limit": _limit(limit), "offset": _offset(offset)}
    if target_chembl_id:
        args["target_chembl_id"] = target_chembl_id
    if pref_name_contains:
        args["pref_name__contains"] = pref_name_contains
    if organism:
        args["organism"] = organism
    if target_type:
        args["target_type"] = target_type
    if selected_fields:
        args["fields"] = selected_fields
    return _tu("ChEMBL_search_targets", args)


def ChEMBL_get_target_activities(
    target_chembl_id: str = "",
    *,
    target_chembl_id__exact: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Get activity data for a ChEMBL target ID."""
    from ._arg_compat import pick

    target_chembl_id = pick(
        target_chembl_id__exact, target_chembl_id, name="target_chembl_id",
    ) or ""
    if not target_chembl_id:
        raise ValueError("ChEMBL_get_target_activities requires a non-empty target_chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_target_activities",
            "target_chembl_id": target_chembl_id,
            "activities": [],
        }
    return _tu(
        "ChEMBL_get_target_activities",
        {
            "target_chembl_id__exact": target_chembl_id,
            "limit": _limit(limit),
            "offset": _offset(offset),
        },
    )


def ChEMBL_get_drug_mechanisms(
    drug_chembl_id: str = "",
    *,
    drug_name: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Get mechanisms of action by ChEMBL drug ID or drug name."""
    if not drug_chembl_id and not drug_name:
        raise ValueError("ChEMBL_get_drug_mechanisms requires drug_chembl_id or drug_name")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_drug_mechanisms",
            "drug_chembl_id": drug_chembl_id,
            "drug_name": drug_name,
            "mechanisms": [],
        }
    args: dict[str, Any] = {"limit": _limit(limit), "offset": _offset(offset)}
    if drug_chembl_id:
        args["drug_chembl_id"] = drug_chembl_id
    if drug_name:
        args["drug_name"] = drug_name
    return _tu("ChEMBL_get_drug_mechanisms", args)


_ASSAY_FIELDS = {
    "assay_chembl_id",
    "description",
    "assay_type",
    "confidence_score",
    "target_chembl_id",
    "assay_organism",
    "bao_label",
}


def ChEMBL_search_assays(
    assay_chembl_id: str = "",
    *,
    assay_type: str = "",
    target_chembl_id: str = "",
    fields: list[str] | tuple[str, ...] | None = None,
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL assays by assay, type, or target filters."""
    selected_fields = _fields(fields, allowed=_ASSAY_FIELDS)
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_assays",
            "assay_chembl_id": assay_chembl_id,
            "assay_type": assay_type,
            "target_chembl_id": target_chembl_id,
            "assays": [],
        }
    args: dict[str, Any] = {"limit": _limit(limit), "offset": _offset(offset)}
    if assay_chembl_id:
        args["assay_chembl_id"] = assay_chembl_id
    if assay_type:
        args["assay_type"] = assay_type
    if target_chembl_id:
        args["target_chembl_id"] = target_chembl_id
    if selected_fields:
        args["fields"] = selected_fields
    return _tu("ChEMBL_search_assays", args)


def ChEMBL_get_target_assays(
    target_chembl_id: str = "",
    *,
    target_chembl_id__exact: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Get assays associated with a ChEMBL target ID."""
    from ._arg_compat import pick

    target_chembl_id = pick(
        target_chembl_id__exact, target_chembl_id, name="target_chembl_id",
    ) or ""
    if not target_chembl_id:
        raise ValueError("ChEMBL_get_target_assays requires a non-empty target_chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_target_assays",
            "target_chembl_id": target_chembl_id,
            "assays": [],
        }
    return _tu(
        "ChEMBL_get_target_assays",
        {
            "target_chembl_id__exact": target_chembl_id,
            "limit": _limit(limit),
            "offset": _offset(offset),
        },
    )


def ChEMBL_get_assay_activities(
    assay_chembl_id: str = "",
    *,
    assay_chembl_id__exact: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Get activity data for a ChEMBL assay ID."""
    from ._arg_compat import pick

    assay_chembl_id = pick(
        assay_chembl_id__exact, assay_chembl_id, name="assay_chembl_id",
    ) or ""
    if not assay_chembl_id:
        raise ValueError("ChEMBL_get_assay_activities requires a non-empty assay_chembl_id")
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_get_assay_activities",
            "assay_chembl_id": assay_chembl_id,
            "activities": [],
        }
    return _tu(
        "ChEMBL_get_assay_activities",
        {
            "assay_chembl_id__exact": assay_chembl_id,
            "limit": _limit(limit),
            "offset": _offset(offset),
        },
    )


def ChEMBL_search_binding_sites(
    target_chembl_id: str = "",
    *,
    site_name_contains: str = "",
    limit: int = 20,
    offset: int = 0,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Search ChEMBL binding sites by target or site name."""
    if not _live:
        return {
            "status": "mocked",
            "source": "ChEMBL_search_binding_sites",
            "target_chembl_id": target_chembl_id,
            "site_name_contains": site_name_contains,
            "binding_sites": [],
        }
    args: dict[str, Any] = {"limit": _limit(limit), "offset": _offset(offset)}
    if target_chembl_id:
        args["target_chembl_id"] = target_chembl_id
    if site_name_contains:
        args["site_name__contains"] = site_name_contains
    return _tu("ChEMBL_search_binding_sites", args)


BINDINGS = [
    ("ChEMBL_search_molecules", ChEMBL_search_molecules),
    ("ChEMBL_get_molecule", ChEMBL_get_molecule),
    ("ChEMBL_search_drugs", ChEMBL_search_drugs),
    ("ChEMBL_get_drug", ChEMBL_get_drug),
    ("ChEMBL_search_similarity", ChEMBL_search_similarity),
    ("ChEMBL_search_substructure", ChEMBL_search_substructure),
    ("ChEMBL_search_compound_structural_alerts", ChEMBL_search_compound_structural_alerts),
    ("ChEMBL_get_molecule_targets", ChEMBL_get_molecule_targets),
    ("ChEMBL_search_targets", ChEMBL_search_targets),
    ("ChEMBL_get_target_activities", ChEMBL_get_target_activities),
    ("ChEMBL_search_activities", ChEMBL_search_activities),
    ("ChEMBL_get_drug_mechanisms", ChEMBL_get_drug_mechanisms),
    ("ChEMBL_search_assays", ChEMBL_search_assays),
    ("ChEMBL_get_target_assays", ChEMBL_get_target_assays),
    ("ChEMBL_get_assay_activities", ChEMBL_get_assay_activities),
    ("ChEMBL_search_binding_sites", ChEMBL_search_binding_sites),
    ("ChEMBL_search_documents", ChEMBL_search_documents),
]
