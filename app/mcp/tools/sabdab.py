"""SAbDab / TheraSAbDab / IEDB BCR wrappers (Step 5).

Thin MCP binding layer. `_live=False` (default) returns deterministic
mock envelopes. `_live=True` routes through `ToolUniverseAdapter` for
the wired subset. ZINC is NOT here — it lives in `zinc.py` and stays
`intentionally_disabled`.

Argument mappings (legacy wrapper → TU schema):
    SAbDab_search_structures(query, limit)         → {query?, limit}
    SAbDab_get_structure(pdb_id)                   → {pdb_id}
    TheraSAbDab_search_by_target(target)           → {target}
    TheraSAbDab_search_therapeutics(query)         → {query}
    iedb_search_bcr_sequences(limit, offset, …)    → {limit, offset, filters?}

Audit doc: `项目文件/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


def _not_implemented(*_a, **_kw):
    raise NotImplementedError


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def SAbDab_search_structures(
    query: str = "", *, limit: int = 50, _live: bool = False
) -> dict[str, Any]:
    """Search SAbDab structural antibody database.

    TU accepts `query` and `antigen` as aliases for the same input; the
    wrapper exposes only the canonical `query`.
    """
    if not _live:
        return {
            "status": "mocked",
            "source": "SAbDab_search_structures",
            "query": query,
            "structures": [],
        }
    args: dict[str, Any] = {"limit": max(1, min(int(limit), 200))}
    if query:
        args["query"] = query
    return _tu("SAbDab_search_structures", args)


def SAbDab_get_structure(pdb_id: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Fetch antibody structure details from SAbDab by PDB ID.

    TU accepts `pdb_code` as an alias; the wrapper exposes only the
    canonical `pdb_id`. Same binding is referenced by Step 5 and Step 7;
    inventory scope filter decides which step may call it.
    """
    if not pdb_id:
        raise ValueError("SAbDab_get_structure requires a non-empty pdb_id")
    pdb_id = pdb_id.lower().strip()
    if not _live:
        return {
            "status": "mocked",
            "source": "SAbDab_get_structure",
            "pdb_id": pdb_id,
            "structure": None,
        }
    return _tu("SAbDab_get_structure", {"pdb_id": pdb_id})


def TheraSAbDab_search_by_target(target: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Find therapeutic antibodies targeting a given antigen."""
    if not target:
        raise ValueError("TheraSAbDab_search_by_target requires a non-empty target")
    if not _live:
        return {
            "status": "mocked",
            "source": "TheraSAbDab_search_by_target",
            "target": target,
            "therapeutics": [],
        }
    return _tu("TheraSAbDab_search_by_target", {"target": target})


def TheraSAbDab_search_therapeutics(query: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Search Thera-SAbDab therapeutic antibodies by name."""
    if not query:
        raise ValueError("TheraSAbDab_search_therapeutics requires a non-empty query")
    if not _live:
        return {
            "status": "mocked",
            "source": "TheraSAbDab_search_therapeutics",
            "query": query,
            "therapeutics": [],
        }
    return _tu("TheraSAbDab_search_therapeutics", {"query": query})


def iedb_search_bcr_sequences(
    *,
    limit: int = 10,
    offset: int = 0,
    filters: dict | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """Search IEDB B-cell receptor / antibody sequences.

    `filters` is forwarded as TU's `filters` PostgREST-style object only
    when the caller actually supplies it; otherwise omitted.
    """
    if not _live:
        return {
            "status": "mocked",
            "source": "iedb_search_bcr_sequences",
            "sequences": [],
        }
    args: dict[str, Any] = {
        "limit": max(1, min(int(limit), 200)),
        "offset": max(0, int(offset)),
    }
    if filters:
        args["filters"] = filters
    return _tu("iedb_search_bcr_sequences", args)


BINDINGS = [
    ("SAbDab_search_structures", SAbDab_search_structures),
    ("SAbDab_get_structure", SAbDab_get_structure),
    ("TheraSAbDab_search_by_target", TheraSAbDab_search_by_target),
    ("TheraSAbDab_search_therapeutics", TheraSAbDab_search_therapeutics),
    ("iedb_search_bcr_sequences", iedb_search_bcr_sequences),
]
