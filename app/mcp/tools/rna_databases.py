"""RNAcentral / Rfam / LNCipedia / miRBase wrappers (RNA sequence DBs).

All eight tools live in ToolUniverse 1.2.2 and route through the adapter
on `_live=True`. Wrappers do NOT import ToolUniverse and do NOT issue
HTTP themselves.

| Wrapper | TU class | Required | Optional (forwarded only when set) |
|---|---|---|---|
| `RNAcentral_search` | `RNAcentralSearchTool` | `query` | `page_size` (default 10) |
| `RNAcentral_get_by_accession` | `RNAcentralGetTool` | `accession` | — |
| `Rfam_search_sequence` | `RfamTool` | `operation`, `sequence` | `max_wait_seconds` (default 120) |
| `Rfam_get_family` | `RfamTool` | `operation`, `family_id` | `format` (json / xml) |
| `LNCipedia_search_lncrna` | `miRNASearchTool` | `query` | `species`, `size` |
| `LNCipedia_get_lncrna` | `miRNAGetTool` | `rnacentral_id` | `taxid` |
| `miRBase_search_mirna` | `miRNASearchTool` | `query` | `species`, `size` |
| `miRBase_get_mirna` | `miRNAGetTool` | `rnacentral_id` | `taxid` |

Rfam's `search_sequence` is sync-poll up to 120 s. That sits within the
acceptable sync budget for this round (vs. ProteinsPlus 900 s and DynaMut2
300 s which we deferred). If the caller wants a tighter budget they can
pass a lower `max_wait_seconds`; we clamp it to a non-negative integer.
"""

from __future__ import annotations

from typing import Any

from ._arg_compat import resolve_operation


def _opt(args: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """Forward only the kwargs the caller actually set (non-None / non-empty)."""
    for k, v in kw.items():
        if v is None:
            continue
        if isinstance(v, str) and not v:
            continue
        args[k] = v
    return args


# ── RNAcentral ──────────────────────────────────────────────────────────────


def RNAcentral_search(
    query: str = "", *, page_size: int | None = None, _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """RNAcentral entity search by query string."""
    if not query:
        raise ValueError("RNAcentral_search requires a non-empty query")
    if not _live:
        return {
            "status": "mocked",
            "source": "RNAcentral_search",
            "query": query,
            "page_size": page_size,
            "results": [],
        }
    from ..tooluniverse_adapter import call_tool

    args = _opt({"query": query}, page_size=page_size)
    return call_tool("RNAcentral_search", args)


def RNAcentral_get_by_accession(
    accession: str = "", *, _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    """RNAcentral entry lookup by accession (e.g. URS0000ABCDEF)."""
    if not accession:
        raise ValueError("RNAcentral_get_by_accession requires a non-empty accession")
    if not _live:
        return {
            "status": "mocked",
            "source": "RNAcentral_get_by_accession",
            "accession": accession,
            "entry": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("RNAcentral_get_by_accession", {"accession": accession})


# ── Rfam ────────────────────────────────────────────────────────────────────


def Rfam_search_sequence(
    sequence: str = "",
    *,
    max_wait_seconds: int | None = None,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Rfam family search for an RNA sequence (TU operation=search_sequence)."""
    if not sequence:
        raise ValueError("Rfam_search_sequence requires a non-empty sequence")
    op = resolve_operation(operation, "search_sequence")
    if not _live:
        return {
            "status": "mocked",
            "source": "Rfam_search_sequence",
            "sequence": sequence,
            "max_wait_seconds": max_wait_seconds,
            "hits": [],
        }
    from ..tooluniverse_adapter import call_tool

    args: dict[str, Any] = {"operation": op, "sequence": sequence}
    if max_wait_seconds is not None:
        args["max_wait_seconds"] = max(0, int(max_wait_seconds))
    return call_tool("Rfam_search_sequence", args)


def Rfam_get_family(
    family_id: str = "",
    *,
    format: str | None = None,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Rfam family record fetch (TU operation=get_family)."""
    if not family_id:
        raise ValueError("Rfam_get_family requires a non-empty family_id")
    op = resolve_operation(operation, "get_family")
    fmt: str | None = None
    if format is not None:
        fmt = str(format).lower()
        if fmt not in {"json", "xml"}:
            raise ValueError(
                "Rfam_get_family format must be one of {'json', 'xml'}"
            )
    if not _live:
        return {
            "status": "mocked",
            "source": "Rfam_get_family",
            "family_id": family_id,
            "format": fmt,
            "family": None,
        }
    from ..tooluniverse_adapter import call_tool

    args: dict[str, Any] = {"operation": op, "family_id": family_id}
    if fmt:
        args["format"] = fmt
    return call_tool("Rfam_get_family", args)


# ── LNCipedia / miRBase (share the miRNA*Tool schema shape) ────────────────


def _mirna_search(
    name: str, query: str, species: str | None, size: int | None, _live: bool,
) -> dict[str, Any]:
    if not query:
        raise ValueError(f"{name} requires a non-empty query")
    if not _live:
        return {
            "status": "mocked",
            "source": name,
            "query": query,
            "species": species,
            "size": size,
            "results": [],
        }
    from ..tooluniverse_adapter import call_tool

    args = _opt({"query": query}, species=species, size=size)
    return call_tool(name, args)


def _mirna_get(
    name: str, rnacentral_id: str, taxid: int | None, _live: bool,
) -> dict[str, Any]:
    if not rnacentral_id:
        raise ValueError(f"{name} requires a non-empty rnacentral_id")
    if not _live:
        return {
            "status": "mocked",
            "source": name,
            "rnacentral_id": rnacentral_id,
            "taxid": taxid,
            "entry": None,
        }
    from ..tooluniverse_adapter import call_tool

    args = _opt({"rnacentral_id": rnacentral_id}, taxid=taxid)
    return call_tool(name, args)


def LNCipedia_search_lncrna(
    query: str = "", *, species: str | None = None, size: int | None = None,
    _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    """LNCipedia lncRNA search."""
    return _mirna_search("LNCipedia_search_lncrna", query, species, size, _live)


def LNCipedia_get_lncrna(
    rnacentral_id: str = "", *, taxid: int | None = None, _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """LNCipedia lncRNA entry lookup by RNAcentral id."""
    return _mirna_get("LNCipedia_get_lncrna", rnacentral_id, taxid, _live)


def miRBase_search_mirna(
    query: str = "", *, species: str | None = None, size: int | None = None,
    _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    """miRBase miRNA search."""
    return _mirna_search("miRBase_search_mirna", query, species, size, _live)


def miRBase_get_mirna(
    rnacentral_id: str = "", *, taxid: int | None = None, _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """miRBase miRNA entry lookup by RNAcentral id."""
    return _mirna_get("miRBase_get_mirna", rnacentral_id, taxid, _live)


BINDINGS = [
    ("RNAcentral_search", RNAcentral_search),
    ("RNAcentral_get_by_accession", RNAcentral_get_by_accession),
    ("Rfam_search_sequence", Rfam_search_sequence),
    ("Rfam_get_family", Rfam_get_family),
    ("LNCipedia_search_lncrna", LNCipedia_search_lncrna),
    ("LNCipedia_get_lncrna", LNCipedia_get_lncrna),
    ("miRBase_search_mirna", miRBase_search_mirna),
    ("miRBase_get_mirna", miRBase_get_mirna),
]
