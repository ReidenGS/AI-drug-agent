"""ZINC compound search wrappers (Step 5/9).

Upstream caveat (from architecture v0.1 + Week 3 audit): current ToolUniverse
`ZINC_*` wrappers actually hit ZINC15 endpoints (which are captcha-gated and
unstable). They MUST NOT be reported as `ZINC22`. Step 5 / Step 9 records use
`source_library="ZINC"` and `source_database_version="unknown"` by default;
callers that confirm the endpoint may upgrade to `"ZINC15"`. `ZINC22` is
never used here.

Mock mode (`_live=False`, default) returns a deterministic shape so the
pipeline graph runs without network; `_live=True` would attempt the real
upstream HTTP call but is intentionally not enabled yet — the wrapper raises
`NotImplementedError` to keep behaviour honest.
"""

from __future__ import annotations

from typing import Any


def _mocked_hit(*, source: str, **payload) -> dict[str, Any]:
    """Common envelope for mock returns.

    `source_database_version` is `"unknown"` because we cannot confirm ZINC22
    from a mock. Callers must NOT upgrade to ZINC22 without explicit evidence.
    """
    return {
        "status": "mocked",
        "source": source,
        "source_database": "ZINC",
        "source_database_version": "unknown",
        "hits": [payload],
    }


def ZINC_search_compounds(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("ZINC_search_compounds requires a non-empty query")
    if _live:
        raise NotImplementedError("ZINC live mode disabled: upstream is captcha-gated")
    return _mocked_hit(source="ZINC_search_compounds", query=query)


def ZINC_get_compound(zinc_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not zinc_id:
        raise ValueError("ZINC_get_compound requires a non-empty zinc_id")
    if _live:
        raise NotImplementedError("ZINC live mode disabled: upstream is captcha-gated")
    return _mocked_hit(source="ZINC_get_compound", zinc_id=zinc_id)


def ZINC_search_by_smiles(smiles: str, *, _live: bool = False) -> dict[str, Any]:
    if not smiles:
        raise ValueError("ZINC_search_by_smiles requires a non-empty smiles")
    if _live:
        raise NotImplementedError("ZINC live mode disabled: upstream is captcha-gated")
    return _mocked_hit(source="ZINC_search_by_smiles", smiles=smiles)


def ZINC_search_by_properties(properties: dict | None = None, *, _live: bool = False) -> dict[str, Any]:
    if _live:
        raise NotImplementedError("ZINC live mode disabled: upstream is captcha-gated")
    return _mocked_hit(source="ZINC_search_by_properties", properties=properties or {})


def ZINC_get_purchasable(zinc_id: str, *, _live: bool = False) -> dict[str, Any]:
    if not zinc_id:
        raise ValueError("ZINC_get_purchasable requires a non-empty zinc_id")
    if _live:
        raise NotImplementedError("ZINC live mode disabled: upstream is captcha-gated")
    return _mocked_hit(source="ZINC_get_purchasable", zinc_id=zinc_id)


BINDINGS = [
    ("ZINC_search_compounds", ZINC_search_compounds),
    ("ZINC_get_compound", ZINC_get_compound),
    ("ZINC_search_by_smiles", ZINC_search_by_smiles),
    ("ZINC_search_by_properties", ZINC_search_by_properties),
    ("ZINC_get_purchasable", ZINC_get_purchasable),
]
