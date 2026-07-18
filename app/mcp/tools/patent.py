"""Patent / prior-art wrappers (Step 14).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` routes through `ToolUniverseAdapter` — there
is NO parallel manual httpx / zip-parser implementation in this file, by
design.

`drugbank_get_drug_references_by_drug_name_or_id` remains
`key_required`: DrugBank needs a paid license, so `_live=True` raises
`NotImplementedError` and the MCPClient surfaces it as
`dependency_unavailable`.

Audit doc: `\u9879\u76ee\u6587\u4ef6/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any

from ..outcome import dependency_unavailable_envelope


def _mocked(*, source: str, **payload: Any) -> dict[str, Any]:
    return {"status": "mocked", "source": source, **payload}


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def PubChem_get_associated_patents_by_CID(cid: str, *, _live: bool = False) -> dict[str, Any]:
    if not cid:
        raise ValueError("PubChem_get_associated_patents_by_CID requires a non-empty cid")
    if not _live:
        return _mocked(
            source="PubChem_get_associated_patents_by_CID", cid=cid, patents=[]
        )
    return _tu("PubChem_get_associated_patents_by_CID", {"cid": cid})


def drugbank_get_drug_references_by_drug_name_or_id(
    drug_name_or_id: str = "",
    *,
    query: str = "",
    case_sensitive: bool | None = None,
    exact_match: bool | None = None,
    limit: int | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    drug_name_or_id = drug_name_or_id or query
    if not drug_name_or_id:
        raise ValueError(
            "drugbank_get_drug_references_by_drug_name_or_id requires a non-empty arg"
        )
    if not _live:
        return _mocked(
            source="drugbank_get_drug_references_by_drug_name_or_id",
            drug_name_or_id=drug_name_or_id,
            references=[],
        )
    # DrugBank is license-gated. We deliberately do NOT route this through
    # ToolUniverse — TU may carry a DrugBank tool but invoking it requires
    # credentials we don't provision in this project.
    return dependency_unavailable_envelope(
        tool_name="drugbank_get_drug_references_by_drug_name_or_id",
        reason_code="drugbank_license_required",
    )


def FDA_OrangeBook_get_patent_info(
    brand_name: str = "", application_number: str = "",
    *, operation: str | None = None,
    _live: bool = False, **_extra: Any,
) -> dict[str, Any]:
    if not brand_name and not application_number:
        raise ValueError(
            "FDA_OrangeBook_get_patent_info requires brand_name or application_number"
        )
    if not _live:
        return _mocked(
            source="FDA_OrangeBook_get_patent_info",
            brand_name=brand_name,
            application_number=application_number,
            records=[],
        )
    args: dict[str, Any] = {}
    if brand_name:
        args["brand_name"] = brand_name
    if application_number:
        args["application_number"] = application_number
    if operation:
        args["operation"] = operation
    return _tu("FDA_OrangeBook_get_patent_info", args)


BINDINGS = [
    ("PubChem_get_associated_patents_by_CID", PubChem_get_associated_patents_by_CID),
    (
        "drugbank_get_drug_references_by_drug_name_or_id",
        drugbank_get_drug_references_by_drug_name_or_id,
    ),
    ("FDA_OrangeBook_get_patent_info", FDA_OrangeBook_get_patent_info),
]
