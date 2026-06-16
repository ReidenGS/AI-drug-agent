"""Patent / prior-art wrappers (Step 14).

Mock mode returns deterministic envelopes so the Step 14 graph runs without
network. `_live=True` raises NotImplementedError — real upstream call paths
need API keys / scraping plumbing that is out of scope here.
"""

from __future__ import annotations

from typing import Any


def _mocked(*, source: str, **payload: Any) -> dict[str, Any]:
    return {"status": "mocked", "source": source, **payload}


def PubChem_get_associated_patents_by_CID(cid: str, *, _live: bool = False) -> dict[str, Any]:
    if not cid:
        raise ValueError("PubChem_get_associated_patents_by_CID requires a non-empty cid")
    if _live:
        raise NotImplementedError("PubChem patents live mode not wired")
    return _mocked(source="PubChem_get_associated_patents_by_CID", cid=cid, patents=[])


def drugbank_get_drug_references_by_drug_name_or_id(
    drug_name_or_id: str, *, _live: bool = False
) -> dict[str, Any]:
    if not drug_name_or_id:
        raise ValueError(
            "drugbank_get_drug_references_by_drug_name_or_id requires a non-empty arg"
        )
    if _live:
        raise NotImplementedError("DrugBank live mode not wired")
    return _mocked(
        source="drugbank_get_drug_references_by_drug_name_or_id",
        drug_name_or_id=drug_name_or_id,
        references=[],
    )


def FDA_OrangeBook_get_patent_info(
    brand_name: str = "", application_number: str = "", *, _live: bool = False
) -> dict[str, Any]:
    if not brand_name and not application_number:
        raise ValueError(
            "FDA_OrangeBook_get_patent_info requires brand_name or application_number"
        )
    if _live:
        raise NotImplementedError("FDA Orange Book live mode not wired")
    return _mocked(
        source="FDA_OrangeBook_get_patent_info",
        brand_name=brand_name,
        application_number=application_number,
        # Real Orange Book returns product / patent / exclusivity tables; in
        # mock mode we record the inputs only.
        records=[],
    )


BINDINGS = [
    ("PubChem_get_associated_patents_by_CID", PubChem_get_associated_patents_by_CID),
    (
        "drugbank_get_drug_references_by_drug_name_or_id",
        drugbank_get_drug_references_by_drug_name_or_id,
    ),
    ("FDA_OrangeBook_get_patent_info", FDA_OrangeBook_get_patent_info),
]
