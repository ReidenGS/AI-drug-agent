"""Optional live smoke for the 3 public-API MCP wrappers.

Default behavior: clean-skip when `MCP_LIVE_TOOLS != "true"` so CI and
local dev never hit external services by accident.

Live behavior (`MCP_LIVE_TOOLS=true`): exercise each allowlisted wrapper
once. Prints a one-line summary per tool — tool name, top-level keys,
and result count only. NEVER prints raw payload, API keys, or prompts.

Example:
  MCP_LIVE_TOOLS=true \\
  MCP_LIVE_TOOL_ALLOWLIST=EuropePMC_search_articles,PubChem_get_associated_patents_by_CID,FDA_OrangeBook_get_patent_info \\
  python3 scripts/run_live_public_wrappers_smoke.py
"""

from __future__ import annotations

import os
import sys

# Make `app` importable when invoked from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.mcp.tools.chembl import ChEMBL_search_activities
from app.mcp.tools.ebi_proteins import EBIProteins_get_features
from app.mcp.tools.evidence import EuropePMC_search_articles
from app.mcp.tools.patent import (
    FDA_OrangeBook_get_patent_info,
    PubChem_get_associated_patents_by_CID,
)
from app.settings import get_settings


# Stable sample inputs for each live wrapper. Kept at module scope so tests
# can import them and verify we don't drift back to unstable choices.
EUROPEPMC_SMOKE_QUERY = "HER2 antibody-drug conjugate"
# CID 5957 = ATP. Used because the xrefs/PatentID response is bounded and
# fast — aspirin (CID 2244) returns thousands of patent links and reliably
# read-times-out on a free network. Either `ok` (some patents) or `empty`
# (zero patents) is acceptable here.
PUBCHEM_SMOKE_CID = "5957"
# LIPITOR is a small-molecule drug consistently present in the FDA Orange
# Book Data Files. HERCEPTIN (a biologic) is NOT in Orange Book and would
# come back empty — that gave a misleading "PASS but empty" signal.
FDA_SMOKE_BRAND_NAME = "LIPITOR"
# CHEMBL1824 = HER2 target — stable, public, plenty of bioactivities.
CHEMBL_SMOKE_TARGET = "CHEMBL1824"
# P00533 = EGFR — densely annotated UniProt entry, stable feature set.
EBI_PROTEINS_SMOKE_ACCESSION = "P00533"

# All migrated wrappers return the ToolUniverseAdapter envelope:
#   {status, source, executor, arguments, payload}
# `payload` shape varies tool-by-tool (TU does not promise a stable
# top-level schema across its 2 000+ tools), so smoke acceptance must NOT
# depend on counting specific keys inside `payload`. Contract instead:
#   - status ∈ {ok, empty} ........... PASS
#   - status == upstream_error ....... FAIL
#   - executor == "tooluniverse" ..... required
#   - source == expected tool name ... required
#   - status == ok → payload must be a non-empty dict / non-None
#
# This is intentionally lax: TU's payload shape can change per tool
# without invalidating the smoke. Real correctness lives in the per-tool
# unit tests, not here.


def _peek_payload_size(payload: object) -> int | str:
    """Best-effort size hint for the summary line — display only, no PASS/FAIL.

    Never returns or prints the payload itself.
    """
    if payload is None:
        return 0
    if isinstance(payload, (list, str, bytes)):
        return len(payload)
    if isinstance(payload, dict):
        for key in (
            "hit_count", "patent_count", "record_count",
            "activity_count", "feature_count",
        ):
            value = payload.get(key)
            if isinstance(value, int):
                return value
        for key in (
            "results", "items", "records", "activities", "features",
            "patents", "data", "annotations",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return f"dict({len(payload)} keys)"
    return "?"


def _summarize(name: str, envelope: dict) -> str:
    status = envelope.get("status", "?")
    executor = envelope.get("executor")
    source = envelope.get("source")
    payload = envelope.get("payload")

    # Hard contract first.
    passed = status in {"ok", "empty"} and executor == "tooluniverse" and source == name
    if passed and status == "ok":
        # ok must come with a real payload — otherwise the tool gave us
        # nothing despite claiming success.
        payload_present = (
            isinstance(payload, dict) and bool(payload)
        ) or (isinstance(payload, list) and bool(payload))
        passed = payload_present

    size = _peek_payload_size(payload) if isinstance(payload, (dict, list)) else "n/a"
    verdict = "PASS" if passed else "FAIL"
    return (
        f"{verdict} {name} status={status} executor={executor} "
        f"source={source} payload_size={size}"
    )


def _should_run() -> bool:
    return os.environ.get("MCP_LIVE_TOOLS", "").lower() in {"1", "true", "yes"}


def main() -> int:
    if not _should_run():
        print(
            "SKIP: MCP_LIVE_TOOLS is not enabled. "
            "Set MCP_LIVE_TOOLS=true plus MCP_LIVE_TOOL_ALLOWLIST=... to run."
        )
        return 0

    settings = get_settings()
    allow = settings.live_tool_allowlist_set()
    if not allow:
        print(
            "SKIP: MCP_LIVE_TOOLS=true but MCP_LIVE_TOOL_ALLOWLIST is empty; "
            "refusing to call upstream services."
        )
        return 0

    print(f"Live smoke active. Allowlist: {sorted(allow)}")
    failures = 0

    cases: list[tuple[str, callable]] = []
    if "EuropePMC_search_articles" in allow:
        cases.append(
            (
                "EuropePMC_search_articles",
                lambda: EuropePMC_search_articles(EUROPEPMC_SMOKE_QUERY, _live=True),
            )
        )
    if "PubChem_get_associated_patents_by_CID" in allow:
        cases.append(
            (
                "PubChem_get_associated_patents_by_CID",
                lambda: PubChem_get_associated_patents_by_CID(PUBCHEM_SMOKE_CID, _live=True),
            )
        )
    if "FDA_OrangeBook_get_patent_info" in allow:
        cases.append(
            (
                "FDA_OrangeBook_get_patent_info",
                lambda: FDA_OrangeBook_get_patent_info(
                    brand_name=FDA_SMOKE_BRAND_NAME, _live=True
                ),
            )
        )
    if "ChEMBL_search_activities" in allow:
        cases.append(
            (
                "ChEMBL_search_activities",
                lambda: ChEMBL_search_activities(
                    target_chembl_id=CHEMBL_SMOKE_TARGET, limit=5, _live=True
                ),
            )
        )
    if "EBIProteins_get_features" in allow:
        cases.append(
            (
                "EBIProteins_get_features",
                lambda: EBIProteins_get_features(
                    EBI_PROTEINS_SMOKE_ACCESSION, _live=True
                ),
            )
        )

    for name, fn in cases:
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name} raised={type(exc).__name__}: {exc}")
            continue
        line = _summarize(name, out)
        if line.startswith("FAIL"):
            failures += 1
        print(line)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
