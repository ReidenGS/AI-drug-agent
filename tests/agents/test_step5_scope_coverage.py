"""Step 5 scoped-tool coverage audit.

Pins the relationship between the *inventory-scoped* Step 5 catalog
(``mcp_client.list_tools(agent_name="candidate_context_agent",
step_id="step_05")``) and the three buckets the Step 5 runtime owns:

- ``registry-covered`` — declared in ``STEP_05_CAPABILITY_REGISTRY`` and
  routed by the deterministic eligibility planner.
- ``synthetic_cdr3_path`` — covered by the runtime extension that
  extracts CDR3 from an antibody full sequence and feeds the IEDB BCR
  filter (``iedb_search_bcr_sequences``).
- ``intentionally_not_covered`` — present in the inventory-scoped
  catalog but deliberately not routed. Each entry carries an explicit
  reason recorded in the handoff so reviewers do not mistake the gap
  for a silent skip.

The test is an audit fence: it must not change the scope, it must not
extend the inventory, and it must not silence a real new tool that
appears in the inventory without an explicit bucket assignment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import _IEDB_BCR_TOOL_NAME
from app.agents.step_05_enrichment_registry import STEP_05_CAPABILITY_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


# Expected coverage table — must stay in sync with the Step 5 handoff
# document and with the registry. Adding or removing a tool in the
# inventory-scoped catalog without updating this table will fail this
# test on purpose.
EXPECTED_REGISTRY_COVERED: frozenset[str] = frozenset({
    "ChEMBL_get_molecule",
    "ChEMBL_search_molecules",
    "ChEMBL_search_similarity",
    "ChEMBL_search_substructure",
    "SAbDab_search_structures",
    "SAbDab_get_structure",
    "TheraSAbDab_search_by_target",
    "TheraSAbDab_search_therapeutics",
    "ZINC_get_compound",
    "ZINC_search_by_smiles",
    "ZINC_search_compounds",
})

EXPECTED_SYNTHETIC_CDR3_PATH: frozenset[str] = frozenset({
    _IEDB_BCR_TOOL_NAME,
})

# Each entry carries the documented reason it is intentionally not
# routed by the registry. The reason strings are not free text — they
# are the canonical buckets used in the project handoff.
EXPECTED_INTENTIONALLY_NOT_COVERED: dict[str, str] = {
    "ChEMBL_get_drug": (
        "Duplicate compound slot already covered by ChEMBL_get_molecule; "
        "not in the prior audit list."
    ),
    "ChEMBL_search_drugs": (
        "Duplicate compound slot already covered by ChEMBL_search_molecules; "
        "not in the prior audit list."
    ),
    "ZINC_get_purchasable": (
        "ZINC family — live disabled / captcha-gated; not routed by the "
        "registry, not synthesised by the runtime."
    ),
    "ZINC_search_by_properties": (
        "ZINC family — live disabled / captcha-gated; not routed by the "
        "registry, not synthesised by the runtime."
    ),
}


EXPECTED_TOTAL_SCOPED_COUNT = 16


def _scoped_step5_tools() -> set[str]:
    """Resolve the inventory-scoped Step 5 catalog through the same path
    the production API uses. Skips when the inventory xlsx is not on
    disk (e.g. a sliced CI checkout) — never silently widens scope."""
    if not DEFAULT_XLSX.exists() and not os.environ.get("TOOL_INVENTORY_XLSX"):
        pytest.skip("Inventory xlsx not present for Step 5 scope audit")
    os.environ.setdefault("STORAGE_MODE", "local")
    os.environ.setdefault("LOCAL_STORAGE_ROOT", "/tmp/step5-scope-audit")
    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ.setdefault(
        "TOOL_INVENTORY_XLSX",
        os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX)),
    )
    from app.settings import get_settings
    from app.deps import get_mcp_client
    get_settings.cache_clear()
    get_mcp_client.cache_clear()
    client = get_mcp_client()
    return set(
        client.list_tools(
            agent_name="candidate_context_agent", step_id="step_05"
        )
    )


# ── 1. The coverage table covers every scoped tool, with no overlap ───


def test_step5_coverage_table_partitions_every_scoped_tool():
    scoped = _scoped_step5_tools()
    assert len(scoped) == EXPECTED_TOTAL_SCOPED_COUNT, (
        f"Step 5 scoped tool count drifted: expected "
        f"{EXPECTED_TOTAL_SCOPED_COUNT}, got {len(scoped)}: {sorted(scoped)}"
    )

    registry = EXPECTED_REGISTRY_COVERED
    synthetic = EXPECTED_SYNTHETIC_CDR3_PATH
    intentional = frozenset(EXPECTED_INTENTIONALLY_NOT_COVERED)

    # No bucket overlap.
    assert registry.isdisjoint(synthetic), registry & synthetic
    assert registry.isdisjoint(intentional), registry & intentional
    assert synthetic.isdisjoint(intentional), synthetic & intentional

    union = registry | synthetic | intentional
    # Every scoped tool must land in exactly one bucket.
    unassigned = scoped - union
    assert not unassigned, (
        "Step 5 inventory-scoped catalog contains tools without an "
        "explicit coverage decision: "
        f"{sorted(unassigned)}. Add each to EXPECTED_REGISTRY_COVERED, "
        "EXPECTED_SYNTHETIC_CDR3_PATH, or "
        "EXPECTED_INTENTIONALLY_NOT_COVERED with a documented reason."
    )
    # And no expected name claims to be in scope while it is not.
    extra = union - scoped
    assert not extra, (
        f"Coverage table references tools that are not in the Step 5 "
        f"scoped catalog: {sorted(extra)}"
    )


# ── 2. Registry-covered set actually matches the metadata registry ────


def test_registry_covered_set_matches_capability_registry():
    """The handoff list and the in-code registry must agree."""
    registry_tool_names = {c.tool_name for c in STEP_05_CAPABILITY_REGISTRY}
    # Registry may carry entries not in the (smaller) inventory-scoped
    # catalog (e.g. ZINC capabilities for environments where ZINC is
    # registered but not scoped). The audit fence is the other
    # direction: every expected registry-covered tool must actually be
    # in the in-code registry.
    missing = EXPECTED_REGISTRY_COVERED - registry_tool_names
    assert not missing, (
        "Coverage table expects these tools to be registry-covered but "
        f"they are absent from STEP_05_CAPABILITY_REGISTRY: {sorted(missing)}"
    )


# ── 3. Synthetic CDR3 path tool wires through the documented constant ─


def test_synthetic_cdr3_path_uses_iedb_constant():
    assert _IEDB_BCR_TOOL_NAME == "iedb_search_bcr_sequences"
    assert _IEDB_BCR_TOOL_NAME in EXPECTED_SYNTHETIC_CDR3_PATH


# ── 4. Intentionally-not-covered entries carry documented reasons ─────


def test_intentionally_not_covered_entries_carry_reason_strings():
    """Each intentional skip must record WHY — readers should not have
    to guess whether a missing entry was an accident."""
    for tool, reason in EXPECTED_INTENTIONALLY_NOT_COVERED.items():
        assert isinstance(reason, str) and reason.strip(), tool
        assert "not routed" in reason or "not in the prior audit list" in reason or "Duplicate" in reason, (
            tool, reason,
        )
