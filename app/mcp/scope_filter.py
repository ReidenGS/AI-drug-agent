"""Filter the MCP-visible tool subset for a given agent / step."""

from __future__ import annotations

from dataclasses import dataclass
from ..services.tool_inventory_service import InventoryEntry


# Agent → step coverage (matches architecture v0.1 tool-call flow doc).
AGENT_STEP_MAP: dict[str, set[str]] = {
    "candidate_context_agent": {"step_05"},
    "developability_agent": {"step_06"},
    "structure_and_design_agent": {"step_07", "step_08", "step_09"},
    "evidence_agent": {"step_13"},
    "patent_ip_agent": {"step_14"},
}


# Architecture-vs-v0.2-inventory carve-outs.
#
# The v0.2 inventory tags every tool with a single canonical step_id, but the
# architecture document and tool-flow doc explicitly route some tools to a
# second step too (e.g. ZINC compound search is "Step 5 candidate context" by
# inventory but also "Step 9 compound library screening" by architecture).
# Listing each (agent, step) override here keeps the carve-outs auditable —
# we never grant an agent a tool that the architecture doesn't sanction.
AGENT_TOOL_OVERRIDES: dict[tuple[str, str], set[str]] = {
    ("structure_and_design_agent", "step_09"): {
        "ZINC_search_compounds",
        "ZINC_get_compound",
        "ZINC_search_by_smiles",
        "ZINC_search_by_properties",
        "ZINC_get_purchasable",
        "ChEMBL_search_molecules",
        "ChEMBL_search_substructure",
        "ChEMBL_search_similarity",
    },
}


@dataclass(slots=True)
class ScopeRequest:
    agent_name: str
    step_id: str
    require_runtime_ok: bool = True


def _step_digits(s: str | None) -> str | None:
    """Normalize a step id to its digit form.

    Inventory rows use bare `"5"`, code uses canonical `"step_05"`. Both must
    compare equal here so inventory-based filtering actually fires.
    """
    if not s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.lstrip("0") or None


def filter_inventory(entries: list[InventoryEntry], req: ScopeRequest) -> list[InventoryEntry]:
    allowed_steps = AGENT_STEP_MAP.get(req.agent_name, set())
    if req.step_id not in allowed_steps:
        return []
    want = _step_digits(req.step_id)
    overrides = AGENT_TOOL_OVERRIDES.get((req.agent_name, req.step_id), set())
    out: list[InventoryEntry] = []
    for e in entries:
        if e.step_id and _step_digits(e.step_id) != want:
            # Architecture-sanctioned carve-out: tool is allowed at this
            # agent's step even though the v0.2 inventory tagged it elsewhere.
            if e.tool_name not in overrides:
                continue
        if req.require_runtime_ok and (e.runtime_status or "").lower() in {"broken", "unstable"}:
            continue
        out.append(e)
    return out
