"""Bind v0.2 inventory entries to FastMCP tool definitions.

Each per-domain module (sabdab.py, chembl.py, ...) lists the canonical tool
names it owns. At register time we intersect those with the inventory: a tool
is registered ONLY if its name appears in v0.2 inventory.
"""

from __future__ import annotations

from typing import Any, Callable

from ...services.tool_inventory_service import ToolInventoryService


# canonical → (module name, doc string). Real implementation supplied later.
ToolBinding = tuple[str, Callable[..., Any]]


def register_all(fastmcp: Any, inventory: ToolInventoryService) -> list[str]:
    allowed = inventory.names()
    registered: list[str] = []
    for name, fn in _all_bindings():
        if name not in allowed:
            continue
        fastmcp.tool(name=name)(fn)  # python-a2a FastMCP decorator-style
        registered.append(name)
    return registered


def _all_bindings() -> list[ToolBinding]:
    from . import (
        sabdab,
        chembl,
        rcsb_pdbe,
        alphafold,
        nvidianim,
        pdbe_pisa,
        proteins_plus,
        ebi_proteins,
        developability_compounds,
        sequence_features,
        variant,
        zinc,
        evidence,
        patent,
    )

    modules = [
        sabdab, chembl, rcsb_pdbe, alphafold, nvidianim, pdbe_pisa,
        proteins_plus, ebi_proteins, developability_compounds,
        sequence_features, variant, zinc, evidence, patent,
    ]
    out: list[ToolBinding] = []
    for m in modules:
        out.extend(getattr(m, "BINDINGS", []))
    return out
