"""AlphaFold prediction wrapper (Steps 5, 7, 8).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` routes through `ToolUniverseAdapter` — TU
owns the real upstream call to https://alphafold.ebi.ac.uk/api/.
"""

from __future__ import annotations

from typing import Any


def alphafold_get_prediction(uniprot: str, *, _live: bool = False) -> dict[str, Any]:
    if not uniprot or not isinstance(uniprot, str):
        raise ValueError("alphafold_get_prediction requires a non-empty UniProt id")
    if not _live:
        return {
            "uniprot": uniprot,
            "status": "mocked",
            "model_url": f"mock://alphafold/{uniprot}.pdb",
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("alphafold_get_prediction", {"qualifier": uniprot})


BINDINGS = [("alphafold_get_prediction", alphafold_get_prediction)]
