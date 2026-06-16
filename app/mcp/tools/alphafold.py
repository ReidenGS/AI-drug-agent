"""AlphaFold prediction wrapper (Steps 5, 7, 8).

Real upstream: https://alphafold.ebi.ac.uk/api/prediction/{uniprot}

Network is optional. By default this wrapper is fully callable but does NOT
trigger a real HTTP request; pass `_live=True` to attempt the real call. Any
network error is propagated as an exception so the `MCPClient` records it as
`failed` / `dependency_unavailable` without crashing the agent.
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
    import httpx  # local import to keep dev-mode cold path light

    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return {"uniprot": uniprot, "status": "live", "predictions": resp.json()}


BINDINGS = [("alphafold_get_prediction", alphafold_get_prediction)]
