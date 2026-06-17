"""DynaMut2 / AlphaMissense / ESM variant wrappers (Steps 6, 9).

Adapter-backed: `AlphaMissense_get_variant_score`. TU
`AlphaMissenseTool` is pure HTTP against `alphamissense.hegelab.org`
+ UniProt fasta — no GPU, no vendor key, no local model weights, no
async polling.

Deferred (`_ni` → NotImplementedError on `_live=True`):

- `DynaMut2_predict_stability`: TU `DynaMut2Tool` is an async REST job
  (upload PDB, poll `prediction_single` up to ~300s). Long synchronous
  blocking call profile matches the ProteinsPlus deferral policy; until
  long-job orchestration is settled we keep mock-only.
- `ESM_generate_protein_sequence` / `ESM_score_variant_sae_batch`: TU
  `ESMTool` requires the vendor `ESM_API_KEY` from
  `forge.evolutionaryscale.ai`, the `esm` PyPI package (plus a git-pinned
  build for SAE features), and remote inference billing. Vendor-key +
  extra-dep profile — not in this migration round's scope.
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def AlphaMissense_get_variant_score(
    uniprot_id: str = "",
    variant: str = "",
    *,
    _live: bool = False,
) -> dict[str, Any]:
    """Get the AlphaMissense pathogenicity score for a single variant.

    TU required: `uniprot_id`, `variant` (protein notation, e.g. ``V600E``
    or ``p.R123H``). TU implementation hits the hegelab AlphaMissense
    REST API via HTTP — no GPU, no vendor key, no local weights.
    """
    if not uniprot_id:
        raise ValueError(
            "AlphaMissense_get_variant_score requires a non-empty uniprot_id"
        )
    if not variant:
        raise ValueError(
            "AlphaMissense_get_variant_score requires a non-empty variant"
        )
    if not _live:
        return {
            "status": "mocked",
            "source": "AlphaMissense_get_variant_score",
            "uniprot_id": uniprot_id,
            "variant": variant,
            "score": None,
            "classification": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool(
        "AlphaMissense_get_variant_score",
        {"uniprot_id": uniprot_id, "variant": variant},
    )


BINDINGS = [
    ("DynaMut2_predict_stability", _ni),
    ("AlphaMissense_get_variant_score", AlphaMissense_get_variant_score),
    ("ESM_generate_protein_sequence", _ni),
    ("ESM_score_variant_sae_batch", _ni),
]
