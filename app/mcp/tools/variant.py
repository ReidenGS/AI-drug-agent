"""DynaMut2 / AlphaMissense / ESM variant wrappers (Steps 6, 9).

All four tools are now thin ToolUniverse adapter bindings:

- `AlphaMissense_get_variant_score`: TU `AlphaMissenseTool` is pure HTTP
  against `alphamissense.hegelab.org` + UniProt fasta — no GPU, no vendor
  key, no local model weights, no async polling. Keeps its existing
  offline mocked-success shape for `_live=False` (unchanged from before this
  migration).
- `DynaMut2_predict_stability`, `ESM_generate_protein_sequence`,
  `ESM_score_variant_sae_batch`: thin `_call_variant_tu` bindings — same
  shape as `nvidianim._call_nim`. `_live=False` raises `NotImplementedError`
  (no offline mock success); `_live=True` forwards to
  `tooluniverse_adapter.call_tool`. No direct vendor HTTP client is added
  here — ToolUniverse itself owns whatever REST/job-polling/vendor-key
  profile each of these tools needs at execution time.
"""

from __future__ import annotations

from typing import Any


def _call_variant_tu(tool_name: str, args: dict[str, Any], *, _live: bool = False) -> dict[str, Any]:
    if not _live:
        raise NotImplementedError(
            f"{tool_name} requires live ToolUniverse execution; enable via MCP live settings and required upstream credentials"
        )
    from ..tooluniverse_adapter import call_tool

    return call_tool(tool_name, args)


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


def DynaMut2_predict_stability(
    operation: str = "",
    pdb_id: str = "",
    chain: str = "",
    mutation: str = "",
    *,
    _live: bool = False,
) -> dict[str, Any]:
    """DynaMut2 mutation stability impact prediction (Step 9).

    Official ToolUniverse required args: ``operation``, ``pdb_id``,
    ``chain``, ``mutation``. ``pdb_id`` must be a real PDB identifier —
    callers must never forward an uploaded file path or storage ref here.
    """
    args: dict[str, Any] = {
        "operation": operation or "",
        "pdb_id": pdb_id or "",
        "chain": chain or "",
        "mutation": mutation or "",
    }
    return _call_variant_tu("DynaMut2_predict_stability", args, _live=_live)


def ESM_generate_protein_sequence(
    prompt_sequence: str = "",
    *,
    model: str | None = None,
    num_steps: int | None = None,
    temperature: float | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """ESM protein sequence completion/generation (Step 9).

    Official ToolUniverse required arg: ``prompt_sequence``. Optional args
    (``model``, ``num_steps``, ``temperature``) are only forwarded when the
    caller set them, so ToolUniverse applies its own documented defaults
    otherwise.
    """
    args: dict[str, Any] = {"prompt_sequence": prompt_sequence or ""}
    if model is not None:
        args["model"] = model
    if num_steps is not None:
        args["num_steps"] = num_steps
    if temperature is not None:
        args["temperature"] = temperature
    return _call_variant_tu("ESM_generate_protein_sequence", args, _live=_live)


def ESM_score_variant_sae_batch(
    sequence: str = "",
    variants: list[Any] | None = None,
    *,
    model: str | None = None,
    sae_model: str | None = None,
    top_k_features: int | None = None,
    window: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """ESM SAE batch variant scoring (Step 9).

    Official ToolUniverse required args: ``sequence``, ``variants``.
    Optional args (``model``, ``sae_model``, ``top_k_features``, ``window``)
    are only forwarded when the caller set them.
    """
    args: dict[str, Any] = {"sequence": sequence or "", "variants": variants or []}
    if model is not None:
        args["model"] = model
    if sae_model is not None:
        args["sae_model"] = sae_model
    if top_k_features is not None:
        args["top_k_features"] = top_k_features
    if window is not None:
        args["window"] = window
    return _call_variant_tu("ESM_score_variant_sae_batch", args, _live=_live)


BINDINGS = [
    ("DynaMut2_predict_stability", DynaMut2_predict_stability),
    ("AlphaMissense_get_variant_score", AlphaMissense_get_variant_score),
    ("ESM_generate_protein_sequence", ESM_generate_protein_sequence),
    ("ESM_score_variant_sae_batch", ESM_score_variant_sae_batch),
]
