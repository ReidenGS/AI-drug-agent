"""DynaMut2 / AlphaMissense / ESM variant wrappers (Steps 6, 9).

Most tools remain thin ToolUniverse adapter bindings:

- `AlphaMissense_get_variant_score`: TU `AlphaMissenseTool` is pure HTTP
  against `alphamissense.hegelab.org` + UniProt fasta — no GPU, no vendor
  key, no local model weights, no async polling. Keeps its existing
  offline mocked-success shape for `_live=False` (unchanged from before this
  migration).
- `DynaMut2_predict_stability`, `ESM_generate_protein_sequence`,
  thin `_call_variant_tu` bindings — same shape as `nvidianim._call_nim`.
  `_live=False` raises `NotImplementedError` (no offline mock success);
  `_live=True` forwards to `tooluniverse_adapter.call_tool`.
- `ESM_score_variant_sae_batch`: direct official Biohub ESM SDK wrapper for
  live mode. ToolUniverse's current ESM SAE binding is incompatible with the
  installed SDK path; this wrapper uses `encode(ESMProtein(...))` before
  `logits(..., LogitsConfig(..., sae_config=SAEConfig(...)))` and returns a
  compact audit-safe envelope instead of raw tensors.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any


_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_PROTEIN_VARIANT_RE = re.compile(
    r"^(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$",
    re.IGNORECASE,
)


def _parse_protein_variant(variant: str) -> tuple[str, int, str, str] | None:
    match = _PROTEIN_VARIANT_RE.fullmatch(variant.strip())
    if match is None:
        return None
    reference_aa = match.group(1).upper()
    position = int(match.group(2))
    variant_aa = match.group(3).upper()
    return reference_aa, position, variant_aa, f"{reference_aa}{position}{variant_aa}"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize_classification(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return {
        "benign": "benign",
        "likely_benign": "benign",
        "ambiguous": "ambiguous",
        "pathogenic": "pathogenic",
        "likely_pathogenic": "pathogenic",
    }.get(value.strip().lower())


def _parse_classification_bin(
    raw_response: dict[str, Any], classification: str,
) -> tuple[set[str], str | None, bool]:
    all_key = f"{classification}_all"
    base_key = classification
    raw_value = raw_response.get(all_key)
    source_key = all_key
    if raw_value in (None, ""):
        raw_value = raw_response.get(base_key)
        source_key = base_key
    if raw_value in (None, ""):
        return set(), None, False
    if not isinstance(raw_value, str) or ":" not in raw_value:
        return set(), None, True
    amino_acid_text = raw_value.split(":", 1)[1].strip()
    if not amino_acid_text:
        return set(), f"raw_response.{source_key}", False
    amino_acids = {item.strip().upper() for item in amino_acid_text.split(",")}
    if any(len(item) != 1 or item not in _STANDARD_AMINO_ACIDS for item in amino_acids):
        return set(), None, True
    return amino_acids, f"raw_response.{source_key}", False


def _alphamissense_identity_error(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        **envelope,
        "status": "upstream_error",
        "normalization_status": "identity_mismatch",
        "normalized_result": None,
        "error_message": "alphamissense_response_identity_mismatch",
    }


def _normalize_alphamissense_envelope(
    envelope: dict[str, Any], *, uniprot_id: str, variant: str,
) -> dict[str, Any]:
    if envelope.get("status") not in {"ok", "empty"}:
        return {
            **envelope,
            "normalization_status": "upstream_error",
            "normalized_result": None,
        }

    parsed_variant = _parse_protein_variant(variant)
    if parsed_variant is None:
        return {
            **envelope,
            "status": "upstream_error",
            "normalization_status": "identity_mismatch",
            "normalized_result": None,
            "error_message": "alphamissense_response_identity_mismatch",
        }
    reference_aa, position, variant_aa, canonical_variant = parsed_variant
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return _alphamissense_identity_error(envelope)

    data = payload.get("data")
    raw_response = data.get("raw_response") if isinstance(data, dict) else None
    if isinstance(raw_response, dict):
        if (
            raw_response.get("uid") != uniprot_id
            or str(raw_response.get("aa") or "").upper() != reference_aa
            or raw_response.get("resi") != position
        ):
            return _alphamissense_identity_error(envelope)

        bins: dict[str, tuple[set[str], str | None]] = {}
        malformed = False
        for classification in ("benign", "ambiguous", "pathogenic"):
            members, source, one_malformed = _parse_classification_bin(
                raw_response, classification
            )
            bins[classification] = (members, source)
            malformed = malformed or one_malformed
        matches = [
            (classification, source)
            for classification, (members, source) in bins.items()
            if variant_aa in members
        ]
        classification = matches[0][0] if len(matches) == 1 and not malformed else None
        classification_source = (
            matches[0][1] if len(matches) == 1 and not malformed else None
        )
        normalized_result = {
            "uniprot_id": uniprot_id,
            "variant": canonical_variant,
            "position": position,
            "reference_aa": reference_aa,
            "variant_aa": variant_aa,
            "classification": classification,
            "classification_source": classification_source,
            "pathogenicity_score": None,
            "pathogenicity_score_source": None,
            "residue_mean_snv_score": _numeric(raw_response.get("mean")),
            "residue_mean_all_substitutions_score": _numeric(
                raw_response.get("mean_all")
            ),
        }
        return {
            **envelope,
            "normalization_status": (
                "normalized" if classification is not None
                else "classification_unavailable"
            ),
            "normalized_result": normalized_result,
        }

    returned_uniprot = payload.get("uniprot_id") or payload.get("uid")
    returned_variant = payload.get("variant")
    returned_parsed = (
        _parse_protein_variant(returned_variant)
        if isinstance(returned_variant, str)
        else None
    )
    if returned_uniprot != uniprot_id or returned_parsed != parsed_variant:
        return _alphamissense_identity_error(envelope)

    score_key = "pathogenicity_score" if "pathogenicity_score" in payload else "score"
    score = _numeric(payload.get(score_key))
    if score is None:
        classification = None
        classification_source = None
    elif payload.get("classification") not in (None, ""):
        classification = _normalize_classification(payload.get("classification"))
        classification_source = (
            "payload.classification" if classification is not None else None
        )
    else:
        classification = (
            "benign" if score < 0.34
            else "pathogenic" if score > 0.564
            else "ambiguous"
        )
        classification_source = f"payload.{score_key}_threshold"
    return {
        **envelope,
        "normalization_status": (
            "normalized" if score is not None and classification is not None
            else "classification_unavailable"
        ),
        "normalized_result": {
            "uniprot_id": uniprot_id,
            "variant": canonical_variant,
            "position": position,
            "reference_aa": reference_aa,
            "variant_aa": variant_aa,
            "classification": classification,
            "classification_source": classification_source,
            "pathogenicity_score": score,
            "pathogenicity_score_source": (
                f"payload.{score_key}" if score is not None else None
            ),
            "residue_mean_snv_score": None,
            "residue_mean_all_substitutions_score": None,
        },
    }


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

    envelope = call_tool(
        "AlphaMissense_get_variant_score",
        {"uniprot_id": uniprot_id, "variant": variant},
    )
    return _normalize_alphamissense_envelope(
        envelope,
        uniprot_id=uniprot_id,
        variant=variant,
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

    Live mode intentionally uses the official Biohub ESM SDK directly because
    the current ToolUniverse binding can call the installed SDK in an obsolete
    order. Offline mode still raises: no mocked success is introduced.
    """
    if not _live:
        raise NotImplementedError(
            "ESM_score_variant_sae_batch requires live Biohub ESM SDK execution; "
            "enable via MCP live settings and ESM_API_KEY"
        )
    sequence_value = sequence or ""
    variants_value = variants or []
    model_name = model or "esmc-6b-2024-12"
    sae_model_name = sae_model or "esmc-6b-2024-12_k64_codebook16384_layer60"
    top_k = 10 if top_k_features is None else top_k_features
    window_size = 8 if window is None else window

    args = {
        "sequence": sequence_value,
        "variants": variants_value,
        "model": model_name,
        "sae_model": sae_model_name,
        "top_k_features": top_k,
        "window": window_size,
    }
    redacted_args = _esm_score_redacted_arguments(args)
    if not sequence_value:
        return _esm_score_upstream_error(
            "ValueError: ESM_score_variant_sae_batch requires a non-empty sequence",
            "ValueError",
            redacted_args,
            model_name=model_name,
            sae_model_name=sae_model_name,
        )
    if not isinstance(variants_value, list) or not variants_value:
        return _esm_score_upstream_error(
            "ValueError: ESM_score_variant_sae_batch requires a non-empty variants list",
            "ValueError",
            redacted_args,
            model_name=model_name,
            sae_model_name=sae_model_name,
        )

    retry_count = 0
    recovered_after_transient_error = False
    try:
        from ..tooluniverse_adapter import _hydrate_env_from_settings

        _hydrate_env_from_settings()
        api_key = os.environ.get("ESM_API_KEY") or ""
        if not api_key:
            return _esm_score_upstream_error(
                "ESM_API_KEY is not configured",
                "MissingCredentialError",
                redacted_args,
                model_name=model_name,
                sae_model_name=sae_model_name,
            )
        ESMProtein, LogitsConfig, SAEConfig, ESMCForgeInferenceClient = _load_esm_sdk_classes()
        logits_out = None
        last_error_message = ""
        last_error_type = ""
        last_retryable = False
        for attempt in range(_ESM_SCORE_MAX_RETRIES + 1):
            try:
                client = ESMCForgeInferenceClient(model=model_name, token=api_key)
                protein = ESMProtein(sequence=sequence_value)
                tensor = client.encode(protein)
                if _is_esm_protein_error(tensor):
                    raise _ESMScoreSdkError(
                        f"ESMProteinError during encode: {tensor}",
                        "ESMProteinError",
                    )
                logits_config = LogitsConfig(
                    sequence=True,
                    sae_config=SAEConfig(models=[sae_model_name], normalize_features=True),
                )
                logits_out = client.logits(tensor, logits_config)
                if _is_esm_protein_error(logits_out):
                    raise _ESMScoreSdkError(
                        f"ESMProteinError during logits: {logits_out}",
                        "ESMProteinError",
                    )
                recovered_after_transient_error = attempt > 0
                retry_count = attempt
                break
            except Exception as exc:  # noqa: BLE001 - classify then retry or surface
                error_message = (
                    exc.message if isinstance(exc, _ESMScoreSdkError)
                    else f"{type(exc).__name__}: {exc}"
                )
                error_type = (
                    exc.error_type if isinstance(exc, _ESMScoreSdkError)
                    else type(exc).__name__
                )
                retryable = _esm_score_error_is_retryable(error_message)
                last_error_message = error_message
                last_error_type = error_type
                last_retryable = retryable
                if retryable and attempt < _ESM_SCORE_MAX_RETRIES:
                    retry_count = attempt + 1
                    _esm_score_sleep(0.2 * (attempt + 1))
                    continue
                return _esm_score_upstream_error(
                    error_message,
                    error_type,
                    redacted_args,
                    model_name=model_name,
                    sae_model_name=sae_model_name,
                    sensitive_values=(sequence_value, api_key),
                    retry_count=attempt,
                    retryable=retryable,
                )
        if logits_out is None:
            return _esm_score_upstream_error(
                last_error_message or "ESM score inference failed",
                last_error_type or "ESMScoreInferenceError",
                redacted_args,
                model_name=model_name,
                sae_model_name=sae_model_name,
                sensitive_values=(sequence_value, api_key),
                retry_count=retry_count,
                retryable=last_retryable,
            )
    except Exception as exc:  # noqa: BLE001 - live upstream/sdk failures become envelope
        return _esm_score_upstream_error(
            f"{type(exc).__name__}: {exc}",
            type(exc).__name__,
            redacted_args,
            model_name=model_name,
            sae_model_name=sae_model_name,
            sensitive_values=(sequence_value, os.environ.get("ESM_API_KEY") or ""),
            retry_count=retry_count,
            retryable=_esm_score_error_is_retryable(str(exc)),
        )

    score_result = _score_esm_variants_from_logits(
        sequence=sequence_value,
        variants=variants_value,
        logits_out=logits_out,
    )
    if score_result.get("status") != "ok":
        return _esm_score_upstream_error(
            str(score_result.get("error_message") or "scoring_unavailable"),
            str(score_result.get("final_error_type") or "scoring_unavailable"),
            redacted_args,
            model_name=model_name,
            sae_model_name=sae_model_name,
            sensitive_values=(sequence_value, os.environ.get("ESM_API_KEY") or ""),
        )

    sequence_sha = hashlib.sha256(sequence_value.encode("utf-8")).hexdigest()[:12]
    return {
        "status": "ok",
        "source": "ESM_score_variant_sae_batch",
        "executor": "biohub_esm_sdk",
        "arguments": redacted_args,
        "model": model_name,
        "sae_model": sae_model_name,
        "sequence_length": len(sequence_value),
        "sequence_sha256_prefix": sequence_sha,
        "variant_count": len(variants_value),
        "variants": _compact_variants(variants_value),
        "variant_scores": score_result["variant_scores"],
        "logits_layout": score_result["logits_layout"],
        "top_k_features": top_k,
        "window": window_size,
        "inference": {
            "encode": "ok",
            "logits": "ok",
            "sae_config": "ok",
        },
        "payload_summary": _compact_sdk_output_summary(logits_out),
        "retry_count": retry_count,
        "retryable": False,
        "recovered_after_transient_error": recovered_after_transient_error,
    }


_ESM_SCORE_MAX_RETRIES = 2
_ESM_SCORE_RETRY_TOKENS = (
    "401 unauthorized",
    "401 failure",
    "failure in encode",
    "failure in logits",
    "timeout",
    "timed out",
    "connection reset",
    "connectionerror",
    "remote disconnected",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "rate limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
)


class _ESMScoreSdkError(RuntimeError):
    def __init__(self, message: str, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


def _esm_score_error_is_retryable(message: str | None) -> bool:
    lowered = str(message or "").lower()
    return any(token in lowered for token in _ESM_SCORE_RETRY_TOKENS)


def _esm_score_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _load_esm_sdk_classes() -> tuple[Any, Any, Any, Any]:
    from esm.sdk.api import ESMProtein, LogitsConfig, SAEConfig  # type: ignore[import-not-found]
    from esm.sdk.forge import ESMCForgeInferenceClient  # type: ignore[import-not-found]

    return ESMProtein, LogitsConfig, SAEConfig, ESMCForgeInferenceClient


def _load_esm_sequence_tokenizer() -> Any:
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer  # type: ignore[import-not-found]

    return EsmSequenceTokenizer()


def _esm_score_upstream_error(
    message: str,
    error_type: str,
    redacted_args: dict[str, Any],
    *,
    model_name: str,
    sae_model_name: str,
    sensitive_values: tuple[str, ...] = (),
    retry_count: int = 0,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": "upstream_error",
        "source": "ESM_score_variant_sae_batch",
        "executor": "biohub_esm_sdk",
        "arguments": redacted_args,
        "model": model_name,
        "sae_model": sae_model_name,
        "error_message": _compact_error(message, sensitive_values=sensitive_values),
        "final_error_type": error_type,
        "retry_count": retry_count,
        "retryable": retryable,
    }


def _compact_error(
    message: str,
    *,
    limit: int = 240,
    sensitive_values: tuple[str, ...] = (),
) -> str:
    text = " ".join(str(message or "").split())
    for value in sensitive_values:
        if value:
            text = text.replace(value, "<redacted>")
    return text[:limit]


def _esm_score_redacted_arguments(args: dict[str, Any]) -> dict[str, Any]:
    sequence = str(args.get("sequence") or "")
    return {
        "sequence": {
            "redacted": True,
            "length": len(sequence),
            "sha256_prefix": hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:12]
            if sequence
            else "",
            "reason": "sequence omitted for compact audit",
        },
        "variants": _compact_variants(args.get("variants") or []),
        "model": args.get("model"),
        "sae_model": args.get("sae_model"),
        "top_k_features": args.get("top_k_features"),
        "window": args.get("window"),
    }


def _compact_variants(variants: Any) -> list[Any]:
    if not isinstance(variants, list):
        return []
    compact: list[Any] = []
    for item in variants:
        if isinstance(item, dict):
            compact.append(
                {
                    key: item.get(key)
                    for key in ("position", "ref_aa", "alt_aa", "variant", "mutation")
                    if key in item
                }
            )
        elif isinstance(item, (str, int, float, bool)) or item is None:
            compact.append(item)
        else:
            compact.append({"type": type(item).__name__})
    return compact


def _compact_sdk_output_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value.keys())[:20]
        return summary
    attrs = [
        name
        for name in ("logits", "embeddings", "hidden_states", "sae", "sequence_logits")
        if hasattr(value, name)
    ]
    if attrs:
        summary["available_attributes"] = attrs
    return summary


def _is_esm_protein_error(value: Any) -> bool:
    return type(value).__name__ == "ESMProteinError"


def _score_esm_variants_from_logits(
    *,
    sequence: str,
    variants: list[Any],
    logits_out: Any,
) -> dict[str, Any]:
    sequence_logits = _extract_sequence_logits(logits_out)
    if sequence_logits is None:
        return {
            "status": "upstream_error",
            "final_error_type": "scoring_unavailable",
            "error_message": "scoring_unavailable: logits.sequence missing from SDK output",
        }

    logits = sequence_logits
    shape = _tensor_shape(logits)
    if len(shape) == 3 and shape[0] == 1:
        logits = logits[0]
        shape = _tensor_shape(logits)
    if len(shape) < 2:
        return {
            "status": "upstream_error",
            "final_error_type": "scoring_unavailable",
            "error_message": f"scoring_unavailable: sequence logits shape {shape!r} is not rank-2",
        }

    row_count = int(shape[0])
    if row_count == len(sequence) + 2:
        layout = "bos_eos"
    elif row_count == len(sequence):
        layout = "residue_only"
    else:
        return {
            "status": "upstream_error",
            "final_error_type": "scoring_unavailable",
            "error_message": (
                "scoring_unavailable: sequence logits first dimension "
                f"{row_count} does not match sequence length {len(sequence)} or length+2"
            ),
        }

    try:
        tokenizer = _load_esm_sequence_tokenizer()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "upstream_error",
            "final_error_type": type(exc).__name__,
            "error_message": f"scoring_unavailable: tokenizer load failed: {exc}",
        }

    scores: list[dict[str, Any]] = []
    for variant in variants:
        scores.append(
            _score_one_esm_variant(
                sequence=sequence,
                variant=variant,
                logits=logits,
                logits_shape=shape,
                layout=layout,
                tokenizer=tokenizer,
            )
        )
    return {"status": "ok", "variant_scores": scores, "logits_layout": layout}


def _extract_sequence_logits(logits_out: Any) -> Any | None:
    logits = logits_out.get("logits") if isinstance(logits_out, dict) else getattr(logits_out, "logits", None)
    if logits is None:
        return None
    if isinstance(logits, dict):
        return logits.get("sequence")
    return getattr(logits, "sequence", None)


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value), *_tensor_shape(value[0]))
    return ()


def _score_one_esm_variant(
    *,
    sequence: str,
    variant: Any,
    logits: Any,
    logits_shape: tuple[int, ...],
    layout: str,
    tokenizer: Any,
) -> dict[str, Any]:
    parsed = _parse_variant_object(variant)
    out: dict[str, Any] = {
        "position": parsed.get("position"),
        "ref_aa": parsed.get("ref_aa"),
        "alt_aa": parsed.get("alt_aa"),
    }
    position = parsed.get("position")
    ref_aa = parsed.get("ref_aa")
    alt_aa = parsed.get("alt_aa")
    if not isinstance(position, int) or position < 1 or position > len(sequence):
        out.update(
            {
                "ref_matches_sequence": False,
                "scoring_status": "invalid_position",
                "error": "position_out_of_range",
            }
        )
        return out
    if not isinstance(ref_aa, str) or len(ref_aa) != 1 or not isinstance(alt_aa, str) or len(alt_aa) != 1:
        out.update(
            {
                "ref_matches_sequence": False,
                "scoring_status": "invalid_variant",
                "error": "ref_aa_and_alt_aa_must_be_single_residue_strings",
            }
        )
        return out

    ref_aa = ref_aa.upper()
    alt_aa = alt_aa.upper()
    out["ref_aa"] = ref_aa
    out["alt_aa"] = alt_aa
    out["ref_matches_sequence"] = sequence[position - 1].upper() == ref_aa
    row_index = position if layout == "bos_eos" else position - 1
    try:
        ref_token_id = _aa_token_id(tokenizer, ref_aa)
        alt_token_id = _aa_token_id(tokenizer, alt_aa)
        ref_logit = _as_float(_tensor_value(logits, row_index, ref_token_id))
        alt_logit = _as_float(_tensor_value(logits, row_index, alt_token_id))
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "scoring_status": "scoring_unavailable",
                "error": _compact_error(str(exc)),
            }
        )
        return out
    out.update(
        {
            "scoring_status": "ok",
            "ref_logit": ref_logit,
            "alt_logit": alt_logit,
            "delta_logit": alt_logit - ref_logit,
            "logits_row": row_index,
            "logits_vocab_size": int(logits_shape[1]),
        }
    )
    return out


def _parse_variant_object(variant: Any) -> dict[str, Any]:
    if not isinstance(variant, dict):
        return {"position": None, "ref_aa": None, "alt_aa": None}
    position = variant.get("position")
    if isinstance(position, str) and position.isdigit():
        position = int(position)
    return {
        "position": position,
        "ref_aa": variant.get("ref_aa"),
        "alt_aa": variant.get("alt_aa"),
    }


def _aa_token_id(tokenizer: Any, aa: str) -> int:
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(aa, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(aa)
        ids = _token_ids_to_list(encoded)
        if len(ids) == 1:
            return int(ids[0])
        if len(ids) >= 3:
            return int(ids[1])
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is not None:
            return int(token_id)
    raise ValueError(f"could not resolve token id for residue {aa!r}")


def _token_ids_to_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return []


def _tensor_value(logits: Any, row: int, col: int) -> Any:
    if hasattr(logits, "__getitem__"):
        return logits[row][col]
    raise TypeError("sequence logits object is not indexable")


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


BINDINGS = [
    ("DynaMut2_predict_stability", DynaMut2_predict_stability),
    ("AlphaMissense_get_variant_score", AlphaMissense_get_variant_score),
    ("ESM_generate_protein_sequence", ESM_generate_protein_sequence),
    ("ESM_score_variant_sae_batch", ESM_score_variant_sae_batch),
]
