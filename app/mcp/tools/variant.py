"""DynaMut2 / AlphaMissense / ESM variant wrappers (Steps 6, 9)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("DynaMut2_predict_stability", _ni),
    ("AlphaMissense_get_variant_score", _ni),
    ("ESM_generate_protein_sequence", _ni),
    ("ESM_score_variant_sae_batch", _ni),
]
