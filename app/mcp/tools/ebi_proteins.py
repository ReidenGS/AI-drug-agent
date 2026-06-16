"""EBI Proteins API wrappers (Step 6)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("EBIProteins_get_epitopes", _ni),
    ("EBIProteins_get_antigen", _ni),
    ("EBIProteins_get_features", _ni),
]
