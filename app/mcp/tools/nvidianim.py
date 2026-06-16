"""NvidiaNIM wrappers (Steps 8, 9)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("NvidiaNIM_alphafold2_multimer", _ni),
    ("NvidiaNIM_openfold3", _ni),
    ("NvidiaNIM_boltz2", _ni),
    ("NvidiaNIM_rfdiffusion", _ni),
    ("NvidiaNIM_proteinmpnn", _ni),
]
