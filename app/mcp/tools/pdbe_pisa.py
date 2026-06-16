"""PDBe PISA / KB interface wrappers (Steps 6, 8)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("PDBePISA_get_interfaces", _ni),
    ("PDBePISA_get_monomer_analysis", _ni),
    ("PDBe_KB_get_interface_residues", _ni),
]
