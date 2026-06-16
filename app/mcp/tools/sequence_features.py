"""PROSITE / GlyGen / iPTMnet / IEDB MHC-I wrappers (Step 6)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("PROSITE_scan_sequence", _ni),
    ("GlyGen_get_glycoprotein", _ni),
    ("GlyGen_get_site", _ni),
    ("iPTMnet_get_ptm_sites", _ni),
    ("IEDB_predict_mhci_binding", _ni),
]
