"""SAbDab / TheraSAbDab / IEDB BCR wrappers (Step 5)."""

from __future__ import annotations


def _not_implemented(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("SAbDab_search_structures", _not_implemented),
    ("SAbDab_get_structure", _not_implemented),
    ("TheraSAbDab_search_by_target", _not_implemented),
    ("TheraSAbDab_search_therapeutics", _not_implemented),
    ("iedb_search_bcr_sequences", _not_implemented),
]
