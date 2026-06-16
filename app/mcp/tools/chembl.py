"""ChEMBL wrappers (Steps 5, 6, 13)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("ChEMBL_search_molecules", _ni),
    ("ChEMBL_get_molecule", _ni),
    ("ChEMBL_search_drugs", _ni),
    ("ChEMBL_get_drug", _ni),
    ("ChEMBL_search_similarity", _ni),
    ("ChEMBL_search_substructure", _ni),
    ("ChEMBL_search_compound_structural_alerts", _ni),
    ("ChEMBL_get_molecule_targets", _ni),
    ("ChEMBL_search_targets", _ni),
    ("ChEMBL_get_target_activities", _ni),
    ("ChEMBL_search_activities", _ni),
    ("ChEMBL_get_drug_mechanisms", _ni),
    ("ChEMBL_search_assays", _ni),
    ("ChEMBL_get_target_assays", _ni),
    ("ChEMBL_get_assay_activities", _ni),
    ("ChEMBL_search_binding_sites", _ni),
    ("ChEMBL_search_documents", _ni),
]
