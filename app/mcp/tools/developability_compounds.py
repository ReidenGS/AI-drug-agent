"""DrugProps / SwissADME / ADMETAI / BindingDB wrappers (Step 6)."""

from __future__ import annotations


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("DrugProps_pains_filter", _ni),
    ("DrugProps_lipinski_filter", _ni),
    ("DrugProps_calculate_qed", _ni),
    ("SwissADME_calculate_adme", _ni),
    ("SwissADME_check_druglikeness", _ni),
    ("ADMETAI_predict_toxicity", _ni),
    ("ADMETAI_predict_physicochemical_properties", _ni),
    ("ADMETAI_predict_solubility_lipophilicity_hydration", _ni),
    ("ADMETAI_predict_CYP_interactions", _ni),
    ("ADMETAI_predict_bioavailability", _ni),
    ("ADMETAI_predict_clearance_distribution", _ni),
    ("ADMETAI_predict_stress_response", _ni),
    ("ADMETAI_predict_nuclear_receptor_activity", _ni),
    ("BindingDB_get_targets_by_compound", _ni),
]
