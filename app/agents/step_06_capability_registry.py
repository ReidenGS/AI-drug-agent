"""Single-source Step 6 capability and inventory classification registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RuntimePolicy = Literal[
    "live_wired", "dependency_unavailable", "future", "unsupported_for_adc_step6"
]


@dataclass(frozen=True)
class Step6Capability:
    tool_name: str
    lane_type: str | None
    required_input_slots: tuple[str, ...]
    accepted_input_slots: tuple[str, ...]
    schema_arg_mapping: dict[str, str]
    coverage_category: str
    priority: int
    redundancy_group: str
    runtime_policy: RuntimePolicy
    output_interpreter_type: str = "none"
    classification: str = "production Step 6 capability"
    known_unavailable_reason: str = ""
    notes: str = ""


def _cap(
    tool: str, lane: str, slots: tuple[str, ...], arg: str, category: str,
    priority: int, *, runtime: RuntimePolicy = "live_wired",
    interpreter: str = "none", redundancy: str = "", reason: str = "",
    notes: str = "",
) -> Step6Capability:
    return Step6Capability(
        tool_name=tool, lane_type=lane, required_input_slots=slots,
        accepted_input_slots=slots, schema_arg_mapping={slots[0]: arg},
        coverage_category=category, priority=priority,
        redundancy_group=redundancy or category, runtime_policy=runtime,
        output_interpreter_type=interpreter,
        known_unavailable_reason=reason, notes=notes,
    )


def _excluded(tool: str, classification: str, *, runtime: RuntimePolicy, notes: str) -> Step6Capability:
    return Step6Capability(
        tool_name=tool, lane_type=None, required_input_slots=(), accepted_input_slots=(),
        schema_arg_mapping={}, coverage_category="excluded", priority=999,
        redundancy_group="", runtime_policy=runtime, classification=classification,
        known_unavailable_reason=notes if runtime == "dependency_unavailable" else "",
        notes=notes,
    )


_SMILES = "payload_linker_compound_liability"
_SEQUENCE = "antibody_protein_sequence_liability"
_ANTIGEN = "antigen_protein_feature_context"
_STRUCTURE = "structure_interface_quality"
_BIOACTIVITY = "compound_bioactivity_prior_context"

STEP_06_CAPABILITY_REGISTRY: tuple[Step6Capability, ...] = (
    _cap("DrugProps_pains_filter", _SMILES, ("smiles",), "smiles", "structural_alert", 10, interpreter="pains"),
    _cap("DrugProps_lipinski_filter", _SMILES, ("smiles",), "smiles", "rule_drug_likeness", 20, interpreter="lipinski"),
    _cap("DrugProps_calculate_qed", _SMILES, ("smiles",), "smiles", "qed", 30, interpreter="qed"),
    _cap("SwissADME_calculate_adme", _SMILES, ("smiles",), "smiles", "adme", 40, interpreter="adme"),
    _cap("SwissADME_check_druglikeness", _SMILES, ("smiles",), "smiles", "extended_drug_likeness", 50, interpreter="adme"),
    _cap("ADMETAI_predict_toxicity", _SMILES, ("smiles",), "smiles", "toxicity", 60, runtime="dependency_unavailable", interpreter="toxicity", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_physicochemical_properties", _SMILES, ("smiles",), "smiles", "physchem", 61, runtime="dependency_unavailable", interpreter="adme", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_solubility_lipophilicity_hydration", _SMILES, ("smiles",), "smiles", "solubility", 62, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_CYP_interactions", _SMILES, ("smiles",), "smiles", "cyp", 63, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_bioavailability", _SMILES, ("smiles",), "smiles", "bioavailability", 64, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_clearance_distribution", _SMILES, ("smiles",), "smiles", "clearance", 65, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_stress_response", _SMILES, ("smiles",), "smiles", "stress_response", 66, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("ADMETAI_predict_nuclear_receptor_activity", _SMILES, ("smiles",), "smiles", "nuclear_receptor", 67, runtime="dependency_unavailable", reason="ADMET-AI local model dependency deferred"),
    _cap("PROSITE_scan_sequence", _SEQUENCE, ("protein_sequence",), "sequence", "sequence_motif", 10, interpreter="motifs"),
    _cap("IEDB_predict_mhci_binding", _SEQUENCE, ("protein_sequence",), "sequence", "immunogenicity", 20),
    _cap("EBIProteins_get_features", _ANTIGEN, ("uniprot_id",), "accession", "protein_features", 10, interpreter="protein_features"),
    _cap("EBIProteins_get_epitopes", _ANTIGEN, ("uniprot_id",), "accession", "epitope_context", 20, interpreter="epitopes"),
    _cap("EBIProteins_get_antigen", _ANTIGEN, ("uniprot_id",), "accession", "antigenicity_context", 30, interpreter="protein_features"),
    _cap("GlyGen_get_glycoprotein", _ANTIGEN, ("uniprot_id",), "uniprot_ac", "glycosylation_context", 40),
    _cap("iPTMnet_get_ptm_sites", _ANTIGEN, ("uniprot_id",), "uniprot_id", "ptm_context", 50),
    _cap("PDBe_KB_get_interface_residues", _ANTIGEN, ("uniprot_id",), "uniprot_accession", "interface_context", 60),
    _cap("ProteinsPlus_profile_structure_quality", _STRUCTURE, ("structure_file",), "pdb_id_or_path", "structure_quality", 10, runtime="dependency_unavailable", interpreter="structure_quality", reason="ProteinsPlus live mode not wired"),
    _cap("PDBePISA_get_interfaces", _STRUCTURE, ("pdb_id",), "pdb_id", "interface_quality", 20),
    _cap("PDBePISA_get_monomer_analysis", _STRUCTURE, ("pdb_id",), "pdb_id", "monomer_quality", 30),
    _cap("ChEMBL_search_activities", _BIOACTIVITY, ("chembl_id",), "molecule_chembl_id", "activity_prior", 10),
    _cap("ChEMBL_search_compound_structural_alerts", _BIOACTIVITY, ("chembl_id",), "molecule_chembl_id", "chembl_structural_alert", 20, interpreter="pains"),
    _cap("ChEMBL_get_molecule_targets", _BIOACTIVITY, ("chembl_id",), "molecule_chembl_id", "target_prior", 30),
    _cap("BindingDB_get_targets_by_compound", _BIOACTIVITY, ("smiles",), "smiles", "binding_target_prior", 40),
    _excluded("DynaMut2_predict_stability", "requires inputs unavailable in this test", runtime="dependency_unavailable", notes="requires explicit mutation and wired DynaMut2 runtime"),
    _excluded("ProteinsPlus_predict_binding_sites", "valid but not yet interpreted", runtime="dependency_unavailable", notes="ProteinsPlus live mode not wired; binding-site prediction is not a Step 6 prefilter output"),
    _excluded("ProteinsPlus_predict_binding_sites_v3", "valid but not yet interpreted", runtime="dependency_unavailable", notes="ProteinsPlus live mode not wired; redundant binding-site alternative"),
    _excluded("GlyGen_get_site", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="requires a GlyGen site_id, not a UniProt accession"),
    _excluded("ChEMBL_get_drug_mechanisms", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="requires confirmed drug ChEMBL ID or drug name; molecule ID alone is insufficient"),
    _excluded("ChEMBL_search_targets", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="target search is discovery, not candidate liability prefilter"),
    _excluded("ChEMBL_get_target_activities", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="requires target_chembl_id"),
    _excluded("ChEMBL_search_assays", "valid but not yet interpreted", runtime="unsupported_for_adc_step6", notes="assay discovery is outside Step 6 prefilter"),
    _excluded("ChEMBL_get_target_assays", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="requires target_chembl_id"),
    _excluded("ChEMBL_get_assay_activities", "requires inputs unavailable in this test", runtime="unsupported_for_adc_step6", notes="requires assay_chembl_id"),
    _excluded("ChEMBL_search_binding_sites", "valid but not yet interpreted", runtime="unsupported_for_adc_step6", notes="binding-site discovery is outside Step 6 prefilter"),
    *tuple(
        _excluded(name, "future", runtime="future", notes="future DNA/RNA/oligo capability excluded from ADC Step 6")
        for name in (
            "IDT_analyze_oligo", "IDT_check_self_dimer", "DNA_calculate_gc_content",
            "DNA_reverse_complement", "Sequence_gc_content", "Sequence_reverse_complement",
            "RNAcentral_search", "RNAcentral_get_by_accession", "Rfam_search_sequence",
            "Rfam_get_family", "LNCipedia_search_lncrna", "LNCipedia_get_lncrna",
            "miRBase_search_mirna", "miRBase_get_mirna",
        )
    ),
)

STEP_06_CAPABILITY_BY_TOOL = {c.tool_name: c for c in STEP_06_CAPABILITY_REGISTRY}


def capabilities_for_lane(lane_type: str) -> list[Step6Capability]:
    return sorted(
        (c for c in STEP_06_CAPABILITY_REGISTRY if c.lane_type == lane_type),
        key=lambda c: (c.priority, c.tool_name),
    )


def eligible_capabilities(
    lane_type: str, *, signals: dict[str, bool], scoped_tools: set[str],
) -> tuple[list[Step6Capability], list[dict]]:
    eligible: list[Step6Capability] = []
    excluded: list[dict] = []
    for cap in capabilities_for_lane(lane_type):
        if cap.tool_name not in scoped_tools:
            excluded.append({"tool_name": cap.tool_name, "reason": "not_in_mcp_step_scope"})
            continue
        if not any(signals.get(slot) for slot in cap.required_input_slots):
            excluded.append({"tool_name": cap.tool_name, "reason": "missing_typed_input"})
            continue
        if cap.runtime_policy != "live_wired":
            excluded.append({
                "tool_name": cap.tool_name,
                "reason": cap.runtime_policy,
                "detail": cap.known_unavailable_reason or cap.notes,
            })
            continue
        eligible.append(cap)
    return eligible, excluded


def deterministic_arguments(tool_name: str, arg_hints: dict[str, Any]) -> dict[str, Any]:
    cap = STEP_06_CAPABILITY_BY_TOOL.get(tool_name)
    if cap is None or not cap.schema_arg_mapping:
        return {}
    for slot in cap.accepted_input_slots:
        value = arg_hints.get(slot)
        if value:
            return {cap.schema_arg_mapping.get(slot) or next(iter(cap.schema_arg_mapping.values())): value}
    return {}


def inventory_classification_summary() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cap in STEP_06_CAPABILITY_REGISTRY:
        key = cap.classification
        if cap.runtime_policy == "dependency_unavailable" and cap.lane_type:
            key = "dependency unavailable"
        elif cap.runtime_policy == "live_wired" and cap.lane_type:
            key = "valid and live-wired"
        out.setdefault(key, []).append(cap.tool_name)
    return {k: sorted(v) for k, v in sorted(out.items())}
