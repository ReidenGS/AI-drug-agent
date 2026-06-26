from __future__ import annotations

from app.agents.step_05_enrichment_registry import (
    STEP_05_CAPABILITY_REGISTRY,
    Step5EnrichmentCapability,
    plan_enrichment_for_record,
    skipped_low_information_chembl_name_queries,
)
from app.schemas.step_05_candidate_context_table import CandidateRecord, Material, Identifier


def _record(
    *,
    candidate_type: str = "compound_component",
    materials: list[Material] | None = None,
    identifiers: list[Identifier] | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id="candidate_registry_test",
        candidate_label="registry test",
        candidate_type=candidate_type,  # type: ignore[arg-type]
        materials=materials or [],
        identifiers=identifiers or [],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


def _mat(material_type: str, value: str, role: str | None = None) -> Material:
    return Material(
        material_id=f"mat_{material_type}",
        material_type=material_type,
        value=value,
        role=role,
    )


def _ident(id_type: str, value: str) -> Identifier:
    return Identifier(id_type=id_type, id_value=value)


def test_custom_registry_entry_plans_without_agent_if_else():
    custom = (
        Step5EnrichmentCapability(
            tool_name="Custom_step5_name_lookup",
            capability_type="compound_name_lookup",
            required_input_slots=("payload_name",),
            accepted_input_slots=("payload_name",),
            schema_arg_mapping={"*": "query"},
            priority=1,
            fallback_group="custom_name",
            output_extractor_type="compound",
        ),
    )
    plans = plan_enrichment_for_record(
        _record(materials=[_mat("payload_name", "MMAE", "payload")]),
        scoped_tools=["Custom_step5_name_lookup"],
        candidate_category="compound_component",
        registry=custom,
    )
    assert len(plans) == 1
    assert plans[0].tool_name == "Custom_step5_name_lookup"
    assert plans[0].query == "MMAE"


def test_tools_outside_step5_scope_are_ignored_even_if_registered():
    plans = plan_enrichment_for_record(
        _record(materials=[_mat("payload_name", "MMAE", "payload")]),
        scoped_tools=[],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert plans == []


def test_required_input_slots_gate_calls_before_mcp():
    plans = plan_enrichment_for_record(
        _record(materials=[_mat("antibody_name", "trastuzumab", "antibody")]),
        scoped_tools=["ChEMBL_search_molecules"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert plans == []


def test_chembl_id_plans_get_molecule():
    plans = plan_enrichment_for_record(
        _record(identifiers=[_ident("chembl_id", "CHEMBL1201585")]),
        scoped_tools=["ChEMBL_get_molecule"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.schema_arg_name, p.query_kind, p.query) for p in plans] == [
        ("ChEMBL_get_molecule", "chembl_id", "chembl_id", "CHEMBL1201585")
    ]


def test_name_material_plans_chembl_search_molecules():
    plans = plan_enrichment_for_record(
        _record(materials=[_mat("payload_name", "monomethyl auristatin E", "payload")]),
        scoped_tools=["ChEMBL_search_molecules"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.schema_arg_name, p.query_kind, p.query_role) for p in plans] == [
        ("ChEMBL_search_molecules", "query", "name", "payload")
    ]


def test_low_information_linker_alias_vc_does_not_plan_chembl_name_lookup():
    record = _record(materials=[_mat("linker_name", "vc", "linker")])
    plans = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_search_molecules"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert plans == []
    assert skipped_low_information_chembl_name_queries(
        record,
        scoped_tools=["ChEMBL_search_molecules"],
        candidate_category="compound_component",
    ) == [("linker_name", "vc", "linker")]


def test_real_chembl_name_queries_survive_quality_gate():
    allowed = [
        ("payload_name", "MMAE", "payload"),
        ("payload_name", "MMAF", "payload"),
        ("payload_name", "DM1", "payload"),
        ("payload_name", "DXd", "payload"),
        ("payload_name", "SN-38", "payload"),
        ("linker_payload_name", "vc-MMAE", "linker_payload"),
        ("linker_name", "valine-citrulline", "linker"),
        ("payload_name", "monomethyl auristatin E", "payload"),
        ("linker_payload_name", "trastuzumab deruxtecan", "linker_payload"),
        ("compound_name", "T-DM1", "compound"),
        ("compound_name", "T-DXd", "compound"),
        ("compound_name", "Enhertu", "compound"),
    ]
    for material_type, value, role in allowed:
        plans = plan_enrichment_for_record(
            _record(materials=[_mat(material_type, value, role)]),
            scoped_tools=["ChEMBL_search_molecules"],
            candidate_category="compound_component",
            registry=STEP_05_CAPABILITY_REGISTRY,
        )
        assert [(p.tool_name, p.query) for p in plans] == [
            ("ChEMBL_search_molecules", value)
        ]


def test_smiles_material_prefers_substructure_over_similarity_by_fallback_group():
    record = _record(materials=[_mat("payload_smiles", "CCO", "payload")])
    both = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_search_substructure", "ChEMBL_search_similarity"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.query) for p in both] == [
        ("ChEMBL_search_substructure", "CCO")
    ]

    fallback = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_search_similarity"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.query) for p in fallback] == [
        ("ChEMBL_search_similarity", "CCO")
    ]


def test_sabdab_target_and_antibody_name_planning():
    target_plans = plan_enrichment_for_record(
        _record(
            candidate_type="target_antigen",
            materials=[_mat("target_antigen_name", "HER2", "target")],
        ),
        scoped_tools=["SAbDab_search_structures"],
        candidate_category="target_antigen",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    antibody_plans = plan_enrichment_for_record(
        _record(
            candidate_type="antibody",
            materials=[_mat("antibody_name", "trastuzumab", "antibody")],
        ),
        scoped_tools=["SAbDab_search_structures"],
        candidate_category="antibody",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.query) for p in target_plans] == [
        ("SAbDab_search_structures", "HER2")
    ]
    assert [(p.tool_name, p.query) for p in antibody_plans] == [
        ("SAbDab_search_structures", "trastuzumab")
    ]


def test_pdb_id_plans_sabdab_get_structure_for_antibody_or_target():
    """SAbDab_get_structure should plan for antibody / target_antigen /
    adc_construct candidates when a pdb_id identifier is present."""
    for category in ("antibody", "target_antigen", "adc_construct"):
        plans = plan_enrichment_for_record(
            _record(
                candidate_type=category,
                identifiers=[_ident("pdb_id", "1N8Z")],
            ),
            scoped_tools=["SAbDab_get_structure"],
            candidate_category=category,
            registry=STEP_05_CAPABILITY_REGISTRY,
        )
        assert [(p.tool_name, p.schema_arg_name, p.query_kind, p.query)
                for p in plans] == [
            ("SAbDab_get_structure", "pdb_id", "pdb_id", "1N8Z")
        ], category


def test_target_antigen_name_plans_therasabdab_search_by_target():
    plans = plan_enrichment_for_record(
        _record(
            candidate_type="target_antigen",
            materials=[_mat("target_antigen_name", "HER2", "target")],
        ),
        scoped_tools=["TheraSAbDab_search_by_target"],
        candidate_category="target_antigen",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.schema_arg_name, p.query, p.query_role)
            for p in plans] == [
        ("TheraSAbDab_search_by_target", "target", "HER2", "target")
    ]


def test_antibody_name_plans_therasabdab_search_therapeutics():
    plans = plan_enrichment_for_record(
        _record(
            candidate_type="antibody",
            materials=[_mat("antibody_name", "trastuzumab", "antibody")],
        ),
        scoped_tools=["TheraSAbDab_search_therapeutics"],
        candidate_category="antibody",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [(p.tool_name, p.schema_arg_name, p.query, p.query_role)
            for p in plans] == [
        ("TheraSAbDab_search_therapeutics", "query", "trastuzumab", "antibody")
    ]


def test_new_therasabdab_tools_only_route_in_their_candidate_category():
    """Symmetric guard: target_antigen_name should NOT plan
    TheraSAbDab_search_therapeutics, and antibody_name should NOT plan
    TheraSAbDab_search_by_target — they answer different questions and
    each is keyed to its own candidate category."""
    target_plans = plan_enrichment_for_record(
        _record(
            candidate_type="target_antigen",
            materials=[_mat("target_antigen_name", "HER2", "target")],
        ),
        scoped_tools=["TheraSAbDab_search_therapeutics"],
        candidate_category="target_antigen",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    antibody_plans = plan_enrichment_for_record(
        _record(
            candidate_type="antibody",
            materials=[_mat("antibody_name", "trastuzumab", "antibody")],
        ),
        scoped_tools=["TheraSAbDab_search_by_target"],
        candidate_category="antibody",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert target_plans == []
    assert antibody_plans == []


def test_iedb_search_bcr_sequences_is_not_routed_by_step5_registry():
    """Audit note (documented in the final report, not just a test
    silently passing): iedb_search_bcr_sequences has only optional
    ``limit`` / ``offset`` / ``select`` / ``filters`` parameters.
    ``filters`` is a free-form object whose schema is not knowable from
    Step 5 typed inputs, so a deterministic argument mapping cannot be
    constructed safely. The registry therefore does NOT route to this
    tool; the agent's eligibility planner returns no plan for it even
    when it is in the scoped catalog. The audit / handoff explicitly
    records this as an intentional skip (NOT a silent fallback)."""
    for materials in (
        [_mat("antibody_name", "trastuzumab", "antibody")],
        [_mat("target_antigen_name", "HER2", "target")],
        [_mat("antibody_heavy_chain_sequence",
              "/runs/x/trastuzumab.fasta", "antibody_sequence_reference")],
    ):
        for category in ("antibody", "target_antigen", "adc_construct"):
            plans = plan_enrichment_for_record(
                _record(candidate_type=category, materials=materials),
                scoped_tools=["iedb_search_bcr_sequences"],
                candidate_category=category,
                registry=STEP_05_CAPABILITY_REGISTRY,
            )
            assert plans == [], (category, materials)


def test_sequence_only_candidate_does_not_fabricate_tool_calls():
    """When a candidate only carries a FASTA material reference (no
    name, no UniProt accession, no PDB ID), the registry must NOT
    invent a tool call. Step 5 does not have a sequence-based UniProt
    or sequence-search wrapper in scope, and we do not synthesise one."""
    record = _record(
        candidate_type="antibody",
        materials=[
            _mat(
                "antibody_heavy_chain_sequence",
                "/runs/x/trastuzumab.fasta",
                "antibody_sequence_reference",
            )
        ],
    )
    full_scope = [
        "SAbDab_search_structures",
        "SAbDab_get_structure",
        "TheraSAbDab_search_by_target",
        "TheraSAbDab_search_therapeutics",
        "ChEMBL_get_molecule",
        "ChEMBL_search_molecules",
        "ChEMBL_search_substructure",
    ]
    plans = plan_enrichment_for_record(
        record,
        scoped_tools=full_scope,
        candidate_category="antibody",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert plans == []


def test_zinc_known_unavailable_policy_is_explicit_when_included():
    plans = plan_enrichment_for_record(
        _record(identifiers=[_ident("zinc_id", "ZINC0000001")]),
        scoped_tools=["ZINC_get_compound"],
        candidate_category="compound_component",
        include_known_unavailable=True,
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert len(plans) == 1
    assert plans[0].tool_name == "ZINC_get_compound"
    assert plans[0].known_live_unavailable is True
    assert "ZINC live disabled" in plans[0].known_unavailable_reason
