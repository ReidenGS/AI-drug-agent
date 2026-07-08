"""Step 14 runtime resolver tests.

The resolver is the ONLY place a real value is read; it resolves compact,
non-sensitive values from the artifacts a request explicitly declared, and
returns a compact unresolved reason otherwise.
"""

from __future__ import annotations

from app.agents.step_14_runtime_resolver import resolve_step14_input_ref
from app.schemas.step_14_patent_request import (
    Step14InputRef,
    Step14PatentRequest,
    Step14PatentScope,
)


def _write_cct(storage, run_id: str, artifact: dict) -> None:
    storage.write_json(storage.run_key(run_id, "candidate_context_table.json"), artifact)


def _write_sq(storage, run_id: str, artifact: dict) -> None:
    storage.write_json(storage.run_key(run_id, "inputs", "structured_query.json"), artifact)


def _req(run_id: str, refs, *, declared, antibody_allowed=False) -> Step14PatentRequest:
    return Step14PatentRequest(
        run_id=run_id,
        user_query="patent search",
        source_artifact_refs=declared,
        input_refs=refs,
        patent_scope=Step14PatentScope(antibody_search_allowed=antibody_allowed),
    )


def test_resolve_pubchem_cid_from_candidate_context_table(local_storage):
    run_id = "run_r1"
    _write_cct(local_storage, run_id, {
        "candidate_records": [
            {
                "candidate_id": "cand_1",
                "candidate_type": "compound_component",
                "identifiers": [{"id_type": "pubchem_cid", "id_value": "123456"}],
                "materials": [],
            }
        ],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", candidate_id="cand_1",
        supports_tool_args=["cid", "pubchem_cid"],
    )
    request = _req(run_id, [ref], declared={"candidate_context_table": "cct"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is True
    assert resolved.value == "123456"
    assert resolved.unresolved_reason is None
    # Audit is value-free.
    assert "value" not in resolved.audit_entry()


def test_resolve_brand_name_from_candidate_material(local_storage):
    run_id = "run_r2"
    _write_cct(local_storage, run_id, {
        "candidate_records": [
            {
                "candidate_id": "cand_1",
                "candidate_type": "compound_component",
                "candidate_label": "Kadcyla",
                "identifiers": [],
                "materials": [{"material_type": "compound_name", "value": "trastuzumab emtansine"}],
            }
        ],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="candidate_context_table",
        source_path="candidate_records[].materials[].value",
        role="brand_name", candidate_id="cand_1",
        supports_tool_args=["brand_name"],
    )
    request = _req(run_id, [ref], declared={"candidate_context_table": "cct"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is True
    assert resolved.value == "trastuzumab emtansine"


def test_resolve_application_number_from_structured_query(local_storage):
    run_id = "run_r3"
    _write_sq(local_storage, run_id, {
        "mentioned_entities": {},
        "referenced_inputs": [
            {"id_type": "application_number", "value": "NDA761139", "source": "user"}
        ],
        "normalized_entities": [],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="structured_query",
        source_path="referenced_inputs[].value",
        role="application_number",
        supports_tool_args=["application_number"],
    )
    request = _req(run_id, [ref], declared={"structured_query": "sq"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is True
    assert resolved.value == "NDA761139"


def test_resolve_payload_text_from_structured_query(local_storage):
    run_id = "run_r4"
    _write_sq(local_storage, run_id, {
        "mentioned_entities": {"payload_text": "MMAE"},
        "referenced_inputs": [],
        "normalized_entities": [],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    request = _req(run_id, [ref], declared={"structured_query": "sq"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is True
    assert resolved.value == "MMAE"


def test_unresolved_when_source_artifact_not_declared(local_storage):
    run_id = "run_r5"
    _write_cct(local_storage, run_id, {"candidate_records": []})
    ref = Step14InputRef(
        ref_id="r1", source_artifact="candidate_context_table",
        source_path="x", role="pubchem_cid", supports_tool_args=["cid"],
    )
    # source_artifact_refs does NOT declare candidate_context_table.
    request = _req(run_id, [ref], declared={"structured_query": "sq"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "source_artifact_not_declared"


def test_unresolved_when_value_not_found(local_storage):
    run_id = "run_r6"
    _write_cct(local_storage, run_id, {
        "candidate_records": [
            {"candidate_id": "cand_1", "candidate_type": "compound_component",
             "identifiers": [], "materials": []}
        ],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", candidate_id="cand_1",
        supports_tool_args=["cid"],
    )
    request = _req(run_id, [ref], declared={"candidate_context_table": "cct"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "value_not_found_for_role"


def test_wrong_source_path_does_not_resolve_application_number(local_storage):
    # structured_query HAS an application_number, but the ref points at the
    # payload_text path → the resolver must NOT wander to referenced_inputs.
    run_id = "run_sp1"
    _write_sq(local_storage, run_id, {
        "mentioned_entities": {},
        "referenced_inputs": [
            {"id_type": "application_number", "value": "NDA761139", "source": "user"}
        ],
        "normalized_entities": [],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="application_number", supports_tool_args=["application_number"],
    )
    request = _req(run_id, [ref], declared={"structured_query": "sq"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "source_path_not_supported_for_role"
    assert resolved.value is None


def test_wrong_source_path_does_not_resolve_pubchem_cid(local_storage):
    # cct HAS a pubchem_cid identifier, but the ref points at the materials
    # value path → the resolver must NOT read the identifier.
    run_id = "run_sp2"
    _write_cct(local_storage, run_id, {
        "candidate_records": [
            {
                "candidate_id": "cand_1",
                "candidate_type": "compound_component",
                "identifiers": [{"id_type": "pubchem_cid", "id_value": "123456"}],
                "materials": [],
            }
        ],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="candidate_context_table",
        source_path="candidate_records[].materials[].value",
        role="pubchem_cid", candidate_id="cand_1",
        supports_tool_args=["cid", "pubchem_cid"],
    )
    request = _req(run_id, [ref], declared={"candidate_context_table": "cct"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "source_path_not_supported_for_role"
    assert resolved.value is None


def test_unknown_source_path_is_rejected(local_storage):
    run_id = "run_sp3"
    _write_sq(local_storage, run_id, {
        "mentioned_entities": {"payload_text": "MMAE"},
        "referenced_inputs": [], "normalized_entities": [],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="structured_query",
        source_path="some.made.up.path", role="payload",
        supports_tool_args=["query"],
    )
    request = _req(run_id, [ref], declared={"structured_query": "sq"})
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "source_path_not_supported_for_role"


def test_antibody_ref_unresolved_when_scope_disallows(local_storage):
    run_id = "run_r7"
    _write_sq(local_storage, run_id, {
        "mentioned_entities": {"antibody_candidate_text": "Trastuzumab"},
        "referenced_inputs": [], "normalized_entities": [],
    })
    ref = Step14InputRef(
        ref_id="r1", source_artifact="structured_query",
        source_path="mentioned_entities.antibody_candidate_text",
        role="antibody", supports_tool_args=["query"],
    )
    request = _req(run_id, [ref], declared={"structured_query": "sq"}, antibody_allowed=False)
    resolved = resolve_step14_input_ref(local_storage, request, ref)
    assert resolved.resolved is False
    assert resolved.unresolved_reason == "antibody_search_not_allowed"

    # Allowed → resolves.
    request_ok = _req(run_id, [ref], declared={"structured_query": "sq"}, antibody_allowed=True)
    resolved_ok = resolve_step14_input_ref(local_storage, request_ok, ref)
    assert resolved_ok.resolved is True
    assert resolved_ok.value == "Trastuzumab"
