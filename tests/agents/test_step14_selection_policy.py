"""Step 14 LLM patent tool-selection policy tests.

The selector chooses tools purely from input-ref roles / supports_tool_args;
the validator drops hallucinated tools / input refs, unsatisfiable plans, and
antibody plans when antibody search is disabled.
"""

from __future__ import annotations

from app.agents.step_14_selection_policy import (
    STEP14_SELECTION_PROMPT_CACHE_LAYOUT_VERSION,
    STEP14_SELECTION_SYSTEM_PROMPT,
    STEP14_SELECTION_USER_PROMPT,
    STEP14_TOOL_SPECS,
    build_step14_selection_catalog,
    build_step14_selection_payload,
    plan_step14_tool_calls,
    schema_arg_for_support,
)
from app.agents.tool_selection_policy import signature_schema_for
import pytest
from pydantic import ValidationError

from app.llm.json_task_validation import build_json_prompt_sections, shape_instruction
from app.llm.provider import MockLLMProvider
from app.schemas.step_14_patent_request import (
    Step14InputRef,
    Step14PatentRequest,
    Step14PatentScope,
)


class _StubLLM:
    """Returns a fixed response regardless of prompt (drift/hallucination tests)."""

    name = "stub"
    model = "stub"

    def __init__(self, response: dict):
        self.response = response

    def generate(self, prompt: str, *, system=None, **kwargs):
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system=None) -> dict:
        self.schema = schema
        return self.response


def _ref(ref_id, role, supports, *, source="candidate_context_table"):
    return Step14InputRef(
        ref_id=ref_id, source_artifact=source, source_path="p",
        role=role, supports_tool_args=supports,
    )


def _req(refs, *, antibody_allowed=False):
    return Step14PatentRequest(
        run_id="run_sel",
        user_query="patent search",
        source_artifact_refs={"candidate_context_table": "cct", "structured_query": "sq"},
        input_refs=refs,
        patent_scope=Step14PatentScope(antibody_search_allowed=antibody_allowed),
    )


# ── schema guard ────────────────────────────────────────────────────────────


def test_request_schema_has_no_runtime_value():
    assert "runtime_value" not in Step14PatentRequest.model_fields
    assert "runtime_value" not in Step14InputRef.model_fields


def test_input_ref_rejects_runtime_value_and_unknown_fields():
    base = dict(
        ref_id="r1", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text", role="payload",
        supports_tool_args=["query"],
    )
    with pytest.raises(ValidationError):
        Step14InputRef(**base, runtime_value="MMAE")
    with pytest.raises(ValidationError):
        Step14InputRef(**base, some_unknown_field=1)


def test_patent_scope_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Step14PatentScope(antibody_search_allowed=True, runtime_value="x")


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Step14PatentRequest(run_id="run_x", runtime_value="x")


# ── single-stage planner: mock produces schema_arg → input_ref_id maps ──────


def _plan(result, tool_name):
    for p in result.tool_plans:
        if p.tool_name == tool_name:
            return p
    return None


def _maps(plan):
    return {(m.schema_arg, m.input_ref_id) for m in plan.argument_mappings}


def test_pubchem_plan_maps_cid_to_ref():
    ref = _ref("r_cid", "pubchem_cid", ["cid", "pubchem_cid"])
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    plan = _plan(result, "PubChem_get_associated_patents_by_CID")
    assert plan is not None and plan.can_invoke is True
    assert _maps(plan) == {("cid", "r_cid")}
    assert plan.missing_required_args == []


def test_fda_plan_maps_brand_name():
    ref = _ref("r_brand", "brand_name", ["brand_name"])
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    plan = _plan(result, "FDA_OrangeBook_get_patent_info")
    assert plan is not None and plan.can_invoke is True
    assert _maps(plan) == {("brand_name", "r_brand")}


def test_fda_plan_maps_application_number():
    ref = _ref("r_app", "application_number", ["application_number"], source="structured_query")
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    plan = _plan(result, "FDA_OrangeBook_get_patent_info")
    assert plan is not None and plan.can_invoke is True
    assert _maps(plan) == {("application_number", "r_app")}


def test_drugbank_plan_maps_to_official_query_arg():
    ref = _ref("r_payload", "payload", ["query", "drug_name_or_id"])
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    plan = _plan(result, "drugbank_get_drug_references_by_drug_name_or_id")
    assert plan is not None and plan.can_invoke is True
    # Official identity arg is `query`, never `drug_name_or_id`.
    assert _maps(plan) == {("query", "r_payload")}
    assert not any(m.schema_arg == "drug_name_or_id" for m in plan.argument_mappings)
    # payload text cannot fill FDA (no brand_name/application_number support).
    assert _plan(result, "FDA_OrangeBook_get_patent_info") is None


def test_argument_mapping_audit_records_support_token():
    ref = _ref("r_cid", "pubchem_cid", ["pubchem_cid"])
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    audit = [a for a in result.argument_mapping_audit
             if a["tool_name"] == "PubChem_get_associated_patents_by_CID"]
    assert audit == [{
        "tool_name": "PubChem_get_associated_patents_by_CID",
        "schema_arg": "cid", "input_ref_id": "r_cid",
        "satisfied_by_support": "pubchem_cid", "can_invoke": True,
    }]


# ── planner validation / rejection rules (via _StubLLM) ─────────────────────


def test_unknown_tool_is_rejected():
    ref = _ref("r1", "pubchem_cid", ["cid"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "Evil_unlisted_tool", "can_invoke": True,
         "argument_mappings": [{"schema_arg": "cid", "input_ref_id": "r1"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    reasons = {p.tool_name: p.reason for p in result.rejected_tool_plans}
    assert reasons["Evil_unlisted_tool"] == "unknown_tool"


def test_unknown_input_ref_id_is_rejected():
    ref = _ref("r1", "pubchem_cid", ["cid"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [{"schema_arg": "cid", "input_ref_id": "ghost_ref"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    assert any("unknown_input_ref_id:ghost_ref" in p.reason for p in result.rejected_tool_plans)


def test_unknown_schema_arg_is_rejected():
    ref = _ref("r1", "pubchem_cid", ["cid"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [{"schema_arg": "not_a_real_arg", "input_ref_id": "r1"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    assert any("unknown_schema_arg:not_a_real_arg" in p.reason for p in result.rejected_tool_plans)


def test_duplicate_schema_arg_is_rejected_no_silent_overwrite():
    refs = [_ref("r1", "pubchem_cid", ["cid"]), _ref("r2", "pubchem_cid", ["cid"])]
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [
             {"schema_arg": "cid", "input_ref_id": "r1"},
             {"schema_arg": "cid", "input_ref_id": "r2"},
         ]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req(refs))
    assert result.tool_plans == []
    assert any("duplicate_schema_arg:cid" in p.reason for p in result.rejected_tool_plans)


def test_ref_cannot_satisfy_schema_arg_is_rejected():
    # A payload ref (supports query) cannot fill PubChem's cid.
    ref = _ref("r1", "payload", ["query"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [{"schema_arg": "cid", "input_ref_id": "r1"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    assert any("input_ref_cannot_satisfy_schema_arg:cid" in p.reason
               for p in result.rejected_tool_plans)


def test_unsatisfied_identity_arg_sets_can_invoke_false():
    # Valid empty plan for PubChem with no cid mapping → kept, uninvokable.
    ref = _ref("r1", "pubchem_cid", ["cid"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": []},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    plan = _plan(result, "PubChem_get_associated_patents_by_CID")
    assert plan is not None
    assert plan.can_invoke is False
    assert plan.missing_required_args == ["cid"]


def test_shape_instruction_uses_literal_value_json():
    shape = shape_instruction("step14_patent_tool_selection")
    assert "literal_value_json" in shape
    # No bare `literal_value` field in the parser-facing shape.
    assert '"literal_value":' not in shape


def test_planner_decodes_literal_value_json_to_literal_value(monkeypatch):
    # Opt into an official DrugBank schema where `limit` is an integer so the
    # config literal is schema-allowed; then a literal_value_json "25" must
    # normalize to literal_value == 25 before runtime execution.
    from app.mcp import tooluniverse_adapter

    official = {
        "description": "DrugBank refs",
        "parameter": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    }
    target = "drugbank_get_drug_references_by_drug_name_or_id"
    monkeypatch.setattr(
        tooluniverse_adapter, "get_tool_specification",
        lambda name: official if name == target else None,
    )
    monkeypatch.setattr(
        tooluniverse_adapter, "get_tool_specifications",
        lambda names: {target: official} if target in names else {},
    )

    ref = _ref("r_pay", "payload", ["query"], source="structured_query")
    llm = _StubLLM({"tool_plans": [
        {"tool_name": target, "can_invoke": True,
         "argument_mappings": [{"schema_arg": "query", "input_ref_id": "r_pay"}],
         "argument_literals": [{"schema_arg": "limit", "literal_value_json": "25"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    plan = _plan(result, target)
    assert plan is not None and plan.can_invoke is True
    assert [(l.schema_arg, l.literal_value) for l in plan.argument_literals] == [("limit", 25)]
    assert plan.argument_literals[0].literal_value == 25  # decoded int, not "25"


def test_planner_rejects_invalid_literal_value_json(monkeypatch):
    from app.mcp import tooluniverse_adapter

    official = {
        "parameter": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    }
    target = "drugbank_get_drug_references_by_drug_name_or_id"
    monkeypatch.setattr(
        tooluniverse_adapter, "get_tool_specification",
        lambda name: official if name == target else None,
    )
    ref = _ref("r_pay", "payload", ["query"], source="structured_query")
    llm = _StubLLM({"tool_plans": [
        {"tool_name": target, "can_invoke": True,
         "argument_mappings": [{"schema_arg": "query", "input_ref_id": "r_pay"}],
         "argument_literals": [{"schema_arg": "limit", "literal_value_json": "not-json{"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    assert any("literal_value_json_invalid:limit" in p.reason for p in result.rejected_tool_plans)


def test_literal_for_identity_arg_is_rejected():
    # The LLM must not smuggle a runtime CID as an argument_literal.
    ref = _ref("r1", "pubchem_cid", ["cid"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [],
         "argument_literals": [{"schema_arg": "cid", "literal_value": "999999"}]},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    assert result.tool_plans == []
    assert any("literal_not_allowed:cid" in p.reason for p in result.rejected_tool_plans)


def test_antibody_ref_rejected_by_default():
    ref = _ref("r1", "antibody", ["query"], source="structured_query")
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    assert result.tool_plans == []
    assert any(p.reason == "antibody_search_not_allowed" for p in result.rejected_tool_plans)


def test_antibody_ref_allowed_when_scope_enables_it():
    ref = _ref("r1", "antibody", ["query"], source="structured_query")
    result = plan_step14_tool_calls(
        llm=MockLLMProvider(), request=_req([ref], antibody_allowed=True)
    )
    plan = _plan(result, "drugbank_get_drug_references_by_drug_name_or_id")
    assert plan is not None and plan.can_invoke is True
    assert _maps(plan) == {("query", "r1")}


def test_empty_plans_when_no_valid_tool():
    ref = _ref("r1", "target", [])
    result = plan_step14_tool_calls(llm=MockLLMProvider(), request=_req([ref]))
    assert result.tool_plans == []


def test_fda_or_semantics_missing_lists_both_identity_options():
    # can_invoke=false path for FDA lists the OR-group compactly.
    ref = _ref("r1", "brand_name", ["brand_name"])
    llm = _StubLLM({"tool_plans": [
        {"tool_name": "FDA_OrangeBook_get_patent_info", "can_invoke": True,
         "argument_mappings": []},
    ]})
    result = plan_step14_tool_calls(llm=llm, request=_req([ref]))
    plan = _plan(result, "FDA_OrangeBook_get_patent_info")
    assert plan is not None and plan.can_invoke is False
    assert plan.missing_required_args == ["brand_name|application_number"]


# ── catalog is officially sourced ───────────────────────────────────────────

_EXPECTED_TOOLS = {
    "PubChem_get_associated_patents_by_CID",
    "FDA_OrangeBook_get_patent_info",
    "drugbank_get_drug_references_by_drug_name_or_id",
}


def test_catalog_contains_exactly_the_three_tools():
    names = {e["tool_name"] for e in build_step14_selection_catalog()}
    assert names == _EXPECTED_TOOLS
    assert set(STEP14_TOOL_SPECS) == _EXPECTED_TOOLS


def test_catalog_schema_source_and_required_args_come_from_signature_schema_for():
    catalog = {e["tool_name"]: e for e in build_step14_selection_catalog()}
    for name, entry in catalog.items():
        schema = signature_schema_for(name)
        assert schema is not None, f"no sourced schema for {name}"
        # schema_source labels the real provenance.
        assert entry["schema_source"] in {
            "official_schema", "signature_schema", "fallback_binding_signature",
        }
        # required args + schema arg names are taken from the sourced schema,
        # not a local hardcoded list.
        assert entry["official_required_args"] == sorted(schema.get("required") or [])
        assert entry["schema_arg_names"] == sorted(
            str(k) for k in (schema.get("properties") or {})
        )
        # No invented output fields — description is a plain string, catalog
        # carries no `output`/`returns` keys.
        assert isinstance(entry["description"], str) and entry["description"]
        assert "output" not in entry and "returns" not in entry


def test_schema_source_falls_back_to_signature_without_tooluniverse():
    # The agents-package conftest disables ToolUniverse by default (offline),
    # so with no official spec the catalog honestly labels schema_source
    # `signature_schema` and takes required args from the wrapper signature.
    from app.agents import tool_selection_policy as tsp

    catalog = {e["tool_name"]: e for e in build_step14_selection_catalog()}
    for name in _EXPECTED_TOOLS:
        assert catalog[name]["schema_source"] == "signature_schema"
        sig = tsp._signature_schema_from_binding(name)
        assert catalog[name]["official_required_args"] == sorted(sig.get("required") or [])


def test_schema_source_is_official_when_tooluniverse_spec_available(monkeypatch):
    # Opt into the official-spec path (production parity): when the TU adapter
    # returns an official parameter schema, the catalog labels schema_source
    # `official_schema`, takes required args + arg names from it, and sources
    # the description from TU.
    from app.mcp import tooluniverse_adapter

    official = {
        "description": "OFFICIAL_TU_PUBCHEM_DESCRIPTION",
        "parameter": {
            "type": "object",
            "properties": {"cid": {"type": "integer"}},
            "required": ["cid"],
        },
    }
    target = "PubChem_get_associated_patents_by_CID"

    monkeypatch.setattr(
        tooluniverse_adapter,
        "get_tool_specification",
        lambda name: official if name == target else None,
    )
    monkeypatch.setattr(
        tooluniverse_adapter,
        "get_tool_specifications",
        lambda names: {target: official} if target in names else {},
    )
    catalog = {e["tool_name"]: e for e in build_step14_selection_catalog()}
    entry = catalog[target]
    assert entry["schema_source"] == "official_schema"
    assert entry["official_required_args"] == ["cid"]
    assert entry["schema_arg_names"] == ["cid"]
    assert entry["description"] == "OFFICIAL_TU_PUBCHEM_DESCRIPTION"
    # The other two (no fake official spec here) honestly stay signature.
    for other in _EXPECTED_TOOLS - {target}:
        assert catalog[other]["schema_source"] == "signature_schema"


def test_acceptable_supports_align_with_sourced_schema_args():
    # Every acceptable_supports token must map (via supports_to_schema_arg) to
    # an argument that exists in the tool's sourced schema properties. This is
    # the alignment guarantee between Step 14's ref-role mapping and the
    # official schema — including the DrugBank drug_name_or_id→query rename.
    for name, spec in STEP14_TOOL_SPECS.items():
        schema = signature_schema_for(name)
        props = set((schema or {}).get("properties") or {})
        for token in spec.acceptable_supports:
            arg = schema_arg_for_support(name, token.upper())  # case-insensitive
            assert arg is not None, f"{name}:{token} has no schema arg mapping"
            assert arg in props, (
                f"{name}: supports token '{token}' maps to '{arg}' which is not "
                f"an official schema arg {sorted(props)}"
            )


def test_drugbank_token_rename_is_documented():
    spec = STEP14_TOOL_SPECS["drugbank_get_drug_references_by_drug_name_or_id"]
    # Official schema arg is `query`; the drug_name_or_id token maps to it and
    # the rename is documented in arg_mapping_notes.
    assert spec.supports_to_schema_arg["drug_name_or_id"] == "query"
    assert "drug_name_or_id" in spec.arg_mapping_notes
    assert "query" in spec.arg_mapping_notes["drug_name_or_id"]


# ── prompt cache layout ─────────────────────────────────────────────────────


def test_stable_prefix_excludes_run_specific_input_refs():
    ref = _ref("r_SENTINEL", "pubchem_cid", ["cid"])
    catalog = build_step14_selection_catalog()
    payload = build_step14_selection_payload(request=_req([ref]), catalog=catalog)
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP14_SELECTION_USER_PROMPT,
        schema=payload,
        system=STEP14_SELECTION_SYSTEM_PROMPT,
    )
    # The 3 tool names + sourced schema metadata live in the stable prefix;
    # run-specific data does not.
    assert "PubChem_get_associated_patents_by_CID" in stable
    assert "schema_source" in stable
    assert "r_SENTINEL" not in stable
    assert "r_SENTINEL" in dynamic


def test_stable_prefix_byte_identical_across_requests_same_catalog():
    def _sections(ref_id, user_query, candidate_id):
        ref = Step14InputRef(
            ref_id=ref_id, source_artifact="candidate_context_table",
            source_path="candidate_records[].identifiers[].id_value",
            role="pubchem_cid", candidate_id=candidate_id,
            supports_tool_args=["cid", "pubchem_cid"],
        )
        req = Step14PatentRequest(
            run_id="run_" + ref_id, user_query=user_query,
            source_artifact_refs={"candidate_context_table": "cct"},
            input_refs=[ref],
        )
        catalog = build_step14_selection_catalog()
        return build_json_prompt_sections(
            prompt=STEP14_SELECTION_USER_PROMPT,
            schema=build_step14_selection_payload(request=req, catalog=catalog),
            system=STEP14_SELECTION_SYSTEM_PROMPT,
        )

    stable_a, dyn_a = _sections("rA", "query alpha", "cand_alpha")
    stable_b, dyn_b = _sections("rB", "query beta", "cand_beta")
    assert stable_a == stable_b and stable_a
    # Dynamic suffix differs and carries the run-specific data.
    assert dyn_a != dyn_b
    for needle in ("rA", "query alpha", "cand_alpha"):
        assert needle not in stable_a
        assert needle in dyn_a


def test_stable_prefix_excludes_candidate_id_source_path_user_query_and_values():
    ref = Step14InputRef(
        ref_id="ref_SENT", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value_SENT",
        role="pubchem_cid", candidate_id="cand_SENT",
        supports_tool_args=["cid"],
    )
    req = Step14PatentRequest(
        run_id="run_SENT", user_query="USERQUERY_SENT",
        source_artifact_refs={"candidate_context_table": "cct"},
        input_refs=[ref],
    )
    catalog = build_step14_selection_catalog()
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP14_SELECTION_USER_PROMPT,
        schema=build_step14_selection_payload(request=req, catalog=catalog),
        system=STEP14_SELECTION_SYSTEM_PROMPT,
    )
    for needle in (
        "run_SENT", "ref_SENT", "cand_SENT", "USERQUERY_SENT",
        "id_value_SENT",  # source_path
    ):
        assert needle not in stable, f"{needle} leaked into stable prefix"
    # And they DO appear after the stable prefix (dynamic suffix).
    for needle in ("ref_SENT", "cand_SENT", "USERQUERY_SENT", "id_value_SENT"):
        assert needle in dynamic


def test_stable_prefix_declares_single_stage_output_shape():
    ref = _ref("r1", "pubchem_cid", ["cid"])
    catalog = build_step14_selection_catalog()
    stable, _ = build_json_prompt_sections(
        prompt=STEP14_SELECTION_USER_PROMPT,
        schema=build_step14_selection_payload(request=_req([ref]), catalog=catalog),
        system=STEP14_SELECTION_SYSTEM_PROMPT,
    )
    for needle in (
        '"tool_plans"', '"argument_mappings"', '"schema_arg"', '"input_ref_id"',
        '"can_invoke"', "schema_arg_names", "official_required_args",
    ):
        assert needle in stable, f"{needle} missing from stable prefix"


def test_prompt_cache_layout_version_is_single_stage():
    result = plan_step14_tool_calls(
        llm=MockLLMProvider(), request=_req([_ref("r1", "pubchem_cid", ["cid"])])
    )
    assert result.prompt_cache_layout_version == STEP14_SELECTION_PROMPT_CACHE_LAYOUT_VERSION
    assert result.prompt_cache_layout_version == "step14_selection_v3"
