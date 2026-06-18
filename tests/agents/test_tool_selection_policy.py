from __future__ import annotations

from typing import Any

import pytest

from app.agents.tool_selection_policy import (
    CAPABILITY_REGISTRY,
    SelectionContext,
    ToolInvocationPlan,
    build_compact_catalog,
    select_and_build_invocations,
    signature_schema_for,
)
from app.mcp import tooluniverse_adapter


class _Client:
    def __init__(self, tools: list[str]):
        self.tools = tools

    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]:
        return list(self.tools)

    def call_tool(self, *, agent_name: str, step_id: str, tool_name: str, **kwargs: Any) -> dict:
        return {"run_status": "success", "payload": {"tool_name": tool_name, "kwargs": kwargs}}


class _LLM:
    name = "test"
    model = "test"

    def __init__(self, stage1: dict, stage2: dict | None = None):
        self.stage1 = stage1
        self.stage2 = stage2 or {"arguments": {"pdb_id_or_path": "1N8Z"}}
        self.stage2_payloads: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        if schema.get("task") == "tool_selection_stage_1":
            return self.stage1
        if schema.get("task") == "tool_selection_stage_2":
            self.stage2_payloads.append(schema)
            return self.stage2
        return {}


def _fallback() -> list[ToolInvocationPlan]:
    return [
        ToolInvocationPlan(
            tool_name="DrugProps_pains_filter",
            selection_reason="fallback",
            arguments={},
            selected_by="deterministic_fallback",
        )
    ]


def test_compact_catalog_does_not_include_full_schema():
    catalog = build_compact_catalog(
        mcp_client=_Client(["ProteinsPlus_profile_structure_quality"]),
        agent_name="developability_agent",
        step_id="step_06",
    )

    dumped = catalog[0].model_dump()
    assert dumped["tool_name"] == "ProteinsPlus_profile_structure_quality"
    assert "full_schema" not in dumped
    assert "properties" not in dumped
    assert "required" not in dumped


def test_stage1_filters_unknown_out_of_scope_and_duplicate_tools():
    llm = _LLM(
        {
            "selections": [
                {"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "ok"},
                {"tool_name": "NotARealTool", "selection_reason": "bad"},
                {"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "dupe"},
            ]
        }
    )

    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_Client(["ProteinsPlus_profile_structure_quality"]),
        llm=llm,
        context=__import__("app.agents.tool_selection_policy", fromlist=["SelectionContext"]).SelectionContext(
            signals={"pdb_id": True},
            arg_hints={"pdb_id_or_path": "1N8Z"},
        ),
        deterministic_fallback=_fallback,
    )

    assert [p.tool_name for p in plans] == ["ProteinsPlus_profile_structure_quality"]
    assert plans[0].selected_by == "llm"
    assert plans[0].arguments == {"pdb_id_or_path": "1N8Z"}


def test_empty_stage1_uses_deterministic_fallback():
    llm = _LLM({"selections": []})

    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_Client(["ProteinsPlus_profile_structure_quality"]),
        llm=llm,
        context=__import__("app.agents.tool_selection_policy", fromlist=["SelectionContext"]).SelectionContext(
            signals={"pdb_id": True},
            arg_hints={"pdb_id_or_path": "1N8Z"},
        ),
        deterministic_fallback=_fallback,
    )

    assert plans[0].selected_by == "deterministic_fallback"
    assert plans[0].tool_name == "DrugProps_pains_filter"


def test_stage2_only_receives_selected_tool_schema():
    llm = _LLM(
        {
            "selections": [
                {"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "structure"}
            ]
        }
    )

    select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_Client(["ProteinsPlus_profile_structure_quality", "DrugProps_pains_filter"]),
        llm=llm,
        context=__import__("app.agents.tool_selection_policy", fromlist=["SelectionContext"]).SelectionContext(
            signals={"pdb_id": True},
            arg_hints={"pdb_id_or_path": "1N8Z"},
        ),
        deterministic_fallback=_fallback,
    )

    assert len(llm.stage2_payloads) == 1
    payload = llm.stage2_payloads[0]
    assert payload["tool_name"] == "ProteinsPlus_profile_structure_quality"
    assert "full_schema" in payload
    assert "DrugProps_pains_filter" not in str(payload["full_schema"])


def test_stage2_missing_required_marks_skipped_without_valid_mapping(monkeypatch):
    """When the official schema declares `pdb_id` required and the LLM
    sends none, Stage 2 marks the plan `skipped` with a clear warning.

    Uses a fake universe so the official-schema-first path exercises a
    spec that DOES have a required arg (wrappers themselves now soak
    extra kwargs via `**_extra` and default most positional args, so
    the signature-fallback path no longer carries a useful `required`
    list for this assertion)."""
    fake = _FakeUniverseForMetadata(
        specs={
            "ProteinsPlus_profile_structure_quality": {
                "name": "ProteinsPlus_profile_structure_quality",
                "description": "TU official",
                "parameter": {
                    "type": "object",
                    "properties": {
                        "pdb_id": {"type": "string", "required": True},
                        "setting": {"type": "string"},
                    },
                    "required": ["pdb_id"],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)
    llm = _LLM(
        {
            "selections": [
                {"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "structure"}
            ]
        },
        stage2={"arguments": {}, "argument_construction_reason": "no context"},
    )

    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_Client(["ProteinsPlus_profile_structure_quality"]),
        llm=llm,
        context=__import__("app.agents.tool_selection_policy", fromlist=["SelectionContext"]).SelectionContext(
            signals={"pdb_id": True},
            arg_hints={},
        ),
        deterministic_fallback=_fallback,
    )

    assert plans[0].tool_name == "ProteinsPlus_profile_structure_quality"
    assert plans[0].validation_status == "skipped"
    assert any("required argument" in w for w in plans[0].validation_warnings)


# ── Stage 1 + Stage 2: ToolUniverse official metadata is the primary source ──


class _FakeUniverseForMetadata:
    """Minimal stand-in matching `_get_universe()`'s API surface."""

    def __init__(
        self,
        *,
        specs: dict[str, dict] | None = None,
        required: dict[str, list[str]] | None = None,
        names: list[str] | None = None,
        raise_on_specs: bool = False,
    ) -> None:
        self._specs = specs or {}
        self._required = required or {}
        self._names = names or list(self._specs.keys())
        self._raise = raise_on_specs
        self.spec_lookups: list[tuple[str, ...]] = []
        self.required_lookups: list[str] = []
        self.run_calls: list[dict] = []

    def load_tools(self, **_kw: Any) -> None:
        return None

    def get_available_tools(self, name_only: bool = True) -> list[str]:
        return list(self._names)

    def get_tool_specification_by_names(self, names: list[str]) -> list[dict]:
        self.spec_lookups.append(tuple(names))
        if self._raise:
            raise RuntimeError("simulated TU metadata blowup")
        return [self._specs[n] for n in names if n in self._specs]

    def get_required_parameters(self, name: str) -> list[str]:
        self.required_lookups.append(name)
        return list(self._required.get(name, []))

    def run_one_function(self, fc, **_kw):
        self.run_calls.append(fc)
        return {"status": "ok"}


def _install_fake(monkeypatch, fake: _FakeUniverseForMetadata) -> None:
    tooluniverse_adapter._reset_for_tests()
    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", lambda: fake)


def test_compact_catalog_prefers_official_description(monkeypatch):
    """ToolUniverse description wins over CAPABILITY_REGISTRY hand-written line."""
    fake = _FakeUniverseForMetadata(
        specs={
            "ChEMBL_search_molecules": {
                "name": "ChEMBL_search_molecules",
                "description": "TU official: query ChEMBL for molecules by name.",
                "parameter": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "required": True},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client(["ChEMBL_search_molecules"]),
        agent_name="candidate_context_agent",
        step_id="step_05",
    )
    assert len(catalog) == 1
    entry = catalog[0]
    assert entry.short_description.startswith("TU official")
    # Project-specific hints still come from CAPABILITY_REGISTRY.
    registry = CAPABILITY_REGISTRY["ChEMBL_search_molecules"]
    assert entry.capability_tags == list(registry["capability_tags"])
    assert entry.coarse_input_requirements == list(
        registry["coarse_input_requirements"]
    )
    # And the bulk lookup was scoped to the allowed-set only.
    assert fake.spec_lookups == [("ChEMBL_search_molecules",)]


def test_compact_catalog_does_not_emit_full_schema(monkeypatch):
    fake = _FakeUniverseForMetadata(
        specs={
            "ChEMBL_search_molecules": {
                "name": "ChEMBL_search_molecules",
                "description": "TU official",
                "parameter": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "required": True}},
                    "required": ["query"],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client(["ChEMBL_search_molecules"]),
        agent_name="candidate_context_agent",
        step_id="step_05",
    )
    dumped = catalog[0].model_dump()
    assert set(dumped) == {
        "tool_name",
        "short_description",
        "capability_tags",
        "coarse_input_requirements",
        "step_id",
        "agent_name",
    }
    assert "properties" not in str(dumped)
    assert "required" not in dumped


def test_compact_catalog_falls_back_to_registry_when_tu_unknown(monkeypatch):
    """Tool TU does NOT carry → CAPABILITY_REGISTRY description used."""
    fake = _FakeUniverseForMetadata(specs={})  # TU recognizes nothing
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client(["DrugProps_pains_filter"]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    registry = CAPABILITY_REGISTRY["DrugProps_pains_filter"]
    assert catalog[0].short_description == registry["short_description"]


def test_compact_catalog_falls_back_to_generic_when_no_registry_no_tu():
    """Custom-service / placeholder name → generic de-snake-cased fallback."""
    catalog = build_compact_catalog(
        mcp_client=_Client(["UnknownPlaceholderTool"]),
        agent_name="candidate_context_agent",
        step_id="step_05",
    )
    assert catalog[0].short_description == "UnknownPlaceholderTool".replace("_", " ")
    assert catalog[0].capability_tags == ["step_05", "candidate_context_agent"]


def test_compact_catalog_survives_tu_import_error(monkeypatch):
    """If `_get_universe` raises (TU not installed), catalog still builds."""
    catalog = build_compact_catalog(
        mcp_client=_Client(["DrugProps_pains_filter"]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    registry = CAPABILITY_REGISTRY["DrugProps_pains_filter"]
    assert catalog[0].short_description == registry["short_description"]


def test_compact_catalog_survives_tu_metadata_exception(monkeypatch):
    """A `get_tool_specification_by_names` blowup falls through to registry."""
    fake = _FakeUniverseForMetadata(
        specs={"DrugProps_pains_filter": {}}, raise_on_specs=True
    )
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client(["DrugProps_pains_filter"]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    registry = CAPABILITY_REGISTRY["DrugProps_pains_filter"]
    assert catalog[0].short_description == registry["short_description"]


def test_compact_catalog_metadata_scope_limited_to_allowed_set(monkeypatch):
    """Selector must NOT query metadata for tools outside the allowed list."""
    fake = _FakeUniverseForMetadata(specs={})
    _install_fake(monkeypatch, fake)
    build_compact_catalog(
        mcp_client=_Client(["DrugProps_pains_filter", "DrugProps_lipinski_filter"]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    assert len(fake.spec_lookups) == 1
    queried = set(fake.spec_lookups[0])
    assert queried == {"DrugProps_pains_filter", "DrugProps_lipinski_filter"}


def test_signature_schema_prefers_official_over_signature(monkeypatch):
    """Stage 2 schema for a TU-known tool comes from the spec, not inspect."""
    official_schema = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "required": True},
            "rules": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "operation": {
                "type": "string",
                "enum": ["check_druglikeness"],
                "required": True,
            },
        },
        "required": ["operation", "smiles"],
    }
    fake = _FakeUniverseForMetadata(
        specs={
            "SwissADME_check_druglikeness": {
                "name": "SwissADME_check_druglikeness",
                "description": "TU official",
                "parameter": official_schema,
            }
        },
        required={"SwissADME_check_druglikeness": ["operation", "smiles"]},
    )
    _install_fake(monkeypatch, fake)
    schema = signature_schema_for("SwissADME_check_druglikeness")
    assert schema is not None
    assert set(schema["required"]) == {"operation", "smiles"}
    # `_live` must NEVER leak to the LLM.
    assert "_live" not in schema["properties"]
    # `["array", "null"]` collapses to "array".
    assert schema["properties"]["rules"]["type"] == "array"
    # Enums propagated for stricter LLM coercion.
    assert schema["properties"]["operation"]["enum"] == ["check_druglikeness"]


def test_signature_schema_falls_back_to_inspect_when_tu_unknown(monkeypatch):
    """Tool not in TU → fall back to wrapper signature (unchanged behavior)."""
    fake = _FakeUniverseForMetadata(specs={})
    _install_fake(monkeypatch, fake)
    schema = signature_schema_for("DrugProps_pains_filter")
    assert schema is not None
    # Wrapper sig fallback: `smiles` exposed, `_live` filtered out.
    assert "smiles" in schema["properties"]
    assert "_live" not in schema["properties"]


def test_signature_schema_survives_tu_import_error():
    """TU not installed → no crash; falls back to signature."""
    schema = signature_schema_for("DrugProps_pains_filter")
    assert schema is not None
    assert "smiles" in schema["properties"]


def test_stage2_only_queries_spec_for_selected_tools(monkeypatch):
    """Progressive disclosure: Stage 2 must not pre-fetch unselected tools."""
    fake = _FakeUniverseForMetadata(
        specs={
            "ChEMBL_search_molecules": {
                "name": "ChEMBL_search_molecules",
                "description": "TU official",
                "parameter": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "required": True}},
                    "required": ["query"],
                },
            },
            "DrugProps_pains_filter": {
                "name": "DrugProps_pains_filter",
                "description": "TU official",
                "parameter": {
                    "type": "object",
                    "properties": {"smiles": {"type": "string", "required": True}},
                    "required": ["smiles"],
                },
            },
        }
    )
    _install_fake(monkeypatch, fake)
    llm = _LLM(
        {
            "selections": [
                {"tool_name": "ChEMBL_search_molecules", "selection_reason": "ok"}
            ]
        },
        stage2={"arguments": {"query": "imatinib"}},
    )
    select_and_build_invocations(
        agent_name="candidate_context_agent",
        step_id="step_05",
        mcp_client=_Client(["ChEMBL_search_molecules", "DrugProps_pains_filter"]),
        llm=llm,
        context=SelectionContext(signals={}, arg_hints={"query": "imatinib"}),
        deterministic_fallback=_fallback,
    )
    # Exactly two metadata lookups: 1 bulk for Stage 1 catalog,
    # 1 single-tool spec for Stage 2 (the selected survivor).
    bulk_lookups = [s for s in fake.spec_lookups if len(s) >= 1]
    stage2_lookups = [s for s in fake.spec_lookups if len(s) == 1]
    assert any(
        set(s) == {"ChEMBL_search_molecules", "DrugProps_pains_filter"}
        for s in bulk_lookups
    ), fake.spec_lookups
    assert ("ChEMBL_search_molecules",) in stage2_lookups
    # DrugProps_pains_filter spec was never fetched as a single-tool Stage-2 lookup.
    assert ("DrugProps_pains_filter",) not in stage2_lookups


def test_out_of_scope_tool_never_gets_metadata_query(monkeypatch):
    """A tool the MCP client does not list must not surface in any TU lookup."""
    fake = _FakeUniverseForMetadata(specs={})
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client(["DrugProps_pains_filter"]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    assert [c.tool_name for c in catalog] == ["DrugProps_pains_filter"]
    for lookup in fake.spec_lookups:
        assert "AlphaMissense_get_variant_score" not in lookup
        assert "MultiAgentLiteratureSearch" not in lookup


def test_get_required_parameters_passthrough(monkeypatch):
    fake = _FakeUniverseForMetadata(
        required={"SwissADME_calculate_adme": ["operation", "smiles"]}
    )
    _install_fake(monkeypatch, fake)
    assert tooluniverse_adapter.get_required_parameters(
        "SwissADME_calculate_adme"
    ) == ["operation", "smiles"]
    assert fake.required_lookups == ["SwissADME_calculate_adme"]


def test_get_required_parameters_falls_back_when_tu_unavailable():
    # Default tests/agents/conftest.py makes _get_universe raise → adapter
    # swallows it and returns [].
    assert tooluniverse_adapter.get_required_parameters("anything") == []


def test_get_tool_specification_falls_back_when_tu_unavailable():
    assert tooluniverse_adapter.get_tool_specification("anything") is None
    assert tooluniverse_adapter.get_tool_specifications(["a", "b"]) == {}


# ── Newly registered (Step 6/8/21) tools — official metadata first ──────────


@pytest.mark.parametrize(
    "tool_name,required_decl",
    [
        ("RNAcentral_search", ["query"]),
        ("Rfam_get_family", ["operation", "family_id"]),
        ("DNA_calculate_gc_content", ["operation", "sequence"]),
        ("dynamic_package_discovery", ["requirements"]),
        ("embedding_database_search", ["database_name", "query"]),
    ],
)
def test_new_tools_stage1_description_from_official_spec(
    monkeypatch, tool_name, required_decl
):
    """For the 18 newly registered tools, Stage 1 catalog uses TU description."""
    fake = _FakeUniverseForMetadata(
        specs={
            tool_name: {
                "name": tool_name,
                "description": f"TU official line for {tool_name}.",
                "parameter": {
                    "type": "object",
                    "properties": {p: {"type": "string"} for p in required_decl},
                    "required": list(required_decl),
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)
    catalog = build_compact_catalog(
        mcp_client=_Client([tool_name]),
        agent_name="developability_agent",
        step_id="step_06",
    )
    assert len(catalog) == 1
    assert catalog[0].short_description.startswith("TU official line for")
    # Stage 1 must not leak the schema.
    dumped = catalog[0].model_dump()
    assert "properties" not in str(dumped)
    assert "required" not in dumped


def test_new_tools_stage2_schema_from_official_spec(monkeypatch):
    """Stage 2 schema for a TU-known tool comes from TU's parameter block."""
    fake = _FakeUniverseForMetadata(
        specs={
            "Rfam_get_family": {
                "name": "Rfam_get_family",
                "description": "TU official",
                "parameter": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["get_family"],
                            "required": True,
                        },
                        "family_id": {"type": "string", "required": True},
                        "format": {
                            "type": "string",
                            "enum": ["json", "xml"],
                            "default": "json",
                        },
                    },
                    "required": ["operation", "family_id"],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)
    schema = signature_schema_for("Rfam_get_family")
    assert schema is not None
    assert set(schema["required"]) == {"operation", "family_id"}
    assert "_live" not in schema["properties"]
    assert schema["properties"]["operation"]["enum"] == ["get_family"]
    assert schema["properties"]["format"]["enum"] == ["json", "xml"]
