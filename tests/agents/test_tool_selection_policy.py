from __future__ import annotations

from typing import Any

from app.agents.tool_selection_policy import (
    ToolInvocationPlan,
    build_compact_catalog,
    select_and_build_invocations,
)


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


def test_stage2_missing_required_marks_skipped_without_valid_mapping():
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
