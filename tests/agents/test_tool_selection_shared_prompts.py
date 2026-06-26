"""Centralized Stage 1 / Stage 2 tool-selection prompt contract tests.

Covers the production prompt contract requirements:

- Single source of truth for Stage 1 / Stage 2 prompt text lives in
  `app/agents/tool_selection_policy.py`. Every step agent that calls
  `select_and_build_invocations` reuses it — none of them define their
  own selector wording.
- Stage 1 prompt commits to: catalog-only, scope-filtered, no argument
  construction, no full ToolUniverse knowledge, return JSON only.
- Stage 1 payload to the LLM contains the compact catalog only — no
  `full_schema` / `parameter` / `properties` keys leak.
- Stage 2 prompt commits to: don't invent missing required IDs, leave
  missing fields in `missing_fields`, don't output `_live`, don't call
  the tool, return JSON only.
- Stage 2 payload contains the schema for the selected tool only and the
  schema has `_live` excluded.
- Different (agent, step) pairs get different catalog contents.
- Out-of-scope tools returned by a hallucinating LLM are filtered before
  Stage 2 ever sees them.
- ZINC is not present in any Step 6 catalog and never gets the
  `zinc22` label anywhere.
- Step 2 SupervisorAgent neither imports the shared prompt constants
  nor exposes any MCP catalog / tool list.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest

from app.agents.tool_selection_policy import (
    SELECTION_STAGE1_MULTI_LANE_SYSTEM_PROMPT,
    SELECTION_STAGE1_MULTI_LANE_USER_PROMPT,
    SELECTION_STAGE1_SYSTEM_PROMPT,
    SELECTION_STAGE1_USER_PROMPT,
    SELECTION_STAGE2_SYSTEM_PROMPT,
    SELECTION_STAGE2_USER_PROMPT,
    SelectionContext,
    ToolInvocationPlan,
    build_compact_catalog,
    select_and_build_invocations,
)
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService

import os


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XLSX = (
    _PROJECT_ROOT.parent / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
)


@pytest.fixture
def inventory_client():
    """LocalMCPClient backed by the real v0.2 inventory so step-scope
    rejection (ZINC is step_05 / step_09, not step_06) actually fires."""
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(_DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


# ── shared prompt content ─────────────────────────────────────────────────


def test_stage1_system_prompt_forbids_argument_construction():
    s = SELECTION_STAGE1_SYSTEM_PROMPT.lower()
    assert "do not construct arguments" in s


def test_stage1_system_prompt_marks_catalog_as_scope_filtered():
    s = SELECTION_STAGE1_SYSTEM_PROMPT.lower()
    # "scope-filtered by agent_name and step_id" is the canonical phrasing.
    assert "scope-filtered" in s
    assert "agent_name" in s and "step_id" in s
    # Catalog-only is explicit.
    assert "use only the `compact_catalog` provided" in s


def test_multilane_stage1_prompt_has_no_tool_count_cap_language():
    prompt = (
        SELECTION_STAGE1_MULTI_LANE_SYSTEM_PROMPT + "\n" +
        SELECTION_STAGE1_MULTI_LANE_USER_PROMPT
    ).lower()
    assert "allowed_tools" in prompt
    assert "tool cap" not in prompt
    assert "per-lane tool cap" not in prompt
    assert "max_tools_per_lane" not in prompt
    assert "at most 2" not in prompt


def test_stage1_system_prompt_blocks_full_tooluniverse_knowledge():
    # Collapse newlines + repeated whitespace so the substring check
    # tolerates the hard-wrapped prompt text.
    import re
    s = re.sub(r"\s+", " ", SELECTION_STAGE1_SYSTEM_PROMPT.lower())
    assert "do not draw on full-tooluniverse knowledge" in s


def test_stage1_system_prompt_allows_empty_selection_on_insufficient_context():
    s = SELECTION_STAGE1_SYSTEM_PROMPT.lower()
    assert "insufficient" in s
    assert "empty `selections`" in s or "empty selections" in s


def test_stage1_system_prompt_requires_json_only():
    s = SELECTION_STAGE1_SYSTEM_PROMPT.lower()
    assert "json" in s
    assert "no markdown" in s and "no tool calls" in s


def test_stage2_system_prompt_forbids_inventing_required_ids():
    s = SELECTION_STAGE2_SYSTEM_PROMPT.lower()
    assert "do not invent missing required" in s
    for forbidden in ("smiles", "pdb id", "uniprot", "pubchem cid", "brand", "chembl"):
        assert forbidden in s, f"Stage 2 system prompt missing mention of {forbidden!r}"


def test_stage2_system_prompt_says_leave_missing_in_missing_fields():
    s = SELECTION_STAGE2_SYSTEM_PROMPT.lower()
    assert "missing_fields" in s


def test_stage2_system_prompt_forbids_live_and_tool_execution():
    s = SELECTION_STAGE2_SYSTEM_PROMPT.lower()
    assert "_live" in s
    assert "do not call the tool" in s


def test_stage1_and_stage2_user_prompts_are_non_empty():
    assert SELECTION_STAGE1_USER_PROMPT.strip()
    assert SELECTION_STAGE2_USER_PROMPT.strip()


# ── all agents that call the selector use the shared constants ────────────


_TOOL_SELECTOR_AGENTS = (
    "app.agents.developability_agent",
    "app.agents.structure_and_design_agent",
    "app.agents.evidence_agent",
    "app.agents.patent_ip_agent",
)


def _module_source(name: str) -> str:
    return Path(importlib.import_module(name).__file__).read_text()


@pytest.mark.parametrize("module_name", _TOOL_SELECTOR_AGENTS)
def test_agents_do_not_define_their_own_selector_prompt(module_name):
    """Agents that delegate to `select_and_build_invocations` must NOT
    duplicate Stage 1 / Stage 2 prompt wording locally. The shared
    constants in `tool_selection_policy` are the only source."""
    src = _module_source(module_name).lower()
    forbidden_phrases = (
        "you are picking mcp tools",
        "construct arguments for the selected tool",
        "you are choosing mcp tools",
        "you are constructing arguments",
        "fill arguments only from the provided context",
    )
    for phrase in forbidden_phrases:
        assert phrase not in src, (
            f"{module_name} contains a local copy of selector prompt text: {phrase!r}"
        )


# ── Step 2 SupervisorAgent never sees selector prompts or MCP catalog ─────


def test_step2_supervisor_does_not_import_shared_selector_prompts():
    src = _module_source("app.agents.supervisor_agent")
    # Symbol-level: supervisor must NOT import any selector machinery.
    for sym in (
        "SELECTION_STAGE1_SYSTEM_PROMPT",
        "SELECTION_STAGE1_USER_PROMPT",
        "SELECTION_STAGE2_SYSTEM_PROMPT",
        "SELECTION_STAGE2_USER_PROMPT",
        "select_and_build_invocations",
        "build_compact_catalog",
        "tool_selection_policy",
        "compact_catalog",
    ):
        assert sym not in src, (
            f"Step 2 SupervisorAgent must not reference {sym!r}"
        )
    # Content-level: supervisor's own system prompt is allowed to mention
    # "MCP tool lists" / "ToolUniverse" in the negative privacy section
    # (it explicitly tells the model it will NOT receive them), so we
    # check the absence of selector-shaped exposure instead. The
    # supervisor must not enumerate any specific MCP tool name; the
    # canonical-ZINC-as-a-tool surface in particular must not appear.
    src_lower = src.lower()
    for forbidden in (
        "zinc_search_compounds",
        "zinc_search_by_smiles",
        "chembl_search_molecules",
        "drugprops_pains_filter",
        "europepmc_search_articles",
        "swissadme_calculate_adme",
    ):
        assert forbidden not in src_lower, (
            f"Step 2 SupervisorAgent must not expose tool {forbidden!r}"
        )


def test_step2_supervisor_module_has_no_mcp_imports():
    """AST-level guard — Step 2 must not import anything from app.mcp."""
    module_path = Path(
        importlib.import_module("app.agents.supervisor_agent").__file__
    )
    tree = ast.parse(module_path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(("app.mcp", "app.a2a")):
                bad.append(f"from {mod} import …")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("app.mcp", "app.a2a")):
                    bad.append(f"import {alias.name}")
    assert bad == [], f"Step 2 SupervisorAgent must not import MCP: {bad}"


# ── recording fake LLM that captures everything sent to it ────────────────


class _RecordingLLM:
    name = "recording"
    model = "recording-v1"

    def __init__(
        self,
        *,
        stage1_response: dict | None = None,
        stage2_response: dict | None = None,
    ) -> None:
        self.stage1_response = stage1_response or {"selections": []}
        self.stage2_response = stage2_response or {"arguments": {}}
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(
        self, prompt: str, *, schema: dict, system: str | None = None
    ) -> dict:
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        task = (schema or {}).get("task")
        if task == "tool_selection_stage_1":
            return self.stage1_response
        if task == "tool_selection_stage_2":
            return self.stage2_response
        return {}


# ── Stage 1 payload contains compact catalog ONLY (no schema leakage) ─────


def _fallback() -> list[ToolInvocationPlan]:
    return []


def test_stage1_payload_contains_no_full_schema(local_storage):
    """LLM Stage 1 must see the compact catalog only."""
    llm = _RecordingLLM(
        stage1_response={"selections": []},
    )
    # Use the no-inventory LocalMCPClient with a known binding so the
    # catalog is non-empty; deterministic_fallback gives an empty list.
    client = LocalMCPClient()
    select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=client,
        llm=llm,
        context=SelectionContext(signals={}, arg_hints={}, note="t"),
        deterministic_fallback=_fallback,
    )
    assert llm.calls, "Stage 1 LLM call expected"
    stage1 = llm.calls[0]
    schema = stage1["schema"]
    assert schema["task"] == "tool_selection_stage_1"
    assert "compact_catalog" in schema
    # No parameter schema / full_schema keys in any catalog entry.
    blob = str(schema)
    assert "full_schema" not in blob
    for cat_entry in schema["compact_catalog"]:
        assert set(cat_entry) == {
            "tool_name",
            "short_description",
            "capability_tags",
            "coarse_input_requirements",
            "step_id",
            "agent_name",
        }
        assert "properties" not in cat_entry
        assert "required" not in cat_entry
    # System prompt forwarded is the shared canonical one.
    assert stage1["system"] == SELECTION_STAGE1_SYSTEM_PROMPT


# ── Stage 2 payload contains the selected tool's schema only ──────────────


def test_stage2_payload_contains_only_selected_tool_schema(local_storage):
    """When Stage 1 picks DrugProps_pains_filter, Stage 2's `full_schema`
    must describe that tool and nothing else; `_live` must not appear."""
    llm = _RecordingLLM(
        stage1_response={
            "selections": [
                {
                    "tool_name": "DrugProps_pains_filter",
                    "selection_reason": "smiles in context",
                }
            ]
        },
        stage2_response={
            "arguments": {"smiles": "CCO"},
            "argument_construction_reason": "filled smiles",
        },
    )
    client = LocalMCPClient()
    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=client,
        llm=llm,
        context=SelectionContext(
            signals={"smiles": True}, arg_hints={"smiles": "CCO"}, note="t"
        ),
        deterministic_fallback=_fallback,
    )

    # Exactly one Stage 2 call for the survivor.
    stage2_calls = [c for c in llm.calls if c["schema"].get("task") == "tool_selection_stage_2"]
    assert len(stage2_calls) == 1
    schema = stage2_calls[0]["schema"]
    assert schema["tool_name"] == "DrugProps_pains_filter"
    full_schema = schema["full_schema"]
    assert "smiles" in full_schema["properties"]
    # `_live` MUST be filtered before the LLM sees it.
    assert "_live" not in full_schema["properties"]
    # System prompt forwarded is the shared canonical Stage 2 one.
    assert stage2_calls[0]["system"] == SELECTION_STAGE2_SYSTEM_PROMPT
    # Plan validates and survived.
    assert plans
    assert plans[0].tool_name == "DrugProps_pains_filter"
    assert plans[0].selection_policy_version == "v1"


# ── different agent/step pairs get different catalog contents ─────────────


def test_different_agent_step_pairs_get_different_catalogs():
    client = LocalMCPClient()
    cat_step5 = build_compact_catalog(
        mcp_client=client,
        agent_name="candidate_context_agent",
        step_id="step_05",
    )
    cat_step6 = build_compact_catalog(
        mcp_client=client,
        agent_name="developability_agent",
        step_id="step_06",
    )
    cat_step13 = build_compact_catalog(
        mcp_client=client,
        agent_name="evidence_agent",
        step_id="step_13",
    )
    # All non-empty in the no-inventory client; AGENT_STEP_MAP gates by
    # step alone. The sets must differ.
    names5 = {e.tool_name for e in cat_step5}
    names6 = {e.tool_name for e in cat_step6}
    names13 = {e.tool_name for e in cat_step13}
    assert names5 and names6 and names13
    # Step 5 owns SAbDab / TheraSAbDab; Step 6 should not (without
    # AGENT_STEP_MAP membership it returns []).
    cat_step6_for_step5_agent = build_compact_catalog(
        mcp_client=client,
        agent_name="candidate_context_agent",
        step_id="step_06",
    )
    assert cat_step6_for_step5_agent == []


# ── out-of-scope LLM picks get filtered before Stage 2 ────────────────────


def test_out_of_scope_llm_selections_filtered_before_stage2():
    llm = _RecordingLLM(
        stage1_response={
            "selections": [
                {"tool_name": "DrugProps_pains_filter", "selection_reason": "ok"},
                {"tool_name": "NotAToolHallucination", "selection_reason": "bad"},
                {"tool_name": "DrugProps_pains_filter", "selection_reason": "dupe"},
            ]
        },
        stage2_response={"arguments": {"smiles": "CCO"}},
    )
    client = LocalMCPClient()
    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=client,
        llm=llm,
        context=SelectionContext(
            signals={"smiles": True}, arg_hints={"smiles": "CCO"}, note="t"
        ),
        deterministic_fallback=_fallback,
    )
    # Only one survivor reaches Stage 2.
    stage2_calls = [c for c in llm.calls if c["schema"].get("task") == "tool_selection_stage_2"]
    assert len(stage2_calls) == 1
    assert stage2_calls[0]["schema"]["tool_name"] == "DrugProps_pains_filter"
    # Plan list deduped down to one entry.
    assert [p.tool_name for p in plans] == ["DrugProps_pains_filter"]


# ── ZINC scope: not in Step 6 catalog; never labeled zinc22 anywhere ──────


def test_zinc_not_in_step6_catalog_and_never_zinc22(inventory_client):
    """Inventory-scoped client: ZINC is canonical step_05 / architecture-
    override step_09 only. Developability_agent @ step_06 must NOT see
    ZINC tools at all in the compact catalog."""
    cat = build_compact_catalog(
        mcp_client=inventory_client,
        agent_name="developability_agent",
        step_id="step_06",
    )
    names = {e.tool_name for e in cat}
    for zinc_tool in (
        "ZINC_search_compounds",
        "ZINC_get_compound",
        "ZINC_search_by_smiles",
        "ZINC_search_by_properties",
        "ZINC_get_purchasable",
    ):
        assert zinc_tool not in names, f"{zinc_tool} leaked into Step 6 catalog"
    for entry in cat:
        blob = str(entry.model_dump()).lower()
        assert "zinc22" not in blob


def test_zinc_label_never_zinc22_in_any_catalog(inventory_client):
    for agent_name, step_id in (
        ("candidate_context_agent", "step_05"),
        ("developability_agent", "step_06"),
        ("structure_and_design_agent", "step_07"),
        ("structure_and_design_agent", "step_08"),
        ("structure_and_design_agent", "step_09"),
        ("evidence_agent", "step_13"),
        ("patent_ip_agent", "step_14"),
    ):
        cat = build_compact_catalog(
            mcp_client=inventory_client, agent_name=agent_name, step_id=step_id
        )
        for entry in cat:
            blob = str(entry.model_dump()).lower()
            assert "zinc22" not in blob, (
                f"zinc22 label leaked into {agent_name}/{step_id} catalog"
            )


# ── existing plan metadata fields preserved ───────────────────────────────


def test_plan_metadata_fields_preserved_through_shared_prompts():
    llm = _RecordingLLM(
        stage1_response={
            "selections": [
                {
                    "tool_name": "DrugProps_pains_filter",
                    "selection_reason": "smiles ready",
                    "priority": 2,
                    "required_context": ["smiles"],
                }
            ]
        },
        stage2_response={
            "arguments": {"smiles": "CCO"},
            "argument_construction_reason": "filled from context",
        },
    )
    client = LocalMCPClient()
    plans = select_and_build_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=client,
        llm=llm,
        context=SelectionContext(
            signals={"smiles": True}, arg_hints={"smiles": "CCO"}, note="t"
        ),
        deterministic_fallback=_fallback,
    )
    assert plans
    p = plans[0]
    assert p.selected_by == "llm"
    assert p.selection_reason == "smiles ready"
    assert p.selection_policy_version == "v1"
    assert p.argument_construction_reason == "filled from context"
    assert p.validation_status in {"ok", "warning"}
    assert isinstance(p.validation_warnings, list)
