"""DevelopabilityAgent — Step 6 with LLM-assisted progressive tool selection.

The agent still owns Step 6 logic: it builds lane contexts from Step 5
candidate records, asks the selector to choose tools from the current
Agent/Step MCP allowed list, then calls tools only through the MCP client.
Raw outputs are stored under `tool_outputs/step_06/{tool_call_id}.json` and
referenced by `tool_call_records[].tool_output_ref`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..agents.tool_selection_policy import (
    SelectionContext,
    ToolInvocationPlan,
    select_and_build_invocations,
)
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.step_06_structured_liability_summary import (
    CandidateLiability,
    LaneResult,
    LaneType,
    StructuredLiabilitySummary,
)
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "developability_agent"
_STEP_ID = "step_06"
_ARTIFACT_KEY = "structured_liability_summary.json"


# Lane → (representative fallback tool, material_type predicates that activate it).
_LANE_TOOLS: tuple[tuple[LaneType, str, tuple[str, ...]], ...] = (
    (
        "payload_linker_compound_liability",
        "DrugProps_pains_filter",
        ("payload_name", "linker_name", "payload_smiles", "linker_smiles", "compound_smiles"),
    ),
    (
        "antibody_protein_sequence_liability",
        "PROSITE_scan_sequence",
        ("antibody_name", "antibody_heavy_chain_sequence", "antibody_light_chain_sequence"),
    ),
    (
        "antigen_protein_feature_context",
        "EBIProteins_get_features",
        ("target_antigen_name", "target_sequence"),
    ),
    (
        "structure_interface_quality",
        "ProteinsPlus_profile_structure_quality",
        ("structure_file", "structure_ref"),
    ),
    (
        "compound_bioactivity_prior_context",
        "ChEMBL_search_activities",
        ("payload_name", "compound_name", "compound_smiles"),
    ),
)


def _material_types(candidate: dict) -> set[str]:
    return {m.get("material_type") for m in (candidate.get("materials") or [])}


def _material_value(candidate: dict, types: Iterable[str]) -> Optional[str]:
    for m in candidate.get("materials") or []:
        if m.get("material_type") in types:
            value = m.get("value")
            if value:
                return str(value)
    return None


def _materials(candidate: dict) -> list[dict]:
    return list(candidate.get("materials") or [])


def _identifiers(candidate: dict) -> list[dict]:
    return list(candidate.get("identifiers") or [])


class DevelopabilityAgent:
    name = _AGENT_NAME

    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        mcp_client: MCPClient,
        llm: LLMProvider | None = None,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client
        self.llm = llm or MockLLMProvider()

    def run(self, run_id: str) -> StructuredLiabilitySummary:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError(
                "Step 6 requires Step 5 candidate_context_table in registry"
            )

        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        candidates = cct.get("candidate_records") or []

        missing_input_flags: list[str] = []
        candidate_liabilities: list[CandidateLiability] = []
        any_lane_ran = False
        any_lane_failed_or_dep = False

        for candidate in candidates:
            lane_results: list[LaneResult] = []
            mat_types = _material_types(candidate)

            for lane_type, fallback_tool, activator_types in _LANE_TOOLS:
                activator_match = mat_types.intersection(activator_types)
                if not activator_match:
                    lane_results.append(
                        LaneResult(
                            lane_type=lane_type,
                            run_status="skipped",
                            input_status="missing",
                            selected_tools=[],
                            tool_call_records=[],
                            liability_flags=[],
                            lane_risk_category="unknown",
                            lane_summary=(
                                f"no candidate materials matched {sorted(activator_types)[0]} family"
                            ),
                        )
                    )
                    continue

                arg_value = _material_value(candidate, activator_types) or ""
                context = _selection_context(candidate, lane_type, arg_value)
                plans = select_and_build_invocations(
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_ID,
                    mcp_client=self.mcp_client,
                    llm=self.llm,
                    context=context,
                    deterministic_fallback=lambda ft=fallback_tool, av=arg_value: [
                        _fallback_plan(ft, av)
                    ],
                    deterministic_argument_mapping=_deterministic_argument_mapping,
                )
                if not plans:
                    plans = [_fallback_plan(fallback_tool, arg_value)]

                tool_records: list[ToolCallRecord] = []
                lane_input_status = "insufficient"
                for plan in plans:
                    tc, one_input_status = self._call_lane_plan(
                        run_id=run_id,
                        candidate_id=candidate.get("candidate_id", "unknown"),
                        plan=plan,
                    )
                    tool_records.append(tc)
                    if one_input_status == "sufficient":
                        lane_input_status = "sufficient"
                    if tc.run_status not in {"skipped", "not_run"}:
                        any_lane_ran = True
                    if tc.run_status in {"failed", "dependency_unavailable"}:
                        any_lane_failed_or_dep = True

                lane_results.append(
                    LaneResult(
                        lane_type=lane_type,
                        run_status=_aggregate_lane_run_status(tool_records),
                        input_status=lane_input_status,
                        selected_tools=[p.tool_name for p in plans],
                        tool_call_records=tool_records,
                        liability_flags=[],  # never embed raw payload here
                        lane_risk_category="unknown",
                        lane_summary=_aggregate_lane_summary(tool_records),
                    )
                )

            cand_status = _candidate_status(lane_results)
            candidate_liabilities.append(
                CandidateLiability(
                    candidate_id=candidate.get("candidate_id", "unknown"),
                    candidate_prefilter_status=cand_status,
                    lane_results=lane_results,
                    candidate_overall_liability_label="unknown",
                    recommended_action="insufficient_data",
                )
            )

        if not candidates:
            missing_input_flags.append("candidate_context_table.candidate_records is empty")

        prefilter_status = _summary_status(
            candidates_present=bool(candidates),
            any_lane_ran=any_lane_ran,
            any_lane_failed_or_dep=any_lane_failed_or_dep,
        )

        notes = (
            "Step 6 used LLM-assisted progressive tool selection bounded by the "
            "inventory-scoped MCP allowed tools list. Raw upstream tool outputs "
            "are stored by reference under tool_outputs/step_06/."
        )

        summary = StructuredLiabilitySummary(
            run_id=run_id,
            created_at=now_iso(),
            prefilter_status=prefilter_status,
            strict_filter_mode=False,
            candidate_liability_results=candidate_liabilities,
            missing_input_flags=missing_input_flags,
            tool_output_artifacts=_collect_output_artifacts(candidate_liabilities),
            notes=notes,
        )

        artifact_id = new_artifact_id("structured_liability_summary")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **summary.model_dump()},
        )
        self.registry.update_active(run_id, structured_liability_summary_id=artifact_id)
        self.workflow_state.mark(run_id, "step_06", "completed")
        return summary

    def _call_lane_plan(
        self,
        *,
        run_id: str,
        candidate_id: str,
        plan: ToolInvocationPlan,
    ) -> tuple[ToolCallRecord, str]:
        tc_id = new_tool_call_id()
        started = now_iso()
        input_status = "sufficient" if plan.arguments else "insufficient"

        if plan.validation_status == "skipped":
            finished = now_iso()
            return ToolCallRecord(
                tool_call_id=tc_id,
                tool_name=plan.tool_name,
                agent_name=_AGENT_NAME,
                step_id=_STEP_ID,
                run_status="skipped",
                started_at=started,
                finished_at=finished,
                tool_input_summary=_tool_input_summary(plan, candidate_id),
                error_message="tool invocation plan validation_status=skipped",
            ), input_status

        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            tool_name=plan.tool_name,
            **plan.arguments,
        )
        finished = now_iso()

        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_06", f"{tc_id}.json"
            )
            self.storage.write_json(
                output_key,
                {
                    "tool_call_id": tc_id,
                    "candidate_id": candidate_id,
                    "tool_name": plan.tool_name,
                    "input": plan.arguments,
                    "output": result["payload"],
                },
            )
            output_ref = output_key

        return ToolCallRecord(
            tool_call_id=tc_id,
            tool_name=plan.tool_name,
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            run_status=result.get("run_status", "pending"),
            started_at=started,
            finished_at=finished,
            tool_input_summary=_tool_input_summary(plan, candidate_id),
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        ), input_status


def _selection_context(candidate: dict, lane_type: LaneType, arg_value: str) -> SelectionContext:
    materials = _materials(candidate)
    identifiers = _identifiers(candidate)
    mat_types = {m.get("material_type") for m in materials}
    id_types = {i.get("id_type") for i in identifiers}

    smiles = _first_material_value(materials, {"payload_smiles", "linker_smiles", "compound_smiles"})
    compound_name = _first_material_value(materials, {"payload_name", "linker_name", "compound_name"})
    sequence = _first_material_value(
        materials,
        {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence", "target_sequence"},
    )
    pdb_like = _first_material_value(materials, {"structure_file", "structure_ref"})
    uniprot = _first_identifier_value(identifiers, {"uniprot_id"})
    chembl = _first_identifier_value(identifiers, {"chembl_id"})

    signals = {
        "smiles": bool(smiles),
        "compound_name": bool(compound_name),
        "protein_sequence": bool(sequence),
        "antibody_sequence": bool(
            mat_types.intersection({"antibody_heavy_chain_sequence", "antibody_light_chain_sequence"})
        ),
        "target_name": "target_antigen_name" in mat_types,
        "uniprot_id": bool(uniprot),
        "pdb_id": "pdb_id" in id_types,
        "structure_file": bool(pdb_like),
        "chembl_id": bool(chembl),
    }
    arg_hints: dict[str, Any] = {
        "query": arg_value,
        "smiles": smiles or arg_value,
        "sequence": sequence or arg_value,
        "protein_sequence": sequence or arg_value,
        "pdb_id_or_path": pdb_like or arg_value,
        "pdb_id": _first_identifier_value(identifiers, {"pdb_id"}) or arg_value,
        "uniprot_id": uniprot or arg_value,
        "compound_name": compound_name or arg_value,
        "chembl_id": chembl or arg_value,
    }
    return SelectionContext(
        signals=signals,
        arg_hints={k: v for k, v in arg_hints.items() if v},
        note=f"lane_type={lane_type}; candidate_id={candidate.get('candidate_id', '')}",
    )


def _fallback_plan(tool_name: str, arg_value: str) -> ToolInvocationPlan:
    return ToolInvocationPlan(
        tool_name=tool_name,
        selection_reason="deterministic Step 6 lane fallback",
        arguments=_deterministic_argument_mapping(tool_name, {"query": arg_value, "smiles": arg_value, "sequence": arg_value, "pdb_id_or_path": arg_value}),
        argument_construction_reason="deterministic lane argument mapping",
        selected_by="deterministic_fallback",
    )


def _deterministic_argument_mapping(tool_name: str, arg_hints: dict) -> dict[str, Any]:
    if tool_name == "ProteinsPlus_profile_structure_quality":
        return {"pdb_id_or_path": arg_hints.get("pdb_id_or_path") or arg_hints.get("pdb_id") or arg_hints.get("query") or ""}
    if tool_name in {"ZINC_search_by_smiles", "DrugProps_pains_filter", "DrugProps_lipinski_filter", "DrugProps_calculate_qed", "SwissADME_calculate_adme", "SwissADME_check_druglikeness", "ADMETAI_predict_toxicity", "ADMETAI_predict_physicochemical_properties"}:
        return {"smiles": arg_hints.get("smiles") or arg_hints.get("query") or ""}
    if tool_name == "PROSITE_scan_sequence":
        return {"sequence": arg_hints.get("sequence") or arg_hints.get("protein_sequence") or arg_hints.get("query") or ""}
    if tool_name == "EBIProteins_get_features":
        return {"query": arg_hints.get("uniprot_id") or arg_hints.get("query") or ""}
    if tool_name == "ChEMBL_search_activities":
        return {"query": arg_hints.get("chembl_id") or arg_hints.get("compound_name") or arg_hints.get("smiles") or arg_hints.get("query") or ""}
    return {"query": arg_hints.get("query") or arg_hints.get("compound_name") or arg_hints.get("smiles") or ""}


def _tool_input_summary(plan: ToolInvocationPlan, candidate_id: str) -> dict[str, Any]:
    return {
        **{k: _short(v) for k, v in plan.arguments.items()},
        "candidate_id": candidate_id,
        "selected_by": plan.selected_by,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
    }


def _first_material_value(materials: list[dict], types: set[str]) -> Optional[str]:
    for m in materials:
        if m.get("material_type") in types and m.get("value"):
            return str(m.get("value"))
    return None


def _first_identifier_value(identifiers: list[dict], types: set[str]) -> Optional[str]:
    for i in identifiers:
        if i.get("id_type") in types and i.get("id_value"):
            return str(i.get("id_value"))
    return None


def _aggregate_lane_run_status(records: list[ToolCallRecord]) -> str:
    if not records:
        return "skipped"
    statuses = {r.run_status for r in records}
    if "success" in statuses:
        return "partial" if len(statuses - {"success"}) else "ok"
    if statuses <= {"skipped", "not_run"}:
        return "skipped"
    if statuses & {"failed", "dependency_unavailable"}:
        return "failed"
    return "partial"


def _aggregate_lane_summary(records: list[ToolCallRecord]) -> str:
    if not records:
        return "no tool invocation planned"
    names = ", ".join(r.tool_name for r in records)
    statuses = ", ".join(sorted({r.run_status for r in records}))
    return f"selected tools [{names}] finished with statuses [{statuses}]; raw outputs stored by reference"


def _candidate_status(lane_results: list[LaneResult]) -> str:
    statuses = {lr.run_status for lr in lane_results}
    if not statuses:
        return "not_run"
    if statuses == {"ok"}:
        return "completed"
    if "ok" in statuses or "partial" in statuses:
        return "partial"
    if statuses <= {"skipped"}:
        return "not_run"
    return "partial"


def _summary_status(
    *, candidates_present: bool, any_lane_ran: bool, any_lane_failed_or_dep: bool
) -> str:
    if not candidates_present:
        return "failed"
    if not any_lane_ran:
        return "completed_with_missing_lanes"
    if any_lane_failed_or_dep:
        return "partial"
    return "completed"


def _collect_output_artifacts(cands: list[CandidateLiability]) -> list[str]:
    out: list[str] = []
    for c in cands:
        for lr in c.lane_results:
            for tc in lr.tool_call_records:
                if tc.tool_output_artifact_id:
                    out.append(tc.tool_output_artifact_id)
    return out


def _short(v: Any) -> Any:
    if isinstance(v, str) and len(v) > 200:
        return v[:200] + "…"
    return v
