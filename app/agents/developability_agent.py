"""DevelopabilityAgent — Step 6 with LLM-assisted progressive tool selection.

The agent still owns Step 6 logic: it builds lane contexts from Step 5
candidate records, asks the selector to choose tools from the current
Agent/Step MCP allowed list, then calls tools only through the MCP client.
Raw outputs are stored under `tool_outputs/step_06/{tool_call_id}.json` and
referenced by `tool_call_records[].tool_output_ref`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..agents.tool_selection_policy import ToolInvocationPlan
from ..agents.step_06_available_fields import project_candidate_available_fields
from ..agents.step_06_capability_registry import (
    STEP_06_CAPABILITY_BY_TOOL,
    STEP_06_CAPABILITY_REGISTRY,
)
from ..agents.step_06_interpretation import (
    aggregate_lane_risk,
    interpret_tool_payload,
    lane_summary as build_lane_summary,
)
from ..agents.step_06_runtime_value_resolver import resolve_runtime_value
from ..agents.step_06_schema_mapping_selector import (
    select_step6_schema_mapped_invocations,
    step6_live_capability_fallback,
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


_LANE_TYPES: tuple[LaneType, ...] = (
    "payload_linker_compound_liability",
    "antibody_protein_sequence_liability",
    "antigen_protein_feature_context",
    "structure_interface_quality",
    "compound_bioactivity_prior_context",
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
        scoped_tools = set(self.mcp_client.list_tools(agent_name=_AGENT_NAME, step_id=_STEP_ID))
        selection_audit: dict[str, Any] = {
            "step_06_mcp_scoped_tool_count": len(scoped_tools),
            "step_06_registry_inventory_tool_count": len(STEP_06_CAPABILITY_REGISTRY),
            "step_06_registry_supported_tool_count": sum(
                1 for c in STEP_06_CAPABILITY_REGISTRY if c.lane_type is not None
            ),
            "step_06_runtime_eligible_tools_by_candidate_lane": {},
            "step_06_stage1_catalog_tool_names": [],
            "step_06_stage1_allowed_tools_by_lane": {},
            "step_06_stage1_scope_tool_names": [],
            "step_06_stage1_disclosed_tool_names": [],
            "step_06_stage1_hidden_tools_with_reason": [],
            "step_06_stage1_disclosure_summary": {},
            "step_06_stage1_selected_tools": [],
            "step_06_stage2_schema_survivors": [],
            "step_06_stage2_mapped_tools": [],
            "step_06_runtime_resolved_tools": [],
            "step_06_executed_tools": [],
            "step_06_recorded_tool_call_tools": [],
            "step_06_suppressed_tools_with_reason": [],
            "step_06_dependency_unavailable_tools": [],
            "step_06_upstream_error_tools": [],
            "step_06_mocked_tools": [],
            "tool_selection_source_distribution": {},
            "argument_construction_source_distribution": {},
            "argument_mapping_source_distribution": {},
        }

        missing_input_flags: list[str] = []
        candidate_liabilities: list[CandidateLiability] = []
        any_lane_ran = False
        any_lane_failed_or_dep = False

        for candidate in candidates:
            lane_results: list[LaneResult] = []
            candidate_id = candidate.get("candidate_id", "unknown")
            projection = project_candidate_available_fields(candidate)
            available_fields = projection.available_fields
            modality_summary = projection.modality_summary

            # ── 1. Partition lanes into "skipped (missing input)" and "active". ──
            active_lanes: list[LaneType] = []
            for lane_type in _LANE_TYPES:
                if not _lane_active_from_modality(lane_type, modality_summary):
                    lane_results.append(
                        LaneResult(
                            lane_type=lane_type,
                            run_status="skipped",
                            input_status="missing",
                            selected_tools=[],
                            tool_call_records=[],
                            liability_flags=[],
                            lane_risk_category="unknown",
                            lane_summary=_lane_missing_summary(lane_type),
                        )
                    )
                    continue
                active_lanes.append(lane_type)

            # ── 2. One Stage 1 + one Stage 2 LLM call across all active lanes. ──
            plans_by_lane: dict[str, list[ToolInvocationPlan]] = {}
            if active_lanes:
                plans_by_lane, disclosure, selector_audit = select_step6_schema_mapped_invocations(
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_ID,
                    mcp_client=self.mcp_client,
                    llm=self.llm,
                    candidate_id=candidate_id,
                    available_fields=available_fields,
                    modality_summary=modality_summary,
                    user_query_summary=_user_query_summary(cct, candidate),
                    deterministic_fallback=step6_live_capability_fallback,
                )
                selection_audit["step_06_stage1_scope_tool_names"].extend(disclosure.scoped_tool_names)
                selection_audit["step_06_stage1_catalog_tool_names"].extend(disclosure.disclosed_tool_names)
                selection_audit["step_06_stage1_disclosed_tool_names"].extend(disclosure.disclosed_tool_names)
                selection_audit["step_06_stage1_hidden_tools_with_reason"].extend(
                    {"candidate_id": candidate_id, **item}
                    for item in disclosure.hidden_tools_with_reason
                )
                selection_audit["step_06_stage1_disclosure_summary"][candidate_id] = {
                    **disclosure.disclosure_summary,
                    "disclosure_tags": disclosure.disclosure_tags,
                }
                selection_audit["step_06_stage1_selected_tools"].extend(
                    selector_audit.get("stage1_selected_tools") or []
                )
                selection_audit["step_06_stage2_schema_survivors"].extend(
                    selector_audit.get("stage2_schema_survivors") or []
                )
                selection_audit["step_06_stage2_mapped_tools"].extend(
                    selector_audit.get("stage2_mapped_tools") or []
                )

            # ── 3. Execute the plans lane by lane and assemble lane results. ────
            for lane_type in active_lanes:
                plans = plans_by_lane.get(lane_type) or []

                tool_records: list[ToolCallRecord] = []
                lane_flags: list[dict] = []
                lane_input_status = "insufficient"
                argument_mapping_audit: list[dict] = []
                for plan in plans:
                    argument_mapping_audit.extend(plan.argument_mapping_audit)
                    tc, one_input_status, payload = self._call_lane_plan(
                        run_id=run_id,
                        candidate_id=candidate_id,
                        candidate=candidate,
                        plan=plan,
                    )
                    tool_records.append(tc)
                    selection_audit["step_06_recorded_tool_call_tools"].append(plan.tool_name)
                    if tc.run_status not in {"skipped", "not_run"}:
                        selection_audit["step_06_executed_tools"].append(plan.tool_name)
                    if plan.argument_field_refs and tc.run_status not in {"skipped", "not_run"}:
                        selection_audit["step_06_runtime_resolved_tools"].append(plan.tool_name)
                    _increment(selection_audit["tool_selection_source_distribution"], plan.tool_selection_source)
                    _increment(selection_audit["argument_construction_source_distribution"], plan.argument_construction_source)
                    _increment(selection_audit["argument_mapping_source_distribution"], plan.argument_construction_source)
                    if isinstance(payload, dict) and payload.get("status") == "upstream_error":
                        selection_audit["step_06_upstream_error_tools"].append(plan.tool_name)
                    if isinstance(payload, dict) and payload.get("status") == "mocked":
                        selection_audit["step_06_mocked_tools"].append(plan.tool_name)
                    if one_input_status == "sufficient":
                        lane_input_status = "sufficient"
                    if tc.run_status not in {"skipped", "not_run"}:
                        any_lane_ran = True
                    if tc.run_status in {"failed", "dependency_unavailable"}:
                        any_lane_failed_or_dep = True
                    if tc.run_status == "success" and payload is not None:
                        lane_flags.extend(
                            interpret_tool_payload(
                                tc.tool_name,
                                payload,
                                source_ref=tc.tool_output_ref,
                                lane_type=lane_type,
                            )
                        )

                any_success = any(r.run_status == "success" for r in tool_records)
                all_dep_unavail = bool(tool_records) and all(
                    r.run_status == "dependency_unavailable" for r in tool_records
                )
                lane_results.append(
                    LaneResult(
                        lane_type=lane_type,
                        run_status=_aggregate_lane_run_status(tool_records),
                        input_status=lane_input_status,
                        selected_tools=[p.tool_name for p in plans],
                        tool_call_records=tool_records,
                        argument_mapping_audit=argument_mapping_audit,
                        liability_flags=lane_flags,
                        lane_risk_category=aggregate_lane_risk(
                            lane_flags,
                            any_success=any_success,
                            all_dependency_unavailable=all_dep_unavail,
                        ),
                        lane_summary=_compose_lane_summary(
                            lane_type=lane_type,
                            candidate=candidate,
                            base_summary=build_lane_summary(
                                tool_records_summary=_aggregate_lane_summary(tool_records),
                                flags=lane_flags,
                                any_success=any_success,
                                lane_type=lane_type,
                            ),
                        ),
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
            selection_audit=_finalize_selection_audit(selection_audit),
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
        candidate: dict,
        plan: ToolInvocationPlan,
    ) -> tuple[ToolCallRecord, str, Any]:
        tc_id = new_tool_call_id()
        started = now_iso()
        runtime_arguments: dict[str, Any] = {}
        resolver_audit: list[dict] = []
        unresolved: list[str] = []
        for arg_name, field_ref in sorted((plan.argument_field_refs or {}).items()):
            resolved = resolve_runtime_value(candidate=candidate, field_ref=field_ref)
            resolver_audit.append({
                "schema_arg": arg_name,
                "field_ref": field_ref,
                "resolve_status": resolved.status,
                "audit_metadata": resolved.audit_metadata,
                "error_message": resolved.error_message,
            })
            if resolved.status == "resolved" and resolved.raw_value not in (None, ""):
                runtime_arguments[arg_name] = resolved.raw_value
            else:
                unresolved.append(arg_name)
        if not plan.argument_field_refs:
            runtime_arguments = dict(plan.arguments)
        input_status = "sufficient" if runtime_arguments else "insufficient"

        if plan.validation_status == "skipped" or unresolved:
            finished = now_iso()
            return ToolCallRecord(
                tool_call_id=tc_id,
                tool_name=plan.tool_name,
                agent_name=_AGENT_NAME,
                step_id=_STEP_ID,
                run_status="skipped",
                started_at=started,
                finished_at=finished,
                tool_input_summary=_tool_input_summary(
                    plan, candidate_id, resolver_audit=resolver_audit
                ),
                error_message=(
                    "tool invocation plan validation_status=skipped"
                    if plan.validation_status == "skipped"
                    else f"runtime field_ref unresolved for args: {sorted(unresolved)}"
                ),
            ), input_status, None

        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            tool_name=plan.tool_name,
            **runtime_arguments,
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
                    "input": _tool_input_summary(
                        plan, candidate_id, resolver_audit=resolver_audit
                    ),
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
            tool_input_summary=_tool_input_summary(
                plan, candidate_id, resolver_audit=resolver_audit
            ),
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        ), input_status, result.get("payload")


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
        "smiles": smiles,
        "sequence": sequence or arg_value,
        "protein_sequence": sequence or arg_value,
        "structure_file": pdb_like,
        "pdb_id_or_path": pdb_like or arg_value,
        "pdb_id": _first_identifier_value(identifiers, {"pdb_id"}) or arg_value,
        "uniprot_id": uniprot or arg_value,
        # `accession` mirrors uniprot_id so Stage 2 / deterministic
        # mapping can satisfy EBIProteins_get_features / _get_epitopes,
        # whose official TU schemas require `accession`, not `query`.
        "accession": uniprot or arg_value,
        "compound_name": compound_name or arg_value,
        "chembl_id": chembl or arg_value,
    }
    return SelectionContext(
        signals=signals,
        arg_hints={k: v for k, v in arg_hints.items() if v},
        note=f"lane_type={lane_type}; candidate_id={candidate.get('candidate_id', '')}",
    )


def _compose_lane_summary(
    *,
    lane_type: LaneType,
    candidate: dict,
    base_summary: str,
) -> str:
    """Wrap the base lane summary with candidate-level provenance hints.

    For the compound bioactivity lane, surface whether the chembl_id used
    by the lane came from a substructure search (upper-bound identity)
    rather than a name-confirmed exact match. Substructure-derived prior
    context is still valuable but should not be misread as exact identity.
    """
    if lane_type != "compound_bioactivity_prior_context":
        return base_summary
    notes = [n for n in (candidate.get("context_notes") or []) if isinstance(n, str)]
    gaps = [g for g in (candidate.get("data_gaps") or []) if isinstance(g, str)]
    substructure = any(
        "substructure-derived" in n.lower() for n in notes
    ) or any(
        "substructure_derived" in g.lower() for g in gaps
    )
    if not substructure:
        return base_summary
    return (
        f"{base_summary}; chembl_id origin=substructure-derived (upper-bound prior "
        "context, not confirmed exact identity)"
    )


def _lane_missing_summary(lane_type: LaneType) -> str:
    """Human-readable skipped-lane summary that matches the lane's real input.

    Most lanes activate off candidate ``materials`` of a specific family; the
    bioactivity prior lane activates off a typed ``chembl_id`` identifier, so
    its skipped summary points at the identifier family instead of a SMILES
    family the lane does NOT use.
    """
    if lane_type == "compound_bioactivity_prior_context":
        return "no candidate identifiers matched chembl_id family"
    if lane_type == "antigen_protein_feature_context":
        return "no candidate identifiers matched uniprot_id family"
    requirements = {
        "payload_linker_compound_liability": "SMILES",
        "antibody_protein_sequence_liability": "protein sequence",
        "structure_interface_quality": "structure file or canonical PDB ID",
    }
    return f"no candidate inputs matched {requirements.get(lane_type, 'typed input')} family"


def _lane_input(candidate: dict, lane_type: LaneType) -> Optional[str]:
    """Return the typed input that can safely drive a Step 6 lane.

    This intentionally does not treat display names as typed scientific
    inputs. A payload/linker name is not a SMILES string, an antibody name
    is not a sequence, and a target name is not a UniProt accession.
    """
    materials = _materials(candidate)
    identifiers = _identifiers(candidate)

    if lane_type == "payload_linker_compound_liability":
        return _first_material_value(
            materials, {"payload_smiles", "linker_smiles", "compound_smiles"}
        )
    if lane_type == "antibody_protein_sequence_liability":
        return _first_material_value(
            materials,
            {
                "antibody_heavy_chain_sequence",
                "antibody_light_chain_sequence",
                "target_sequence",
            },
        )
    if lane_type == "antigen_protein_feature_context":
        return _first_identifier_value(identifiers, {"uniprot_id", "accession"})
    if lane_type == "structure_interface_quality":
        return (
            _first_material_value(materials, {"structure_file", "structure_ref"})
            or _first_identifier_value(identifiers, {"pdb_id"})
        )
    if lane_type == "compound_bioactivity_prior_context":
        return (
            _first_identifier_value(identifiers, {"chembl_id"})
            or _first_material_value(materials, {"payload_smiles", "linker_smiles", "compound_smiles"})
        )
    return None


def _coverage_plan(
    tool_name: str,
    arg_hints: dict[str, Any],
    *,
    selection_source: str = "coverage_policy",
) -> ToolInvocationPlan:
    args = deterministic_arguments(tool_name, arg_hints)
    return ToolInvocationPlan(
        tool_name=tool_name,
        selection_reason=(
            "coverage category required by Step 6 production policy"
            if selection_source == "coverage_policy"
            else "deterministic fallback after Stage 1 failure or empty selection"
        ),
        arguments=args,
        argument_construction_reason="registry deterministic argument mapping",
        selected_by="deterministic_fallback",
        tool_selection_source=selection_source,  # type: ignore[arg-type]
        argument_construction_source="deterministic_mapping",
        stage2_skipped=True,
        validation_status="ok" if args else "skipped",
    )


def _apply_coverage_policy(
    plans: list[ToolInvocationPlan], eligible: list[Any], arg_hints: dict[str, Any],
) -> tuple[list[ToolInvocationPlan], list[dict]]:
    kept: list[ToolInvocationPlan] = []
    suppressed: list[dict] = []
    seen_tools: set[str] = set()
    seen_redundancy: set[str] = set()
    for plan in sorted(plans, key=lambda p: (STEP_06_CAPABILITY_BY_TOOL.get(p.tool_name).priority if STEP_06_CAPABILITY_BY_TOOL.get(p.tool_name) else 999, p.tool_name)):
        cap = STEP_06_CAPABILITY_BY_TOOL.get(plan.tool_name)
        if cap is None or plan.tool_name in seen_tools:
            continue
        if cap.redundancy_group and cap.redundancy_group in seen_redundancy:
            suppressed.append({
                "tool_name": plan.tool_name,
                "reason": f"semantic_redundancy_group:{cap.redundancy_group}",
            })
            continue
        kept.append(plan)
        seen_tools.add(plan.tool_name)
        if cap.redundancy_group:
            seen_redundancy.add(cap.redundancy_group)
    covered = {
        STEP_06_CAPABILITY_BY_TOOL[p.tool_name].coverage_category
        for p in kept if p.validation_status != "skipped" and p.tool_name in STEP_06_CAPABILITY_BY_TOOL
    }
    for cap in eligible:
        if cap.coverage_category in covered or cap.tool_name in seen_tools:
            continue
        plan = _coverage_plan(cap.tool_name, arg_hints)
        kept.append(plan)
        seen_tools.add(cap.tool_name)
        covered.add(cap.coverage_category)
    return kept, suppressed


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _finalize_selection_audit(audit: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "step_06_stage1_catalog_tool_names", "step_06_stage1_selected_tools",
        "step_06_stage2_schema_survivors", "step_06_executed_tools",
        "step_06_upstream_error_tools", "step_06_mocked_tools",
        "step_06_stage1_scope_tool_names", "step_06_stage1_disclosed_tool_names",
        "step_06_stage2_mapped_tools", "step_06_runtime_resolved_tools",
        "step_06_recorded_tool_call_tools",
    ):
        audit[key] = sorted(set(audit.get(key) or []))
    dependency = audit.get("step_06_dependency_unavailable_tools") or []
    audit["step_06_dependency_unavailable_tools"] = list({
        (d.get("candidate_id"), d.get("lane_type"), d.get("tool_name")): d
        for d in dependency
    }.values())
    return audit


def _tool_input_summary(
    plan: ToolInvocationPlan,
    candidate_id: str,
    *,
    resolver_audit: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        **{k: _short(v) for k, v in plan.arguments.items()},
        "candidate_id": candidate_id,
        "selected_by": plan.selected_by,
        "tool_selection_source": plan.tool_selection_source,
        "argument_construction_source": plan.argument_construction_source,
        "stage2_skipped": plan.stage2_skipped,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
        "argument_field_refs": dict(plan.argument_field_refs),
        "argument_mapping_audit": list(plan.argument_mapping_audit),
        "missing_required_fields": list(plan.missing_required_fields),
        "runtime_resolver_audit": resolver_audit or [],
    }


def _lane_active_from_modality(lane_type: LaneType, modality_summary: Any) -> bool:
    if lane_type == "payload_linker_compound_liability":
        return bool(
            modality_summary.has_payload_smiles
            or modality_summary.has_linker_smiles
            or modality_summary.has_compound_smiles
        )
    if lane_type == "compound_bioactivity_prior_context":
        return bool(
            modality_summary.has_payload_smiles
            or modality_summary.has_linker_smiles
            or modality_summary.has_compound_smiles
            or modality_summary.has_compound_identifier
        )
    if lane_type == "antibody_protein_sequence_liability":
        return bool(
            modality_summary.has_antibody_heavy_sequence
            or modality_summary.has_antibody_light_sequence
            or modality_summary.has_antibody_sequence
            or modality_summary.has_antigen_sequence
            or modality_summary.has_protein_sequence
            or modality_summary.has_uploaded_fasta_ref
            or modality_summary.has_cdr3_ref_or_marker
        )
    if lane_type == "antigen_protein_feature_context":
        return bool(modality_summary.has_uniprot_id)
    if lane_type == "structure_interface_quality":
        return bool(
            modality_summary.has_pdb_id
            or modality_summary.has_uploaded_structure_ref
        )
    return False


def _user_query_summary(cct: dict, candidate: dict) -> str:
    hints = cct.get("downstream_query_hints") or []
    compact_hints = []
    for item in hints:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        role = item.get("role")
        if entity and role:
            compact_hints.append(f"{role}:{entity}")
    notes = [
        n for n in (candidate.get("context_notes") or [])
        if isinstance(n, str) and len(n) <= 160
    ]
    return "; ".join([*compact_hints[:8], *notes[:4]])


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
