"""DevelopabilityAgent — Step 6 schema-field mapped tool selection.

Step 6 projects Step 5 candidate records into LLM-safe available fields and
candidate modality summaries, discloses a modality-relevant Step 6 catalog,
asks Stage 1 to choose relevant tools, asks Stage 2 to map tool schemas to
field refs, then resolves raw values only at MCP-call runtime. Raw outputs
are stored under `tool_outputs/step_06/{tool_call_id}.json` and referenced
by `tool_call_records[].tool_output_ref`.
"""

from __future__ import annotations

from typing import Any

from ..agents.tool_selection_policy import ToolInvocationPlan
from ..agents.step_06_available_fields import (
    AvailableField,
    project_candidate_available_fields,
)
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
            "step_06_stage1_catalog_tool_names": [],
            "step_06_stage1_scope_tool_names": [],
            "step_06_stage1_disclosed_tool_names": [],
            "step_06_stage1_hidden_tools_with_reason": [],
            "step_06_stage1_disclosure_summary": {},
            "step_06_stage1_selected_tools": [],
            "step_06_stage2_schema_survivors": [],
            "step_06_stage2_mapped_tools": [],
            "step_06_stage2_uninvokable_tools": [],
            "step_06_stage2_uninvokable_tool_details": [],
            "step_06_selection_progress": [],
            "step_06_resolver_failed_tools": [],
            "step_06_runtime_chain_expanded_tools": [],
            "step_06_runtime_chain_expansion_details": [],
            "step_06_runtime_resolved_tools": [],
            "step_06_executed_tools": [],
            "step_06_recorded_tool_call_tools": [],
            "step_06_dependency_unavailable_tools": [],
            "step_06_upstream_error_tools": [],
            "step_06_mocked_tools": [],
            "tool_selection_source_distribution": {},
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
            force_fail_open = (
                modality_summary.ambiguous_or_unknown
                or not modality_summary.modality_tags
            )

            # ── 1. Partition lanes into "skipped (missing input)" and "active". ──
            active_lanes: list[LaneType] = []
            for lane_type in _LANE_TYPES:
                if not force_fail_open and not _lane_active_from_modality(lane_type, modality_summary):
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
                # `step_06_stage1_catalog_tool_names` is the LLM-visible
                # compact catalog AFTER progressive disclosure. The full
                # scoped Step 6 surface remains in
                # `step_06_stage1_scope_tool_names`.
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
                selection_audit["step_06_stage2_uninvokable_tools"].extend(
                    selector_audit.get("stage2_uninvokable_tools") or []
                )
                selection_audit["step_06_stage2_uninvokable_tool_details"].extend(
                    selector_audit.get("stage2_uninvokable_tool_details") or []
                )
                selection_audit["step_06_selection_progress"].extend(
                    selector_audit.get("selection_progress") or []
                )

            # ── 3. Execute the plans lane by lane and assemble lane results. ────
            for lane_type in active_lanes:
                plans = plans_by_lane.get(lane_type) or []
                selection_audit["step_06_selection_progress"].append({
                    "candidate_id": candidate_id,
                    "lane_type": lane_type,
                    "stage": "runtime",
                    "status": "started",
                    "selected_tool_names": [plan.tool_name for plan in plans],
                })

                tool_records: list[ToolCallRecord] = []
                lane_flags: list[dict] = []
                lane_input_status = "insufficient"
                argument_mapping_audit: list[dict] = []
                for plan in plans:
                    for plan_to_run, chain_expansion in _expand_chain_sequences_for_runtime(
                        candidate=candidate,
                        candidate_id=candidate_id,
                        lane_type=lane_type,
                        plan=plan,
                    ):
                        if chain_expansion:
                            selection_audit["step_06_runtime_chain_expanded_tools"].append(
                                plan_to_run.tool_name
                            )
                            selection_audit["step_06_runtime_chain_expansion_details"].append(
                                chain_expansion
                            )
                        argument_mapping_audit.extend(plan_to_run.argument_mapping_audit)
                        tc, one_input_status, payload = self._call_lane_plan(
                            run_id=run_id,
                            candidate_id=candidate_id,
                            candidate=candidate,
                            plan=plan_to_run,
                            chain_expansion=chain_expansion,
                        )
                        tool_records.append(tc)
                        selection_audit["step_06_recorded_tool_call_tools"].append(plan_to_run.tool_name)
                        if tc.run_status not in {"skipped", "not_run"}:
                            selection_audit["step_06_executed_tools"].append(plan_to_run.tool_name)
                        if plan_to_run.argument_field_refs and tc.run_status not in {"skipped", "not_run"}:
                            selection_audit["step_06_runtime_resolved_tools"].append(plan_to_run.tool_name)
                        if (
                            isinstance(tc.tool_input_summary, dict)
                            and any(
                                entry.get("resolve_status") in {"missing", "unresolved"}
                                for entry in tc.tool_input_summary.get("runtime_resolver_audit", [])
                            )
                        ):
                            selection_audit["step_06_resolver_failed_tools"].append(plan_to_run.tool_name)
                        _increment(selection_audit["tool_selection_source_distribution"], plan_to_run.tool_selection_source)
                        _increment(selection_audit["argument_mapping_source_distribution"], plan_to_run.argument_construction_source)
                        if isinstance(payload, dict) and payload.get("status") == "upstream_error":
                            selection_audit["step_06_upstream_error_tools"].append(plan_to_run.tool_name)
                        if isinstance(payload, dict) and payload.get("status") == "mocked":
                            selection_audit["step_06_mocked_tools"].append(plan_to_run.tool_name)
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
                any_timeout = any(
                    r.run_status in {"failed", "error"} and
                    "timeout" in str(r.error_message or "").lower()
                    for r in tool_records
                )
                runtime_status = (
                    "skipped" if not plans
                    else "timeout" if any_timeout
                    else "error" if any(r.run_status in {"failed", "dependency_unavailable"} for r in tool_records)
                    else "ok"
                )
                selection_audit["step_06_selection_progress"].append({
                    "candidate_id": candidate_id,
                    "lane_type": lane_type,
                    "stage": "runtime",
                    "status": runtime_status,
                    "selected_tool_names": [plan.tool_name for plan in plans],
                })
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
        chain_expansion: dict[str, Any] | None = None,
    ) -> tuple[ToolCallRecord, str, Any]:
        tc_id = new_tool_call_id()
        started = now_iso()
        runtime_arguments: dict[str, Any] = {}
        resolver_audit: list[dict] = []
        unresolved: list[str] = []
        for arg_name, field_ref in sorted((plan.argument_field_refs or {}).items()):
            resolved = resolve_runtime_value(
                candidate=candidate,
                field_ref=field_ref,
                storage=self.storage,
            )
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
                    plan,
                    candidate_id,
                    resolver_audit=resolver_audit,
                    chain_expansion=chain_expansion,
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
                        plan,
                        candidate_id,
                        resolver_audit=resolver_audit,
                        chain_expansion=chain_expansion,
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
                plan,
                candidate_id,
                resolver_audit=resolver_audit,
                chain_expansion=chain_expansion,
            ),
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        ), input_status, result.get("payload")


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


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _finalize_selection_audit(audit: dict[str, Any]) -> dict[str, Any]:
    uninv = audit.get("step_06_stage2_uninvokable_tools") or []
    if uninv and isinstance(uninv[0], dict):
        deduped = []
        seen: set[tuple[Any, ...]] = set()
        for item in uninv:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("candidate_id"),
                item.get("lane_type"),
                item.get("tool_name"),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        audit["step_06_stage2_uninvokable_tools"] = deduped
    elif isinstance(uninv, list):
        audit["step_06_stage2_uninvokable_tools"] = sorted(set(uninv))

    if "step_06_stage2_uninvokable_tools" in audit and not isinstance(audit["step_06_stage2_uninvokable_tools"], list):
        audit["step_06_stage2_uninvokable_tools"] = []

    for key in (
        "step_06_stage1_catalog_tool_names", "step_06_stage1_selected_tools",
        "step_06_stage2_schema_survivors", "step_06_executed_tools",
        "step_06_upstream_error_tools", "step_06_mocked_tools",
        "step_06_stage1_scope_tool_names", "step_06_stage1_disclosed_tool_names",
        "step_06_stage2_mapped_tools", "step_06_runtime_resolved_tools",
        "step_06_recorded_tool_call_tools", "step_06_resolver_failed_tools",
        "step_06_runtime_chain_expanded_tools",
        "step_06_stage2_uninvokable_tools",
    ):
        audit[key] = sorted(set(audit.get(key) or []))
    uninv_details = audit.get("step_06_stage2_uninvokable_tool_details")
    if isinstance(uninv_details, list):
        deduped_details: list[dict[str, Any]] = []
        seen = set[str]()
        for entry in uninv_details:
            if not isinstance(entry, dict):
                continue
            key = str(sorted(entry.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped_details.append(entry)
        audit["step_06_stage2_uninvokable_tool_details"] = deduped_details
    else:
        audit["step_06_stage2_uninvokable_tool_details"] = []
    chain_expansion_details = audit.get("step_06_runtime_chain_expansion_details")
    if isinstance(chain_expansion_details, list):
        deduped_chain_expansions: list[dict[str, Any]] = []
        seen = set[str]()
        for entry in chain_expansion_details:
            if not isinstance(entry, dict):
                continue
            key = str(sorted(entry.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped_chain_expansions.append(entry)
        audit["step_06_runtime_chain_expansion_details"] = deduped_chain_expansions
    else:
        audit["step_06_runtime_chain_expansion_details"] = []
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
    chain_expansion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
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
    if chain_expansion:
        summary.update({
            "runtime_chain_expansion": True,
            "original_field_ref": chain_expansion.get("original_field_ref"),
            "expanded_field_ref": chain_expansion.get("expanded_field_ref"),
            "chain_role": chain_expansion.get("chain_role"),
            "expansion_reason": chain_expansion.get("expansion_reason"),
        })
    return summary


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


def _expand_chain_sequences_for_runtime(
    *,
    candidate: dict,
    candidate_id: str,
    lane_type: LaneType,
    plan: ToolInvocationPlan,
) -> list[tuple[ToolInvocationPlan, dict[str, Any] | None]]:
    if not plan.argument_field_refs:
        return [(plan, None)]

    projection = project_candidate_available_fields(candidate)
    fields_by_ref = {field.field_ref: field for field in projection.available_fields}
    chain_ref_by_role: dict[str, str] = {}
    for field in projection.available_fields:
        if not _is_antibody_chain_sequence_field(field):
            continue
        role = _normalize_antibody_chain_role(field=field)
        if role in {"heavy", "light"}:
            chain_ref_by_role[role] = field.field_ref
    if len(chain_ref_by_role) < 2:
        return [(plan, None)]

    chainable_args: dict[str, str] = {}
    for arg, field_ref in plan.argument_field_refs.items():
        if str(arg).lower() not in {"sequence", "protein_sequence"}:
            continue
        field = fields_by_ref.get(field_ref)
        if not _is_antibody_chain_sequence_field(field):
            continue
        chain_role = _normalize_antibody_chain_role(field=field)
        if chain_role not in chain_ref_by_role:
            continue
        chainable_args[arg] = field_ref

    if not chainable_args:
        return [(plan, None)]

    expanded: list[tuple[ToolInvocationPlan, dict[str, Any]]] = []
    for chain_role, expanded_ref in chain_ref_by_role.items():
        updated_mapping = dict(plan.argument_field_refs)
        for arg in chainable_args:
            updated_mapping[arg] = expanded_ref
        reason = "single-sequence antibody tool expanded across available heavy/light chain sequence fields"
        original_ref = chainable_args[next(iter(chainable_args))]
        updated_mapping_audit: list[dict[str, Any]] = []
        chainable_arg_set = set(chainable_args)
        for entry in plan.argument_mapping_audit or []:
            if not isinstance(entry, dict):
                continue
            updated_entry = dict(entry)
            schema_arg = str(updated_entry.get("schema_arg") or "")
            if schema_arg in chainable_arg_set:
                updated_entry["field_ref"] = expanded_ref
                updated_entry["runtime_chain_expansion"] = True
                updated_entry["original_field_ref"] = chainable_args[schema_arg]
                updated_entry["expanded_field_ref"] = expanded_ref
                updated_entry["chain_role"] = chain_role
                updated_entry["expansion_reason"] = reason
            updated_mapping_audit.append(updated_entry)

        for arg in chainable_args:
            if not any(
                isinstance(entry, dict) and str(entry.get("schema_arg") or "") == arg
                for entry in updated_mapping_audit
            ):
                updated_mapping_audit.append({
                    "tool_name": plan.tool_name,
                    "schema_arg": arg,
                    "field_ref": expanded_ref,
                    "mapping_source": plan.argument_construction_source,
                    "argument_mapping_reason": "",
                    "runtime_chain_expansion": True,
                    "original_field_ref": chainable_args[arg],
                    "expanded_field_ref": expanded_ref,
                    "chain_role": chain_role,
                    "expansion_reason": reason,
                })

        updated_plan = plan.model_copy(update={"argument_field_refs": updated_mapping})
        if updated_mapping_audit:
            updated_plan = updated_plan.model_copy(
                update={"argument_mapping_audit": updated_mapping_audit}
            )
        expanded.append((
            updated_plan,
            {
                "candidate_id": candidate_id,
                "lane_type": lane_type,
                "tool_name": plan.tool_name,
                "original_field_ref": original_ref,
                "expanded_field_ref": expanded_ref,
                "chain_role": chain_role,
                "expansion_reason": reason,
            },
        ))
    return expanded


def _is_antibody_chain_sequence_field(field: AvailableField | None) -> bool:
    if field is None:
        return False
    return (
        field.field_type == "protein_sequence"
        and field.value_kind in {"protein_sequence", "uploaded_fasta_ref"}
        and _normalize_antibody_chain_role(field=field) in {"heavy", "light"}
    )


def _normalize_antibody_chain_role(field: AvailableField) -> str | None:
    if field.material_type in {"antibody_heavy_chain_sequence", "antibody_heavy_cdr3_sequence"}:
        return "heavy"
    if field.material_type in {"antibody_light_chain_sequence", "antibody_light_cdr3_sequence"}:
        return "light"
    if field.chain_role in {"heavy", "light"}:
        return field.chain_role
    return None


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
