"""LangGraph nodes for ADC Step 1-14.

Implementation status:
- Step 1-4: real services (deterministic).
- Step 5-6: real agents (`CandidateContextAgent`, `DevelopabilityAgent`).
- Step 7-9: real agent (`StructureAndDesignAgent`).
- Step 10-12: real deterministic services (`ScoringHandoffService`,
  `ScoringValidationService`, `RankingService`).
- Step 13-14: real agents (`EvidenceAgent`, `PatentIPAgent`), this
  iteration. Architecture goal is Step 13 ∥ Step 14 after Step 12; the MVP
  runs them sequentially in `build_step1_14_graph` for simplicity.

Per architecture v0.1 + IO Schema v0.1:
- step_01_intake          → services.intake_service.IntakeService
- step_02_structured_query→ services.structured_query_service.StructuredQueryService
                            (composes agents.supervisor_agent.SupervisorAgent)
- step_03_input_readiness → services.input_readiness_service.InputReadinessService
- step_04_workflow_setup  → services.workflow_setup_service.WorkflowSetupService
- step_05_candidate_ctx   → agents.candidate_context_agent.CandidateContextAgent
- step_06_developability  → agents.developability_agent.DevelopabilityAgent
- step_07/08/09           → agents.structure_and_design_agent.StructureAndDesignAgent
- step_10_scoring_handoff → external Yufei AEE (placeholder)
- step_11_scoring_valid   → services.scoring_validation_service.ScoringValidationService
- step_12_ranking         → services.ranking_service.RankingService
- step_13_evidence        → agents.evidence_agent.EvidenceAgent (parallel)
- step_14_patent_ip       → agents.patent_ip_agent.PatentIPAgent  (parallel)
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..agents.candidate_context_agent import CandidateContextAgent
from ..agents.developability_agent import DevelopabilityAgent
from ..agents.evidence_agent import EvidenceAgent
from ..agents.patent_ip_agent import PatentIPAgent
from ..agents.structure_and_design_agent import StructureAndDesignAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import MCPClient
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.input_readiness_service import InputReadinessService
from ..services.intake_service import IntakeService
from ..services.ranking_service import RankingService
from ..services.scoring_handoff_service import ScoringHandoffService
from ..services.scoring_validation_service import ScoringValidationService
from ..services.storage_service import Storage
from ..services.structured_query_service import StructuredQueryService
from ..services.workflow_setup_service import (
    WorkflowSetupService,
    execution_decision,
    planned_step_for,
)
from ..services.workflow_state_service import WorkflowStateService


def _load_plan(storage: Storage, run_id: str) -> dict | None:
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    if not storage.exists(key):
        return None
    return storage.read_json(key)


class PipelineState(TypedDict, total=False):
    run_id: str
    intake_request: dict           # input fed at graph entry (raw_user_query, context, files)
    artifacts: dict[str, str]      # artifact_type -> artifact_id
    results: dict[str, Any]        # step-key -> result payload (summary)


def make_node_step_01(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    intake = IntakeService(storage=storage, registry=registry, workflow_state=workflow_state)

    def node_step_01(state: PipelineState) -> PipelineState:
        req = dict(state.get("intake_request") or {})
        rec = intake.submit(**req)
        out = dict(state)
        out["run_id"] = rec.run_id
        artifacts = dict(out.get("artifacts") or {})
        artifacts["raw_request_record"] = registry.get(rec.run_id).active_artifacts.raw_request_record_id or ""
        out["artifacts"] = artifacts
        results = dict(out.get("results") or {})
        results["step_01"] = {"run_id": rec.run_id}
        out["results"] = results
        return out

    return node_step_01


def make_node_step_02(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    supervisor: SupervisorAgent,
):
    svc = StructuredQueryService(
        storage=storage, registry=registry, workflow_state=workflow_state, supervisor=supervisor
    )

    def node_step_02(state: PipelineState) -> PipelineState:
        sq = svc.parse(state["run_id"])
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["structured_query"] = (
            registry.get(state["run_id"]).active_artifacts.structured_query_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_02"] = sq.model_dump()
        return out

    return node_step_02


def make_node_step_03(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    svc = InputReadinessService(storage=storage, registry=registry, workflow_state=workflow_state)

    def node_step_03(state: PipelineState) -> PipelineState:
        ready = svc.check(state["run_id"])
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["input_readiness_status"] = (
            registry.get(state["run_id"]).active_artifacts.input_readiness_status_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_03"] = {"input_readiness_status": ready.input_readiness_status}
        return out

    return node_step_03


def make_node_step_04(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    svc = WorkflowSetupService(storage=storage, registry=registry, workflow_state=workflow_state)

    def node_step_04(state: PipelineState) -> PipelineState:
        plan = svc.plan(state["run_id"])
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["run_step_plan"] = (
            registry.get(state["run_id"]).active_artifacts.run_step_plan_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_04"] = {"plan_status": plan.plan_status}
        return out

    return node_step_04


def make_node_step_05(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
):
    agent = CandidateContextAgent(
        storage=storage,
        registry=registry,
        workflow_state=workflow_state,
        mcp_client=mcp_client,
        llm=llm,
    )

    def node_step_05(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, "step_05_candidate_context")
        if not decision.allow:
            # `workflow_state` schema has no `blocked` status — we reuse
            # `skipped` and surface the reason in `results.step_05` so the
            # caller cannot mistake gated runs for executed runs.
            workflow_state.mark(run_id, "step_05", "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"]["step_05"] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out

        table = agent.run(run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["candidate_context_table"] = (
            registry.get(run_id).active_artifacts.candidate_context_table_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_05"] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            "context_build_status": table.context_build_status,
            "candidate_count": len(table.candidate_records),
            "tool_call_count": len(table.tool_call_records),
        }
        return out

    return node_step_05


def make_node_step_06(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
):
    agent = DevelopabilityAgent(
        storage=storage,
        registry=registry,
        workflow_state=workflow_state,
        mcp_client=mcp_client,
        llm=llm or MockLLMProvider(),
    )

    def node_step_06(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, "step_06_developability")
        if not decision.allow:
            workflow_state.mark(run_id, "step_06", "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"]["step_06"] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out

        summary = agent.run(run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["structured_liability_summary"] = (
            registry.get(run_id).active_artifacts.structured_liability_summary_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_06"] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            "prefilter_status": summary.prefilter_status,
            "candidate_count": len(summary.candidate_liability_results),
        }
        return out

    return node_step_06


def _make_structure_node(
    *,
    step_key: str,
    step_id_for_decision: str,
    artifact_key: str,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    runner,
    llm: LLMProvider | None = None,
):
    """Shared factory for Step 7/8/9 nodes. `runner(agent, run_id) -> result`."""
    agent = StructureAndDesignAgent(
        storage=storage,
        registry=registry,
        workflow_state=workflow_state,
        mcp_client=mcp_client,
        llm=llm or MockLLMProvider(),
    )

    def node(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, step_id_for_decision)
        if not decision.allow:
            workflow_state.mark(run_id, step_key, "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"][step_key] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out

        result = runner(agent, run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        active = registry.get(run_id).active_artifacts
        out["artifacts"][artifact_key] = getattr(active, f"{artifact_key}_id") or ""
        out.setdefault("results", {})
        out["results"][step_key] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            "status_summary": _status_summary(result),
        }
        return out

    return node


def _status_summary(result) -> dict:
    """Compact step-result summary for the graph state log."""
    out = {}
    for attr in (
        "structure_preparation_status",
        "structure_modeling_status",
        "screening_status",
    ):
        v = getattr(result, attr, None)
        if v is not None:
            out[attr] = v
    return out


def make_node_step_07(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
):
    return _make_structure_node(
        step_key="step_07",
        step_id_for_decision="step_07_structure_input",
        artifact_key="prepared_structure_input_package",
        storage=storage, registry=registry, workflow_state=workflow_state, mcp_client=mcp_client,
        runner=lambda agent, run_id: agent.run_step_7(run_id),
    )


def make_node_step_08(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
):
    return _make_structure_node(
        step_key="step_08",
        step_id_for_decision="step_08_structure_evaluation",
        artifact_key="structure_prediction_and_interface_results",
        storage=storage, registry=registry, workflow_state=workflow_state, mcp_client=mcp_client,
        runner=lambda agent, run_id: agent.run_step_8(run_id),
    )


def make_node_step_09(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
):
    return _make_structure_node(
        step_key="step_09",
        step_id_for_decision="step_09_structure_design",
        artifact_key="structure_variant_and_compound_screening",
        storage=storage, registry=registry, workflow_state=workflow_state, mcp_client=mcp_client,
        runner=lambda agent, run_id: agent.run_step_9(run_id),
        llm=llm,
    )


def _make_gated_service_node(
    *,
    step_key: str,
    step_id_for_decision: str,
    artifact_attr: str,
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    runner,
    artifact_state_key: str,
    status_field: str,
):
    """Shared factory for deterministic-service nodes (Step 10/11/12).

    `runner(run_id) -> result_model`. The status_field is read off the result
    to surface a short summary in the graph state.
    """

    def node(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, step_id_for_decision)
        if not decision.allow:
            workflow_state.mark(run_id, step_key, "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"][step_key] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out

        result = runner(run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        active = registry.get(run_id).active_artifacts
        out["artifacts"][artifact_state_key] = getattr(active, artifact_attr) or ""
        out.setdefault("results", {})
        out["results"][step_key] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            status_field: getattr(result, status_field, None),
        }
        return out

    return node


def make_node_step_10(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    svc = ScoringHandoffService(storage=storage, registry=registry, workflow_state=workflow_state)
    return _make_gated_service_node(
        step_key="step_10",
        step_id_for_decision="step_10_scoring_handoff",
        artifact_attr="scoring_handoff_id",
        artifact_state_key="scoring_handoff_package",
        status_field="handoff_status",
        storage=storage, registry=registry, workflow_state=workflow_state,
        runner=lambda run_id: svc.prepare(run_id),
    )


def make_node_step_11(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    svc = ScoringValidationService(storage=storage, registry=registry, workflow_state=workflow_state)
    return _make_gated_service_node(
        step_key="step_11",
        step_id_for_decision="step_11_scoring_validation",
        artifact_attr="scoring_validation_id",
        artifact_state_key="scoring_validation",
        status_field="validation_status",
        storage=storage, registry=registry, workflow_state=workflow_state,
        runner=lambda run_id: svc.validate(run_id),
    )


def make_node_step_12(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
):
    svc = RankingService(storage=storage, registry=registry, workflow_state=workflow_state)
    return _make_gated_service_node(
        step_key="step_12",
        step_id_for_decision="step_12_ranking",
        artifact_attr="ranking_table_id",
        artifact_state_key="ranking_table",
        status_field="ranking_status",
        storage=storage, registry=registry, workflow_state=workflow_state,
        runner=lambda run_id: svc.build_ranking_table(run_id),
    )


def make_node_step_13(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
):
    agent = EvidenceAgent(
        storage=storage, registry=registry,
        workflow_state=workflow_state, mcp_client=mcp_client,
        llm=llm or MockLLMProvider(),
    )

    def node(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, "step_13_evidence")
        if not decision.allow:
            workflow_state.mark(run_id, "step_13", "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"]["step_13"] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out
        table = agent.run(run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["scientific_evidence_table"] = (
            registry.get(run_id).active_artifacts.scientific_evidence_table_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_13"] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            "review_status": table.review_status,
            "evidence_record_count": len(table.evidence_records),
            "tool_call_count": len(table.tool_call_records),
        }
        return out

    return node


def make_node_step_14(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    mcp_client: MCPClient,
    llm: LLMProvider | None = None,
):
    agent = PatentIPAgent(
        storage=storage, registry=registry,
        workflow_state=workflow_state, mcp_client=mcp_client,
        llm=llm or MockLLMProvider(),
    )

    def node(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        plan = _load_plan(storage, run_id)
        decision = execution_decision(plan, "step_14_patent_ip")
        if not decision.allow:
            workflow_state.mark(run_id, "step_14", "skipped")
            out = dict(state)
            out.setdefault("results", {})
            out["results"]["step_14"] = {
                "executed": False,
                "plan_status": decision.plan_status,
                "planned_status": decision.planned_status,
                "reason": decision.reason,
            }
            return out
        table = agent.run(run_id)
        out = dict(state)
        out.setdefault("artifacts", {})
        out["artifacts"]["patent_prior_art_table"] = (
            registry.get(run_id).active_artifacts.patent_prior_art_table_id or ""
        )
        out.setdefault("results", {})
        out["results"]["step_14"] = {
            "executed": True,
            "plan_status": decision.plan_status,
            "planned_status": decision.planned_status,
            "patent_review_status": table.patent_review_status,
            "patent_record_count": len(table.patent_records),
            "tool_call_count": len(table.tool_call_records),
        }
        return out

    return node
