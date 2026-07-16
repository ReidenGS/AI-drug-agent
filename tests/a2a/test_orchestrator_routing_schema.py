import pytest
from pydantic import ValidationError

from app.schemas.worker_routing_plan import (
    OrchestratorRoutingProposal,
    WorkerRoutingPlan,
)
from app.utils.ids import new_routing_decision_id, new_routing_plan_id, new_task_id


def test_empty_dispatch_proposal_parses_for_deterministic_audit():
    assert (
        OrchestratorRoutingProposal(
            loop_decision="dispatch_next_workers",
            decisions=[],
            decision_summary="No routes proposed.",
        ).decisions
        == []
    )


@pytest.mark.parametrize(
    "field", ["agent_id", "capability_id", "objective", "selection_reason"]
)
def test_empty_decision_fields_rejected(field):
    data = {
        "agent_id": "a",
        "capability_id": "c",
        "objective": "o",
        "selection_reason": "r",
        "priority": "normal",
    }
    data[field] = ""
    with pytest.raises(ValidationError):
        OrchestratorRoutingProposal(
            loop_decision="dispatch_next_workers",
            decisions=[data],
            decision_summary="Valid summary.",
        )


def test_empty_decision_summary_rejected():
    with pytest.raises(ValidationError):
        OrchestratorRoutingProposal(
            loop_decision="route_to_final_response",
            decisions=[],
            decision_summary="",
        )


def test_extra_and_negative_counts_rejected_and_llm_failed_allowed():
    with pytest.raises(ValidationError):
        OrchestratorRoutingProposal(
            loop_decision="request_user_input",
            decisions=[],
            decision_summary="Need user input.",
            extra=True,
        )
    base = dict(
        run_id="r",
        routing_plan_id="p",
        planned_at="t",
        loop_decision=None,
        routing_status="llm_failed",
        llm_selection_source="mock",
    )
    assert WorkerRoutingPlan(**base).loop_decision is None
    with pytest.raises(ValidationError):
        WorkerRoutingPlan(**base, ready_task_count=-1)


def test_generated_ids_are_unique_and_prefixed():
    for fn, prefix in (
        (new_routing_plan_id, "wrp_"),
        (new_routing_decision_id, "route_"),
        (new_task_id, "task_"),
    ):
        values = {fn() for _ in range(100)}
        assert len(values) == 100
        assert all(x.startswith(prefix) for x in values)
