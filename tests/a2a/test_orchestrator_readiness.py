"""Strict semantic checks for the active Step 3 routing authority."""

from __future__ import annotations

import pytest

from app.a2a.orchestrator_readiness import (
    OrchestratorReadinessError,
    require_ready_input_readiness,
)
from tests.a2a.test_orchestrator_routing_service import _seed_inputs


def test_ready_with_optional_gap_remains_valid(
    local_storage, registry_service
):
    run_id = _seed_inputs(
        local_storage, registry_service, run_id="run_optional_readiness"
    )
    key = local_storage.run_key(
        run_id, "inputs/input_readiness_status.json"
    )
    body = local_storage.read_json(key)
    body["missing_input_checklist"] = [
        {
            "field": "structured_query.missing_slots.constraint",
            "severity": "optional",
            "message": "Optional test-only preference.",
            "category": "constraints",
            "recoverable": True,
        }
    ]
    local_storage.write_json(key, body)

    status = require_ready_input_readiness(
        run_id=run_id,
        registry=registry_service,
        storage=local_storage,
    )

    assert status.input_readiness_status == "ready"
    assert [item.severity for item in status.missing_input_checklist] == [
        "optional"
    ]


@pytest.mark.parametrize(
    "semantic_update",
    [
        {
            "missing_input_checklist": [
                {
                    "field": "warning_gap",
                    "severity": "warning",
                    "message": "test-only warning",
                    "category": "other",
                }
            ]
        },
        {"blocking_reasons": ["test-only blocker"]},
        {
            "clarification_requests": [
                {
                    "request_id": "clr_warning_test",
                    "slot_name": "other",
                    "slot_category": "other",
                    "severity": "warning",
                    "question": "Provide more information.",
                    "resolved": False,
                }
            ]
        },
    ],
)
def test_ready_semantic_inconsistencies_fail_closed(
    local_storage, registry_service, semantic_update
):
    run_id = _seed_inputs(
        local_storage, registry_service, run_id="run_semantic_readiness"
    )
    key = local_storage.run_key(
        run_id, "inputs/input_readiness_status.json"
    )
    body = local_storage.read_json(key)
    body.update(semantic_update)
    local_storage.write_json(key, body)

    with pytest.raises(
        OrchestratorReadinessError,
        match="^input_readiness_status_semantic_invalid$",
    ):
        require_ready_input_readiness(
            run_id=run_id,
            registry=registry_service,
            storage=local_storage,
        )
