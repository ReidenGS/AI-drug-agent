"""Verify `new_run_id()` collisions are vanishingly rare and that two intakes
don't overwrite each other on disk."""

from __future__ import annotations

import concurrent.futures
import re

from app.services.intake_service import IntakeService
from app.utils.ids import new_run_id


# Format: run_YYYYMMDD_<8 hex>. Keep `run_` prefix mandatory so downstream
# string checks ("startswith('run_')") keep working.
_RUN_ID_RE = re.compile(r"^run_\d{8}_[0-9a-f]{8}$")


def test_new_run_id_format():
    rid = new_run_id()
    assert _RUN_ID_RE.match(rid), f"unexpected run_id format: {rid!r}"


def test_new_run_id_does_not_repeat_across_many_calls():
    ids = {new_run_id() for _ in range(2000)}
    assert len(ids) == 2000, "run_id collisions detected within 2000 calls"


def test_concurrent_new_run_id_calls_are_unique():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        ids = set(pool.map(lambda _: new_run_id(), range(256)))
    assert len(ids) == 256


def test_two_intake_submits_get_different_run_ids(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec1 = intake.submit(
        raw_user_query="run one",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    rec2 = intake.submit(
        raw_user_query="run two",
        user_provided_context={"target_or_antigen_text": "EGFR"},
    )
    assert rec1.run_id != rec2.run_id
    # Storage paths land under distinct run directories — no overwrite.
    raw1 = local_storage.read_json(
        local_storage.run_key(rec1.run_id, "inputs/raw_request_record.json")
    )
    raw2 = local_storage.read_json(
        local_storage.run_key(rec2.run_id, "inputs/raw_request_record.json")
    )
    assert raw1["raw_user_query"] == "run one"
    assert raw2["raw_user_query"] == "run two"


def test_two_intake_submits_do_not_corrupt_registry_or_workflow_state(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec_a = intake.submit(
        raw_user_query="A", user_provided_context={"target_or_antigen_text": "X"},
    )
    rec_b = intake.submit(
        raw_user_query="B", user_provided_context={"target_or_antigen_text": "Y"},
    )

    reg_a = registry_service.get(rec_a.run_id)
    reg_b = registry_service.get(rec_b.run_id)
    assert reg_a.run_id == rec_a.run_id
    assert reg_b.run_id == rec_b.run_id
    assert reg_a.run_artifact_registry_id != reg_b.run_artifact_registry_id

    ws_a = workflow_state_service.get(rec_a.run_id)
    ws_b = workflow_state_service.get(rec_b.run_id)
    assert ws_a["run_id"] == rec_a.run_id
    assert ws_b["run_id"] == rec_b.run_id
    assert ws_a["steps"]["step_01"] == "completed"
    assert ws_b["steps"]["step_01"] == "completed"
