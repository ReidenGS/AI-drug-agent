"""Compact graph-resumption state adapter (LangGraph checkpointer seam).

We deliberately do NOT wire a LangGraph checkpointer/memory into the graphs
this round: every builder in ``adc_graph.py`` calls ``StateGraph(...).compile()``
with no ``checkpointer=``, and adding one would require threading a
``thread_id`` config through every ``invoke`` call site plus reworking
``PipelineState`` — a large graph change for no functional gain right now.
The artifact store (registry + workflow_state + clarification_state) already
provides run continuation, and snapshot/hydrate already provides resume.

This module is the small, documented adapter a future checkpointer would
serialize: it builds a COMPACT resume-state dict from the artifact store.
The artifact store stays the source of truth — this is only a
resume/conversation-state helper.

Privacy contract: the compact state contains ONLY ids, the LLM
``canonical_query`` (already sanitized + length-capped by the Step 2
normalizer), a slot-name/severity summary, and clarification request ids. It
NEVER includes raw artifact bodies, tool payloads, prompts, LLM responses,
API keys, or full PDB/CIF/FASTA/protein sequence/CDR3.
"""

from __future__ import annotations

from typing import Any

from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService


def build_compact_resume_state(
    storage: Storage,
    registry: ArtifactRegistryService,
    workflow_state: WorkflowStateService,
    run_id: str,
) -> dict[str, Any]:
    """Return a compact, privacy-safe resume-state dict for ``run_id``.

    Reads only small fields from the artifact store; never inlines raw
    content. Suitable to hand to a future LangGraph checkpointer as the
    serialized conversation/run state.
    """
    reg = registry.get(run_id)
    try:
        wf = workflow_state.get(run_id)
    except Exception:  # noqa: BLE001 — resume state must never hard-fail
        wf = {}

    canonical_query = None
    missing_slots_summary: list[dict] = []
    sq_key = storage.run_key(run_id, "inputs/structured_query.json")
    if storage.exists(sq_key):
        try:
            sq = storage.read_json(sq_key)
        except Exception:  # noqa: BLE001
            sq = {}
        cq = sq.get("canonical_query")
        if isinstance(cq, str) and cq.strip():
            canonical_query = cq.strip()
        for s in sq.get("missing_slots") or []:
            if isinstance(s, dict):
                missing_slots_summary.append(
                    {"slot_name": s.get("slot_name"), "severity": s.get("severity")}
                )

    clarification_request_ids: list[str] = []
    ir_key = storage.run_key(run_id, "inputs/input_readiness_status.json")
    if storage.exists(ir_key):
        try:
            ir = storage.read_json(ir_key)
        except Exception:  # noqa: BLE001
            ir = {}
        for c in ir.get("clarification_requests") or []:
            if isinstance(c, dict) and c.get("request_id"):
                clarification_request_ids.append(c["request_id"])

    return {
        "run_id": run_id,
        "current_step": wf.get("current_step"),
        "active_artifacts": reg.active_artifacts.model_dump(),
        "canonical_query": canonical_query,
        "missing_slots_summary": missing_slots_summary,
        "clarification_request_ids": clarification_request_ids,
    }
