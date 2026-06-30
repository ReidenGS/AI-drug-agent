"""Step 3 clarification multi-turn loop — minimal backend closed loop.

The frontend shows the user the Step 3 `response`; the user supplies only the
missing information. This service records that answer and builds the NEXT
turn's Step 1 input by carrying the previous intent + the answers forward, so
the Step 2 LLM can remember the original request and re-parse. Step 3 still
NEVER calls an LLM, and there is NO LangGraph memory / checkpointer — state
is persisted as ordinary artifacts via LocalStorage + ArtifactRegistryService.

Design principles honored here:

- The user's answer is NOT parsed into business fields by this service. It is
  carried as ``user_provided_context.clarification_answers`` (plus compact
  ``previous_*`` context) into a fresh revision run; the Step 2 LLM re-parses.
- The original run's Step 2 / Step 3 artifacts are never overwritten — the
  revision is a NEW run linked from the ``ClarificationState`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..schemas.step_01_raw_request_record import RawRequestRecord
from ..schemas.step_02_structured_query import StructuredQuery
from ..schemas.step_03_clarification import ClarificationAnswer, ClarificationState
from ..schemas.step_03_input_readiness import InputReadinessStatus
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .intake_service import IntakeService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_RAW_KEY = "inputs/raw_request_record.json"
_SQ_KEY = "inputs/structured_query.json"
_READINESS_KEY = "inputs/input_readiness_status.json"
_STATE_KEY_TMPL = ("clarification", "{artifact_id}.json")


@dataclass
class ClarificationRoundResult:
    """Outcome of a full submit + re-parse round."""

    state: ClarificationState
    next_run_id: str
    structured_query: Optional[StructuredQuery] = None
    input_readiness_status: Optional[InputReadinessStatus] = None


def _compact_missing_slots(slots: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in slots or []:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "slot_name": s.get("slot_name"),
                "slot_category": s.get("slot_category"),
                "severity": s.get("severity"),
            }
        )
    return out


def _compact_requests(requests: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in requests or []:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "request_id": r.get("request_id"),
                "slot_name": r.get("slot_name"),
                "slot_category": r.get("slot_category"),
                "severity": r.get("severity"),
                "question": r.get("question"),
            }
        )
    return out


class ClarificationService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.intake = IntakeService(storage, registry, workflow_state)

    # ── public API ──────────────────────────────────────────────────────────

    def submit_clarification_answer(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
    ) -> ClarificationState:
        """Record clarification answers and create the next revision run.

        Validates each answer's ``request_id`` against the source Step 3
        ``clarification_requests``; rejects unknown ids and duplicate answers
        for the same request (deterministic — raises ``ValueError``). Writes a
        ``ClarificationState`` artifact and a fresh revision ``RawRequestRecord``
        (new run id) carrying the previous intent + answers. Does NOT call an
        LLM and does NOT overwrite the original run's Step 2 / Step 3 outputs.
        """
        reg = self.registry.get(run_id)
        raw_id = reg.active_artifacts.raw_request_record_id
        sq_id = reg.active_artifacts.structured_query_id
        readiness_id = reg.active_artifacts.input_readiness_status_id
        if not raw_id or not sq_id or not readiness_id:
            raise WorkflowStateError(
                "clarification requires Step 1 + Step 2 + Step 3 artifacts in registry"
            )

        raw = self.storage.read_json(self.storage.run_key(run_id, _RAW_KEY))
        sq = self.storage.read_json(self.storage.run_key(run_id, _SQ_KEY))
        readiness = self.storage.read_json(self.storage.run_key(run_id, _READINESS_KEY))

        requests = readiness.get("clarification_requests") or []
        if not requests:
            raise ValueError(
                "run has no Step 3 clarification_requests to answer "
                "(nothing to clarify or pre-clarification artifact)"
            )
        requests_by_id = {
            r.get("request_id"): r for r in requests if isinstance(r, dict)
        }

        normalized = self._normalize_answers(answers, requests_by_id)

        resolved_ids = [a.request_id for a in normalized]
        unresolved_ids = [
            rid for rid in requests_by_id if rid not in set(resolved_ids)
        ]

        next_run_id, next_raw = self._build_next_run(
            run_id=run_id,
            raw=raw,
            sq=sq,
            requests=requests,
            answers=normalized,
        )

        state = ClarificationState(
            run_id=run_id,
            source_input_readiness_status_id=readiness_id,
            source_structured_query_id=sq_id,
            source_raw_request_record_id=raw_id,
            clarification_answers=normalized,
            resolved_request_ids=resolved_ids,
            unresolved_request_ids=unresolved_ids,
            next_run_id=next_run_id,
            next_raw_request_record_id=next_raw.run_artifact_registry_id,
            created_at=now_iso(),
        )

        artifact_id = new_artifact_id("clarification_state")
        self.storage.write_json(
            self.storage.run_key(
                run_id, _STATE_KEY_TMPL[0], _STATE_KEY_TMPL[1].format(artifact_id=artifact_id)
            ),
            {"artifact_id": artifact_id, **state.model_dump()},
        )
        # Non-destructive: original Step 2/3 artifacts are untouched; we only
        # add a pointer to the latest clarification state.
        self.registry.update_active(run_id, clarification_state_id=artifact_id)
        return state

    def reparse_next_turn(
        self, state: ClarificationState, supervisor: Any
    ) -> tuple[StructuredQuery, InputReadinessStatus]:
        """Run Step 2 (parse) + Step 3 (readiness) on the revision run.

        `supervisor` is a ``SupervisorAgent`` (the only LLM-using piece — and
        it is Step 2, not Step 3). Step 3 readiness stays deterministic.
        """
        from .input_readiness_service import InputReadinessService
        from .structured_query_service import StructuredQueryService

        if not state.next_run_id:
            raise ValueError("clarification state has no next_run_id to re-parse")

        sq = StructuredQueryService(
            self.storage, self.registry, self.workflow_state, supervisor
        ).parse(state.next_run_id)
        readiness = InputReadinessService(
            self.storage, self.registry, self.workflow_state
        ).check(state.next_run_id)
        return sq, readiness

    def submit_and_reparse(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
        supervisor: Any,
    ) -> ClarificationRoundResult:
        """Convenience: submit answers, then re-parse Step 2/3 on the revision."""
        state = self.submit_clarification_answer(run_id, answers)
        sq, readiness = self.reparse_next_turn(state, supervisor)
        return ClarificationRoundResult(
            state=state,
            next_run_id=state.next_run_id,  # type: ignore[arg-type]
            structured_query=sq,
            input_readiness_status=readiness,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_answers(
        answers: list[ClarificationAnswer] | list[dict],
        requests_by_id: dict[str, dict],
    ) -> list[ClarificationAnswer]:
        normalized: list[ClarificationAnswer] = []
        seen: set[str] = set()
        for raw_answer in answers or []:
            if isinstance(raw_answer, ClarificationAnswer):
                ans = raw_answer
            elif isinstance(raw_answer, dict):
                ans = ClarificationAnswer(
                    request_id=str(raw_answer.get("request_id") or ""),
                    answer_text=str(raw_answer.get("answer_text") or ""),
                    answered_at=str(raw_answer.get("answered_at") or now_iso()),
                    source=str(raw_answer.get("source") or "user"),
                    target_slot_name=raw_answer.get("target_slot_name"),
                    target_slot_category=raw_answer.get("target_slot_category"),
                )
            else:
                raise ValueError("each clarification answer must be a dict or ClarificationAnswer")

            if not ans.request_id:
                raise ValueError("clarification answer is missing request_id")
            if ans.request_id not in requests_by_id:
                raise ValueError(
                    f"unknown clarification request_id: {ans.request_id!r}"
                )
            if ans.request_id in seen:
                # Deterministic policy: reject duplicate answers for the same
                # request in a single submission rather than silently override.
                raise ValueError(
                    f"duplicate clarification answer for request_id: {ans.request_id!r}"
                )
            seen.add(ans.request_id)
            if not ans.answer_text.strip():
                raise ValueError(
                    f"clarification answer for {ans.request_id!r} is empty"
                )
            # Backfill the slot identity from the matching request when absent.
            req = requests_by_id[ans.request_id]
            if ans.target_slot_name is None:
                ans = ans.model_copy(update={"target_slot_name": req.get("slot_name")})
            if ans.target_slot_category is None:
                ans = ans.model_copy(
                    update={"target_slot_category": req.get("slot_category")}
                )
            normalized.append(ans)
        if not normalized:
            raise ValueError("no clarification answers were provided")
        return normalized

    def _build_next_run(
        self,
        *,
        run_id: str,
        raw: dict,
        sq: dict,
        requests: list[dict],
        answers: list[ClarificationAnswer],
    ) -> tuple[str, RawRequestRecord]:
        original_query = str(raw.get("raw_user_query") or "").strip()
        task_intent = sq.get("task_intent") or {}
        previous_task_intent = {
            "primary_intent": task_intent.get("primary_intent"),
            "secondary_intents": task_intent.get("secondary_intents") or [],
        }

        # Carry the original context fields forward (do not clobber unrelated
        # keys), then layer the clarification carry-over on top.
        next_ctx: dict[str, Any] = dict(raw.get("user_provided_context") or {})
        next_ctx["previous_task_intent"] = previous_task_intent
        # Carry the previous LLM canonical_query so the next Step 2 turn can
        # UPDATE it with the answers rather than re-deriving from scratch. The
        # service only transports it — it never composes a business query.
        prev_canonical = sq.get("canonical_query")
        if isinstance(prev_canonical, str) and prev_canonical.strip():
            next_ctx["previous_canonical_query"] = prev_canonical.strip()
        next_ctx["previous_missing_slots"] = _compact_missing_slots(
            sq.get("missing_slots") or []
        )
        next_ctx["previous_clarification_requests"] = _compact_requests(requests)
        next_ctx["clarification_answers"] = [
            {
                "request_id": a.request_id,
                "slot_name": a.target_slot_name,
                "slot_category": a.target_slot_category,
                "answer_text": a.answer_text,
                "answered_at": a.answered_at,
            }
            for a in answers
        ]

        # Keep the ORIGINAL query and append only a compact marker naming the
        # answered slots — never the (possibly long) answer values, which stay
        # in the structured clarification_answers block above.
        answered_slots = [a.target_slot_name or "answer" for a in answers]
        marker = (
            f"\n\n[Clarification follow-up provided for: {', '.join(answered_slots)}]"
            if answered_slots
            else ""
        )
        next_query = (original_query + marker) if original_query else marker.strip()

        next_run_id = self.intake.allocate_run_id()
        next_raw = self.intake.submit(
            raw_user_query=next_query or original_query,
            entry_source=raw.get("entry_source") or "api",
            submitted_by=raw.get("submitted_by"),
            user_provided_context=next_ctx,
            uploaded_files=raw.get("uploaded_files") or [],
            run_id=next_run_id,
        )
        return next_run_id, next_raw
