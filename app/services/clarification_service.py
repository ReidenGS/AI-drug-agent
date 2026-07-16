"""Same-run Step 3 clarification revisions with persisted replay authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from app.a2a.orchestrator_readiness import (
    OrchestratorReadinessError,
    ValidatedInputReadinessAuthority,
    load_input_readiness_authority,
)

from ..schemas.step_01_raw_request_record import RawRequestRecord
from ..schemas.step_02_structured_query import StructuredQuery
from ..schemas.step_03_clarification import ClarificationAnswer, ClarificationState
from ..schemas.step_03_input_readiness import InputReadinessStatus
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .input_readiness_service import (
    InputReadinessService,
    input_readiness_history_key,
)
from .raw_request_authority import RawRequestAuthorityError, load_active_raw_request
from .storage_service import Storage
from .structured_query_service import (
    StructuredQueryService,
    structured_query_history_key,
)
from .workflow_state_service import WorkflowStateService

_STATE_DIR = "clarification"
_STATE_KEY = "clarification/{revision_id}.json"
_STRUCTURED_CANONICAL_KEY = "inputs/structured_query.json"
_READINESS_CANONICAL_KEY = "inputs/input_readiness_status.json"


@dataclass(frozen=True)
class ClarificationRoundResult:
    """Compact in-process result of one same-run clarification revision."""

    state: ClarificationState
    run_id: str
    structured_query: Optional[StructuredQuery] = None
    input_readiness_status: Optional[InputReadinessStatus] = None


class ClarificationRequestError(ValueError):
    """Fixed compact user clarification input failure (HTTP 422)."""


class ClarificationConflictError(RuntimeError):
    """Fixed compact same-run authority conflict (HTTP 409)."""


class ClarificationReparseError(RuntimeError):
    """Fixed compact Step 2/3 internal failure (HTTP 503)."""


def _compact_missing_slots(slots: list[dict]) -> list[dict]:
    return [
        {
            "slot_name": item.get("slot_name"),
            "slot_category": item.get("slot_category"),
            "severity": item.get("severity"),
        }
        for item in slots or []
        if isinstance(item, dict)
    ]


def _compact_requests(requests: list[dict]) -> list[dict]:
    return [
        {
            "request_id": item.get("request_id"),
            "slot_name": item.get("slot_name"),
            "slot_category": item.get("slot_category"),
            "severity": item.get("severity"),
            "question": item.get("question"),
        }
        for item in requests or []
        if isinstance(item, dict)
    ]


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

    def submit_clarification_answer(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
    ) -> ClarificationState:
        """Create or replay one persisted same-run clarification revision."""

        self._ensure_not_routed(run_id)
        existing = self._matching_revision(run_id, answers)
        if existing is not None:
            self._materialize_revision_source_history(existing)
            if existing.revision_status == "completed":
                self._validate_completed_authority(existing)
            else:
                self._validate_recovery_authority(existing)
            return existing
        authority = self._load_source_authority(run_id)
        requests = self._eligible_requests(authority)
        revisions = self._load_revisions(run_id)
        if any(
            revision.source_input_readiness_status_id
            == authority.readiness_artifact_id
            for revision in revisions
        ):
            raise ClarificationConflictError(
                "clarification_submission_conflict"
            )

        source_ids = self._active_source_ids(run_id)
        if source_ids != (
            authority.raw_request_artifact_id,
            authority.structured_query_artifact_id,
            authority.readiness_artifact_id,
        ):
            raise OrchestratorReadinessError(
                "input_readiness_status_source_mismatch"
            )
        self._materialize_active_source_history(authority)
        normalized = self._normalize_answers(run_id, answers, requests)
        revision_id = new_artifact_id("clarification_state")
        timestamp = now_iso()
        resolved = [answer.request_id for answer in normalized]
        state = ClarificationState(
            run_id=run_id,
            revision_id=revision_id,
            revision_number=len(revisions) + 1,
            source_input_readiness_status_id=authority.readiness_artifact_id,
            source_structured_query_id=authority.structured_query_artifact_id,
            source_raw_request_record_id=authority.raw_request_artifact_id,
            clarification_answers=normalized,
            resolved_request_ids=resolved,
            unresolved_request_ids=[
                request_id
                for request_id in requests
                if request_id not in set(resolved)
            ],
            submission_fingerprint=self._submission_fingerprint(
                source_raw_request_record_id=authority.raw_request_artifact_id,
                source_structured_query_id=authority.structured_query_artifact_id,
                source_input_readiness_status_id=authority.readiness_artifact_id,
                answers=normalized,
            ),
            revision_status="submitted",
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._write_state(state)
        self.registry.update_active(
            run_id, clarification_state_id=revision_id
        )
        if self._active_source_ids(run_id) != source_ids:
            raise ClarificationConflictError(
                "clarification_authority_changed"
            )
        return state

    def submit_and_reparse(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
        supervisor: Any,
    ) -> ClarificationRoundResult:
        """Create/recover one revision and run missing Step 2/3 phases."""

        state = self.submit_clarification_answer(run_id, answers)
        return self._reparse(state, supervisor)

    def _reparse(
        self, state: ClarificationState, supervisor: Any
    ) -> ClarificationRoundResult:
        self._ensure_not_routed(state.run_id)
        if state.revision_status == "completed":
            return self._completed_result(state)

        source_raw, source_structured, source_readiness = (
            self._load_revision_sources(state)
        )
        structured = None
        if state.output_structured_query_id is None:
            effective = self._effective_step2_input(
                source_raw,
                source_structured,
                source_readiness,
                state,
            )
            try:
                structured = StructuredQueryService(
                    self.storage,
                    self.registry,
                    self.workflow_state,
                    supervisor,
                ).parse_effective(state.run_id, effective)
                active_structured = self.registry.get(
                    state.run_id
                ).active_artifacts.structured_query_id
                if not active_structured:
                    raise RuntimeError("clarification_step2_output_missing")
                state = state.model_copy(
                    update={
                        "revision_status": "step2_completed",
                        "output_structured_query_id": active_structured,
                        "failure_code": None,
                        "updated_at": now_iso(),
                    }
                )
                self._write_state(state)
            except Exception:
                failed = state.model_copy(
                    update={
                        "revision_status": "reparse_failed",
                        "failure_code": "clarification_step2_failed",
                        "updated_at": now_iso(),
                    }
                )
                self._write_state(failed)
                raise ClarificationReparseError(
                    "clarification_reparse_failed"
                ) from None
        else:
            structured = self._load_structured_history(
                state.run_id, state.output_structured_query_id
            )
            active_structured = self.registry.get(
                state.run_id
            ).active_artifacts.structured_query_id
            if active_structured != state.output_structured_query_id:
                raise ClarificationConflictError(
                    "clarification_recovery_authority_mismatch"
                )

        try:
            readiness = InputReadinessService(
                self.storage, self.registry, self.workflow_state
            ).check(state.run_id)
            active_readiness = self.registry.get(
                state.run_id
            ).active_artifacts.input_readiness_status_id
            if not active_readiness:
                raise RuntimeError("clarification_step3_output_missing")
            state = state.model_copy(
                update={
                    "revision_status": "completed",
                    "output_input_readiness_status_id": active_readiness,
                    "failure_code": None,
                    "updated_at": now_iso(),
                }
            )
            self._write_state(state)
        except Exception:
            failed = state.model_copy(
                update={
                    "revision_status": "reparse_failed",
                    "failure_code": "clarification_step3_failed",
                    "updated_at": now_iso(),
                }
            )
            self._write_state(failed)
            raise ClarificationReparseError(
                "clarification_reparse_failed"
            ) from None
        return ClarificationRoundResult(
            state=state,
            run_id=state.run_id,
            structured_query=structured,
            input_readiness_status=readiness,
        )

    def _completed_result(
        self, state: ClarificationState
    ) -> ClarificationRoundResult:
        self._validate_completed_authority(state)
        if (
            state.output_structured_query_id is None
            or state.output_input_readiness_status_id is None
        ):
            raise ClarificationConflictError(
                "clarification_revision_invalid"
            )
        return ClarificationRoundResult(
            state=state,
            run_id=state.run_id,
            structured_query=self._load_structured_history(
                state.run_id, state.output_structured_query_id
            ),
            input_readiness_status=self._load_readiness_history(
                state.run_id, state.output_input_readiness_status_id
            ),
        )

    def _validate_completed_authority(self, state: ClarificationState) -> None:
        authority = load_input_readiness_authority(
            run_id=state.run_id,
            registry=self.registry,
            storage=self.storage,
        )
        if (
            state.output_structured_query_id is None
            or state.output_input_readiness_status_id is None
            or authority.raw_request_artifact_id
            != state.source_raw_request_record_id
            or authority.structured_query_artifact_id
            != state.output_structured_query_id
            or authority.readiness_artifact_id
            != state.output_input_readiness_status_id
        ):
            raise ClarificationConflictError(
                "clarification_recovery_authority_mismatch"
            )

    def _validate_recovery_authority(self, state: ClarificationState) -> None:
        self._load_revision_sources(state)
        active = self.registry.get(state.run_id).active_artifacts
        expected_structured = (
            state.output_structured_query_id
            or state.source_structured_query_id
        )
        if (
            active.raw_request_record_id != state.source_raw_request_record_id
            or active.structured_query_id != expected_structured
            or active.input_readiness_status_id
            != state.source_input_readiness_status_id
        ):
            raise ClarificationConflictError(
                "clarification_recovery_authority_mismatch"
            )

    def _matching_revision(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
    ) -> ClarificationState | None:
        semantic = self._untrusted_answer_semantics(answers)
        revisions = self._load_revisions(run_id)
        active_revision_id = self.registry.get(
            run_id
        ).active_artifacts.clarification_state_id
        if revisions and active_revision_id != revisions[-1].revision_id:
            raise ClarificationConflictError(
                "clarification_revision_authority_mismatch"
            )
        matches = [
            revision
            for revision in revisions
            if self._stored_answer_semantics(revision) == semantic
        ]
        if matches:
            if matches[-1].revision_id != active_revision_id:
                raise ClarificationConflictError(
                    "clarification_submission_conflict"
                )
            return matches[-1]
        if revisions and revisions[-1].revision_status != "completed":
            raise ClarificationConflictError(
                "clarification_submission_conflict"
            )
        return None

    def _load_source_authority(
        self, run_id: str
    ) -> ValidatedInputReadinessAuthority:
        authority = load_input_readiness_authority(
            run_id=run_id,
            registry=self.registry,
            storage=self.storage,
        )
        if authority.readiness.input_readiness_status != "needs_user_input":
            raise ClarificationRequestError("clarification_source_not_ready")
        return authority

    @staticmethod
    def _eligible_requests(
        authority: ValidatedInputReadinessAuthority,
    ) -> dict[str, dict]:
        requests = {
            request.request_id: request.model_dump(mode="json")
            for request in authority.readiness.clarification_requests
            if not request.resolved
            and request.severity in {"blocking", "warning"}
        }
        if not requests:
            raise ClarificationRequestError("clarification_request_invalid")
        return requests

    def _normalize_answers(
        self,
        run_id: str,
        answers: list[ClarificationAnswer] | list[dict],
        requests_by_id: dict[str, dict],
    ) -> list[ClarificationAnswer]:
        normalized: list[ClarificationAnswer] = []
        seen: set[str] = set()
        for raw_answer in answers or []:
            try:
                answer = (
                    raw_answer
                    if isinstance(raw_answer, ClarificationAnswer)
                    else ClarificationAnswer.model_validate(raw_answer, strict=True)
                )
            except (ValidationError, TypeError):
                raise ClarificationRequestError(
                    "clarification_request_invalid"
                ) from None
            if (
                not answer.request_id
                or answer.request_id not in requests_by_id
                or answer.request_id in seen
                or not isinstance(answer.answer_text, str)
                or not answer.answer_text.strip()
            ):
                raise ClarificationRequestError(
                    "clarification_request_invalid"
                )
            seen.add(answer.request_id)
            request = requests_by_id[answer.request_id]
            answer_text = answer.answer_text.strip()
            update = {
                "answer_text": answer_text,
                "source": "user",
                "target_slot_name": request.get("slot_name"),
                "target_slot_category": request.get("slot_category"),
            }
            answer = answer.model_copy(update=update)
            normalized.append(answer)
        if not normalized:
            raise ClarificationRequestError("clarification_request_invalid")
        return sorted(normalized, key=lambda item: item.request_id)

    @staticmethod
    def _untrusted_answer_semantics(
        answers: list[ClarificationAnswer] | list[dict],
    ) -> tuple[tuple[str, str, str], ...]:
        semantic: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for raw in answers or []:
            if isinstance(raw, ClarificationAnswer):
                request_id = raw.request_id
                answer_text = raw.answer_text
                source = raw.source
            elif isinstance(raw, dict):
                request_id = raw.get("request_id")
                answer_text = raw.get("answer_text")
                source = raw.get("source") or "user"
            else:
                raise ClarificationRequestError(
                    "clarification_request_invalid"
                )
            if (
                not isinstance(request_id, str)
                or not request_id
                or request_id in seen
                or not isinstance(answer_text, str)
                or not answer_text.strip()
                or source != "user"
            ):
                raise ClarificationRequestError(
                    "clarification_request_invalid"
                )
            seen.add(request_id)
            text = answer_text.strip()
            semantic.append((request_id, text, "user"))
        if not semantic:
            raise ClarificationRequestError("clarification_request_invalid")
        return tuple(sorted(semantic))

    @staticmethod
    def _stored_answer_semantics(
        state: ClarificationState,
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            sorted(
                (
                    answer.request_id,
                    answer.answer_text.strip(),
                    answer.source,
                )
                for answer in state.clarification_answers
            )
        )

    @staticmethod
    def _submission_fingerprint(
        *,
        source_raw_request_record_id: str,
        source_structured_query_id: str,
        source_input_readiness_status_id: str,
        answers: list[ClarificationAnswer],
    ) -> str:
        payload = {
            "source_ids": [
                source_raw_request_record_id,
                source_structured_query_id,
                source_input_readiness_status_id,
            ],
            "answers": [
                {
                    "request_id": answer.request_id,
                    "answer_text": answer.answer_text,
                    "source": answer.source,
                    "slot_name": answer.target_slot_name,
                    "slot_category": answer.target_slot_category,
                }
                for answer in answers
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return f"clarification_submission_{hashlib.sha256(encoded).hexdigest()}"

    def _effective_step2_input(
        self,
        raw: RawRequestRecord,
        structured: StructuredQuery,
        readiness: InputReadinessStatus,
        state: ClarificationState,
    ) -> dict:
        payload = raw.model_dump(mode="json")
        payload["artifact_id"] = state.source_raw_request_record_id
        context = dict(payload.get("user_provided_context") or {})
        intent = structured.task_intent.model_dump(mode="json")
        context.update(
            {
                "previous_task_intent": {
                    "primary_intent": intent.get("primary_intent"),
                    "secondary_intents": intent.get("secondary_intents") or [],
                },
                "previous_canonical_query": structured.canonical_query,
                "previous_missing_slots": _compact_missing_slots(
                    [item.model_dump(mode="json") for item in structured.missing_slots]
                ),
                "previous_clarification_requests": _compact_requests(
                    [
                        item.model_dump(mode="json")
                        for item in readiness.clarification_requests
                    ]
                ),
                "clarification_answers": [
                    {
                        "request_id": answer.request_id,
                        "slot_name": answer.target_slot_name,
                        "slot_category": answer.target_slot_category,
                        "answer_text": answer.answer_text,
                        "answered_at": answer.answered_at,
                    }
                    for answer in self._cumulative_answers(state)
                ],
            }
        )
        payload["user_provided_context"] = context
        return payload


    def _cumulative_answers(
        self, current: ClarificationState
    ) -> list[ClarificationAnswer]:
        by_slot: dict[str, ClarificationAnswer] = {}
        for revision in self._load_revisions(current.run_id):
            if (
                revision.revision_id != current.revision_id
                and revision.revision_status == "completed"
            ):
                for answer in revision.clarification_answers:
                    by_slot[
                        answer.target_slot_name or answer.request_id
                    ] = answer
        for answer in current.clarification_answers:
            by_slot[answer.target_slot_name or answer.request_id] = answer
        return list(by_slot.values())

    def _load_revision_sources(
        self, state: ClarificationState
    ) -> tuple[RawRequestRecord, StructuredQuery, InputReadinessStatus]:
        try:
            raw = load_active_raw_request(
                run_id=state.run_id,
                registry=self.registry,
                storage=self.storage,
            )
        except RawRequestAuthorityError as exc:
            raise OrchestratorReadinessError(str(exc)) from None
        if self.registry.get(
            state.run_id
        ).active_artifacts.raw_request_record_id != state.source_raw_request_record_id:
            raise ClarificationConflictError(
                "clarification_recovery_authority_mismatch"
            )
        structured = self._load_structured_history(
            state.run_id, state.source_structured_query_id
        )
        readiness = self._load_readiness_history(
            state.run_id, state.source_input_readiness_status_id
        )
        if (
            structured.source_raw_request_ref.raw_request_record_id
            != state.source_raw_request_record_id
            or readiness.source_refs.raw_request_record_id
            != state.source_raw_request_record_id
            or readiness.source_refs.structured_query_id
            != state.source_structured_query_id
            or readiness.input_readiness_status != "needs_user_input"
        ):
            raise ClarificationConflictError(
                "clarification_recovery_authority_mismatch"
            )
        return raw, structured, readiness

    def _load_structured_history(
        self, run_id: str, artifact_id: str
    ) -> StructuredQuery:
        body = self._load_versioned_body(
            run_id=run_id,
            artifact_id=artifact_id,
            history_key=structured_query_history_key(artifact_id),
            canonical_key=_STRUCTURED_CANONICAL_KEY,
            active_id=self.registry.get(run_id).active_artifacts.structured_query_id,
            error_code="structured_query_history_invalid",
        )
        try:
            return StructuredQuery.model_validate(
                {key: value for key, value in body.items() if key != "artifact_id"},
                strict=True,
            )
        except ValidationError:
            raise ClarificationConflictError(
                "structured_query_history_invalid"
            ) from None

    def _load_readiness_history(
        self, run_id: str, artifact_id: str
    ) -> InputReadinessStatus:
        body = self._load_versioned_body(
            run_id=run_id,
            artifact_id=artifact_id,
            history_key=input_readiness_history_key(artifact_id),
            canonical_key=_READINESS_CANONICAL_KEY,
            active_id=self.registry.get(
                run_id
            ).active_artifacts.input_readiness_status_id,
            error_code="input_readiness_status_history_invalid",
        )
        try:
            return InputReadinessStatus.model_validate(
                {key: value for key, value in body.items() if key != "artifact_id"},
                strict=True,
            )
        except ValidationError:
            raise ClarificationConflictError(
                "input_readiness_status_history_invalid"
            ) from None

    def _load_versioned_body(
        self,
        *,
        run_id: str,
        artifact_id: str,
        history_key: str,
        canonical_key: str,
        active_id: str | None,
        error_code: str,
    ) -> dict:
        key = self.storage.run_key(run_id, history_key)
        if not self.storage.exists(key) and active_id == artifact_id:
            key = self.storage.run_key(run_id, canonical_key)
        try:
            body = self.storage.read_json(key)
        except Exception:
            raise ClarificationConflictError(error_code) from None
        if (
            not isinstance(body, dict)
            or body.get("artifact_id") != artifact_id
            or body.get("run_id") != run_id
        ):
            raise ClarificationConflictError(error_code)
        return body

    def _materialize_active_source_history(
        self, authority: ValidatedInputReadinessAuthority
    ) -> None:
        run_id = authority.raw_request.run_id
        structured_body = self._validated_canonical_body(
            run_id=run_id,
            artifact_id=authority.structured_query_artifact_id,
            canonical_key=_STRUCTURED_CANONICAL_KEY,
            model=StructuredQuery,
            error_code="structured_query_history_invalid",
        )
        readiness_body = self._validated_canonical_body(
            run_id=run_id,
            artifact_id=authority.readiness_artifact_id,
            canonical_key=_READINESS_CANONICAL_KEY,
            model=InputReadinessStatus,
            error_code="input_readiness_status_history_invalid",
        )
        self._materialize_history_body(
            run_id,
            structured_query_history_key(authority.structured_query_artifact_id),
            structured_body,
            "structured_query_history_invalid",
        )
        self._materialize_history_body(
            run_id,
            input_readiness_history_key(authority.readiness_artifact_id),
            readiness_body,
            "input_readiness_status_history_invalid",
        )

    def _materialize_revision_source_history(
        self, state: ClarificationState
    ) -> None:
        structured = self._load_versioned_body(
            run_id=state.run_id,
            artifact_id=state.source_structured_query_id,
            history_key=structured_query_history_key(
                state.source_structured_query_id
            ),
            canonical_key=_STRUCTURED_CANONICAL_KEY,
            active_id=self.registry.get(
                state.run_id
            ).active_artifacts.structured_query_id,
            error_code="structured_query_history_invalid",
        )
        readiness = self._load_versioned_body(
            run_id=state.run_id,
            artifact_id=state.source_input_readiness_status_id,
            history_key=input_readiness_history_key(
                state.source_input_readiness_status_id
            ),
            canonical_key=_READINESS_CANONICAL_KEY,
            active_id=self.registry.get(
                state.run_id
            ).active_artifacts.input_readiness_status_id,
            error_code="input_readiness_status_history_invalid",
        )
        try:
            structured_model = StructuredQuery.model_validate(
                {
                    key: value
                    for key, value in structured.items()
                    if key != "artifact_id"
                },
                strict=True,
            )
            readiness_model = InputReadinessStatus.model_validate(
                {
                    key: value
                    for key, value in readiness.items()
                    if key != "artifact_id"
                },
                strict=True,
            )
        except ValidationError:
            raise ClarificationConflictError(
                "clarification_revision_authority_invalid"
            ) from None
        if (
            structured_model.source_raw_request_ref.raw_request_record_id
            != state.source_raw_request_record_id
            or readiness_model.source_refs.raw_request_record_id
            != state.source_raw_request_record_id
            or readiness_model.source_refs.structured_query_id
            != state.source_structured_query_id
        ):
            raise ClarificationConflictError(
                "clarification_revision_authority_invalid"
            )
        self._materialize_history_body(
            state.run_id,
            structured_query_history_key(state.source_structured_query_id),
            structured,
            "structured_query_history_invalid",
        )
        self._materialize_history_body(
            state.run_id,
            input_readiness_history_key(
                state.source_input_readiness_status_id
            ),
            readiness,
            "input_readiness_status_history_invalid",
        )

    def _validated_canonical_body(
        self,
        *,
        run_id: str,
        artifact_id: str,
        canonical_key: str,
        model: type[StructuredQuery] | type[InputReadinessStatus],
        error_code: str,
    ) -> dict:
        try:
            body = self.storage.read_json(
                self.storage.run_key(run_id, canonical_key)
            )
            if (
                not isinstance(body, dict)
                or body.get("artifact_id") != artifact_id
                or body.get("run_id") != run_id
            ):
                raise ValueError
            model.model_validate(
                {key: value for key, value in body.items() if key != "artifact_id"},
                strict=True,
            )
        except Exception:
            raise ClarificationConflictError(error_code) from None
        return body

    def _materialize_history_body(
        self, run_id: str, relative_key: str, body: dict, error_code: str
    ) -> None:
        key = self.storage.run_key(run_id, relative_key)
        if self.storage.exists(key):
            try:
                existing = self.storage.read_json(key)
            except Exception:
                raise ClarificationConflictError(error_code) from None
            if existing != body:
                raise ClarificationConflictError(error_code)
            return
        self.storage.write_json(key, body)

    def _validate_revision_authority(self, state: ClarificationState) -> None:
        try:
            readiness = self._load_readiness_history(
                state.run_id, state.source_input_readiness_status_id
            )
            if (
                readiness.source_refs.raw_request_record_id
                != state.source_raw_request_record_id
                or readiness.source_refs.structured_query_id
                != state.source_structured_query_id
                or readiness.input_readiness_status != "needs_user_input"
            ):
                raise ValueError
            eligible = {
                request.request_id: request
                for request in readiness.clarification_requests
                if not request.resolved
                and request.severity in {"blocking", "warning"}
            }
            answer_ids = [answer.request_id for answer in state.clarification_answers]
            if len(answer_ids) != len(set(answer_ids)):
                raise ValueError
            for answer in state.clarification_answers:
                request = eligible.get(answer.request_id)
                if (
                    request is None
                    or answer.source != "user"
                    or answer.target_slot_name != request.slot_name
                    or answer.target_slot_category != request.slot_category
                ):
                    raise ValueError
                if not answer.answer_text.strip():
                    raise ValueError
            if state.resolved_request_ids != answer_ids:
                raise ValueError
            unresolved = [
                request.request_id
                for request in readiness.clarification_requests
                if request.request_id in eligible
                and request.request_id not in set(answer_ids)
            ]
            if state.unresolved_request_ids != unresolved:
                raise ValueError
            expected_fingerprint = self._submission_fingerprint(
                source_raw_request_record_id=state.source_raw_request_record_id,
                source_structured_query_id=state.source_structured_query_id,
                source_input_readiness_status_id=(
                    state.source_input_readiness_status_id
                ),
                answers=state.clarification_answers,
            )
            if state.submission_fingerprint != expected_fingerprint:
                raise ValueError
            phase = (
                state.revision_status,
                state.output_structured_query_id is not None,
                state.output_input_readiness_status_id is not None,
                state.failure_code,
            )
            valid_phases = {
                ("submitted", False, False, None),
                ("step2_completed", True, False, None),
                ("completed", True, True, None),
                (
                    "reparse_failed",
                    False,
                    False,
                    "clarification_step2_failed",
                ),
                (
                    "reparse_failed",
                    True,
                    False,
                    "clarification_step3_failed",
                ),
            }
            if phase not in valid_phases:
                raise ValueError
        except Exception:
            raise ClarificationConflictError(
                "clarification_revision_authority_invalid"
            ) from None

    def _write_state(self, state: ClarificationState) -> None:
        self.storage.write_json(
            self.storage.run_key(
                state.run_id,
                _STATE_KEY.format(revision_id=state.revision_id),
            ),
            {"artifact_id": state.revision_id, **state.model_dump(mode="json")},
        )

    def _load_revisions(self, run_id: str) -> list[ClarificationState]:
        prefix = self.storage.run_key(run_id, _STATE_DIR)
        revisions: list[ClarificationState] = []
        for key in self.storage.list_prefix(prefix):
            if not key.endswith(".json"):
                continue
            try:
                body = self.storage.read_json(key)
                revision = ClarificationState.model_validate(
                    {
                        name: value
                        for name, value in body.items()
                        if name != "artifact_id"
                    },
                    strict=True,
                )
            except Exception:
                raise ClarificationConflictError(
                    "clarification_revision_authority_invalid"
                ) from None
            if (
                body.get("artifact_id") != revision.revision_id
                or revision.run_id != run_id
            ):
                raise ClarificationConflictError(
                    "clarification_revision_authority_invalid"
                )
            self._validate_revision_authority(revision)
            revisions.append(revision)
        numbers = [revision.revision_number for revision in revisions]
        if len(numbers) != len(set(numbers)):
            raise ClarificationConflictError(
                "clarification_revision_authority_invalid"
            )
        return sorted(revisions, key=lambda item: item.revision_number)

    def _active_source_ids(self, run_id: str) -> tuple[str | None, ...]:
        active = self.registry.get(run_id).active_artifacts
        return (
            active.raw_request_record_id,
            active.structured_query_id,
            active.input_readiness_status_id,
        )

    def _ensure_not_routed(self, run_id: str) -> None:
        active = self.registry.get(run_id).active_artifacts
        forbidden = (
            active.run_step_plan_id,
            active.worker_discovery_snapshot_id,
            active.worker_routing_plan_id,
            active.worker_routing_plan_control_id,
            active.candidate_context_table_id,
            active.structured_liability_summary_id,
            active.prepared_structure_input_package_id,
            active.structure_prediction_and_interface_results_id,
            active.structure_variant_and_compound_screening_id,
            active.scoring_handoff_id,
            active.scoring_validation_id,
            active.ranking_table_id,
            active.scientific_evidence_table_id,
            active.patent_prior_art_table_id,
        )
        if any(value is not None for value in forbidden):
            raise ClarificationConflictError(
                "clarification_run_already_routed"
            )


__all__ = [
    "ClarificationConflictError",
    "ClarificationRequestError",
    "ClarificationReparseError",
    "ClarificationRoundResult",
    "ClarificationService",
]
