"""Generic HTTP A2A worker transport (Turn B).

Reusable transport layer shared by every ADC worker (Step 5 now; Step 6 /
structure in later turns). It is a *thin A2A adapter*: it speaks python-a2a on
the wire and hands a validated :class:`WorkerExecutionRequest` to an injected
worker core. It owns NO domain logic.

    A2AClient(url).send_task_async(task)
      -> HTTP POST /a2a/tasks/send            (real localhost TCP socket)
      -> python_a2a.A2AServer route
      -> A2AWorkerAdapter.handle_task(task)   (this module — generic adapter)
      -> WorkerExecutionRequest
      -> core.execute_request(request)        (per-worker domain core)
      -> WorkerExecutionResult
      -> A2A task response (encoded into Task.artifacts)

Responsibilities of this generic adapter (see orchestrator_routing_design.md
"Worker A2A Adapter Boundary" + worker_execution_contract.md "Thin A2A Adapter"):

- Parse ``Task.message.content.text`` -> JSON -> ``WorkerExecutionRequest``.
- Require + validate ``Task.metadata`` as ``A2ATaskMetadata`` and enforce it is
  fully identity-consistent with the request body.
- Enforce the receiving worker's ``agent_id`` / ``capability_id``.
- Enforce every ``privacy_constraints`` flag is still true.
- Delegate to ``core.execute_request(request)``.
- Encode the ``WorkerExecutionResult`` into ``Task.artifacts`` and set
  ``TaskStatus`` from ``result_status``.

It MUST NOT: decide routing/planned_status, read run_step_plan/worker_routing_plan,
call worker business methods, run MCP/LLM logic, fall back to a local direct call,
or swallow a transport/validation failure into a fake success.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import ValidationError
from python_a2a import AgentCard, A2AServer, Task, TaskState, TaskStatus
from python_a2a.server.http import create_flask_app

from .agent_cards import AgentContractError, parse_adc_agent_contract
from .contracts import (
    A2ATaskMetadata,
    ToolCallSummary,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)

# success/partial -> A2A COMPLETED task; every other result_status -> A2A FAILED.
# (worker_execution_contract.md status semantics: partial is still a completed
# A2A task; request-level validation / blockers are failed tasks.)
_A2A_COMPLETED_RESULT_STATUSES = frozenset({"success", "partial"})

# The 7 identity fields the compact Task.metadata must echo from the request body.
_METADATA_IDENTITY_FIELDS = (
    "run_id",
    "task_id",
    "routing_plan_id",
    "routing_decision_id",
    "agent_id",
    "capability_id",
    "created_by",
)

# The 6 identity fields the worker core's result must echo from the request.
_RESULT_IDENTITY_FIELDS = (
    "run_id",
    "task_id",
    "routing_plan_id",
    "routing_decision_id",
    "agent_id",
    "capability_id",
)


@runtime_checkable
class WorkerCore(Protocol):
    """Contract a per-worker domain core must satisfy to be served over A2A."""

    agent_id: str
    capability_ids: frozenset[str]

    @property
    def agent_card(self) -> AgentCard: ...

    def health(self) -> dict[str, Any]: ...

    def execute_request(self, request: WorkerExecutionRequest) -> WorkerExecutionResult:
        """Validate worker-specific inputs and run the domain task.

        MUST return a compact :class:`WorkerExecutionResult` for every controlled
        outcome (success / partial / validation_failed / blocked / tool_failed).
        Truly unexpected exceptions may propagate; the adapter converts them into
        a compact ``tool_failed`` result without leaking the raw exception.
        """
        ...


class WorkerRequestRejected(Exception):
    """Raised inside adapter/core validation to signal a compact, safe failure.

    ``result_status`` / ``error_code`` / ``message`` are compact strings that
    flow into the returned WorkerExecutionResult — never a raw payload.
    """

    def __init__(self, *, result_status: str, error_code: str, message: str) -> None:
        super().__init__(message)
        self.result_status = result_status
        self.error_code = error_code
        self.compact_message = message


class WorkerServerConfigurationError(RuntimeError):
    """Raised (fail-fast, at Server construction) when the injected WorkerCore
    and its published AgentCard are inconsistent or the card is not servable."""


def _validate_core_card_consistency(core: WorkerCore) -> None:
    """Fail fast if the worker core and its AgentCard disagree.

    Runs on the pristine card BEFORE ``A2AServer.__init__`` mutates
    ``capabilities`` — so a misconfiguration is caught at startup, never after
    the HTTP server is already accepting tasks. Never rewrites the AgentCard from
    core attributes and never silently reconciles the two.
    """
    try:
        contract = parse_adc_agent_contract(core.agent_card)
    except AgentContractError as exc:
        raise WorkerServerConfigurationError(
            f"worker AgentCard has no valid adc_agent_contract: {exc}"
        ) from exc

    if contract.agent_role != "worker":
        raise WorkerServerConfigurationError(
            f"worker AgentCard agent_role must be 'worker', got '{contract.agent_role}'"
        )
    if contract.agent_id != core.agent_id:
        raise WorkerServerConfigurationError(
            f"AgentCard agent_id '{contract.agent_id}' != core.agent_id '{core.agent_id}'"
        )
    card_capability_ids = {cap.capability_id for cap in contract.capabilities}
    if card_capability_ids != set(core.capability_ids):
        raise WorkerServerConfigurationError(
            "AgentCard capability ids do not match core.capability_ids: "
            f"card={sorted(card_capability_ids)} core={sorted(core.capability_ids)}"
        )
    if not contract.routable:
        raise WorkerServerConfigurationError(
            f"worker '{core.agent_id}' AgentCard is not routable; cannot serve it over HTTP"
        )
    if contract.status != "active":
        raise WorkerServerConfigurationError(
            f"worker '{core.agent_id}' AgentCard status must be 'active', got '{contract.status}'"
        )
    if effective_url_port(str(getattr(core.agent_card, "url", "") or "")) is None:
        raise WorkerServerConfigurationError(
            f"worker '{core.agent_id}' AgentCard url must be a valid HTTP(S) URL "
            "with a determinable effective port"
        )


class A2AWorkerAdapter(A2AServer):
    """Generic thin A2A adapter — a real ``python_a2a.A2AServer`` subclass.

    Overriding ``handle_task`` (the documented worker entrypoint that the
    A2AServer HTTP route invokes) is the proper python-a2a mechanism; we do NOT
    monkeypatch ``server.handle_task``.
    """

    def __init__(self, core: WorkerCore) -> None:
        # Fail fast on any core/card misconfiguration before the HTTP server is
        # assembled — never discover it only when a Task arrives.
        _validate_core_card_consistency(core)
        self._core = core
        super().__init__(agent_card=core.agent_card, google_a2a_compatible=False)

    # python-a2a worker entrypoint. Never raises; always returns a Task carrying
    # a compact WorkerExecutionResult.
    def handle_task(self, task: Task) -> Task:  # type: ignore[override]
        result = self._process(task)
        completed = result.result_status in _A2A_COMPLETED_RESULT_STATUSES
        task.artifacts = [
            {"parts": [{"type": "text", "text": result.model_dump_json()}]}
        ]
        task.status = TaskStatus(
            state=TaskState.COMPLETED if completed else TaskState.FAILED
        )
        return task

    def _process(self, task: Task) -> WorkerExecutionResult:
        try:
            request = self._parse_and_validate(task)
        except WorkerRequestRejected as exc:
            return _failure_result_from_task(
                task,
                agent_id=self._core.agent_id,
                capability_id=_representative_capability(self._core),
                result_status=exc.result_status,
                error_code=exc.error_code,
                error_summary=exc.compact_message,
            )

        # The outer python-a2a transport identity must be the same task identity
        # that the now-validated ADC request + metadata carry. Never rewrite
        # either side and never dispatch the core on a mismatch.
        if str(getattr(task, "id", "") or "") != request.task_id:
            return _failure_result_from_request(
                request,
                result_status="validation_failed",
                error_code="task_transport_id_mismatch",
                error_summary=(
                    "python_a2a.Task.id does not match WorkerExecutionRequest.task_id"
                ),
            )

        # Validated envelope — delegate to the worker domain core.
        try:
            result = self._core.execute_request(request)
        except WorkerRequestRejected as exc:
            return _failure_result_from_request(
                request,
                result_status=exc.result_status,
                error_code=exc.error_code,
                error_summary=exc.compact_message,
            )
        except Exception as exc:  # noqa: BLE001 — safety net; never leak raw payloads
            return _failure_result_from_request(
                request,
                result_status="tool_failed",
                error_code="worker_execution_error",
                error_summary=f"unexpected worker execution error: {type(exc).__name__}",
            )

        # The core MUST return a WorkerExecutionResult. A dict / None / anything
        # else is a worker bug — surface it as a compact failure rather than
        # crashing (AttributeError -> HTTP 500) or coercing it into a fake success.
        if not isinstance(result, WorkerExecutionResult):
            return _failure_result_from_request(
                request,
                result_status="tool_failed",
                error_code="worker_result_schema_invalid",
                error_summary=(
                    "worker core returned a non-WorkerExecutionResult "
                    f"({type(result).__name__})"
                ),
            )

        # Pydantic's model_copy/model_construct APIs can deliberately bypass
        # validation while still producing a WorkerExecutionResult instance.
        # Revalidate the serialized value before reading identity fields so an
        # invalid payload version or a missing field becomes a compact failure,
        # never an AttributeError/HTTP 500 and never an emitted non-v1 result.
        try:
            result = WorkerExecutionResult.model_validate(result.model_dump())
        except Exception:  # noqa: BLE001 — compact schema failure, no raw details
            return _failure_result_from_request(
                request,
                result_status="tool_failed",
                error_code="worker_result_schema_invalid",
                error_summary="worker core returned an invalid WorkerExecutionResult",
            )

        # A success/partial result cannot honestly complete an A2A task unless
        # the core also reports execution_status=completed.
        if (
            result.result_status in _A2A_COMPLETED_RESULT_STATUSES
            and result.execution_status != "completed"
        ):
            return _failure_result_from_request(
                request,
                result_status="tool_failed",
                error_code="worker_result_schema_invalid",
                error_summary=(
                    "worker core returned inconsistent execution_status/result_status"
                ),
            )

        # The core's result MUST carry the same identity as the dispatched request.
        # Never forward a mis-identified result to the Orchestrator, and never
        # silently overwrite it into a fake success.
        identity_mismatches = [
            field
            for field in _RESULT_IDENTITY_FIELDS
            if getattr(result, field) != getattr(request, field)
        ]
        if identity_mismatches:
            return _failure_result_from_request(
                request,
                result_status="tool_failed",
                error_code="worker_result_identity_mismatch",
                error_summary=(
                    "worker core result identity disagrees with request on: "
                    f"{sorted(identity_mismatches)}"
                ),
            )
        return result

    def _parse_and_validate(self, task: Task) -> WorkerExecutionRequest:
        # 1. message text -> JSON object.
        text = _extract_message_text(task)
        if not text:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="malformed_task_no_message_text",
                message="A2A task message contained no text payload",
            )
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="malformed_task_json",
                message="A2A task message text was not valid JSON",
            )
        if not isinstance(raw, dict):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="malformed_task_json",
                message="A2A task message JSON was not an object",
            )

        # 2. WorkerExecutionRequest schema (also enforces payload_type/version).
        try:
            request = WorkerExecutionRequest.model_validate(raw)
        except ValidationError:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="request_schema_invalid",
                message="WorkerExecutionRequest failed schema validation",
            )

        # 3. Task.metadata is REQUIRED, must validate, and must be identity-consistent.
        self._validate_required_metadata(task, request)

        # 4. Routing identity must match this worker's AgentCard.
        if request.agent_id != self._core.agent_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="agent_id_mismatch",
                message=(
                    f"task agent_id '{request.agent_id}' does not match worker "
                    f"'{self._core.agent_id}'"
                ),
            )
        if request.capability_id not in self._core.capability_ids:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="capability_id_mismatch",
                message=(
                    f"task capability_id '{request.capability_id}' is not served by "
                    f"worker '{self._core.agent_id}'"
                ),
            )

        # 5. Privacy constraints must all remain enabled.
        pc = request.privacy_constraints
        if not all(
            (
                pc.no_raw_sequence,
                pc.no_raw_fasta,
                pc.no_raw_pdb_cif,
                pc.no_raw_a3m,
                pc.no_api_keys,
                pc.no_raw_tooluniverse_payload,
                pc.no_full_prompt,
                pc.no_raw_llm_response,
            )
        ):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="privacy_constraints_disabled",
                message="request privacy_constraints must all remain enabled",
            )

        return request

    def _validate_required_metadata(
        self, task: Task, request: WorkerExecutionRequest
    ) -> None:
        meta = getattr(task, "metadata", None)
        if not isinstance(meta, dict) or not meta:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="task_metadata_missing",
                message="A2A task metadata is required but was missing",
            )
        try:
            parsed = A2ATaskMetadata.model_validate(meta)
        except ValidationError:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="task_metadata_invalid",
                message="A2A task metadata failed schema validation",
            )
        # payload type must agree between metadata and request body.
        if parsed.adc_payload_type != request.payload_type:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="task_metadata_payload_type_mismatch",
                message="task metadata adc_payload_type disagrees with request payload_type",
            )
        # payload version must agree; only the locked "v1" is accepted (both
        # A2ATaskMetadata.adc_payload_version and WorkerExecutionRequest.payload_version
        # are Literal["v1"], so an unsupported version is already rejected at schema
        # validation — this is the explicit, non-silent cross-check on top of that).
        if parsed.adc_payload_version != request.payload_version:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="task_metadata_payload_version_mismatch",
                message="task metadata adc_payload_version disagrees with request payload_version",
            )
        mismatches = [
            field
            for field in _METADATA_IDENTITY_FIELDS
            if getattr(parsed, field) != getattr(request, field)
        ]
        if mismatches:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="task_metadata_body_mismatch",
                message=f"task metadata disagrees with request body on: {sorted(mismatches)}",
            )


# ── result builders (shared) ─────────────────────────────────────────────────
def _failure_result_from_request(
    request: WorkerExecutionRequest,
    *,
    result_status: str,
    error_code: str,
    error_summary: str,
) -> WorkerExecutionResult:
    return WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=request.run_id,
        task_id=request.task_id,
        routing_plan_id=request.routing_plan_id,
        routing_decision_id=request.routing_decision_id,
        agent_id=request.agent_id,
        capability_id=request.capability_id,
        execution_status="failed",
        result_status=result_status,  # type: ignore[arg-type]
        error_code=error_code,
        output_artifact_refs={},
        compact_summary={},
        tool_call_summary=ToolCallSummary(),
        skipped_or_failed_tools=[],
        error_summary=error_summary,
    )


def _failure_result_from_task(
    task: Task,
    *,
    agent_id: str,
    capability_id: str,
    result_status: str,
    error_code: str,
    error_summary: str,
) -> WorkerExecutionResult:
    """Build a compact failure result for a task that failed BEFORE the request
    body could be trusted (e.g. malformed body).

    Correlation identity is taken from ``Task.metadata`` ONLY when the metadata
    itself passes ``A2ATaskMetadata`` schema validation — so a malformed body
    with valid metadata still returns a fully-correlated result. If the metadata
    is invalid/missing, a compact ``unknown`` fallback is used and no unvalidated
    metadata string is copied into the result.
    """
    meta = getattr(task, "metadata", None)
    parsed = None
    if isinstance(meta, dict) and meta:
        try:
            parsed = A2ATaskMetadata.model_validate(meta)
        except ValidationError:
            parsed = None

    if parsed is not None:
        return WorkerExecutionResult(
            payload_type="worker_execution_result",
            payload_version="v1",
            run_id=parsed.run_id,
            task_id=parsed.task_id,
            routing_plan_id=parsed.routing_plan_id,
            routing_decision_id=parsed.routing_decision_id,
            agent_id=parsed.agent_id,
            capability_id=parsed.capability_id,
            execution_status="failed",
            result_status=result_status,  # type: ignore[arg-type]
            error_code=error_code,
            output_artifact_refs={},
            compact_summary={},
            tool_call_summary=ToolCallSummary(),
            skipped_or_failed_tools=[],
            error_summary=error_summary,
        )

    task_id = str(getattr(task, "id", "") or "") or "unknown_task"
    return WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id="unknown_run",
        task_id=task_id,
        agent_id=agent_id,
        capability_id=capability_id,
        execution_status="failed",
        result_status=result_status,  # type: ignore[arg-type]
        error_code=error_code,
        output_artifact_refs={},
        compact_summary={},
        tool_call_summary=ToolCallSummary(),
        skipped_or_failed_tools=[],
        error_summary=error_summary,
    )


# ── helpers ──────────────────────────────────────────────────────────────────
def _representative_capability(core: WorkerCore) -> str:
    caps = sorted(core.capability_ids)
    return caps[0] if caps else ""


def _extract_message_text(task: Task) -> Optional[str]:
    """Extract the JSON text from a python-a2a Task message (both the standard
    ``content.text`` and Google ``parts[].text`` shapes)."""
    message = getattr(task, "message", None)
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    if isinstance(content, str):
        return content
    parts = message.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    return None


# ── Flask app assembly + production entrypoint ───────────────────────────────
class WorkerPortInUseError(RuntimeError):
    """Raised (fail-fast) when the configured worker host/port is already bound."""


def effective_url_port(url: str) -> Optional[int]:
    """Effective TCP port an AgentCard ``url`` advertises.

    Uses the explicit port when present; otherwise falls back to the scheme
    default (http -> 80, https -> 443). Returns ``None`` when the port cannot be
    determined (unknown scheme, no port).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        explicit_port = parsed.port
    except ValueError:
        return None
    if explicit_port is not None:
        return explicit_port
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def assert_advertised_url_matches_port(url: str, port: int, *, agent_id: str = "") -> None:
    """Fail fast if the AgentCard-advertised URL port does not match the bind port.

    ``http://step5-worker:8005`` + bind 8005 passes; ``http://step5-worker`` (no
    port -> effective 80) + bind 9000 fails so discovery never drifts from the
    real service port.
    """
    eff = effective_url_port(url)
    if eff is None:
        raise ValueError(
            f"advertised AgentCard url for worker '{agent_id}' must be a valid "
            "HTTP(S) URL with a determinable effective port."
        )
    if eff != port:
        raise ValueError(
            f"advertised AgentCard url port {eff} for worker '{agent_id}' does not "
            f"match bind port {port}; the advertised URL must match the service port."
        )


def create_worker_flask_app(core: WorkerCore):
    """Assemble the Flask app serving ``core`` over real HTTP A2A.

    Endpoints:
      - GET  /health        compact worker status
      - GET  /agent-card    the worker's python-a2a AgentCard (JSON)
      - POST /a2a/tasks/send (+ /tasks/send)   A2A task endpoint (python-a2a)
      - GET  /a2a/agent.json, /.well-known/agent.json   (python-a2a discovery)
    """
    from flask import jsonify

    adapter = A2AWorkerAdapter(core)
    app = create_flask_app(adapter)

    @app.route("/health", methods=["GET"])
    def _health():
        return jsonify(core.health())

    @app.route("/agent-card", methods=["GET"])
    def _agent_card():
        return jsonify(core.agent_card.to_dict())

    return app


def _assert_port_available(host: str, port: int, *, agent_id: str) -> None:
    """Fail fast with :class:`WorkerPortInUseError` if ``host:port`` is taken.

    A pre-bind probe is used because werkzeug's ``BaseWSGIServer`` swallows the
    bind ``OSError`` and calls ``sys.exit(1)`` (a ``SystemExit`` that would
    otherwise escape ``make_server``). The probe does NOT set ``SO_REUSEADDR`` so
    it reliably fails while another process is actively listening on the port.
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise WorkerPortInUseError(
            f"worker '{agent_id}' cannot bind {host}:{port} "
            f"({exc.strerror or exc}); refusing to fall back to another port."
        ) from exc
    finally:
        probe.close()


def serve_worker_http(
    core: WorkerCore,
    *,
    host: str,
    port: int,
) -> None:
    """Bind + serve the worker. Fails fast (no silent fallback) if the configured
    host/port is already in use — never picks a different port, never falls back
    to an in-process call, and never leaks a ``SystemExit`` to the caller."""
    from werkzeug.serving import make_server

    _assert_port_available(host, port, agent_id=core.agent_id)
    assert_advertised_url_matches_port(
        str(getattr(core.agent_card, "url", "") or ""),
        port,
        agent_id=core.agent_id,
    )
    app = create_worker_flask_app(core)
    try:
        server = make_server(host, port, app)
    except (OSError, SystemExit) as exc:  # defence in depth against a TOCTOU race
        raise WorkerPortInUseError(
            f"worker '{core.agent_id}' cannot bind {host}:{port} "
            f"({exc}); refusing to fall back to another port."
        ) from exc
    server.serve_forever()  # pragma: no cover - blocking production loop


__all__ = [
    "WorkerCore",
    "WorkerRequestRejected",
    "WorkerPortInUseError",
    "WorkerServerConfigurationError",
    "A2AWorkerAdapter",
    "create_worker_flask_app",
    "serve_worker_http",
    "effective_url_port",
    "assert_advertised_url_matches_port",
]
