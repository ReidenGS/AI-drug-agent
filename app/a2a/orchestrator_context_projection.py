"""Privacy-safe compact context for the Step 4 routing LLM."""

from __future__ import annotations

import re
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_PATTERNS = (
    (re.compile(r"(?m)^>[^\n]*(?:\n[\w.*:-]+)+"), "[REDACTED_ALIGNMENT]"),
    (
        re.compile(
            r"(?im)^(?:HEADER|ATOM  |HETATM|MODEL |ENDMDL|data_|loop_|_atom_site).*$"
        ),
        "[REDACTED_STRUCTURE]",
    ),
    (
        re.compile(
            r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b|(?:nvidia|biohub|api)[_ -]?(?:key|token)\s*[:=]\s*\S+|bearer\s+\S+"
        ),
        "[REDACTED_CREDENTIAL]",
    ),
    (re.compile(r"(?:/[^\s]+){2,}|[A-Za-z]:\\[^\s]+"), "[REDACTED_PATH]"),
    (
        re.compile(r"\b(?:[ACDEFGHIKLMNPQRSTVWY]{12,}|[ACGTUN]{12,})\b"),
        "[REDACTED_BIOLOGICAL_MATERIAL]",
    ),
    (
        re.compile(
            r"(?i)raw[_ ]tooluniverse[_ ]payload|full[_ ]prompt|raw[_ ]llm[_ ]response"
        ),
        "[REDACTED_PRIVATE_PAYLOAD]",
    ),
)


def redact_routing_text(value: Any) -> str:
    text = str(value or "") if isinstance(value, (str, int, float, bool)) else ""
    for pattern, marker in _PATTERNS:
        text = pattern.sub(marker, text)
    return " ".join(text.split())


def contains_unsafe_routing_text(value: str) -> bool:
    return any(pattern.search(str(value or "")) for pattern, _ in _PATTERNS)


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else None


def _safe_ids(values: Any) -> list[str]:
    return (
        [item for value in values if (item := _safe_id(value))]
        if isinstance(values, list)
        else []
    )


def _project_current(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"completed_routes": [], "compact_statuses": [], "warning_codes": []}
    completed = []
    for route in value.get("completed_routes", []):
        if not isinstance(route, dict):
            continue
        agent = _safe_id(route.get("agent_id"))
        capability = _safe_id(route.get("capability_id"))
        status = _safe_id(route.get("status"))
        if agent and capability and status:
            completed.append(
                {
                    "agent_id": agent,
                    "capability_id": capability,
                    "status": status,
                    "output_artifact_names": _safe_ids(
                        route.get("output_artifact_names", [])
                    ),
                }
            )
    return {
        "completed_routes": completed,
        "available_agent_ids": _safe_ids(value.get("available_agent_ids", [])),
        "unavailable_agent_ids": _safe_ids(
            value.get("unavailable_agent_ids", [])
        ),
        "compact_statuses": _safe_ids(value.get("compact_statuses", [])),
        "warning_codes": _safe_ids(value.get("warning_codes", [])),
    }


def project_orchestrator_context(
    *,
    structured_query: dict[str, Any],
    readiness: dict[str, Any],
    available_artifacts: list[dict[str, Any]],
    current_routing_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_intent = (
        structured_query.get("task_intent")
        if isinstance(structured_query.get("task_intent"), dict)
        else {}
    )
    canonical_query = structured_query.get("canonical_query") or ""
    user_goal = task_intent.get("user_goal_summary") or ""
    intent_parts = []
    if user_goal:
        intent_parts.append(f"User goal: {redact_routing_text(user_goal)}")
    if canonical_query and canonical_query != user_goal:
        intent_parts.append(
            f"Canonical query: {redact_routing_text(canonical_query)}"
        )
    intent = " ".join(intent_parts) or redact_routing_text(canonical_query)
    missing_slots = []
    for slot in structured_query.get("missing_slots", []):
        if isinstance(slot, dict) and (name := _safe_id(slot.get("slot_name"))):
            missing_slots.append(name)
    blocking_categories: set[str] = set()
    warning_categories: set[str] = set()
    for gap in readiness.get("missing_input_checklist", []):
        if not isinstance(gap, dict) or not (category := _safe_id(gap.get("category"))):
            continue
        if gap.get("severity") == "blocking":
            blocking_categories.add(category)
        elif gap.get("severity") == "warning":
            warning_categories.add(category)
    artifacts = []
    for item in available_artifacts:
        if not isinstance(item, dict) or not (
            name := _safe_id(item.get("artifact_name"))
        ):
            continue
        artifacts.append(
            {
                "artifact_name": name,
                "available": bool(item.get("available")),
                "present_field_names": _safe_ids(item.get("present_field_names", [])),
            }
        )
    result = {
        "compact_user_intent": redact_routing_text(intent),
        "structured_intent": {
            "primary_intent": _safe_id(task_intent.get("primary_intent")),
            "secondary_intents": _safe_ids(task_intent.get("secondary_intents", [])),
            "requested_outputs": _safe_ids(
                structured_query.get("requested_outputs", [])
            ),
        },
        "input_readiness_summary": {
            "input_readiness_status": _safe_id(readiness.get("input_readiness_status")),
            "missing_slot_names": missing_slots,
            "blocking_gap_categories": sorted(blocking_categories),
            "warning_gap_categories": sorted(warning_categories),
        },
        "available_artifact_summary": artifacts,
        "current_routing_context": _project_current(current_routing_context),
    }
    if contains_unsafe_routing_text(str(result)):
        raise ValueError("orchestrator_context_privacy_validation_failed")
    return result
