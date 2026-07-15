"""Step 6 interpretation helpers.

Turn raw ToolUniverse-backed tool payloads into compact per-lane
`liability_flags` and a `lane_risk_category`. The full raw payload stays in
`tool_outputs/step_06/{tool_call_id}.json`; only short structured evidence is
copied into the normalized record, and each flag carries a `source_ref`
pointing back to that file.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

from .step_06_capability_registry import STEP_06_CAPABILITY_BY_TOOL

Severity = Literal["low", "medium", "high"]
LaneRisk = Literal["low", "medium", "high", "unknown"]

_MAX_FLAGS_PER_TOOL = 3
_QED_THRESHOLD = 0.40

# Motif name fragments that escalate severity.
_HIGH_RISK_MOTIF_TOKENS = ("GLYCOSYL", "DEAMID", "OXID", "ISOMER", "CLEAVAGE", "FREE_CYS")


def _truncate(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _list_items(payload: dict, keys: Iterable[str]) -> list[dict]:
    for k in keys:
        v = payload.get(k)
        if isinstance(v, list) and v:
            return [x if isinstance(x, dict) else {"value": x} for x in v]
    return []


def _item_label(item: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return str(item[k])
    return None


def _flag(
    *,
    flag_type: str,
    severity: Severity,
    evidence: str,
    source_tool: str,
    source_ref: Optional[str],
) -> dict:
    return {
        "flag_type": flag_type,
        "severity": severity,
        "evidence_summary": _truncate(evidence),
        "source_tool": source_tool,
        "source_ref": source_ref,
    }


def _motif_severity(name: str) -> Severity:
    upper = name.upper()
    if any(tok in upper for tok in _HIGH_RISK_MOTIF_TOKENS):
        return "high"
    return "medium"


def _interpret_pains(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    alerts = _list_items(payload, ("alerts", "pains_alerts", "hits"))
    out: list[dict] = []
    for item in alerts[:_MAX_FLAGS_PER_TOOL]:
        name = _item_label(item, ("alert_name", "alert", "name", "id", "value")) or "alert"
        out.append(
            _flag(
                flag_type="pains_alert",
                severity="high",
                evidence=f"PAINS/structural alert: {name}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        )
    return out


def _interpret_lipinski(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    violations = payload.get("violations")
    count = 0
    if isinstance(violations, list):
        count = len(violations)
    elif isinstance(violations, int):
        count = violations
    passes = payload.get("passes")
    if count or passes is False:
        return [
            _flag(
                flag_type="lipinski_violation",
                severity="medium",
                evidence=f"Lipinski violation count: {count or 'unspecified'}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        ]
    return []


def _interpret_qed(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    qed = payload.get("qed")
    try:
        qed_val = float(qed) if qed is not None else None
    except (TypeError, ValueError):
        qed_val = None
    if qed_val is not None and qed_val < _QED_THRESHOLD:
        return [
            _flag(
                flag_type="low_qed",
                severity="medium",
                evidence=f"QED={qed_val:.2f} below threshold {_QED_THRESHOLD:.2f}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        ]
    return []


def _interpret_adme(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    issues = _list_items(payload, ("warnings", "alerts", "violations", "issues"))
    if not issues:
        return []
    label = _item_label(issues[0], ("name", "alert", "warning", "type", "value")) or "ADME warning"
    return [
        _flag(
            flag_type="adme_alert",
            severity="medium",
            evidence=f"{len(issues)} ADME/druglikeness alert(s); first: {label}",
            source_tool=source_tool,
            source_ref=source_ref,
        )
    ]


def _interpret_toxicity(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    preds = payload.get("predictions") or payload.get("toxicity") or payload.get("alerts")
    if not preds:
        return []
    if isinstance(preds, dict):
        risks = [(k, v) for k, v in preds.items() if isinstance(v, (int, float)) and v >= 0.5]
        if not risks:
            return []
        top = sorted(risks, key=lambda kv: kv[1], reverse=True)[0]
        return [
            _flag(
                flag_type="toxicity_alert",
                severity="high",
                evidence=f"toxicity endpoint {top[0]}={top[1]}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        ]
    if isinstance(preds, list) and preds:
        label = _item_label(preds[0] if isinstance(preds[0], dict) else {"value": preds[0]},
                            ("endpoint", "name", "value")) or "toxicity"
        return [
            _flag(
                flag_type="toxicity_alert",
                severity="high",
                evidence=f"{len(preds)} toxicity prediction(s); first: {label}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        ]
    return []


def _interpret_motifs(payload: dict, source_tool: str, source_ref: Optional[str]) -> list[dict]:
    items = _list_items(payload, ("motifs", "matches", "scan_hits", "hits"))
    out: list[dict] = []
    for item in items[:_MAX_FLAGS_PER_TOOL]:
        name = _item_label(item, ("name", "motif", "pattern", "id")) or "motif"
        out.append(
            _flag(
                flag_type="motif_match",
                severity=_motif_severity(name),
                evidence=f"sequence motif match: {name}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        )
    return out


def _interpret_protein_features(
    payload: dict, source_tool: str, source_ref: Optional[str]
) -> list[dict]:
    items = _list_items(payload, ("features", "feature_list", "hits"))
    if not items:
        return []
    first = _item_label(items[0], ("type", "feature", "name", "category")) or "feature"
    return [
        _flag(
            flag_type="protein_feature_context",
            severity="low",
            evidence=f"{len(items)} annotated feature(s); first: {first}",
            source_tool=source_tool,
            source_ref=source_ref,
        )
    ]


def _interpret_epitopes(
    payload: dict, source_tool: str, source_ref: Optional[str]
) -> list[dict]:
    items = _list_items(payload, ("epitopes", "predictions", "hits"))
    if not items:
        return []
    return [
        _flag(
            flag_type="epitope_context",
            severity="low",
            evidence=f"{len(items)} epitope prediction(s)",
            source_tool=source_tool,
            source_ref=source_ref,
        )
    ]


def _interpret_structure_quality(
    payload: dict, source_tool: str, source_ref: Optional[str]
) -> list[dict]:
    quality = payload.get("quality") or payload.get("quality_label")
    score = payload.get("quality_score")
    bad_label = isinstance(quality, str) and quality.lower() in {"low", "poor", "bad"}
    bad_score = isinstance(score, (int, float)) and score < 0.5
    if bad_label or bad_score:
        return [
            _flag(
                flag_type="low_structure_quality",
                severity="medium",
                evidence=f"structure quality={quality or score}",
                source_tool=source_tool,
                source_ref=source_ref,
            )
        ]
    return []


# Tool-name → (interpreter, allowed lane_types).
# A given small-molecule interpreter must never fire on the antibody lane,
# and vice versa, even if a tool happens to be routed cross-lane.
_SMALL_MOLECULE_LANE = "payload_linker_compound_liability"
_ANTIBODY_LANE = "antibody_protein_sequence_liability"
_ANTIGEN_LANE = "antigen_protein_feature_context"
_STRUCTURE_LANE = "structure_interface_quality"

_INTERPRETER_BY_TYPE: dict[str, Any] = {
    "pains": _interpret_pains,
    "lipinski": _interpret_lipinski,
    "qed": _interpret_qed,
    "adme": _interpret_adme,
    "toxicity": _interpret_toxicity,
    "motifs": _interpret_motifs,
    "protein_features": _interpret_protein_features,
    "epitopes": _interpret_epitopes,
    "structure_quality": _interpret_structure_quality,
}


def interpret_tool_payload(
    tool_name: str,
    payload: Any,
    *,
    source_ref: Optional[str],
    lane_type: Optional[str] = None,
) -> list[dict]:
    """Return compact liability flags for one tool payload.

    Lane-scoped: small-molecule interpreters (PAINS/Lipinski/QED/SwissADME/
    ADMETAI) only fire on the small-molecule lane; motif/feature interpreters
    only fire on antibody/antigen lanes; structure-quality only on the
    structure lane. Cross-lane routing yields zero flags so antibody
    developability is never reduced to small-molecule heuristics.
    """
    if not isinstance(payload, dict):
        return []
    capability = STEP_06_CAPABILITY_BY_TOOL.get(tool_name)
    if capability is None:
        return []
    if lane_type is not None and capability.lane_type != lane_type:
        return []
    fn = _INTERPRETER_BY_TYPE.get(capability.output_interpreter_type)
    if fn is None:
        return []
    try:
        return fn(payload, tool_name, source_ref)
    except Exception:  # noqa: BLE001 — interpretation never breaks the pipeline
        return []


def aggregate_lane_risk(
    flags: list[dict], *, any_success: bool, all_dependency_unavailable: bool
) -> LaneRisk:
    severities = {f.get("severity") for f in flags}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    if "low" in severities:
        return "low"
    if all_dependency_unavailable or not any_success:
        return "unknown"
    return "low"


# Per-lane ADC-developability aspects that Step 6 does NOT assess; surfaced
# in lane_summary so downstream steps / reviewers see the gap explicitly.
_LANE_UNASSESSED_NOTES: dict[str, str] = {
    _ANTIBODY_LANE: (
        "ADC-specific antibody developability (DAR, N297 glycosylation, "
        "Fc linker attachment site, heavy/light chain pairing) is not "
        "assessed in Step 6 and requires downstream assessment"
    ),
    _ANTIGEN_LANE: (
        "antigen-side conjugation context (epitope vs linker attachment, "
        "DAR impact on binding) is not assessed in Step 6 and requires "
        "downstream assessment"
    ),
    _STRUCTURE_LANE: (
        "ADC conjugate-level structural quality (DAR-dependent pose, "
        "linker geometry, Fc impact) is not assessed in Step 6 and "
        "requires downstream assessment"
    ),
}


# ── Reviewer-facing structured interpretation ──────────────────────────────
#
# Translate lane execution + interpreted flags into explicit assessment_status
# / risk_label / not_assessed_reason / interpreted_findings /
# missing_or_unassessed_items. Pure functions over already-computed inputs —
# they do NOT change tool selection, schema mapping, MCP calls, or runtime
# resolution, and never read or emit raw payloads / sequences.

# Per-lane required typed input that, when absent, makes the lane
# `not_assessed_missing_input`. Used to build structured gap entries.
_LANE_REQUIRED_INPUT: dict[str, dict[str, str]] = {
    _SMALL_MOLECULE_LANE: {
        "item": "payload/linker compound SMILES",
        "reason": "no payload/linker SMILES available for this candidate",
        "suggested_next_input": "a payload or linker SMILES string",
    },
    _ANTIBODY_LANE: {
        "item": "antibody heavy/light chain sequence",
        "reason": "no antibody protein sequence available for this candidate",
        "suggested_next_input": "antibody heavy and/or light chain amino-acid sequence",
    },
    _ANTIGEN_LANE: {
        "item": "antigen UniProt / accession",
        "reason": "no target/antigen UniProt accession available for this candidate",
        "suggested_next_input": "a target/antigen UniProt accession",
    },
    _STRUCTURE_LANE: {
        "item": "structure / PDB reference",
        "reason": "no structure or PDB reference available for this candidate",
        "suggested_next_input": "a PDB ID or uploaded structure file",
    },
    "compound_bioactivity_prior_context": {
        "item": "compound ChEMBL ID",
        "reason": "no compound ChEMBL ID available for this candidate",
        "suggested_next_input": "a compound ChEMBL ID",
    },
}

# severity → reviewer risk label.
_SEVERITY_TO_LABEL = {"high": "high", "medium": "review", "low": "low"}


def lane_missing_input_item(lane_type: str) -> dict:
    """Structured (not free-text) missing-input descriptor for a lane."""
    spec = _LANE_REQUIRED_INPUT.get(
        lane_type,
        {
            "item": "required typed input",
            "reason": "required input not available for this candidate",
            "suggested_next_input": "the input this lane consumes",
        },
    )
    return {
        "item": spec["item"],
        "reason": spec["reason"],
        "blocking": False,
        "suggested_next_input": spec["suggested_next_input"],
    }


def interpreted_findings_from_flags(
    flags: list[dict], tool_records: Optional[list[Any]] = None
) -> list[dict]:
    """Map compact liability flags to reviewer-facing interpreted findings.

    Carries only short evidence + source references (tool name + tool_call_id
    by reference), never the raw payload.
    """
    ref_to_call_id: dict[str, str] = {}
    for record in tool_records or []:
        ref = getattr(record, "tool_output_ref", None)
        call_id = getattr(record, "tool_call_id", None)
        if ref and call_id:
            ref_to_call_id[ref] = call_id
    out: list[dict] = []
    for flag in flags:
        source_ref = flag.get("source_ref")
        source_tool = flag.get("source_tool")
        call_ids = [ref_to_call_id[source_ref]] if source_ref in ref_to_call_id else []
        out.append(
            {
                "finding_type": flag.get("flag_type"),
                "label": _SEVERITY_TO_LABEL.get(str(flag.get("severity")), "review"),
                "evidence_summary": flag.get("evidence_summary"),
                "source_tools": [source_tool] if source_tool else [],
                "source_tool_call_ids": call_ids,
            }
        )
    return out


def all_tool_records_skipped_for_missing_typed_input(
    tool_records: Optional[list[Any]],
) -> bool:
    """Whether every planned call was skipped for concrete typed-input gaps.

    Policy-only skips do not qualify: every record must carry either a
    non-empty ``missing_required_fields`` list or a runtime resolver audit with
    an unresolved/missing required reference.
    """

    records = list(tool_records or [])
    if not records:
        return False
    for record in records:
        if getattr(record, "run_status", None) not in {"skipped", "not_run"}:
            return False
        summary = getattr(record, "tool_input_summary", None)
        if not isinstance(summary, dict):
            return False
        missing_required = summary.get("missing_required_fields")
        resolver_audit = summary.get("runtime_resolver_audit")
        has_missing_required = isinstance(missing_required, list) and bool(
            missing_required
        )
        has_unresolved_ref = isinstance(resolver_audit, list) and any(
            isinstance(entry, dict)
            and entry.get("resolve_status") in {"missing", "unresolved"}
            for entry in resolver_audit
        )
        if not (has_missing_required or has_unresolved_ref):
            return False
    return True


def missing_typed_input_item_from_tool_records(
    tool_records: Optional[list[Any]],
) -> dict:
    """Build one compact gap item from declared/resolver field names only."""

    missing_field_names: set[str] = set()
    for record in tool_records or []:
        summary = getattr(record, "tool_input_summary", None)
        if not isinstance(summary, dict):
            continue
        missing_required = summary.get("missing_required_fields")
        if isinstance(missing_required, list):
            missing_field_names.update(
                str(name) for name in missing_required if isinstance(name, str) and name
            )
        resolver_audit = summary.get("runtime_resolver_audit")
        if isinstance(resolver_audit, list):
            missing_field_names.update(
                str(entry.get("schema_arg"))
                for entry in resolver_audit
                if isinstance(entry, dict)
                and entry.get("resolve_status") in {"missing", "unresolved"}
                and isinstance(entry.get("schema_arg"), str)
                and entry.get("schema_arg")
            )
    return {
        "item": "required typed tool input",
        "reason": "selected tool requirements could not be resolved from available typed inputs",
        "blocking": False,
        "suggested_next_input": "the typed identifier or runtime reference required by the selected tool",
        "missing_field_names": sorted(missing_field_names),
    }


def derive_lane_assessment(
    *,
    lane_type: str,
    plans_present: bool,
    flags: list[dict],
    lane_risk_category: str,
    any_success: bool,
    all_dependency_unavailable: bool,
    has_upstream_error: bool,
    any_failed: bool,
    tool_records: Optional[list[Any]] = None,
) -> dict:
    """Compute additive reviewer-facing fields for ONE active lane."""
    findings = interpreted_findings_from_flags(flags, tool_records)
    # Always surface the ADC-developability aspects Step 6 does not assess.
    not_assessed_note = _LANE_UNASSESSED_NOTES.get(lane_type)
    extra_unassessed: list[dict] = []
    if not_assessed_note:
        extra_unassessed.append(
            {
                "item": "ADC-specific developability aspects",
                "reason": not_assessed_note,
                "blocking": False,
                "suggested_next_input": "downstream ADC-stage assessment",
            }
        )

    if not plans_present:
        return {
            "assessment_status": "not_assessed_missing_input",
            "risk_label": "not_assessed",
            "not_assessed_reason": (
                "no invokable tool for this lane (required typed input could "
                "not be resolved)"
            ),
            "interpreted_findings": [],
            "missing_or_unassessed_items": [lane_missing_input_item(lane_type)],
        }
    if has_upstream_error:
        return {
            "assessment_status": "partial_upstream_error",
            "risk_label": "review",
            "not_assessed_reason": (
                "one or more tool calls returned an upstream_error envelope; "
                "lane is not cleanly assessed"
            ),
            "interpreted_findings": findings,
            "missing_or_unassessed_items": extra_unassessed,
        }
    if all_dependency_unavailable:
        return {
            "assessment_status": "not_assessed_dependency_unavailable",
            "risk_label": "not_assessed",
            "not_assessed_reason": "tool dependency unavailable in this runtime",
            "interpreted_findings": [],
            "missing_or_unassessed_items": extra_unassessed,
        }
    if not any_failed and all_tool_records_skipped_for_missing_typed_input(
        tool_records
    ):
        return {
            "assessment_status": "not_assessed_missing_input",
            "risk_label": "not_assessed",
            "not_assessed_reason": (
                "all planned tool invocations were skipped because required "
                "typed or invokable input could not be resolved"
            ),
            "interpreted_findings": [],
            "missing_or_unassessed_items": [
                missing_typed_input_item_from_tool_records(tool_records),
                *extra_unassessed,
            ],
        }
    if not any_success:
        return {
            "assessment_status": "failed",
            "risk_label": "unknown",
            "not_assessed_reason": "tool call(s) did not produce a usable output",
            "interpreted_findings": findings,
            "missing_or_unassessed_items": extra_unassessed,
        }
    if flags:
        risk = "high" if lane_risk_category == "high" else "review"
        return {
            "assessment_status": "signal_detected",
            "risk_label": risk,
            "not_assessed_reason": None,
            "interpreted_findings": findings,
            "missing_or_unassessed_items": extra_unassessed,
        }
    # Ran successfully, no interpreted liability signal.
    return {
        "assessment_status": "no_signal",
        "risk_label": "low",
        "not_assessed_reason": None,
        "interpreted_findings": [],
        "missing_or_unassessed_items": extra_unassessed,
    }


def derive_missing_lane_assessment(lane_type: str) -> dict:
    """Reviewer-facing fields for a lane skipped due to missing input."""
    item = lane_missing_input_item(lane_type)
    return {
        "assessment_status": "not_assessed_missing_input",
        "risk_label": "not_assessed",
        "not_assessed_reason": item["reason"],
        "interpreted_findings": [],
        "missing_or_unassessed_items": [item],
    }


_ASSESSED_STATUSES = {"assessed", "no_signal", "signal_detected"}


def derive_candidate_interpretation(lane_results: list[Any]) -> dict:
    """Aggregate lane assessments into a candidate-level interpretation.

    A candidate with only some lanes assessed is explicitly `partial` context
    and never reported as fully acceptable.
    """
    statuses = [getattr(lr, "assessment_status", "not_assessed_missing_input") for lr in lane_results]
    risk_labels = [getattr(lr, "risk_label", "not_assessed") for lr in lane_results]
    assessed = [s for s in statuses if s in _ASSESSED_STATUSES]
    assessed_count = len(assessed)
    not_assessed_count = len(statuses) - assessed_count

    if assessed_count == 0:
        completeness = "none"
    elif not_assessed_count == 0:
        completeness = "complete"
    else:
        completeness = "partial"

    has_high = "high" in risk_labels
    has_signal = any(s == "signal_detected" for s in statuses)
    has_upstream = any(s == "partial_upstream_error" for s in statuses)

    if has_high:
        label = "high-risk"
    elif has_signal or has_upstream or completeness == "partial":
        # Any liability signal, an upstream error, OR incomplete context →
        # not fully acceptable, needs reviewer attention.
        label = "review"
    elif completeness == "complete":
        label = "acceptable"
    else:
        label = "unknown"

    if has_high:
        action = "deprioritize"
    elif completeness == "none":
        action = "insufficient_data"
    elif completeness == "complete" and not (has_signal or has_upstream):
        action = "continue"
    else:
        action = "continue_with_review"

    # Aggregate structured gaps from lanes (dedup by (item, reason)).
    aggregated: list[dict] = []
    seen: set[tuple] = set()
    for lr in lane_results:
        for item in getattr(lr, "missing_or_unassessed_items", []) or []:
            key = (item.get("item"), item.get("reason"))
            if key in seen:
                continue
            seen.add(key)
            aggregated.append(item)

    assessed_lanes = [
        getattr(lr, "lane_type", "?") for lr in lane_results
        if getattr(lr, "assessment_status", "") in _ASSESSED_STATUSES
    ]
    not_assessed_lanes = [
        getattr(lr, "lane_type", "?") for lr in lane_results
        if getattr(lr, "assessment_status", "") not in _ASSESSED_STATUSES
    ]
    summary = (
        f"{assessed_count} of {len(lane_results)} lane(s) assessed "
        f"[{', '.join(assessed_lanes) or 'none'}]; "
        f"not assessed [{', '.join(not_assessed_lanes) or 'none'}]; "
        f"context {completeness}; "
        + (
            "liability signal detected" if has_signal
            else "no liability signal from assessed lanes"
        )
        + ("; upstream_error present" if has_upstream else "")
        + (
            "; context incomplete — not fully acceptable"
            if completeness != "complete" else ""
        )
    )
    return {
        "context_completeness": completeness,
        "assessed_lane_count": assessed_count,
        "not_assessed_lane_count": not_assessed_count,
        "candidate_overall_liability_label": label,
        "recommended_action": action,
        "interpretation_summary": summary,
        "missing_or_unassessed_items": aggregated,
    }


def lane_summary(
    *,
    tool_records_summary: str,
    flags: list[dict],
    any_success: bool,
    lane_type: Optional[str] = None,
) -> str:
    base = tool_records_summary
    if flags:
        types = sorted({f["flag_type"] for f in flags})
        body = f"{base}; interpreted {len(flags)} liability flag(s) across types {types}"
    elif any_success:
        body = f"{base}; no interpreted liability signal from successful tool outputs"
    else:
        body = base
    note = _LANE_UNASSESSED_NOTES.get(lane_type or "")
    if note:
        body = f"{body}; {note}"
    return body
