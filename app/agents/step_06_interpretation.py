"""Step 6 interpretation helpers.

Turn raw ToolUniverse-backed tool payloads into compact per-lane
`liability_flags` and a `lane_risk_category`. The full raw payload stays in
`tool_outputs/step_06/{tool_call_id}.json`; only short structured evidence is
copied into the normalized record, and each flag carries a `source_ref`
pointing back to that file.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

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

_INTERPRETERS: dict[str, tuple[Any, frozenset[str]]] = {
    "DrugProps_pains_filter": (_interpret_pains, frozenset({_SMALL_MOLECULE_LANE})),
    "DrugProps_lipinski_filter": (_interpret_lipinski, frozenset({_SMALL_MOLECULE_LANE})),
    "DrugProps_calculate_qed": (_interpret_qed, frozenset({_SMALL_MOLECULE_LANE})),
    "SwissADME_calculate_adme": (_interpret_adme, frozenset({_SMALL_MOLECULE_LANE})),
    "SwissADME_check_druglikeness": (_interpret_adme, frozenset({_SMALL_MOLECULE_LANE})),
    "ADMETAI_predict_toxicity": (_interpret_toxicity, frozenset({_SMALL_MOLECULE_LANE})),
    "ADMETAI_predict_physicochemical_properties": (
        _interpret_adme,
        frozenset({_SMALL_MOLECULE_LANE}),
    ),
    "PROSITE_scan_sequence": (
        _interpret_motifs,
        frozenset({_ANTIBODY_LANE, _ANTIGEN_LANE}),
    ),
    "EBIProteins_get_features": (
        _interpret_protein_features,
        frozenset({_ANTIBODY_LANE, _ANTIGEN_LANE}),
    ),
    "EBIProteins_get_epitopes": (_interpret_epitopes, frozenset({_ANTIGEN_LANE})),
    "EBIProteins_get_antigen": (
        _interpret_protein_features,
        frozenset({_ANTIGEN_LANE}),
    ),
    "ProteinsPlus_profile_structure_quality": (
        _interpret_structure_quality,
        frozenset({_STRUCTURE_LANE}),
    ),
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
    entry = _INTERPRETERS.get(tool_name)
    if entry is None:
        return []
    fn, allowed_lanes = entry
    if lane_type is not None and lane_type not in allowed_lanes:
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
