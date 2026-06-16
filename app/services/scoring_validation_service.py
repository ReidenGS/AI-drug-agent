"""Step 11 — deterministic scoring table validation.

Expected external input shape (Yufei AEE module writes this file under
`{run_id}/inputs/external_scoring_result.json`):

```json
{
  "scored_at": "2026-06-15T...",
  "candidates": [
    {
      "candidate_id": "candidate_xxxx",
      "total_score": 7.4,
      "dimensions": {
        "docking_score": -8.2,
        "developability_score": 6.5,
        "evidence_score": 7.0,
        "patent_risk_score": 2.1
      },
      "notes": "optional free text"
    }
  ]
}
```

If the file is absent, `validation_status="awaiting_external_input"` and no
candidates are validated. If present, each row is checked for required
fields, numeric scores in `[0, 10]` (or `[-20, 20]` for raw docking scores),
and candidate_id membership in Step 10's `candidate_ids`. Raw rows stay in
the input file; the validation artifact only carries:
- `validated_candidate_ids` that survived
- `issues[]` (severity + field + message)
- `external_scoring_input_ref` pointing back to the storage key
"""

from __future__ import annotations

from typing import Any

from ..schemas.step_11_scoring_validation import ScoringValidation, ValidationIssue
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "scoring_validation.json"
_EXTERNAL_INPUT_KEY = "inputs/external_scoring_result.json"

_REQUIRED_FIELDS = ("candidate_id", "total_score")
# Total scores are normalized to 0-10 per AEE convention.
_TOTAL_SCORE_RANGE = (0.0, 10.0)
# Docking is allowed to be wider (kcal/mol-style).
_DIMENSION_RANGES: dict[str, tuple[float, float]] = {
    "developability_score": (0.0, 10.0),
    "evidence_score": (0.0, 10.0),
    "patent_risk_score": (0.0, 10.0),
    "docking_score": (-50.0, 50.0),
}


class ScoringValidationService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def validate(self, run_id: str) -> ScoringValidation:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.scoring_handoff_id:
            raise WorkflowStateError("Step 11 requires Step 10 scoring_handoff_package")

        handoff = self.storage.read_json(
            self.storage.run_key(run_id, "scoring_handoff_package.json")
        )
        known_candidate_ids = set(handoff.get("candidate_ids") or [])

        input_key = self.storage.run_key(run_id, _EXTERNAL_INPUT_KEY)
        if not self.storage.exists(input_key):
            result = ScoringValidation(
                run_id=run_id,
                created_at=now_iso(),
                validation_status="awaiting_external_input",
                external_scoring_input_ref=None,
                scoring_table_storage_path="",
                validated_candidate_ids=[],
                issues=[],
                row_count=0,
                notes=(
                    "No external scoring result found at "
                    f"`{_EXTERNAL_INPUT_KEY}`. Drop the Yufei AEE output there "
                    "to advance Step 11."
                ),
            )
            return self._persist(run_id, result)

        external = self.storage.read_json(input_key)
        rows = external.get("candidates") or []
        issues: list[ValidationIssue] = []
        validated_ids: list[str] = []

        for idx, row in enumerate(rows):
            cid = row.get("candidate_id")
            row_label = cid or f"row[{idx}]"

            row_issues: list[ValidationIssue] = []
            for field in _REQUIRED_FIELDS:
                if row.get(field) is None:
                    row_issues.append(
                        ValidationIssue(
                            candidate_id=cid,
                            severity="error",
                            field=field,
                            message=f"required field `{field}` missing in {row_label}",
                        )
                    )

            total = row.get("total_score")
            if total is not None and not _is_number(total):
                row_issues.append(
                    ValidationIssue(
                        candidate_id=cid, severity="error", field="total_score",
                        message=f"total_score must be numeric, got {type(total).__name__}",
                    )
                )
            elif _is_number(total) and not _in_range(total, _TOTAL_SCORE_RANGE):
                row_issues.append(
                    ValidationIssue(
                        candidate_id=cid, severity="warning", field="total_score",
                        message=f"total_score={total} outside expected {_TOTAL_SCORE_RANGE}",
                    )
                )

            for dim_name, dim_value in (row.get("dimensions") or {}).items():
                if not _is_number(dim_value):
                    row_issues.append(
                        ValidationIssue(
                            candidate_id=cid, severity="error",
                            field=f"dimensions.{dim_name}",
                            message=f"dimension `{dim_name}` must be numeric",
                        )
                    )
                    continue
                bounds = _DIMENSION_RANGES.get(dim_name)
                if bounds and not _in_range(dim_value, bounds):
                    row_issues.append(
                        ValidationIssue(
                            candidate_id=cid, severity="warning",
                            field=f"dimensions.{dim_name}",
                            message=(
                                f"dimension `{dim_name}`={dim_value} outside {bounds}"
                            ),
                        )
                    )

            if cid and cid not in known_candidate_ids:
                row_issues.append(
                    ValidationIssue(
                        candidate_id=cid, severity="warning", field="candidate_id",
                        message=(
                            f"candidate_id `{cid}` not in Step 10 handoff candidate_ids"
                        ),
                    )
                )

            if cid and not any(i.severity == "error" for i in row_issues):
                validated_ids.append(cid)
            issues.extend(row_issues)

        has_error = any(i.severity == "error" for i in issues)
        has_warn = any(i.severity == "warning" for i in issues)
        if has_error and not validated_ids:
            status = "failed"
        elif has_error or has_warn:
            status = "completed_with_warnings"
        else:
            status = "completed"

        result = ScoringValidation(
            run_id=run_id,
            created_at=now_iso(),
            validation_status=status,  # type: ignore[arg-type]
            external_scoring_input_ref=input_key,
            scoring_table_storage_path=input_key,
            validated_candidate_ids=validated_ids,
            issues=issues,
            row_count=len(rows),
            notes=None,
        )
        return self._persist(run_id, result)

    def _persist(self, run_id: str, result: ScoringValidation) -> ScoringValidation:
        artifact_id = new_artifact_id("scoring_validation")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **result.model_dump()},
        )
        self.registry.update_active(run_id, scoring_validation_id=artifact_id)
        self.workflow_state.mark(run_id, "step_11", "completed")
        return result


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _in_range(v: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return lo <= float(v) <= hi
