"""Opt-in live Gemini Step 2 benchmark smoke.

Runs the professor's six Step 2 benchmark prompts through the real
GeminiProvider via SupervisorAgent. This script intentionally skips
cleanly unless both LLM_PROVIDER=gemini and GEMINI_API_KEY are set.

Output is compact by design: no API keys, no full prompts, and no raw
Gemini responses are printed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from app.deps import get_llm_provider  # noqa: E402
from app.llm.gemini_provider import GeminiProvider  # noqa: E402
from app.schemas.step_01_raw_request_record import (  # noqa: E402
    RawRequestRecord,
    UserProvidedContext,
)
from app.settings import get_settings  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    query: str
    ctx: dict[str, Any]
    validator: Callable[[Any], list[str]]


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="her2_tdm1_tdxd",
            query=(
                "We want to evaluate the HER2 ADC benchmark comparing T-DM1 vs "
                "T-DXd (Enhertu) for breast cancer treatment."
            ),
            ctx={},
            validator=validate_case_1_her2_tdm1_tdxd,
        ),
        BenchmarkCase(
            name="trop2_mmae_new_adc",
            query="Design a new TROP2 ADC with MMAE payload for solid tumors.",
            ctx={},
            validator=validate_case_2_trop2_mmae,
        ),
        BenchmarkCase(
            name="her2_pdb_zinc",
            query=(
                "For HER2, use PDB 1N8Z and screen ZINC12345678 plus related "
                "ZINC compounds for possible payload candidates."
            ),
            ctx={},
            validator=validate_case_3_her2_pdb_zinc,
        ),
        BenchmarkCase(
            name="cldn18_patent_deruxtecan",
            query=(
                "Run a patent and IP review for a CLDN18.2 ADC using a "
                "deruxtecan-like payload."
            ),
            ctx={},
            validator=validate_case_4_cldn18_patent,
        ),
        BenchmarkCase(
            name="literature_trastuzumab_mmae",
            query="Please review the literature on trastuzumab and MMAE for me.",
            ctx={},
            validator=validate_case_5_literature_trastuzumab_mmae,
        ),
        BenchmarkCase(
            name="chembl_zinc_her2_payload_candidates",
            query=(
                "Screen ChEMBL1201585 and ZINC12345678 as possible HER2 payload "
                "candidates and summarize developability evidence."
            ),
            ctx={},
            validator=validate_case_6_chembl_zinc_payloads,
        ),
    ]


def raw_record_for_case(case: BenchmarkCase, index: int) -> dict[str, Any]:
    rec = RawRequestRecord(
        run_id=f"gemini_step2_bench_{index}",
        run_artifact_registry_id=f"gemini_step2_bench_registry_{index}",
        created_at="2026-06-18T00:00:00Z",
        raw_user_query=case.query,
        user_provided_context=UserProvidedContext(**case.ctx),
    )
    out = rec.model_dump()
    out["artifact_id"] = f"raw_request_record_gemini_step2_bench_{index}"
    return out


def validate_case_1_her2_tdm1_tdxd(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "existing_adc_evaluation":
        failures.append("invalid primary_intent")
    if "literature_review" not in _secondary_intents(sq):
        failures.append("missing literature_review secondary intent")
    if not _has_normalized(sq, "ERBB2"):
        failures.append("missing ERBB2 normalization")
    if not _has_decomposition(sq, "T-DM1"):
        failures.append("missing T-DM1 decomposition")
    if not (_has_decomposition(sq, "T-DXd") or _has_decomposition(sq, "Enhertu")):
        failures.append("missing T-DXd/Enhertu decomposition")
    if not _has_any_requested(sq, {"evidence_summary", "literature_review_summary"}):
        failures.append("missing evidence/literature requested output")
    return failures


def validate_case_2_trop2_mmae(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "new_adc_design":
        failures.append("invalid primary_intent")
    if not _has_normalized(sq, "TACSTD2"):
        failures.append("missing TACSTD2 normalization")
    if not (
        _has_normalized(sq, "monomethyl auristatin E")
        or _has_normalized_contains(sq, "mmae")
    ):
        failures.append("missing MMAE normalization")
    questions = " ".join(_clarification_questions(sq)).lower()
    if "antibody" not in questions and "linker" not in questions:
        failures.append("missing antibody/linker clarification question")
    return failures


def validate_case_3_her2_pdb_zinc(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "structure_analysis":
        failures.append("invalid primary_intent")
    if "compound_screening" not in _secondary_intents(sq):
        failures.append("missing compound_screening secondary intent")
    if not _has_ref_type(sq, "pdb_id"):
        failures.append("missing pdb_id referenced input")
    if not _has_ref_type(sq, "zinc_id"):
        failures.append("missing zinc_id referenced input")
    if not _has_any_requested(
        sq, {"structure_validation_report", "compound_screening_results"}
    ):
        failures.append("missing structure/screening requested output")
    return failures


def validate_case_4_cldn18_patent(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "patent_ip_review":
        failures.append("invalid primary_intent")
    if not (
        _has_normalized(sq, "CLDN18 isoform 2")
        or _has_normalized_contains(sq, "cldn18.2")
        or _has_normalized_contains(sq, "claudin18.2")
    ):
        failures.append("missing CLDN18.2 normalization")
    if "patent_or_ip_summary" not in _requested_outputs(sq):
        failures.append("missing patent_or_ip_summary requested output")
    return failures


def validate_case_5_literature_trastuzumab_mmae(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "literature_review":
        failures.append("invalid primary_intent")
    if not _has_any_requested(sq, {"literature_review_summary", "evidence_summary"}):
        failures.append("missing literature/evidence requested output")
    if not _clarification_questions(sq):
        failures.append("missing clarification question for unclear HER2/ADC context")
    return failures


def validate_case_6_chembl_zinc_payloads(sq: Any) -> list[str]:
    failures: list[str] = []
    if _primary_intent(sq) != "compound_screening":
        failures.append("invalid primary_intent")
    if not _has_ref_type(sq, "chembl_id"):
        failures.append("missing chembl_id referenced input")
    if not _has_ref_type(sq, "zinc_id"):
        failures.append("missing zinc_id referenced input")
    secondary = _secondary_intents(sq)
    if not ({"developability_assessment", "literature_review"} & set(secondary)):
        failures.append("missing developability/literature secondary intent")
    if "compound_screening_results" not in _requested_outputs(sq):
        failures.append("missing compound_screening_results requested output")
    return failures


def compact_summary(case_name: str, sq: Any, failures: list[str]) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "status": "PASS" if not failures else "FAIL",
        "primary_intent": _primary_intent(sq),
        "secondary_intents": _secondary_intents(sq),
        "requested_outputs_count": len(_requested_outputs(sq)),
        "normalized_entities_count": len(_normalized_entities(sq)),
        "decompositions_count": len(_decompositions(sq)),
        "clarification_questions_count": len(_clarification_questions(sq)),
        "reason": "; ".join(failures) if failures else "",
    }


def print_summary(summary: dict[str, Any]) -> None:
    reason_suffix = f" reason={summary['reason']}" if summary["reason"] else ""
    print(
        "case_name={case_name} status={status} primary_intent={primary_intent} "
        "secondary_intents={secondary_intents} requested_outputs_count={requested_outputs_count} "
        "normalized_entities_count={normalized_entities_count} "
        "decompositions_count={decompositions_count} "
        "clarification_questions_count={clarification_questions_count}{reason_suffix}".format(
            **summary,
            reason_suffix=reason_suffix,
        )
    )


def main() -> int:
    settings = get_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print(
            "SKIP: set LLM_PROVIDER=gemini and GEMINI_API_KEY to run the live "
            "Gemini Step 2 benchmark smoke."
        )
        return 0

    provider = get_llm_provider()
    if not isinstance(provider, GeminiProvider):
        raise RuntimeError(f"Expected GeminiProvider, got {type(provider).__name__}")

    agent = SupervisorAgent(llm=provider)
    any_failed = False
    for index, case in enumerate(benchmark_cases(), start=1):
        try:
            sq = agent.parse_raw_to_structured_query(raw_record_for_case(case, index))
            failures = case.validator(sq)
        except Exception as exc:  # noqa: BLE001 - compact smoke failure report
            sq = {}
            failures = [_exception_reason(exc)]
        if failures:
            any_failed = True
        print_summary(compact_summary(case.name, sq, failures))
    return 1 if any_failed else 0


def _primary_intent(sq: Any) -> str | None:
    task_intent = _get(sq, "task_intent") or {}
    return _get(task_intent, "primary_intent")


def _secondary_intents(sq: Any) -> list[str]:
    task_intent = _get(sq, "task_intent") or {}
    value = _get(task_intent, "secondary_intents") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _requested_outputs(sq: Any) -> list[str]:
    value = _get(sq, "requested_outputs") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _normalized_entities(sq: Any) -> list[Any]:
    value = _get(sq, "normalized_entities") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _decompositions(sq: Any) -> list[Any]:
    value = _get(sq, "entity_decompositions") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _clarification_questions(sq: Any) -> list[str]:
    value = _get(sq, "clarification_questions") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _referenced_inputs(sq: Any) -> list[Any]:
    value = _get(sq, "referenced_inputs") or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _has_normalized(sq: Any, canonical_name: str) -> bool:
    needle = canonical_name.lower()
    return any(str(_get(ne, "canonical_name") or "").lower() == needle for ne in _normalized_entities(sq))


def _has_normalized_contains(sq: Any, text: str) -> bool:
    needle = text.lower()
    for ne in _normalized_entities(sq):
        if needle in str(_get(ne, "canonical_name") or "").lower():
            return True
        if needle in str(_get(ne, "original_text") or "").lower():
            return True
    return False


def _has_decomposition(sq: Any, original_text: str) -> bool:
    needle = original_text.lower()
    return any(needle in str(_get(d, "original_text") or "").lower() for d in _decompositions(sq))


def _has_ref_type(sq: Any, id_type: str) -> bool:
    return any(_get(ref, "id_type") == id_type for ref in _referenced_inputs(sq))


def _has_any_requested(sq: Any, values: set[str]) -> bool:
    return bool(set(_requested_outputs(sq)) & values)


def _exception_reason(exc: Exception) -> str:
    exc_type = type(exc).__name__
    text = str(exc)
    lower = text.lower()
    if "503" in text and "unavailable" in lower:
        return f"{exc_type}: 503 UNAVAILABLE"
    if "429" in text or "quota" in lower:
        return f"{exc_type}: quota/rate limit"
    if "ssl" in lower:
        return f"{exc_type}: SSL transport error"
    if "timeout" in lower or "timed out" in lower:
        return f"{exc_type}: timeout"
    first_line = text.splitlines()[0] if text else ""
    sanitized = first_line.replace("{", "").replace("}", "")
    return f"{exc_type}: {sanitized[:120]}"


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


if __name__ == "__main__":
    raise SystemExit(main())
