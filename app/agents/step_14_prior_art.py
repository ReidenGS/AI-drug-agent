"""Step 14 prior-art extraction, dedup, and IP relevance scoring.

Turns raw patent-tool payloads (PubChem `patents`, DrugBank `references`,
Orange Book `records` / `products` / `patent_records`) into compact merged
records. Raw payload stays in `tool_outputs/step_14/{tool_call_id}.json`;
only short structured fields are copied to the normalized record, and each
record carries `source_refs` pointing back to those raw files.

This module computes **evidence/IP relevance scoring over patent hits**, not
Step 12 candidate ranking. Ranking of ADC candidates remains owned by
Step 12 (`ranking_table.json`), which is never written here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Common keys we look at when extracting hits from a raw patent payload.
_HIT_LIST_KEYS = (
    "patents",
    "patent_records",
    "records",
    "products",
    "results",
    "hits",
    "references",
    "items",
)

_PATENT_NUMBER_KEYS = (
    "patent_number",
    "publication_number",
    "patent_id",
    "PatentID",
    "number",
    "id",
)
_TITLE_KEYS = ("title", "patent_title", "name")
_ASSIGNEE_KEYS = ("assignee", "applicant", "owner", "company")
_DATE_KEYS = ("publication_date", "publishedDate", "filing_date", "date", "year")
_LINK_KEYS = ("url", "link", "patent_url", "publication_url")
_CLAIM_KEYS = ("claim_focus", "claim_summary", "claim", "first_claim")
_JURISDICTION_KEYS = ("jurisdiction", "country", "office")

# Tokens we treat as ADC-relevant signal in titles/claim snippets.
_PAYLOAD_TOKENS = (
    "mmae", "mmaf", "dm1", "dm4", "dxd", "calicheamicin", "exatecan", "duocarmycin",
    "pbd", "auristatin", "maytansine",
)
_LINKER_TOKENS = (
    "vc-", "vc ", "valine-citrulline", "cleavable", "non-cleavable",
    "smcc", "spdb", "mc-", "thioether",
)
_CONJUGATION_TOKENS = (
    "conjugation", "conjugate", "antibody-drug conjugate", "adc",
    "dar", "drug-antibody ratio", "n297", "glycan", "site-specific",
    "thiol", "cysteine",
)
_USE_TOKENS = (
    "treatment", "therapy", "use for", "indication", "method of treating",
    "breast cancer", "tumor", "tumour", "carcinoma",
)


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pick(d: dict, keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


def _coerce_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    s = str(value)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


def _normalize_patent_number(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"[\s\-_/]+", "", str(value)).lower()
    return s or None


def _normalize_title(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    s = re.sub(r"[^\w\s]", "", s)
    return s or None


def _normalize_assignee(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"\s+", "", str(value).strip().lower())
    return s or None


@dataclass
class _RawHit:
    title: Optional[str]
    patent_number: Optional[str]
    assignee: Optional[str]
    year: Optional[int]
    publication_date: Optional[str]
    link: Optional[str]
    claim_focus: Optional[str]
    jurisdiction: Optional[str]
    source_tool: str
    source_database: str  # "PubChem" / "DrugBank" / "FDA_OrangeBook" / "other"
    source_ref: Optional[str]
    query_role: str
    query_term: str
    query_term_source: str
    candidate_id: str = ""


@dataclass
class MergedHit:
    title: Optional[str] = None
    patent_number: Optional[str] = None
    assignee: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    link: Optional[str] = None
    claim_focus: Optional[str] = None
    jurisdiction: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    source_database: str = "other"
    query_role: Optional[str] = None
    query_term: Optional[str] = None
    query_term_source: Optional[str] = None
    candidate_id: str = ""
    score: float = 0.0
    rationale: list[str] = field(default_factory=list)


def extract_hits(
    payload: Any,
    *,
    source_tool: str,
    source_database: str,
    source_ref: Optional[str],
    query_role: str,
    query_term: str,
    query_term_source: str,
    candidate_id: str = "",
) -> list[_RawHit]:
    """Extract compact prior-art hits from a single tool payload.

    Returns an empty list when the payload contains no recognizable hit list
    (e.g. default mock envelopes). Raw fields like full `description`,
    `claims`, `abstract` are intentionally NOT propagated.
    """
    if not isinstance(payload, dict):
        return []
    raw_list = None
    for k in _HIT_LIST_KEYS:
        v = payload.get(k)
        if isinstance(v, list) and v:
            raw_list = v
            break
    if not raw_list:
        return []
    out: list[_RawHit] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        title = _pick(item, _TITLE_KEYS)
        patent_number = _pick(item, _PATENT_NUMBER_KEYS)
        if title is None and patent_number is None:
            continue
        out.append(
            _RawHit(
                title=str(title).strip() if title is not None else None,
                patent_number=str(patent_number).strip() if patent_number is not None else None,
                assignee=(str(_pick(item, _ASSIGNEE_KEYS)).strip()
                          if _pick(item, _ASSIGNEE_KEYS) is not None else None),
                year=_coerce_year(_pick(item, _DATE_KEYS)),
                publication_date=(str(_pick(item, _DATE_KEYS))
                                  if _pick(item, _DATE_KEYS) is not None else None),
                link=(str(_pick(item, _LINK_KEYS))
                      if _pick(item, _LINK_KEYS) is not None else None),
                claim_focus=(_truncate(str(_pick(item, _CLAIM_KEYS)))
                             if _pick(item, _CLAIM_KEYS) is not None else None),
                jurisdiction=(str(_pick(item, _JURISDICTION_KEYS))
                              if _pick(item, _JURISDICTION_KEYS) is not None else None),
                source_tool=source_tool,
                source_database=source_database,
                source_ref=source_ref,
                query_role=query_role,
                query_term=query_term,
                query_term_source=query_term_source,
                candidate_id=candidate_id,
            )
        )
    return out


def _score_hit(hit: _RawHit | MergedHit) -> tuple[float, list[str]]:
    score = 0.0
    rationale: list[str] = []
    title = (getattr(hit, "title", None) or "").lower()
    claim = (getattr(hit, "claim_focus", None) or "").lower()
    term = (getattr(hit, "query_term", None) or "").lower()
    haystack = f"{title} {claim}"

    if term and term in title:
        score += 2.0
        rationale.append("exact query term in title (+2)")
    elif term and term in claim:
        score += 1.0
        rationale.append("query term in claim focus (+1)")

    if any(tok in title for tok in _PAYLOAD_TOKENS):
        score += 2.0
        rationale.append("payload token in title (+2)")
    elif any(tok in haystack for tok in _PAYLOAD_TOKENS):
        score += 1.0
        rationale.append("payload token in claim (+1)")

    if any(tok in title for tok in _LINKER_TOKENS):
        score += 1.5
        rationale.append("linker token in title (+1.5)")

    if any(tok in title for tok in _CONJUGATION_TOKENS):
        score += 1.5
        rationale.append("conjugation/ADC token in title (+1.5)")

    if any(tok in haystack for tok in _USE_TOKENS):
        score += 0.5
        rationale.append("use/indication token (+0.5)")

    if getattr(hit, "patent_number", None):
        score += 1.0
        rationale.append("patent_number present (+1)")

    year = getattr(hit, "year", None) or getattr(hit, "publication_year", None)
    if isinstance(year, int) and year >= 2018:
        score += 1.0
        rationale.append("recent (≥2018) (+1)")

    return score, rationale


def _to_merged(hit: _RawHit) -> MergedHit:
    base_score, rationale = _score_hit(hit)
    m = MergedHit(
        title=hit.title,
        patent_number=hit.patent_number,
        assignee=hit.assignee,
        publication_year=hit.year,
        publication_date=hit.publication_date,
        link=hit.link,
        claim_focus=hit.claim_focus,
        jurisdiction=hit.jurisdiction,
        sources=[hit.source_database] if hit.source_database else [],
        source_refs=[hit.source_ref] if hit.source_ref else [],
        source_database=hit.source_database or "other",
        query_role=hit.query_role,
        query_term=hit.query_term,
        query_term_source=hit.query_term_source,
        candidate_id=hit.candidate_id,
        score=base_score,
        rationale=list(rationale),
    )
    return m


def _merge_into(existing: MergedHit, hit: _RawHit) -> None:
    if hit.source_database and hit.source_database not in existing.sources:
        existing.sources.append(hit.source_database)
    if hit.source_ref and hit.source_ref not in existing.source_refs:
        existing.source_refs.append(hit.source_ref)
    if existing.title is None and hit.title:
        existing.title = hit.title
    if existing.patent_number is None and hit.patent_number:
        existing.patent_number = hit.patent_number
    if existing.assignee is None and hit.assignee:
        existing.assignee = hit.assignee
    if existing.publication_year is None and hit.year:
        existing.publication_year = hit.year
    if existing.publication_date is None and hit.publication_date:
        existing.publication_date = hit.publication_date
    if existing.link is None and hit.link:
        existing.link = hit.link
    if existing.claim_focus is None and hit.claim_focus:
        existing.claim_focus = hit.claim_focus
    if existing.jurisdiction is None and hit.jurisdiction:
        existing.jurisdiction = hit.jurisdiction


def dedup_and_sort_by_relevance(hits: list[_RawHit]) -> list[MergedHit]:
    """Dedup patent hits by `patent_number` (preferred) or
    `(normalized_title, normalized_assignee)`, then sort by deterministic
    IP-relevance score (descending).

    Operates on patent hits only; this is NOT Step 12 candidate ranking.
    """
    by_key: dict[tuple[str, str], MergedHit] = {}
    unkeyed: list[MergedHit] = []
    for h in hits:
        npn = _normalize_patent_number(h.patent_number)
        nt = _normalize_title(h.title)
        na = _normalize_assignee(h.assignee)
        if npn:
            key = ("pn", npn)
        elif nt and na:
            key = ("ta", f"{nt}|{na}")
        elif nt:
            key = ("t", nt)
        else:
            unkeyed.append(_to_merged(h))
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = _to_merged(h)
        else:
            _merge_into(existing, h)
    merged = list(by_key.values()) + unkeyed
    # Cross-source bonus, capped at +2.
    for m in merged:
        bonus = min(2.0, max(0, len(set(m.sources)) - 1))
        if bonus:
            m.score += bonus
            m.rationale.append(f"cross-source support (+{bonus:.1f})")
    merged.sort(key=lambda m: (-m.score, (m.title or "").lower()))
    return merged
