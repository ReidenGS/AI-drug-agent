"""CDR3 extraction adapter (IMGT) for Step 5.

Step 5 needs to turn a full antibody heavy/light chain sequence into a
single CDR3 sequence so the IEDB BCR lookup can be queried by a typed
field — never by the full VH/VL string. We do that here, locally,
without external API calls, without an LLM, and without naive regex /
motif slicing of the variable domain.

Dependency policy (audited):

- We try ``abnumber`` first because its API surface (``Chain(seq,
  scheme="imgt")``) is the most ergonomic and it already encapsulates
  ANARCI under the hood. If abnumber is importable AND functional we
  read ``Chain.cdr3_seq`` and ``Chain.chain_type`` (``H`` / ``K`` /
  ``L``).
- If abnumber is unavailable but ANARCI is installed, we fall back to
  ``anarci.run_anarci`` and slice the residues whose IMGT numbering
  falls in 105–117 (the IMGT CDR3 range).
- If neither dependency is available, we return
  ``status="dependency_unavailable"``. The Step 5 agent records that as
  a clear data_gap; it does NOT silently fall back to running IEDB
  against the full VH/VL.
- No external API client is written here. No HTTP call, no LLM, no
  regex, no motif anchoring (``CDR3 = WGXG…``-style cuts).

Privacy / safety:

- ``Cdr3Result.cdr3_sequence`` is the literal CDR3 string and must
  ONLY be used to construct an MCP tool argument. The caller must
  NEVER persist it into normalized artifacts, LLM payloads, logs, or
  audit dicts. Use ``cdr3_length`` + ``cdr3_sha256_prefix`` for those
  channels instead.
- We never log full sequences or full CDR3 strings from this module.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


CDR3_SCHEME = "IMGT"

STATUS_SUCCESS = "success"
STATUS_DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
STATUS_NO_VARIABLE_DOMAIN = "no_variable_domain"
STATUS_EXTRACTION_FAILED = "extraction_failed"

CHAIN_TYPE_HEAVY = "heavy"
CHAIN_TYPE_LIGHT = "light"
CHAIN_TYPE_UNKNOWN = "unknown"

ChainRole = Literal["antibody_heavy", "antibody_light", "unknown"]

# Minimum length we will even attempt — VH/VL domains are ~110 aa, so
# shorter sequences are almost certainly not a variable domain and we
# want the adapter to fail clearly instead of letting the numbering
# library return a partial / garbage CDR3.
_MIN_SEQUENCE_LENGTH = 70

# IMGT CDR3 numbering range (inclusive).
_IMGT_CDR3_START = 105
_IMGT_CDR3_END = 117


_VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


@dataclass
class Cdr3Result:
    """Compact CDR3 extraction outcome.

    Hard contract: ``cdr3_sequence`` is for transient internal use only
    (constructing the IEDB MCP filter). Callers must never copy it into
    normalized artifacts, LLM prompts, audit dicts, or log output.
    Use ``cdr3_length`` and ``cdr3_sha256_prefix`` for those channels.
    """

    status: str
    chain_type: str = CHAIN_TYPE_UNKNOWN
    numbering_scheme: str = CDR3_SCHEME
    cdr3_sequence: str = ""
    cdr3_length: int = 0
    cdr3_sha256_prefix: str = ""
    source_sequence_length: int = 0
    backend: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: str = ""

    def to_compact_audit(self) -> dict:
        """Audit-safe projection — never includes ``cdr3_sequence``."""
        return {
            "status": self.status,
            "chain_type": self.chain_type,
            "numbering_scheme": self.numbering_scheme,
            "cdr3_length": self.cdr3_length,
            "cdr3_sha256_prefix": self.cdr3_sha256_prefix,
            "source_sequence_length": self.source_sequence_length,
            "backend": self.backend,
            "warnings": list(self.warnings),
            "notes": self.notes,
        }


def _clean_sequence(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return "".join(raw.split()).upper()


def _sha256_prefix(value: str, *, prefix_chars: int = 12) -> str:
    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:prefix_chars]


def _map_abnumber_chain_type(token: str) -> str:
    """Map abnumber chain_type letter to our heavy/light label."""
    t = (token or "").strip().upper()
    if t == "H":
        return CHAIN_TYPE_HEAVY
    if t in ("K", "L"):
        return CHAIN_TYPE_LIGHT
    return CHAIN_TYPE_UNKNOWN


def _map_anarci_chain_type(token: str) -> str:
    """Map ANARCI ``chain_type`` letter to our heavy/light label."""
    t = (token or "").strip().upper()
    if t == "H":
        return CHAIN_TYPE_HEAVY
    if t in ("K", "L"):
        return CHAIN_TYPE_LIGHT
    return CHAIN_TYPE_UNKNOWN


def _expected_chain_label(role: ChainRole) -> str:
    if role == "antibody_heavy":
        return CHAIN_TYPE_HEAVY
    if role == "antibody_light":
        return CHAIN_TYPE_LIGHT
    return CHAIN_TYPE_UNKNOWN


def _build_result(
    *,
    status: str,
    backend: str,
    sequence_len: int,
    chain_type: str = CHAIN_TYPE_UNKNOWN,
    cdr3: str = "",
    warnings: Optional[list[str]] = None,
    notes: str = "",
) -> Cdr3Result:
    return Cdr3Result(
        status=status,
        chain_type=chain_type,
        numbering_scheme=CDR3_SCHEME,
        cdr3_sequence=cdr3,
        cdr3_length=len(cdr3),
        cdr3_sha256_prefix=_sha256_prefix(cdr3) if cdr3 else "",
        source_sequence_length=sequence_len,
        backend=backend,
        warnings=list(warnings or []),
        notes=notes,
    )


# ── abnumber backend ─────────────────────────────────────────────────────


def _try_extract_via_abnumber(
    sequence: str, expected_chain_role: ChainRole
) -> Optional[Cdr3Result]:
    """Return a result if abnumber is importable and runs cleanly,
    else ``None`` so the caller can try the next backend."""
    try:
        from abnumber import Chain as _AbnumberChain  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — module not installed or import-time error
        return None
    try:
        chain = _AbnumberChain(sequence, scheme="imgt")
    except Exception as exc:  # noqa: BLE001
        return _build_result(
            status=STATUS_EXTRACTION_FAILED,
            backend="abnumber",
            sequence_len=len(sequence),
            warnings=[f"abnumber raised: {type(exc).__name__}"],
        )
    raw_chain_type = getattr(chain, "chain_type", "") or ""
    chain_type = _map_abnumber_chain_type(raw_chain_type)
    cdr3 = getattr(chain, "cdr3_seq", "") or ""
    if not cdr3:
        return _build_result(
            status=STATUS_NO_VARIABLE_DOMAIN,
            backend="abnumber",
            sequence_len=len(sequence),
            chain_type=chain_type,
            warnings=["abnumber returned empty cdr3_seq"],
        )
    warnings: list[str] = []
    expected = _expected_chain_label(expected_chain_role)
    if expected != CHAIN_TYPE_UNKNOWN and chain_type != expected:
        warnings.append(
            f"abnumber chain_type {chain_type!r} != expected {expected!r}"
        )
    return _build_result(
        status=STATUS_SUCCESS,
        backend="abnumber",
        sequence_len=len(sequence),
        chain_type=chain_type,
        cdr3=cdr3,
        warnings=warnings,
        notes="cdr3 derived via abnumber IMGT numbering",
    )


# ── ANARCI backend ───────────────────────────────────────────────────────


def _try_extract_via_anarci(
    sequence: str, expected_chain_role: ChainRole
) -> Optional[Cdr3Result]:
    """Return a result if anarci is importable and yields a numbering,
    else ``None`` to let the caller try other backends."""
    try:
        from anarci import run_anarci  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        numbered, _alignment, _hits = run_anarci(
            [("query", sequence)], scheme="imgt"
        )
    except Exception as exc:  # noqa: BLE001
        return _build_result(
            status=STATUS_EXTRACTION_FAILED,
            backend="anarci",
            sequence_len=len(sequence),
            warnings=[f"anarci raised: {type(exc).__name__}"],
        )
    if not numbered or not numbered[0]:
        return _build_result(
            status=STATUS_NO_VARIABLE_DOMAIN,
            backend="anarci",
            sequence_len=len(sequence),
        )
    domain = numbered[0][0]
    residues = domain[0] if isinstance(domain, tuple) else domain
    chain_meta = domain[1] if isinstance(domain, tuple) and len(domain) > 1 else {}
    raw_chain_type = ""
    if isinstance(chain_meta, dict):
        raw_chain_type = chain_meta.get("chain_type", "") or ""
    chain_type = _map_anarci_chain_type(raw_chain_type)
    cdr3_chars: list[str] = []
    for ((position, _ins), aa) in residues or []:
        try:
            pos_int = int(position)
        except (TypeError, ValueError):
            continue
        if not _IMGT_CDR3_START <= pos_int <= _IMGT_CDR3_END:
            continue
        if isinstance(aa, str) and aa not in ("-", "", " "):
            cdr3_chars.append(aa)
    cdr3 = "".join(cdr3_chars)
    if not cdr3:
        return _build_result(
            status=STATUS_NO_VARIABLE_DOMAIN,
            backend="anarci",
            sequence_len=len(sequence),
            chain_type=chain_type,
            warnings=["anarci numbering yielded no residues in IMGT 105–117"],
        )
    warnings: list[str] = []
    expected = _expected_chain_label(expected_chain_role)
    if expected != CHAIN_TYPE_UNKNOWN and chain_type != expected:
        warnings.append(
            f"anarci chain_type {chain_type!r} != expected {expected!r}"
        )
    return _build_result(
        status=STATUS_SUCCESS,
        backend="anarci",
        sequence_len=len(sequence),
        chain_type=chain_type,
        cdr3=cdr3,
        warnings=warnings,
        notes="cdr3 derived via anarci IMGT numbering (positions 105–117)",
    )


# ── public entry point ──────────────────────────────────────────────────


def extract_cdr3(
    sequence: str,
    *,
    expected_chain_role: ChainRole = "unknown",
) -> Cdr3Result:
    """Try installed antibody-numbering backends; never invent CDR3.

    Returns a :class:`Cdr3Result` whose ``status`` is one of
    ``success`` / ``dependency_unavailable`` / ``no_variable_domain`` /
    ``extraction_failed``. The caller is responsible for treating
    ``cdr3_sequence`` as a transient internal value (see module
    docstring).
    """
    cleaned = _clean_sequence(sequence)
    if not cleaned:
        return _build_result(
            status=STATUS_EXTRACTION_FAILED,
            backend="",
            sequence_len=0,
            warnings=["empty input sequence"],
        )
    if len(cleaned) < _MIN_SEQUENCE_LENGTH:
        return _build_result(
            status=STATUS_NO_VARIABLE_DOMAIN,
            backend="",
            sequence_len=len(cleaned),
            warnings=[
                f"sequence shorter than minimum variable-domain length "
                f"({_MIN_SEQUENCE_LENGTH})"
            ],
        )
    if not _VALID_AA_RE.fullmatch(cleaned):
        return _build_result(
            status=STATUS_EXTRACTION_FAILED,
            backend="",
            sequence_len=len(cleaned),
            warnings=["sequence contains non-standard amino acid characters"],
        )

    for backend in (
        _try_extract_via_abnumber,
        _try_extract_via_anarci,
    ):
        result = backend(cleaned, expected_chain_role)
        if result is None:
            continue  # backend not installed → try next
        # Even failure results from an installed backend are authoritative;
        # we do not silently keep trying after a real attempt.
        return result

    return _build_result(
        status=STATUS_DEPENDENCY_UNAVAILABLE,
        backend="",
        sequence_len=len(cleaned),
        warnings=[
            "no antibody numbering dependency available (abnumber / anarci)"
        ],
        notes=(
            "Install abnumber (preferred) or anarci to enable Step 5 CDR3 "
            "extraction. Step 5 will record this as a data_gap and will "
            "NOT query IEDB with the full VH/VL sequence."
        ),
    )
