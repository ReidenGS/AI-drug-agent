"""Hardened reference-only contract for unified patent/evidence planning."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .orchestrator_execution_state import RunId
from .patent_evidence_contract import (
    KNOWN_PATENT_EVIDENCE_SUPPORT_TOKENS,
    PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES,
    PatentEvidenceInputRole,
)


SearchLane = Literal["evidence", "patent"]
SafeIdentifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,127}$")]
SafeSourcePath = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.\[\]-]*$",
    ),
]
SafeArtifactRef = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_./:-]*$",
    ),
]


class PatentEvidenceInputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: SafeIdentifier
    source_artifact: SafeIdentifier
    source_path: SafeSourcePath
    role: PatentEvidenceInputRole
    candidate_id: Optional[SafeIdentifier] = None
    supports_tool_args: list[SafeIdentifier] = Field(default_factory=list)

    @field_validator("supports_tool_args")
    @classmethod
    def _known_unique_supports(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("supports_tool_args must not contain duplicates")
        unknown = sorted(set(values) - KNOWN_PATENT_EVIDENCE_SUPPORT_TOKENS)
        if unknown:
            raise ValueError("supports_tool_args contains unknown support token")
        return values

    @model_validator(mode="after")
    def _supports_match_typed_role(self) -> "PatentEvidenceInputRef":
        if any(
            self.role not in PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES[token]
            for token in self.supports_tool_args
        ):
            raise ValueError("supports_tool_args is incompatible with ref role")
        return self


class PatentEvidenceSearchScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_lanes: list[SearchLane] = Field(
        default_factory=lambda: ["evidence", "patent"], min_length=1
    )
    allowed_roles: list[PatentEvidenceInputRole] = Field(
        default_factory=lambda: [
            "linker_payload",
            "payload",
            "linker",
            "compound",
            "target",
            "complete_adc",
            "antibody",
            "pubchem_cid",
            "brand_name",
            "application_number",
            "drug_name",
            "query",
            "pmid",
            "pmids",
            "document_id",
            "title",
        ],
        min_length=1,
    )
    antibody_search_allowed: bool = False

    @field_validator("requested_lanes", "allowed_roles")
    @classmethod
    def _unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("scope lists must not contain duplicates")
        return values


class PatentEvidenceRequest(BaseModel):
    """Compact metadata and references only; never resolved runtime values."""

    model_config = ConfigDict(extra="forbid")

    run_id: RunId
    user_query: Optional[str] = None
    source_artifact_refs: dict[SafeIdentifier, SafeArtifactRef] = Field(
        default_factory=dict
    )
    input_refs: list[PatentEvidenceInputRef] = Field(default_factory=list)
    search_scope: PatentEvidenceSearchScope = Field(
        default_factory=PatentEvidenceSearchScope
    )
    request_notes: list[str] = Field(default_factory=list)

    @field_validator("user_query")
    @classmethod
    def _compact_non_control_query(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in value):
            raise ValueError("user_query contains unsafe control characters")
        return value.strip()

    @field_validator("request_notes")
    @classmethod
    def _compact_notes(cls, values: list[str]) -> list[str]:
        for value in values:
            if (
                not value.strip()
                or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value)
            ):
                raise ValueError("request_notes contains unsafe text")
        return values

    @model_validator(mode="after")
    def _unique_ref_ids(self) -> "PatentEvidenceRequest":
        ref_ids = [ref.ref_id for ref in self.input_refs]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("input ref_id values must be unique within the request")
        return self
