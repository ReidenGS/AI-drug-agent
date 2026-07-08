"""Step 14 — patent request (request-based PatentIPAgent entrypoint).

`Step14PatentRequest` is the minimal, reference-only request contract for
`PatentIPAgent.run_from_request`. Following the ADC_Pipeline_IO_Schema_v0.3
Step 14 principle, the request carries ONLY:

- a compact `user_query` (Step 2 `canonical_query` / orchestrator summary —
  never a full prompt or raw LLM response),
- `source_artifact_refs` (artifact name → ref/path the runtime may read),
- `input_refs` (reference handles describing WHERE a real value lives, plus
  the tool args each ref could satisfy),
- a `patent_scope` gate.

The request NEVER carries a real runtime value: no PubChem CID, brand name,
payload / linker text, raw sequence, PDB body, API key, or raw patent
payload. There is deliberately NO ``runtime_value`` field. Real values are
resolved separately by the Step 14 runtime resolver from
``source_artifact`` + ``source_path`` against run storage.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# Reference roles a Step 14 input ref can describe. Identifier-style roles
# (pubchem_cid / brand_name / application_number / drug_name) and entity-style
# roles (payload / linker / … / antibody). `antibody` is gated by
# `Step14PatentScope.antibody_search_allowed`.
Step14InputRefRole = Literal[
    "pubchem_cid",
    "brand_name",
    "application_number",
    "drug_name",
    "payload",
    "linker",
    "linker_payload",
    "compound",
    "target",
    "complete_adc",
    "antibody",
]


class Step14InputRef(BaseModel):
    """A reference to WHERE a real patent-query value lives — never the value.

    The runtime resolver reads ``source_artifact`` + ``source_path`` from run
    storage to obtain the real value. ``supports_tool_args`` declares which
    Step 14 tool arguments this reference could satisfy (e.g. ``["cid",
    "pubchem_cid"]``); the LLM selector uses it to choose tools WITHOUT ever
    seeing the resolved value.
    """

    model_config = ConfigDict(extra="forbid")

    ref_id: str
    source_artifact: str
    source_path: str
    role: Step14InputRefRole
    candidate_id: Optional[str] = None
    supports_tool_args: list[str] = Field(default_factory=list)


class Step14PatentScope(BaseModel):
    """Step 14 search scope. Antibody search stays OFF by default so patent
    routing is never antibody-centered unless the caller explicitly opts in."""

    model_config = ConfigDict(extra="forbid")

    allowed_roles: list[str] = Field(
        default_factory=lambda: [
            "linker_payload",
            "payload",
            "linker",
            "compound",
            "target",
            "complete_adc",
        ]
    )
    antibody_search_allowed: bool = False


class Step14PatentRequest(BaseModel):
    """Minimal request-based entrypoint for Step 14 patent routing.

    Carries references + a compact query summary only. There is NO
    ``runtime_value`` field by design — real values are resolved by the
    Step 14 runtime resolver from ``source_artifact`` + ``source_path``.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_query: Optional[str] = None
    source_artifact_refs: dict[str, str] = Field(default_factory=dict)
    input_refs: list[Step14InputRef] = Field(default_factory=list)
    patent_scope: Step14PatentScope = Field(default_factory=Step14PatentScope)
    request_notes: list[str] = Field(default_factory=list)
