"""SupervisorAgent — Step 2 parsing, Step 4 branch coordination, inter-step routing.

Step 2: takes the persisted `raw_request_record` and produces a canonical
`structured_query` (per ADC_Pipeline_IO_Schema_v0.1.md). All non-LLM fields
(run_id, parsed_at, source_raw_request_ref) are filled by this agent so the
LLM never has to invent them.
"""

from __future__ import annotations

from typing import Any

from ..llm.provider import LLMProvider
from ..schemas.step_01_raw_request_record import RawRequestRecord
from ..schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
    MentionedEntities,
)
from ..utils.time import now_iso


SUPERVISOR_SYSTEM_PROMPT = """You are the ADC pipeline Supervisor.
Parse the user's free-text ADC design request into a structured_query per the
canonical schema. Extract target/antigen, antibody candidate, payload, linker,
and any explicit user constraints. Do NOT invent identifiers; leave unknowns
as null. Output JSON matching the provided schema only.
""".strip()


class SupervisorAgent:
    name = "supervisor_agent"

    def __init__(self, llm: LLMProvider, mcp_client: Any | None = None) -> None:
        self.llm = llm
        self.mcp_client = mcp_client

    def parse_raw_to_structured_query(self, raw_request_record: dict) -> StructuredQuery:
        # Defensive: ensure the payload at least parses as a raw_request_record.
        RawRequestRecord.model_validate(
            {k: v for k, v in raw_request_record.items() if k != "artifact_id"}
        )

        prompt = (
            "Parse this raw_request_record into the structured_query inner fields. "
            "Do not include run_id, parsed_at, or source_raw_request_ref — those "
            "are filled by the orchestrator."
        )
        llm_payload = self.llm.generate_json(
            prompt,
            schema={"raw_request_record": raw_request_record},
            system=SUPERVISOR_SYSTEM_PROMPT,
        )

        # Agent fills the deterministic fields, never the LLM.
        sq = StructuredQuery(
            run_id=raw_request_record["run_id"],
            parsed_at=now_iso(),
            source_raw_request_ref=SourceRawRequestRef(
                raw_request_record_id=raw_request_record.get("artifact_id")
                or raw_request_record["run_artifact_registry_id"]
            ),
            task_intent=TaskIntent(**(llm_payload.get("task_intent") or {"task_type": "adc_design"})),
            mentioned_entities=MentionedEntities(**(llm_payload.get("mentioned_entities") or {})),
            referenced_inputs=llm_payload.get("referenced_inputs") or [],
            requested_outputs=llm_payload.get("requested_outputs") or [],
            user_constraints=llm_payload.get("user_constraints") or [],
            parse_warnings=llm_payload.get("parse_warnings") or [],
        )
        return sq

    def run(self, *, run_id: str, step_id: str, payload: dict) -> dict:  # noqa: D401
        raise NotImplementedError
