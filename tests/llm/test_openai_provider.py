"""OpenAIProvider JSON generation tests.

These tests stub the SDK call inside the provider (``_generate_content``)
so they never hit the OpenAI API. They verify shared JSON-task validation
flows through the OpenAI surface the same way it flows through Gemini.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest

from app.a2a.orchestrator_routing_prompt import (
    ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    ORCHESTRATOR_ROUTING_USER_TASK,
)
from app.llm.openai_provider import (
    OpenAIProvider,
    OpenAIProviderError,
    _OrchestratorRoutingResponse,
    _PROMPT_CACHE_LAYOUT_VERSION,
    _RESPONSE_MODEL_FOR_TASK,
    _ToolSelectionStage1Response,
    _Step6SchemaMappingStage1Response,
    _Step6SchemaMappingStage2ParserResponse,
    _Step9SchemaMappingStage2Response,
    _Step9ToolSelectionStage1Response,
    _Step14PatentToolSelectionResponse,
    _PatentEvidenceToolSelectionResponse,
    _prompt_cache_key,
)


def _provider_with_responses(responses: list[Any]) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-fake-key", max_retries=2)
    calls = iter(responses)

    def _fake_generate_content(
        prompt: str,
        *,
        system: str | None = None,
        task: str,
    ) -> Any:
        return next(calls)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


_EXEC_DIR = Path(__file__).resolve().parents[2]


def _run_usage_subprocess(source: str) -> tuple[list[str], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_EXEC_DIR)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_EXEC_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    combined = "\n".join((result.stdout, result.stderr))
    lines = [
        line
        for line in combined.splitlines()
        if line.startswith("OPENAI_USAGE_EVENT=")
    ]
    return lines, combined


_FAKE_SUCCESS_CALL = r'''
from types import SimpleNamespace
provider = OpenAIProvider(api_key="sk-subprocess-private", max_retries=0)
message = SimpleNamespace(content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}')
usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15)
provider._generate_content = lambda *_args, **_kwargs: SimpleNamespace(
    choices=[SimpleNamespace(message=message)], usage=usage
)
provider.generate_json("FASTA_PRIVATE_SUBPROCESS", schema={"raw_request_record": {}})
'''


def _chat_response(content: str) -> Any:
    """Build a fake Chat Completions response with one choice."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


# ── basic JSON happy paths ─────────────────────────────────────────────────


def test_structured_query_accepts_valid_json():
    provider = _provider_with_responses([
        _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2"}}'
        )
    ])
    out = provider.generate_json(
        "parse", schema={"raw_request_record": {"raw_user_query": "HER2"}}
    )
    assert out["task_intent"]["task_type"] == "adc_design"
    assert out["mentioned_entities"]["target_or_antigen_text"] == "HER2"
    # Shared normalization filled empty lists for absent optional fields.
    assert out["referenced_inputs"] == []
    assert out["parse_warnings"] == []


def test_structured_query_promotes_component_name_alias():
    provider = _provider_with_responses([
        _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"payload_text":"vc-MMAE"},'
            '"entity_decompositions":[{"original_text":"vc-MMAE",'
            '"components":[{"component_name":"valine-citrulline",'
            '"component_type":"linker","inferred":true}]}]}'
        )
    ])
    out = provider.generate_json(
        "parse", schema={"task": "structured_query", "raw_request_record": {}}
    )
    comp = out["entity_decompositions"][0]["components"][0]
    assert comp["canonical_name"] == "valine-citrulline"
    assert comp["component_type"] == "linker"
    assert comp["role"] == "linker"


def test_stage1_multi_lane_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":['
            '{"lane_type":"payload_linker_compound_liability",'
            ' "tool_name":"DrugProps_pains_filter",'
            ' "selection_reason":"smiles present"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_stage2_multi_tool_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter",'
            '"arguments":{"smiles":"CCO"}}]}'
        )
    ])
    out = provider.generate_json(
        "args", schema={"task": "tool_selection_stage_2_multi_tool"}
    )
    assert out["tools"][0]["arguments"] == {"smiles": "CCO"}


def test_stage1_single_tool_path_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}')
    ])
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_step6_schema_mapping_stage1_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":[{"tool_name":"DrugProps_pains_filter",'
            '"selection_reason":"smiles"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "step6_schema_mapping_stage_1"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_step6_schema_mapping_stage2_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter",'
            '"can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],'
            '"argument_mapping_reason":"mapped"}]}'
        )
    ])
    out = provider.generate_json(
        "map", schema={"task": "step6_schema_mapping_stage_2"}
    )
    assert out["tools"][0]["argument_mapping"]["smiles"].startswith("candidate:")


def test_step9_tool_selection_stage1_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":[{"tool_name":"ZINC_search_by_smiles",'
            '"lane_type":"compound_screening",'
            '"selection_reason":"smiles available"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "step9_tool_selection_stage_1"}
    )
    assert out["selections"][0]["lane_type"] == "compound_screening"


def test_step9_schema_mapping_stage2_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"tool_name":"ZINC_search_by_smiles",'
            '"lane_type":"compound_screening","can_invoke":true,'
            '"argument_mappings":[{"schema_arg":"smiles","field_ref":"material:m1"}],'
            '"argument_literals":[],"missing_required_fields":[],'
            '"skip_reason":"","argument_mapping_reason":"mapped"}]}'
        )
    ])
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert out["tools"][0]["argument_mappings"][0]["schema_arg"] == "smiles"


# ── error paths ───────────────────────────────────────────────────────────


def test_invalid_top_level_shape_raises_clear_error_without_prompt_leakage():
    bad = '{"arguments":{}}'  # missing `selections` for stage_1
    provider = _provider_with_responses([_chat_response(bad)] * 3)
    with pytest.raises(OpenAIProviderError) as excinfo:
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    msg = str(excinfo.value)
    # Compact error referencing the contract, not the prompt or response body.
    assert "tool_selection_stage_1" in msg
    assert "selections" in msg
    # Make sure secrets and raw text don't leak through.
    assert "sk-" not in msg
    assert "System instructions" not in msg


def test_malformed_json_retries_and_returns_corrected_payload():
    provider = _provider_with_responses([
        _chat_response('{"selections": ['),  # malformed
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    ])
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_response_without_choices_raises_clear_error():
    provider = _provider_with_responses(
        [SimpleNamespace(choices=[])] * 3
    )
    with pytest.raises(OpenAIProviderError, match="no choices"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})


def test_stage1_multi_lane_rejects_selection_missing_lane_type():
    provider = _provider_with_responses(
        [_chat_response('{"selections":[{"tool_name":"DrugProps_pains_filter"}]}')] * 3
    )
    with pytest.raises(OpenAIProviderError, match="multi_lane.*lane_type"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_step6_schema_mapping_stage1_rejects_malformed_selections():
    provider = _provider_with_responses(
        [_chat_response('{"selections":"not-a-list"}')] * 3
    )
    with pytest.raises(OpenAIProviderError, match="step6_schema_mapping_stage_1.*selections"):
        provider.generate_json(
            "pick", schema={"task": "step6_schema_mapping_stage_1"}
        )


def test_step6_schema_mapping_stage2_rejects_malformed_tools():
    provider = _provider_with_responses(
        [_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter",'
            '"can_invoke":"yes","argument_mapping":{},'
            '"missing_required_fields":[]}]}'
        )] * 3
    )
    with pytest.raises(OpenAIProviderError, match="step6_schema_mapping_stage_2.*can_invoke"):
        provider.generate_json(
            "map", schema={"task": "step6_schema_mapping_stage_2"}
        )


def test_no_api_key_raises_value_error():
    with pytest.raises(ValueError, match="non-empty api_key"):
        OpenAIProvider(api_key="")


# ── usage event recording ──────────────────────────────────────────────────


def _chat_response_with_usage(
    content: str, *, prompt: int | None = None,
    completion: int | None = None, total: int | None = None,
) -> Any:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_usage_events_record_token_counts_per_attempt():
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2"}}',
            prompt=120, completion=45, total=165,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt == {
        "provider": "openai",
        "model": provider.model,
        "task": "structured_query",
        "attempt": 0,
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
        # The stub response does not carry prompt_tokens_details, so
        # the cache field degrades to None. A dedicated test below
        # exercises the present-and-populated path.
        "cached_prompt_tokens": None,
    }


def test_usage_log_records_one_canonical_compact_event(caplog):
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=12,
            completion=3,
            total=15,
        )
    ])
    provider.generate_json("safe prompt", schema={"raw_request_record": {}})

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("OPENAI_USAGE_EVENT=")
    ]
    assert len(messages) == 1
    payload = json.loads(messages[0].removeprefix("OPENAI_USAGE_EVENT="))
    assert payload == provider.usage_events[0]
    assert set(payload) == {
        "provider",
        "model",
        "task",
        "attempt",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    }
    assert messages[0] == "OPENAI_USAGE_EVENT=" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def test_usage_events_recorded_for_each_retry_attempt():
    """Each provider call costs real tokens. A retry triggered by a
    malformed first response must still log the first attempt."""
    provider = _provider_with_responses([
        _chat_response_with_usage(
            "not json at all", prompt=50, completion=10, total=60,
        ),
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=80, completion=12, total=92,
        ),
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert [e["attempt"] for e in provider.usage_events] == [0, 1]
    assert provider.usage_events[0]["total_tokens"] == 60
    assert provider.usage_events[1]["total_tokens"] == 92


def test_usage_log_records_every_paid_retry_attempt(caplog):
    provider = _provider_with_responses([
        _chat_response_with_usage("not json", prompt=20, completion=2, total=22),
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=25,
            completion=4,
            total=29,
        ),
    ])
    provider.generate_json("safe prompt", schema={"raw_request_record": {}})

    log_lines = [record.getMessage() for record in caplog.records]
    payloads = [
        json.loads(line.removeprefix("OPENAI_USAGE_EVENT="))
        for line in log_lines
        if line.startswith("OPENAI_USAGE_EVENT=")
    ]
    assert [payload["attempt"] for payload in payloads] == [0, 1]
    assert [payload["total_tokens"] for payload in payloads] == [22, 29]
    log_blob = "\n".join(log_lines)
    assert "not json" not in log_blob
    assert "sk-fake-key" not in log_blob


def test_usage_event_degrades_to_null_when_response_has_no_usage_field():
    """Some SDK / proxy responses omit the usage block. The event must
    still be appended with token fields set to None — never crash."""
    message = SimpleNamespace(content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}')
    choice = SimpleNamespace(message=message)
    bare_response = SimpleNamespace(choices=[choice])  # no `usage`
    provider = _provider_with_responses([bare_response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] is None
    assert evt["completion_tokens"] is None
    assert evt["total_tokens"] is None


def test_usage_log_missing_usage_is_null_and_contains_no_sensitive_content(caplog):
    prompt_sentinel = (
        "FASTA_SEQUENCE_PRIVATE_ACDEFGHIK "
        "FULL_PROMPT_PRIVATE TOOLUNIVERSE_RAW_PRIVATE "
        "https://private-endpoint.invalid /private/storage/path"
    )
    response_sentinel = "ATOM_PRIVATE_PDB_RESPONSE"
    message = SimpleNamespace(
        content=(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"'
            + response_sentinel
            + '"}}'
        )
    )
    provider = OpenAIProvider(api_key="sk-private-usage-log-key", max_retries=0)
    provider._generate_content = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        choices=[SimpleNamespace(message=message)]
    )
    provider.generate_json(prompt_sentinel, schema={"raw_request_record": {}})

    log_lines = [record.getMessage() for record in caplog.records]
    usage_messages = [
        line
        for line in log_lines
        if line.startswith("OPENAI_USAGE_EVENT=")
    ]
    assert len(usage_messages) == 1
    payload = json.loads(usage_messages[0].removeprefix("OPENAI_USAGE_EVENT="))
    assert payload["prompt_tokens"] is None
    assert payload["completion_tokens"] is None
    assert payload["total_tokens"] is None
    assert payload["cached_prompt_tokens"] is None
    log_blob = "\n".join(log_lines)
    for sentinel in (
        "sk-private-usage-log-key",
        prompt_sentinel,
        response_sentinel,
        "Authorization",
        "FULL_PROMPT_PRIVATE",
        "TOOLUNIVERSE_RAW_PRIVATE",
        "private-endpoint",
        "/private/storage/path",
    ):
        assert sentinel not in log_blob


def test_usage_events_carry_no_prompt_response_or_api_key():
    """Defensive content check: the compact event must never include
    prompt body, response body, schema payload, or the API key."""
    secret_prompt = "USER PROMPT BODY WITH A SECRET HER2 SCAFFOLD"
    secret_response = (
        '{"task_intent":{"task_type":"adc_design"},'
        '"mentioned_entities":{"target_or_antigen_text":"HER2_SECRET"}}'
    )
    provider = OpenAIProvider(api_key="sk-very-secret-key", max_retries=0)

    def _fake_generate_content(
        prompt: str,
        *,
        system: str | None = None,
        task: str,
    ) -> Any:
        # The provider must NOT carry `prompt` through to the event.
        return _chat_response_with_usage(secret_response, prompt=10, completion=5, total=15)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(secret_prompt, schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-very-secret-key" not in blob
    assert "USER PROMPT BODY" not in blob
    assert "SECRET HER2 SCAFFOLD" not in blob
    assert "HER2_SECRET" not in blob


def test_usage_telemetry_visible_with_default_subprocess_logging():
    lines, combined = _run_usage_subprocess(
        "import logging\n"
        "assert logging.getLogger().getEffectiveLevel() >= logging.WARNING\n"
        "from app.llm.openai_provider import OpenAIProvider\n"
        + _FAKE_SUCCESS_CALL
    )

    assert len(lines) == 1
    payload = json.loads(lines[0].removeprefix("OPENAI_USAGE_EVENT="))
    assert payload["attempt"] == 0
    assert payload["total_tokens"] == 15
    assert "sk-subprocess-private" not in combined
    assert "FASTA_PRIVATE_SUBPROCESS" not in combined


@pytest.mark.parametrize("configure_order", ["before_import", "after_import"])
def test_usage_telemetry_visible_once_with_uvicorn_default_logging(configure_order):
    if configure_order == "before_import":
        prelude = (
            "from uvicorn import Config\n"
            "Config('app.main:app').configure_logging()\n"
            "from app.llm.openai_provider import OpenAIProvider\n"
        )
    else:
        prelude = (
            "import app.llm.openai_provider as provider_module\n"
            "from uvicorn import Config\n"
            "Config('app.main:app').configure_logging()\n"
            "OpenAIProvider = provider_module.OpenAIProvider\n"
        )
    lines, combined = _run_usage_subprocess(prelude + _FAKE_SUCCESS_CALL)

    assert len(lines) == 1
    assert json.loads(lines[0].removeprefix("OPENAI_USAGE_EVENT="))["attempt"] == 0
    assert "sk-subprocess-private" not in combined
    assert "FASTA_PRIVATE_SUBPROCESS" not in combined


def test_usage_telemetry_module_reload_does_not_duplicate_handler():
    lines, _combined = _run_usage_subprocess(
        "import importlib\n"
        "import app.llm.openai_provider as provider_module\n"
        "provider_module = importlib.reload(provider_module)\n"
        "provider_module = importlib.reload(provider_module)\n"
        "OpenAIProvider = provider_module.OpenAIProvider\n"
        + _FAKE_SUCCESS_CALL
    )

    assert len(lines) == 1
    assert json.loads(lines[0].removeprefix("OPENAI_USAGE_EVENT="))["attempt"] == 0


def test_usage_telemetry_subprocess_retry_emits_exact_attempts_without_leak():
    source = r'''
from types import SimpleNamespace
from app.llm.openai_provider import OpenAIProvider
provider = OpenAIProvider(api_key="sk-retry-private", max_retries=1)
responses = iter([
    SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    ),
    SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}'
        ))],
        usage=SimpleNamespace(prompt_tokens=14, completion_tokens=3, total_tokens=17),
    ),
])
provider._generate_content = lambda *_args, **_kwargs: next(responses)
provider.generate_json("A3M_PRIVATE_RETRY_PROMPT", schema={"raw_request_record": {}})
'''
    lines, combined = _run_usage_subprocess(source)

    assert len(lines) == 2
    payloads = [
        json.loads(line.removeprefix("OPENAI_USAGE_EVENT=")) for line in lines
    ]
    assert [payload["attempt"] for payload in payloads] == [0, 1]
    assert [payload["total_tokens"] for payload in payloads] == [12, 17]
    for sentinel in (
        "sk-retry-private",
        "A3M_PRIVATE_RETRY_PROMPT",
        "not json",
    ):
        assert sentinel not in combined


# ── cached prompt tokens (OpenAI auto prompt caching) ─────────────────────


def _chat_response_with_cache(
    content: str, *, prompt: int, completion: int, total: int,
    cached: int | None,
) -> Any:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    details = (
        SimpleNamespace(cached_tokens=cached) if cached is not None else None
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=details,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_usage_event_reads_cached_prompt_tokens_from_prompt_tokens_details():
    provider = _provider_with_responses([
        _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=1000, completion=200, total=1200, cached=512,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] == 1000
    assert evt["cached_prompt_tokens"] == 512
    assert evt["total_tokens"] == 1200


def test_usage_event_cached_prompt_tokens_none_when_no_details():
    provider = _provider_with_responses([
        _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=1000, completion=50, total=1050, cached=None,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] is None


def test_usage_event_dict_shaped_usage_with_details_dict():
    """Some proxies return usage as a dict instead of an SDK object;
    the cache reader must still degrade safely."""
    message = SimpleNamespace(
        content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}'
    )
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], usage={
        "prompt_tokens": 800,
        "completion_tokens": 40,
        "total_tokens": 840,
        "prompt_tokens_details": {"cached_tokens": 256},
    })
    provider = _provider_with_responses([response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] == 256


def test_cached_field_does_not_leak_prompt_or_response(monkeypatch):
    """Regression: the cache field is a single int; even when present
    the event must not carry prompt body, response body, or API key."""
    provider = OpenAIProvider(api_key="sk-cache-secret", max_retries=0)

    def _fake_generate_content(
        prompt: str,
        *,
        system: str | None = None,
        task: str,
    ) -> Any:
        return _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2_CACHE_SECRET"}}',
            prompt=10, completion=5, total=15, cached=4,
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json("PROMPT CACHE BODY", schema={"raw_request_record": {}})
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-cache-secret" not in blob
    assert "PROMPT CACHE BODY" not in blob
    assert "HER2_CACHE_SECRET" not in blob


# ── system prompt is NOT duplicated into the user message ─────────────────


def test_openai_system_prompt_is_not_embedded_into_user_message():
    """``OpenAIProvider`` must forward ``system`` as a dedicated
    role=system message and NOT also prepend it into the user message
    body. Embedding it twice burns prompt tokens on every Step 2 /
    Step 5 call."""
    sentinel_system = "SENTINEL_SYSTEM_BLOCK_OPENAI_only"
    user_prompt_body = "USER_PROMPT_SENTINEL_for_openai"

    captured: dict[str, Any] = {}

    provider = OpenAIProvider(api_key="sk-fake-dedup", max_retries=0)

    def _fake_generate_content(
        prompt: str,
        *,
        system: str | None = None,
        task: str,
    ) -> Any:
        captured["prompt"] = prompt
        captured["system"] = system
        return _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}'
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(
        user_prompt_body,
        schema={"raw_request_record": {"raw_user_query": "HER2"}},
        system=sentinel_system,
    )

    # System sentinel travels via the dedicated `system` kwarg.
    assert captured["system"] == sentinel_system
    # And does NOT appear in the user-message body — so Chat Completions
    # never receives the same system block twice.
    assert sentinel_system not in captured["prompt"], captured["prompt"][:200]
    # The user-message body still carries the user prompt + the expected
    # task-shape preamble produced by ``build_json_prompt`` so the LLM
    # has everything it needs.
    assert user_prompt_body in captured["prompt"]
    assert "expected top-level shape" in captured["prompt"].lower() or (
        "task_intent" in captured["prompt"]
    ), captured["prompt"][:200]


def test_openai_handles_none_system_without_embedding_anything():
    """When the caller does not pass a system block, the user-message
    body still must not embed an empty `System instructions:` header."""
    user_prompt_body = "USER_PROMPT_NO_SYSTEM"
    captured: dict[str, Any] = {}
    provider = OpenAIProvider(api_key="sk-fake-no-sys", max_retries=0)

    def _fake_generate_content(
        prompt: str,
        *,
        system: str | None = None,
        task: str,
    ) -> Any:
        captured["prompt"] = prompt
        captured["system"] = system
        return _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}'
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(
        user_prompt_body,
        schema={"raw_request_record": {}},
        system=None,
    )
    assert captured["system"] is None
    assert "System instructions" not in captured["prompt"]
    assert user_prompt_body in captured["prompt"]


# ── official structured-output parser path + json_object fallback ─────────


def _fake_openai_client(*, parse=None, create=None, has_parse: bool = True):
    """Build a fake OpenAI client exposing `beta.chat.completions.parse` and
    `chat.completions.create`, recording calls to each."""
    calls: dict[str, list[dict[str, Any]]] = {"parse": [], "create": []}
    completions = SimpleNamespace()

    if has_parse:
        def _parse(**kw: Any) -> Any:
            calls["parse"].append(kw)
            return parse(**kw) if callable(parse) else parse
        completions.parse = _parse

    def _create(**kw: Any) -> Any:
        calls["create"].append(kw)
        return create(**kw) if callable(create) else create
    completions.create = _create

    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(beta=SimpleNamespace(chat=chat), chat=chat)
    client._calls = calls  # type: ignore[attr-defined]
    return client


def _parsed_response(parsed_obj: Any, *, usage: Any = None) -> Any:
    message = SimpleNamespace(parsed=parsed_obj, content=None, refusal=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _provider_with_client(client: Any, *, max_retries: int = 2) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-fake-parser", max_retries=max_retries)
    provider._client = client  # type: ignore[assignment]  # bypass real SDK construction
    return provider


def test_gpt55_structured_parser_receives_prompt_cache_kwargs():
    parsed = _ToolSelectionStage1Response(
        selections=[{"tool_name": "DrugProps_calculate_qed"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    provider.model = "gpt-5.5"

    provider.generate_json("dynamic query A", schema={"task": "tool_selection_stage_1"})

    kwargs = client._calls["parse"][0]
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["prompt_cache_key"] == _prompt_cache_key(
        model="gpt-5.5", task="tool_selection_stage_1"
    )


def test_gpt55_json_object_receives_prompt_cache_kwargs():
    client = _fake_openai_client(
        create=_chat_response(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}'
        )
    )
    provider = _provider_with_client(client)
    provider.model = "gpt-5.5"

    provider.generate_json("dynamic query B", schema={"raw_request_record": {}})

    kwargs = client._calls["create"][0]
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["prompt_cache_key"] == _prompt_cache_key(
        model="gpt-5.5", task="structured_query"
    )


def test_parser_fallback_reuses_identical_prompt_cache_key():
    client = _fake_openai_client(
        parse=lambda **_kw: (_ for _ in ()).throw(
            TypeError("response_format invalid schema")
        ),
        create=_chat_response(
            json.dumps({
                "selections": [{"tool_name": "DrugProps_calculate_qed"}]
            })
        ),
    )
    provider = _provider_with_client(client)
    provider.model = "gpt-5.5-2026-07-15"

    provider.generate_json("dynamic query", schema={"task": "tool_selection_stage_1"})

    parse_kwargs = client._calls["parse"][0]
    create_kwargs = client._calls["create"][0]
    assert parse_kwargs["prompt_cache_key"] == create_kwargs["prompt_cache_key"]
    assert parse_kwargs["prompt_cache_key"] == _prompt_cache_key(
        model="gpt-5.5-2026-07-15",
        task="tool_selection_stage_1",
    )
    assert parse_kwargs["prompt_cache_retention"] == "24h"
    assert create_kwargs["prompt_cache_retention"] == "24h"


def test_concurrent_generate_json_keeps_parser_cache_and_usage_task_local():
    entered_parse = Barrier(2, timeout=5)
    parsed_by_model = {
        _ToolSelectionStage1Response: _ToolSelectionStage1Response(
            selections=[{"tool_name": "DrugProps_calculate_qed"}]
        ),
        _OrchestratorRoutingResponse: _OrchestratorRoutingResponse.model_validate(
            _orchestrator_response()
        ),
    }

    def _parse(**kwargs: Any) -> Any:
        entered_parse.wait()
        return _parsed_response(parsed_by_model[kwargs["response_format"]])

    client = _fake_openai_client(parse=_parse)
    provider = _provider_with_client(client)
    provider.model = "gpt-5.5"

    with ThreadPoolExecutor(max_workers=2) as executor:
        stage1_future = executor.submit(
            provider.generate_json,
            "stage1 dynamic prompt",
            schema={"task": "tool_selection_stage_1"},
        )
        orchestrator_future = executor.submit(
            provider.generate_json,
            "orchestrator dynamic prompt",
            schema={"task": "orchestrator_worker_routing"},
        )
        stage1_result = stage1_future.result()
        orchestrator_result = orchestrator_future.result()

    assert stage1_result["selections"][0]["tool_name"] == (
        "DrugProps_calculate_qed"
    )
    assert orchestrator_result == _orchestrator_response()

    calls_by_model = {
        call["response_format"]: call for call in client._calls["parse"]
    }
    stage1_call = calls_by_model[_ToolSelectionStage1Response]
    orchestrator_call = calls_by_model[_OrchestratorRoutingResponse]
    assert stage1_call["prompt_cache_key"] == _prompt_cache_key(
        model="gpt-5.5",
        task="tool_selection_stage_1",
    )
    assert orchestrator_call["prompt_cache_key"] == _prompt_cache_key(
        model="gpt-5.5",
        task="orchestrator_worker_routing",
    )
    assert stage1_call["prompt_cache_key"] != orchestrator_call["prompt_cache_key"]
    assert stage1_call["prompt_cache_retention"] == "24h"
    assert orchestrator_call["prompt_cache_retention"] == "24h"
    assert {event["task"] for event in provider.usage_events} == {
        "tool_selection_stage_1",
        "orchestrator_worker_routing",
    }
    assert not hasattr(provider, "_active_task")


def test_prompt_cache_key_ignores_dynamic_prompt_and_changes_by_namespace(caplog):
    parsed = _ToolSelectionStage1Response(
        selections=[{"tool_name": "DrugProps_calculate_qed"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    provider.model = "gpt-5.5"
    dynamic_sentinels = (
        "QUERY_PRIVATE_ONE",
        "QUERY_PRIVATE_TWO",
        "sk-private-cache-test",
        "ATOM_PRIVATE_CACHE_TEST",
    )

    for prompt in dynamic_sentinels[:2]:
        provider.generate_json(prompt, schema={"task": "tool_selection_stage_1"})

    keys = [call["prompt_cache_key"] for call in client._calls["parse"]]
    assert len(set(keys)) == 1
    assert keys[0].startswith("adcpc_") and len(keys[0]) == 38
    assert _prompt_cache_key(model="gpt-5.5", task="different_task") != keys[0]
    assert _prompt_cache_key(
        model="gpt-5.5",
        task="tool_selection_stage_1",
        layout_version=_PROMPT_CACHE_LAYOUT_VERSION + "-next",
    ) != keys[0]
    log_blob = "\n".join(record.getMessage() for record in caplog.records)
    usage_blob = json.dumps(provider.usage_events)
    for sentinel in (*dynamic_sentinels, keys[0]):
        assert sentinel not in log_blob
        assert sentinel not in usage_blob


def test_non_gpt55_model_omits_prompt_cache_retention():
    parsed = _ToolSelectionStage1Response(
        selections=[{"tool_name": "DrugProps_calculate_qed"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    provider.model = "gpt-4.1-mini"

    provider.generate_json("dynamic query", schema={"task": "tool_selection_stage_1"})

    kwargs = client._calls["parse"][0]
    assert "prompt_cache_key" in kwargs
    assert "prompt_cache_retention" not in kwargs


# ── strict-schema compliance (real production requirement, not fake client) ─


def _strict_schema_violations(schema: Any, path: str = "root") -> list[str]:
    """Return object/array nodes that violate OpenAI strict structured output:
    an object with `additionalProperties` != False, or an array with an
    unconstrained (empty) `items`."""
    problems: list[str] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties={schema.get('additionalProperties')!r}")
        if schema.get("type") == "array" and not schema.get("items"):
            problems.append(f"{path}: unconstrained items={schema.get('items')!r}")
        for key, value in schema.items():
            problems += _strict_schema_violations(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            problems += _strict_schema_violations(value, f"{path}[{i}]")
    return problems


def test_parser_models_produce_real_strict_schema():
    """Every parser-supported response model must yield a genuinely strict
    schema via the SDK's own `to_strict_json_schema`: no `additionalProperties:
    true` and no unconstrained array items anywhere. This is the real
    production gate — not a fake-client assertion that `parse` was called."""
    from openai.lib._pydantic import to_strict_json_schema

    # Currently ACTIVE parser tasks. step6_schema_mapping_stage_2 is
    # temporarily disabled from the parser path (json_object instead) while
    # the Step 6 Stage 2 prompt still asks for the legacy dynamic-dict shape.
    assert set(_RESPONSE_MODEL_FOR_TASK) == {
        "tool_selection_stage_1",
        "step6_schema_mapping_stage_1",
        "step9_tool_selection_stage_1",
        "step9_tool_schema_mapping_stage_2",
        "step14_patent_tool_selection",
        "patent_evidence_tool_selection",
        "orchestrator_worker_routing",
    }
    assert "step6_schema_mapping_stage_2" not in _RESPONSE_MODEL_FOR_TASK
    for task, model in _RESPONSE_MODEL_FOR_TASK.items():
        schema = to_strict_json_schema(model)
        violations = _strict_schema_violations(schema)
        assert not violations, f"{task}/{model.__name__} strict-schema violations: {violations}"


def test_disabled_step6_stage2_parser_model_is_still_strict_for_future_use():
    """The Step 6 Stage 2 list-of-pairs parser model is TEMPORARILY DISABLED
    from the active parser path but kept for future re-enable. It must still
    produce a genuinely strict schema so re-enabling is a one-line change."""
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(_Step6SchemaMappingStage2ParserResponse)
    violations = _strict_schema_violations(schema)
    assert not violations, f"disabled step6 stage2 model strict-schema violations: {violations}"


def test_step6_stage2_uses_json_object_path_not_parser():
    """Even when the parser IS available, step6_schema_mapping_stage_2 must go
    through the json_object path (returning the legacy dynamic-dict shape),
    because it is disabled from the parser registry."""
    client = _fake_openai_client(
        has_parse=True,
        create=_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter","can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],"argument_mapping_reason":"mapped"}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("map", schema={"task": "step6_schema_mapping_stage_2"})
    assert client._calls["parse"] == []  # parser NOT used for disabled task
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_structured_query_not_routed_to_strict_parser():
    """structured_query (deeply-nested / variant fields) must NOT be routed to
    the strict parser — it stays on the json_object path."""
    assert "structured_query" not in _RESPONSE_MODEL_FOR_TASK


# ── behavioral parser tests (supported tasks: stage_1 / step6 stage_1) ─────


def test_openai_stage1_uses_structured_parser_and_returns_dict():
    parsed = _ToolSelectionStage1Response(
        selections=[{"tool_name": "DrugProps_calculate_qed"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)

    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})

    assert len(client._calls["parse"]) == 1
    assert client._calls["create"] == []  # parser succeeded → no fallback
    assert client._calls["parse"][0]["response_format"] is _ToolSelectionStage1Response
    assert isinstance(out, dict)
    assert not isinstance(out, _ToolSelectionStage1Response)
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_step6_stage1_uses_structured_parser_and_validated():
    parsed = _Step6SchemaMappingStage1Response(
        selections=[{"tool_name": "DrugProps_pains_filter", "selection_reason": "smiles"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "step6_schema_mapping_stage_1"})
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step6SchemaMappingStage1Response
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def _orchestrator_response(**overrides: Any) -> dict[str, Any]:
    decision = {
        "agent_id": "step_06_developability_agent",
        "capability_id": "step_06_developability",
        "objective": "Assess developability.",
        "selection_reason": "The user requested developability assessment.",
        "priority": "normal",
    }
    decision.update(overrides)
    return {
        "loop_decision": "dispatch_next_workers",
        "decisions": [decision],
        "decision_summary": "Dispatch the developability worker.",
    }


def _orchestrator_prompt_schema() -> dict[str, Any]:
    return {
        "task": "orchestrator_worker_routing",
        "compact_card_catalog": [
            {
                "agent_id": "step_06_developability_agent",
                "capabilities": [
                    {"capability_id": "step_06_developability"}
                ],
            }
        ],
        "compact_user_intent": "Assess developability",
        "structured_intent": {},
        "input_readiness_summary": {"input_readiness_status": "ready"},
        "available_artifact_summary": [],
        "current_routing_context": {},
    }


def test_openai_orchestrator_uses_official_parser_and_returns_external_dict():
    parsed = _OrchestratorRoutingResponse.model_validate(_orchestrator_response())
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)

    out = provider.generate_json(
        "route", schema={"task": "orchestrator_worker_routing"}
    )

    assert len(client._calls["parse"]) == 1
    assert client._calls["create"] == []
    assert client._calls["parse"][0]["response_format"] is _OrchestratorRoutingResponse
    assert isinstance(out, dict)
    assert out == _orchestrator_response()


def test_openai_orchestrator_system_role_is_not_duplicated_in_user_message():
    parsed = _OrchestratorRoutingResponse.model_validate(_orchestrator_response())
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)

    provider.generate_json(
        ORCHESTRATOR_ROUTING_USER_TASK,
        schema=_orchestrator_prompt_schema(),
        system=ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    )

    messages = client._calls["parse"][0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == ORCHESTRATOR_ROUTING_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert ORCHESTRATOR_ROUTING_SYSTEM_PROMPT not in user
    assert ORCHESTRATOR_ROUTING_USER_TASK in user
    assert user.count('"input_situation"') == 2
    assert "Compact AgentCard catalog JSON:" in user
    assert "step_06_developability_agent" in user
    assert sum(
        message["content"].count(ORCHESTRATOR_ROUTING_SYSTEM_PROMPT)
        for message in messages
    ) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent_id", ""),
        ("capability_id", ""),
        ("objective", ""),
        ("selection_reason", ""),
        ("priority", "urgent"),
    ],
)
def test_openai_orchestrator_parser_model_validates_decision_fields(field, value):
    with pytest.raises(ValueError):
        _OrchestratorRoutingResponse.model_validate(
            _orchestrator_response(**{field: value})
        )


def test_openai_orchestrator_parser_model_rejects_empty_summary():
    with pytest.raises(ValueError):
        _OrchestratorRoutingResponse.model_validate(
            {**_orchestrator_response(), "decision_summary": ""}
        )


@pytest.mark.parametrize("field", ["objective", "selection_reason"])
def test_openai_orchestrator_empty_decision_field_retries(field):
    bad = _orchestrator_response(**{field: ""})
    good = _OrchestratorRoutingResponse.model_validate(_orchestrator_response())
    sequence = iter([_parsed_response(bad), _parsed_response(good)])
    client = _fake_openai_client(parse=lambda **kw: next(sequence))
    provider = _provider_with_client(client, max_retries=1)

    out = provider.generate_json(
        "route", schema={"task": "orchestrator_worker_routing"}
    )

    assert len(client._calls["parse"]) == 2
    assert out["decisions"][0][field]


def test_openai_orchestrator_empty_summary_retries():
    bad = {**_orchestrator_response(), "decision_summary": ""}
    good = _OrchestratorRoutingResponse.model_validate(_orchestrator_response())
    sequence = iter([_parsed_response(bad), _parsed_response(good)])
    client = _fake_openai_client(parse=lambda **kw: next(sequence))
    provider = _provider_with_client(client, max_retries=1)

    out = provider.generate_json(
        "route", schema={"task": "orchestrator_worker_routing"}
    )

    assert len(client._calls["parse"]) == 2
    assert out["decision_summary"]


def test_openai_orchestrator_parser_unavailable_uses_json_object_fallback():
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response(json.dumps(_orchestrator_response())),
    )
    provider = _provider_with_client(client)

    out = provider.generate_json(
        "route", schema={"task": "orchestrator_worker_routing"}
    )

    assert client._calls["parse"] == []
    assert len(client._calls["create"]) == 1
    assert out == _orchestrator_response()


def test_openai_orchestrator_schema_incompatibility_uses_fallback():
    def _incompatible(**kw: Any) -> Any:
        raise TypeError("response_format invalid schema")

    client = _fake_openai_client(
        parse=_incompatible,
        create=_chat_response(json.dumps(_orchestrator_response())),
    )
    provider = _provider_with_client(client)

    out = provider.generate_json(
        "route", schema={"task": "orchestrator_worker_routing"}
    )

    assert len(client._calls["parse"]) == 1
    assert len(client._calls["create"]) == 1
    assert out == _orchestrator_response()


@pytest.mark.parametrize(
    "error", [RuntimeError("network down"), RuntimeError("auth failed"), RuntimeError("rate limit")]
)
def test_openai_orchestrator_runtime_errors_do_not_silent_fallback(error):
    def _fail(**kw: Any) -> Any:
        raise error

    client = _fake_openai_client(
        parse=_fail,
        create=_chat_response(json.dumps(_orchestrator_response())),
    )
    provider = _provider_with_client(client)

    with pytest.raises(RuntimeError, match=str(error)):
        provider.generate_json(
            "route", schema={"task": "orchestrator_worker_routing"}
        )
    assert client._calls["create"] == []


def test_openai_orchestrator_usage_event_is_compact_and_private():
    usage = SimpleNamespace(prompt_tokens=90, completion_tokens=20, total_tokens=110)
    parsed = _OrchestratorRoutingResponse.model_validate(_orchestrator_response())
    client = _fake_openai_client(parse=_parsed_response(parsed, usage=usage))
    provider = _provider_with_client(client)

    provider.generate_json(
        "FULL_PROMPT_SECRET",
        schema={"task": "orchestrator_worker_routing", "private": "RAW_SCHEMA_SECRET"},
        system="SYSTEM_SECRET",
    )

    assert provider.usage_events == [
        {
            "provider": "openai",
            "model": provider.model,
            "task": "orchestrator_worker_routing",
            "attempt": 0,
            "prompt_tokens": 90,
            "completion_tokens": 20,
            "total_tokens": 110,
            "cached_prompt_tokens": None,
        }
    ]
    blob = json.dumps(provider.usage_events)
    for forbidden in (
        "FULL_PROMPT_SECRET",
        "SYSTEM_SECRET",
        "RAW_SCHEMA_SECRET",
        "sk-fake-parser",
        "response_format",
    ):
        assert forbidden not in blob


def test_openai_step14_uses_structured_parser_and_normalizes_literals():
    # The single-stage step14 planner parser returns tool_plans with
    # argument_mappings; literal_value_json is decoded to literal_value in the
    # external dict the validator/runtime consume.
    parsed = _Step14PatentToolSelectionResponse(
        tool_plans=[{
            "tool_name": "drugbank_get_drug_references_by_drug_name_or_id",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "query", "input_ref_id": "r_payload"}],
            "argument_literals": [{"schema_arg": "limit", "literal_value_json": "25"}],
            "missing_required_args": [],
            "selection_reason": "payload ref fills query",
        }]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json("plan", schema={"task": "step14_patent_tool_selection"})
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step14PatentToolSelectionResponse
    plan = out["tool_plans"][0]
    assert plan["argument_mappings"] == [{"schema_arg": "query", "input_ref_id": "r_payload"}]
    # literal_value_json decoded to a real literal_value.
    assert plan["argument_literals"] == [{"schema_arg": "limit", "literal_value": 25}]
    assert "selected_tool_plans" not in out  # single-stage shape, not Turn A


def test_step14_strict_parser_schema_uses_literal_value_json_only():
    from openai.lib._pydantic import to_strict_json_schema

    schema_json = json.dumps(to_strict_json_schema(_Step14PatentToolSelectionResponse))
    assert "literal_value_json" in schema_json
    # The strict parser shape must NOT expose a bare `literal_value` field.
    assert '"literal_value"' not in schema_json


def test_openai_patent_evidence_strict_parser_includes_lane_assessments():
    parsed = _PatentEvidenceToolSelectionResponse(
        lane_assessments=[
            {"search_lane": "evidence", "status": "planned", "reason": "query ref"},
            {
                "search_lane": "patent",
                "status": "missing_inputs",
                "reason": "no patent identifier ref",
            },
        ],
        tool_plans=[
            {
                "tool_name": "EuropePMC_search_articles",
                "can_invoke": True,
                "argument_mappings": [
                    {"schema_arg": "query", "input_ref_id": "r_query"}
                ],
                "argument_literals": [],
                "missing_required_args": [],
                "selection_reason": "query ref",
            }
        ],
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json(
        "plan", schema={"task": "patent_evidence_tool_selection"}
    )
    assert client._calls["parse"][0]["response_format"] is _PatentEvidenceToolSelectionResponse
    assert out["lane_assessments"][1]["status"] == "missing_inputs"


def test_openai_step9_stage1_uses_structured_parser_and_returns_dict():
    parsed = _Step9ToolSelectionStage1Response(
        selections=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "selection_reason": "smiles available",
        }]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json(
        "pick", schema={"task": "step9_tool_selection_stage_1"}
    )
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step9ToolSelectionStage1Response
    assert isinstance(out, dict)
    assert not isinstance(out, _Step9ToolSelectionStage1Response)
    assert out["selections"][0]["tool_name"] == "ZINC_search_by_smiles"


def test_openai_step9_stage2_uses_structured_parser_and_returns_dict():
    parsed = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "mapped",
        }]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step9SchemaMappingStage2Response
    assert isinstance(out, dict)
    assert not isinstance(out, _Step9SchemaMappingStage2Response)
    assert out["tools"][0]["argument_mappings"] == [
        {"schema_arg": "smiles", "field_ref": "material:m1"}
    ]
    assert out["tools"][0]["argument_literals"] == {}


def test_openai_step9_stage2_parser_converts_literal_value_json_to_parsed_dict():
    parsed = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "sequence", "field_ref": "material:heavy_chain"},
            ],
            "argument_literals": [
                {
                    "schema_arg": "variants",
                    "literal_value_json": '[{"position":777,"ref_aa":"V","alt_aa":"L"}]',
                },
                {"schema_arg": "model", "literal_value_json": '"esmc-6b-2024-12"'},
            ],
            "missing_required_fields": [],
        }]
    )
    external = parsed.to_external_dict()
    literals = external["tools"][0]["argument_literals"]
    assert literals["variants"] == [{"position": 777, "ref_aa": "V", "alt_aa": "L"}]
    assert literals["model"] == "esmc-6b-2024-12"
    assert isinstance(literals["variants"], list)
    assert not isinstance(literals["variants"], str)


def test_openai_step9_stage2_parser_rejects_invalid_literal_value_json():
    parsed = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "sequence", "field_ref": "material:heavy_chain"},
            ],
            "argument_literals": [
                {"schema_arg": "variants", "literal_value_json": "[not-json"},
            ],
            "missing_required_fields": [],
        }]
    )
    with pytest.raises(OpenAIProviderError, match="invalid literal_value_json"):
        parsed.to_external_dict()


def test_openai_step9_stage2_parser_rejects_duplicate_literal_schema_arg():
    parsed = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "sequence", "field_ref": "material:heavy_chain"},
            ],
            "argument_literals": [
                {"schema_arg": "variants", "literal_value_json": "[]"},
                {"schema_arg": "variants", "literal_value_json": "[]"},
            ],
            "missing_required_fields": [],
        }]
    )
    with pytest.raises(OpenAIProviderError, match="duplicate schema_arg `variants`"):
        parsed.to_external_dict()


def test_step9_stage2_strict_parser_schema_uses_literal_value_json_only():
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(_Step9SchemaMappingStage2Response)
    blob = json.dumps(schema)
    assert "literal_value_json" in blob
    assert "argument_json_literals" not in blob
    assert '"literal_value"' not in blob
    violations = _strict_schema_violations(schema)
    assert not violations


def test_openai_step9_stage2_duplicate_schema_arg_raises_then_retry_succeeds():
    dup = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
                {"schema_arg": "smiles", "field_ref": "material:m2"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    good = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    seq = iter([_parsed_response(dup), _parsed_response(good)])
    client = _fake_openai_client(parse=lambda **kw: next(seq))
    provider = _provider_with_client(client, max_retries=1)
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert len(client._calls["parse"]) == 2
    assert out["tools"][0]["argument_mappings"] == [
        {"schema_arg": "smiles", "field_ref": "material:m1"}
    ]


def test_openai_parser_output_flows_through_validate_task_shape():
    """Parser output still runs the shared per-task validator: a stage_1 entry
    missing tool_name is rejected (with retry) rather than silently accepted."""
    bad = _ToolSelectionStage1Response()  # empty selections is valid-empty
    # Build a parsed object whose model_dump omits tool_name by using a dict.
    bad_dict = {"selections": [{"selection_reason": "no tool_name here"}]}
    client = _fake_openai_client(parse=_parsed_response(bad_dict))
    provider = _provider_with_client(client, max_retries=0)
    with pytest.raises(OpenAIProviderError, match="tool_selection_stage_1.*tool_name"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    del bad


def test_openai_parser_accepts_dict_parsed_shape():
    """Some SDK/proxy variants set `message.parsed` to a dict; still a dict out."""
    client = _fake_openai_client(
        parse=_parsed_response({"selections": [{"tool_name": "DrugProps_calculate_qed"}]})
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert isinstance(out, dict)
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_falls_back_to_json_object_when_no_parser_api():
    """SDK without a `parse` method → fall back to json_object create."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert client._calls["parse"] == []  # no parse method existed
    assert len(client._calls["create"]) == 1  # fell back to json_object
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_falls_back_on_parser_incompatibility_error():
    """Parser raising an incompatibility error → json_object fallback."""
    def _parse_boom(**kw: Any) -> Any:
        raise TypeError("invalid schema for response_format: additionalProperties")

    client = _fake_openai_client(
        parse=_parse_boom,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert len(client._calls["parse"]) == 1  # attempted
    assert len(client._calls["create"]) == 1  # then fell back
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_fallback_records_usage_events_and_no_leak():
    """Fallback path still records compact usage and leaks nothing."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response_with_usage(
            '{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}',
            prompt=70, completion=15, total=85,
        ),
    )
    provider = _provider_with_client(client)
    provider.generate_json(
        "USER_PROMPT_SECRET", schema={"task": "tool_selection_stage_1"}, system="SYS_SECRET",
    )
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["task"] == "tool_selection_stage_1"
    assert evt["total_tokens"] == 85
    import json as _json
    blob = _json.dumps(evt)
    for secret in ("sk-fake-parser", "USER_PROMPT_SECRET", "SYS_SECRET"):
        assert secret not in blob


def test_openai_parser_records_usage_events_and_no_leak():
    parsed = _ToolSelectionStage1Response(selections=[{"tool_name": "DrugProps_calculate_qed"}])
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=30, total_tokens=230)
    client = _fake_openai_client(parse=_parsed_response(parsed, usage=usage))
    provider = _provider_with_client(client)
    provider.generate_json(
        "USER_PROMPT_SECRET", schema={"task": "tool_selection_stage_1"}, system="SYS_SECRET",
    )
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] == 200 and evt["total_tokens"] == 230
    import json as _json
    blob = _json.dumps(evt)
    for secret in ("sk-fake-parser", "USER_PROMPT_SECRET", "SYS_SECRET"):
        assert secret not in blob


def test_openai_parser_genuine_error_propagates_no_silent_fallback():
    """A non-incompatibility error from the parser is NOT masked as a
    fallback — it propagates (no fake success), and create is never called."""
    def _parse_network(**kw: Any) -> Any:
        raise RuntimeError("connection reset by peer")

    client = _fake_openai_client(
        parse=_parse_network,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client, max_retries=0)
    with pytest.raises(RuntimeError, match="connection reset"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert client._calls["create"] == []  # never silently fell back


def test_openai_structured_query_stays_on_json_object_path():
    """structured_query is NOT strict-parser-routed: it uses json_object and
    still runs validate + normalize (adc_candidate → ranked_candidates)."""
    client = _fake_openai_client(
        parse=_parsed_response({"should": "not be used"}),
        create=_chat_response(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{},'
            '"requested_outputs":["adc_candidate","report"]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("parse", schema={"task": "structured_query"})
    assert client._calls["parse"] == []  # parser NOT used for structured_query
    assert len(client._calls["create"]) == 1
    assert out["requested_outputs"] == ["ranked_candidates", "report"]


# ── Step 6 Stage 2 parser model: TEMPORARILY DISABLED, kept for future ──────
# These exercise the KEPT list-of-pairs parser model + `to_external_dict`
# directly (NOT the provider routing, which is json_object for this task now),
# so the conversion + duplicate-guard code stays covered for a future
# re-enable. See `test_step6_stage2_uses_json_object_path_not_parser` for the
# active routing behavior.


def test_future_step6_stage2_model_list_of_pairs_converts_to_dynamic_dict():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "candidate:c1:material:m1:value"},
            ],
            "argument_literals": [
                {"schema_arg": "operation", "literal_value": "calculate_adme"},
            ],
            "missing_required_fields": [],
            "argument_mapping_reason": "smiles mapped; operation is a schema literal",
        }]
    )
    external = parsed.to_external_dict()
    tool = external["tools"][0]
    assert tool["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}
    assert tool["argument_literals"] == {"operation": "calculate_adme"}
    assert tool["can_invoke"] is True
    assert tool["missing_required_fields"] == []
    assert tool["argument_mapping_reason"].startswith("smiles mapped")


def test_future_step6_stage2_model_empty_literals_becomes_empty_dict():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "DrugProps_pains_filter",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "candidate:c1:material:m1:value"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    external = parsed.to_external_dict()
    assert external["tools"][0]["argument_literals"] == {}
    assert external["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_future_step6_stage2_model_can_invoke_false_empty_mappings_missing_fields():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "PDBePISA_get_interfaces",
            "can_invoke": False,
            "argument_mappings": [],
            "argument_literals": [],
            "missing_required_fields": ["pdb_id"],
            "argument_mapping_reason": "no pdb_id available",
        }]
    )
    external = parsed.to_external_dict()
    tool = external["tools"][0]
    assert tool["can_invoke"] is False
    assert tool["argument_mapping"] == {}
    assert tool["argument_literals"] == {}
    assert tool["missing_required_fields"] == ["pdb_id"]


def test_future_step6_stage2_model_duplicate_schema_arg_raises_no_silent_overwrite():
    """Duplicate argument_mapping schema_arg raises (never a silent overwrite)
    at conversion time, so a future re-enable keeps that guarantee."""
    dup = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "ref_a"},
                {"schema_arg": "smiles", "field_ref": "ref_b"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    with pytest.raises(OpenAIProviderError, match="duplicate argument_mapping"):
        dup.to_external_dict()


def test_openai_step6_stage2_fallback_json_object_accepts_old_dict_shape():
    """When the parser is unavailable, the json_object fallback still returns
    the existing dynamic-dict shape unchanged (no list-of-pairs conversion)."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter","can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],"argument_mapping_reason":"mapped"}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("map", schema={"task": "step6_schema_mapping_stage_2"})
    assert client._calls["parse"] == []
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_openai_step6_stage2_output_passes_validate_task_shape():
    """The converted external dict passes the shared step6 stage2 validator
    (bad shape would raise). Here a valid tool round-trips."""
    from app.llm.json_task_validation import validate_task_shape
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "smiles", "field_ref": "candidate:x"}],
            "argument_literals": [{"schema_arg": "operation", "literal_value": "calculate_adme"}],
            "missing_required_fields": [],
        }]
    )
    external = parsed.to_external_dict()
    # Direct validator round-trip (same call the provider makes).
    validated = validate_task_shape(external, "step6_schema_mapping_stage_2", error_factory=OpenAIProviderError)
    assert validated["tools"][0]["can_invoke"] is True


def test_openai_unsupported_task_skips_parser_uses_json_object():
    """A task without a structured-output model goes straight to json_object;
    the parser is never attempted."""
    client = _fake_openai_client(
        parse=_parsed_response(_ToolSelectionStage1Response()),  # would be wrong if used
        create=_chat_response(
            '{"tools":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter","arguments":{"smiles":"CCO"}}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("args", schema={"task": "tool_selection_stage_2_multi_tool"})
    assert client._calls["parse"] == []  # unsupported task → no parser
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["arguments"] == {"smiles": "CCO"}


def test_openai_parser_refusal_is_surfaced_then_retried():
    """A structured-output refusal raises a compact error and is retried."""
    refusal_msg = SimpleNamespace(parsed=None, content=None, refusal="I can't help with that")
    refusal_resp = SimpleNamespace(choices=[SimpleNamespace(message=refusal_msg)], usage=None)
    good = _parsed_response(_ToolSelectionStage1Response(selections=[{"tool_name": "DrugProps_calculate_qed"}]))
    parse_seq = iter([refusal_resp, good])
    client = _fake_openai_client(parse=lambda **kw: next(parse_seq))
    provider = _provider_with_client(client, max_retries=1)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert len(client._calls["parse"]) == 2  # retried after refusal
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"
    # both attempts recorded in usage_events
    assert [e["attempt"] for e in provider.usage_events] == [0, 1]


def test_openai_usage_events_shape_unchanged_after_dedup():
    """Defensive sanity: the dedup fix does not strip any field from
    ``usage_events``. Every recorded entry still has the 8 documented
    keys."""
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=100, completion=20, total=120,
        )
    ])
    provider.generate_json(
        "parse", schema={"raw_request_record": {}},
        system="SOME SYSTEM PROMPT",
    )
    assert len(provider.usage_events) == 1
    assert set(provider.usage_events[0].keys()) == {
        "provider", "model", "task", "attempt",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cached_prompt_tokens",
    }
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-fake-key" not in blob
    assert "SOME SYSTEM PROMPT" not in blob
