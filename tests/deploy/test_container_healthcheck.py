"""Unit tests for the worker-only container health probe."""

from __future__ import annotations

import ast
import io
import urllib.error
from pathlib import Path

import pytest

from app.a2a.container_healthcheck import check_worker_health, main


class _Response(io.BytesIO):
    def __init__(self, body: bytes, *, status: int = 200):
        super().__init__(body)
        self.status = status


def _response(body: str, *, status: int = 200) -> _Response:
    return _Response(body.encode("utf-8"), status=status)


def _check(monkeypatch, response=None, error=None):
    calls = []

    def _urlopen(url, *, timeout):
        calls.append((url, timeout))
        if error is not None:
            raise error
        return response

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    result = check_worker_health(
        url="http://worker.invalid/health",
        expected_agent_id="expected_agent",
        expected_capability_ids=["expected_capability"],
        timeout=2.5,
    )
    assert calls == [("http://worker.invalid/health", 2.5)]
    return result


def test_correct_health_response_succeeds(monkeypatch):
    result = _check(
        monkeypatch,
        _response(
            '{"status":"ok","agent_id":"expected_agent",'
            '"capabilities":["expected_capability"]}'
        ),
    )
    assert result.ok is True
    assert result.code == "ok"


@pytest.mark.parametrize(
    "body,code",
    [
        (
            '{"status":"ok","agent_id":"wrong",'
            '"capabilities":["expected_capability"]}',
            "agent_id_mismatch",
        ),
        (
            '{"status":"ok","agent_id":"expected_agent","capabilities":[]}',
            "capabilities_mismatch",
        ),
        (
            '{"status":"ok","agent_id":"expected_agent",'
            '"capabilities":["expected_capability","extra"]}',
            "capabilities_mismatch",
        ),
        (
            '{"status":"degraded","agent_id":"expected_agent",'
            '"capabilities":["expected_capability"]}',
            "status_not_ok",
        ),
        ("not-json", "response_not_json"),
    ],
)
def test_invalid_health_responses_fail(monkeypatch, body, code):
    result = _check(monkeypatch, _response(body))
    assert result.ok is False
    assert result.code == code


def test_non_200_fails(monkeypatch):
    result = _check(monkeypatch, _response("{}", status=503))
    assert result == type(result)(False, "http_status_mismatch")


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        urllib.error.URLError("credential-bearing endpoint must not be printed"),
        ConnectionError("raw connection detail"),
    ],
)
def test_timeout_and_connection_failures_are_compact(monkeypatch, error):
    result = _check(monkeypatch, error=error)
    assert result.ok is False
    assert result.code == "connection_failed"


def test_cli_failure_prints_only_compact_code(monkeypatch, capsys):
    monkeypatch.setattr(
        "app.a2a.container_healthcheck.check_worker_health",
        lambda **_kwargs: type("Result", (), {"ok": False, "code": "agent_id_mismatch"})(),
    )
    exit_code = main(
        [
            "--url",
            "http://user:secret@worker.invalid/health",
            "--agent-id",
            "expected_agent",
            "--capability",
            "expected_capability",
            "--timeout",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "container_healthcheck_failed:agent_id_mismatch\n"
    assert "secret" not in captured.err
    assert "worker.invalid" not in captured.err


def test_module_has_no_domain_llm_mcp_or_a2a_execution_calls():
    path = Path(__file__).resolve().parents[2] / "app/a2a/container_healthcheck.py"
    tree = ast.parse(path.read_text())
    imports = []
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
    assert not any(
        token in module
        for module in imports
        for token in ("agents", "llm", "mcp", "python_a2a")
    )
    assert not {
        "send_task_async",
        "execute_request",
        "call_tool",
        "generate",
        "generate_json",
    } & called
