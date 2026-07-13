"""Turn E — Docker Compose deployment skeleton tests.

Static + import-level checks over the real ``docker-compose.yml`` / ``Dockerfile``
/ ``.dockerignore`` and the worker entrypoint modules. These do NOT build images
or start containers (that real smoke is run separately); they lock the skeleton's
contract so it cannot silently drift:

- exactly the four business services, correct names;
- no mem0 / postgres / pgvector / vector-db service;
- workers internal-only (expose, no host ports); orchestrator the only host port;
- shared local-artifact volume + identical LOCAL_STORAGE_ROOT everywhere;
- orchestrator points at the Docker-internal worker URLs;
- worker entrypoints delegate to the existing ``run_step*_worker`` (python_a2a
  A2AServer path) and never call a domain agent directly or scan/switch ports;
- Dockerfile/.dockerignore keep secrets / local artifacts / raw bio data out;
- no Mem0 / PostgreSQL integration code was introduced.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from app.a2a.worker_server import effective_url_port

_EXEC_DIR = Path(__file__).resolve().parents[2]
_COMPOSE = _EXEC_DIR / "docker-compose.yml"
_DOCKERFILE = _EXEC_DIR / "Dockerfile"
_DOCKERIGNORE = _EXEC_DIR / ".dockerignore"

_BUSINESS_SERVICES = {"orchestrator", "step5-worker", "step6-worker", "structure-worker"}
_WORKERS = {"step5-worker", "step6-worker", "structure-worker"}
_INVENTORY_CONTAINER_PATH = "/opt/adc/inventory/ToolUniversity_inventory_v0.2.xlsx"
_STORAGE_ROOT = "/data/localstore"


@pytest.fixture(scope="module")
def compose() -> dict:
    with _COMPOSE.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def compose_text() -> str:
    return _COMPOSE.read_text()


def _env(service: dict) -> dict:
    env = service.get("environment", {})
    assert isinstance(env, dict), "environment must be a mapping"
    return env


def _volume_pairs(service: dict) -> list[tuple[str, str]]:
    """Return (source, target) pairs for named-volume and bind mounts."""
    out = []
    for v in service.get("volumes", []):
        if isinstance(v, str):
            src, _, tgt = v.partition(":")
            out.append((src, tgt.split(":")[0]))
        elif isinstance(v, dict):
            out.append((str(v.get("source")), str(v.get("target"))))
    return out


# ── 1. service names ─────────────────────────────────────────────────────────
def test_exact_four_business_services(compose):
    assert set(compose["services"]) == _BUSINESS_SERVICES


# ── 2-4. no mem0 / postgres / pgvector / vector db ───────────────────────────
# These check the PARSED services / images / envs (not comment text — the file's
# own comments and DEPLOYMENT doc legitimately name these as "not included").
def _service_images(compose) -> list[str]:
    return [str(svc.get("image", "")).lower() for svc in compose["services"].values()]


def test_no_mem0_service(compose):
    assert "mem0" not in compose["services"]
    assert all("mem0" not in img for img in _service_images(compose))


def test_no_postgres_service(compose):
    assert "postgres" not in compose["services"]
    assert all("postgres" not in img for img in _service_images(compose))
    for svc in compose["services"].values():
        assert "DATABASE_URL" not in _env(svc)


def test_no_pgvector_or_vector_db(compose):
    # Only the four business services exist; none uses a vector-db image.
    assert set(compose["services"]) == _BUSINESS_SERVICES
    for img in _service_images(compose):
        for banned in ("pgvector", "weaviate", "qdrant", "milvus", "chroma", "vector"):
            assert banned not in img, f"unexpected vector-db image token: {banned}"


# ── 5. workers internal-only (expose, no host ports); orchestrator host port ──
def test_workers_use_expose_no_host_ports(compose):
    for name in _WORKERS:
        svc = compose["services"][name]
        assert "ports" not in svc, f"{name} must not publish host ports"
        assert svc.get("expose"), f"{name} must declare internal expose"
    # exact expose ports
    assert compose["services"]["step5-worker"]["expose"] == ["8005"]
    assert compose["services"]["step6-worker"]["expose"] == ["8006"]
    assert compose["services"]["structure-worker"]["expose"] == ["8009"]


def test_only_orchestrator_maps_a_host_port(compose):
    assert "ports" in compose["services"]["orchestrator"]
    ports = compose["services"]["orchestrator"]["ports"]
    assert len(ports) == 1
    # host side is env-driven with a default; container side is 8000.
    assert "8000" in str(ports[0])
    assert "ORCHESTRATOR_HOST_PORT" in str(ports[0])


# ── 6. orchestrator points at Docker-internal worker URLs ────────────────────
def test_orchestrator_internal_worker_urls(compose):
    env = _env(compose["services"]["orchestrator"])
    assert env["STEP5_WORKER_URL"] == "http://step5-worker:8005"
    assert env["STEP6_WORKER_URL"] == "http://step6-worker:8006"
    assert env["STRUCTURE_WORKER_URL"] == "http://structure-worker:8009"


# ── 7-8. shared artifact volume + identical LOCAL_STORAGE_ROOT ────────────────
def test_all_services_share_same_local_artifact_volume(compose):
    sources = set()
    targets = set()
    for name in _BUSINESS_SERVICES:
        pairs = _volume_pairs(compose["services"][name])
        named = [(s, t) for s, t in pairs if s == "adc_local_store"]
        assert len(named) == 1, f"{name} must mount the shared named volume once"
        sources.add(named[0][0])
        targets.add(named[0][1])
    assert sources == {"adc_local_store"}, "single shared volume source"
    assert targets == {_STORAGE_ROOT}, "identical mount target everywhere"
    # No explicit global name: Compose project-name isolation owns the real name.
    assert compose["volumes"]["adc_local_store"] is None


def test_all_services_share_same_logical_network_without_global_name(compose):
    for name in _BUSINESS_SERVICES:
        assert compose["services"][name]["networks"] == ["adc_a2a_net"]
    assert compose["networks"]["adc_a2a_net"] == {"driver": "bridge"}


def test_all_services_same_local_storage_root(compose):
    roots = {_env(compose["services"][n])["LOCAL_STORAGE_ROOT"] for n in _BUSINESS_SERVICES}
    assert roots == {_STORAGE_ROOT}
    modes = {_env(compose["services"][n])["STORAGE_MODE"] for n in _BUSINESS_SERVICES}
    assert modes == {"local"}


def test_all_services_same_inventory_path(compose):
    for name in _BUSINESS_SERVICES:
        env = _env(compose["services"][name])
        assert env["TOOL_INVENTORY_XLSX"] == _INVENTORY_CONTAINER_PATH
        # inventory is a read-only bind, never copied into the image.
        binds = [
            v for v in compose["services"][name]["volumes"]
            if isinstance(v, dict) and v.get("target") == _INVENTORY_CONTAINER_PATH
        ]
        assert len(binds) == 1
        assert binds[0]["read_only"] is True
        assert binds[0]["source"].endswith("ToolUniversity_inventory_v0.2.xlsx")


# ── 9. Dockerfile / .dockerignore keep secrets + artifacts out ───────────────
def test_dockerignore_excludes_secrets_and_artifacts():
    text = _DOCKERIGNORE.read_text()
    for needle in (".env", ".git", ".localstore", "*.fasta", "*.pdb", "*.cif", "*.a3m", "tests/"):
        assert needle in text, f".dockerignore must exclude {needle}"


def test_dockerfile_has_no_credentials_and_uses_supported_runtime():
    text = _DOCKERFILE.read_text()
    assert "FROM python:3.12-slim" in text
    low = text.lower()
    for banned in ("api_key", "aws_secret", "password", "secret_key", "-----begin"):
        assert banned not in low, f"Dockerfile must not contain credential token: {banned}"
    assert "'torch==2.13.0+cpu'" in text
    assert "https://download.pytorch.org/whl/cpu" in text
    assert "'.[deployment,admet]'" in text
    assert "PIP_RETRIES" not in text
    assert "vendor/wheels" not in text
    for other_a2a in ("a2a-sdk", "google-a2a", "pip install a2a"):
        assert other_a2a not in low


def test_deployment_python_and_esm_source_are_documented_and_unique():
    pyproject = (_EXEC_DIR / "pyproject.toml").read_text()
    deployment = (_EXEC_DIR / "DEPLOYMENT_A2A.md").read_text()
    immutable_url = (
        "https://github.com/evolutionaryscale/esm/archive/"
        "ba4d7124864eed323a93bf3cfefcd958f573b75a.tar.gz"
    )
    assert 'requires-python = ">=3.11"' in pyproject
    assert pyproject.count(immutable_url) == 1
    assert immutable_url not in _DOCKERFILE.read_text()
    assert "Python `>=3.11`" in deployment
    assert "Python 3.12" in deployment
    assert "`>=3.12,<3.13`" in deployment


def test_target_architecture_boundary_is_documented_without_compose_pin(compose):
    deployment = (_EXEC_DIR / "DEPLOYMENT_A2A.md").read_text()
    low = deployment.lower()
    normalized = " ".join(low.split())
    for required in (
        "linux/arm64",
        "linux/amd64",
        "uname -m",
        "docker image inspect",
        "architecture=amd64",
        "official cpu index",
        "gpu dependency guard",
        "pip check",
        "qemu",
        "fail fast",
    ):
        assert required in normalized
    assert "does **not** claim" in deployment
    assert "aws x86_64/amd64 host, has been validated" in low

    assert all("platform" not in service for service in compose["services"].values())
    assert "platform:" not in _COMPOSE.read_text().lower()


# ── 10-14. worker entrypoints ─────────────────────────────────────────────────
def test_worker_entrypoints_call_existing_run_functions():
    import app.a2a.step5_worker_main as m5
    import app.a2a.step6_worker_main as m6
    import app.a2a.structure_worker_main as ms
    from app.a2a.step5_worker import run_step5_worker
    from app.a2a.step6_worker import run_step6_worker
    from app.a2a.structure_worker import run_structure_worker

    assert m5.run_step5_worker is run_step5_worker
    assert m6.run_step6_worker is run_step6_worker
    assert ms.run_structure_worker is run_structure_worker
    for m in (m5, m6, ms):
        assert callable(m.main)


def test_worker_entrypoints_do_not_call_domain_agents_or_scan_ports():
    # Inspect real code (not docstrings): the entrypoints must not IMPORT domain
    # agents, dispatch tasks, or scan/switch ports. Docstrings may mention agent
    # names to explain what they do NOT do, so we check imports/behaviour, not
    # substrings of prose.
    import ast

    for fname in ("step5_worker_main.py", "step6_worker_main.py", "structure_worker_main.py"):
        path = _EXEC_DIR / "app" / "a2a" / fname
        src = path.read_text()
        tree = ast.parse(src)

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        # No agents package, no socket module (no port scanning).
        assert not any("agents" in m for m in imported), f"{fname} must not import domain agents"
        assert "socket" not in imported, f"{fname} must not import socket (no port scanning)"

        # No dispatch / direct-execution CALLS in code.
        called = {
            n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        for banned in ("send_task_async", "execute_request", "run", "run_from_artifacts"):
            assert banned not in called, f"{fname} must not call {banned}()"
        # Must delegate to the existing run_step*_worker entrypoint.
        assert "run_step5_worker" in src or "run_step6_worker" in src or "run_structure_worker" in src


def test_compose_advertised_url_matches_worker_bind_port(compose):
    for name, url_key in (
        ("step5-worker", "STEP5_WORKER_URL"),
        ("step6-worker", "STEP6_WORKER_URL"),
        ("structure-worker", "STRUCTURE_WORKER_URL"),
    ):
        env = _env(compose["services"][name])
        advertised = env[url_key]
        bind_port = int(env["WORKER_BIND_PORT"])
        assert effective_url_port(advertised) == bind_port
        assert env["WORKER_BIND_HOST"] == "0.0.0.0"


def test_worker_commands_use_module_entrypoints_not_inline_python(compose):
    expected = {
        "step5-worker": ["python", "-m", "app.a2a.step5_worker_main"],
        "step6-worker": ["python", "-m", "app.a2a.step6_worker_main"],
        "structure-worker": ["python", "-m", "app.a2a.structure_worker_main"],
    }
    for name, cmd in expected.items():
        got = compose["services"][name]["command"]
        assert got == cmd, f"{name} command must be the module entrypoint"
        # No long inline `python -c` business logic in the service command.
        assert "-c" not in got


# ── 15. no silent mock/offline deployment configuration ─────────────────────
def test_no_mock_success_or_validation_bypass_config(compose_text):
    low = compose_text.lower()
    for banned in (
        "mock_success",
        "fake_success",
        "bypass",
        "skip_validation",
        "disable_inventory",
        "disable_validation",
    ):
        assert banned not in low, f"compose must not configure {banned}"
    assert 'LLM_PROVIDER: "mock"' not in compose_text
    assert 'MCP_LIVE_TOOLS: "false"' not in compose_text
    assert "${LLM_PROVIDER:?LLM_PROVIDER must be set explicitly}" in compose_text
    assert "${MCP_LIVE_TOOLS:?MCP_LIVE_TOOLS must be set explicitly}" in compose_text


def test_worker_healthchecks_use_exact_identity_module(compose):
    expected = {
        "step5-worker": (
            "http://127.0.0.1:8005/health",
            "step_05_candidate_context_agent",
            "step_05_candidate_context",
        ),
        "step6-worker": (
            "http://127.0.0.1:8006/health",
            "step_06_developability_agent",
            "step_06_developability",
        ),
        "structure-worker": (
            "http://127.0.0.1:8009/health",
            "structure_and_design_agent",
            "structure_design_workflow",
        ),
    }
    for service_name, (url, agent_id, capability) in expected.items():
        command = compose["services"][service_name]["healthcheck"]["test"]
        assert command[:4] == [
            "CMD",
            "python",
            "-m",
            "app.a2a.container_healthcheck",
        ]
        assert command[command.index("--url") + 1] == url
        assert command[command.index("--agent-id") + 1] == agent_id
        assert command[command.index("--capability") + 1] == capability
        assert "-c" not in command


# ── 16. docker compose required env + project resource isolation ────────────
def _compose_env(*, explicit_modes: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["COMPOSE_DISABLE_ENV_FILE"] = "1"
    env.pop("LLM_PROVIDER", None)
    env.pop("MCP_LIVE_TOOLS", None)
    if explicit_modes:
        env["LLM_PROVIDER"] = "mock"
        env["MCP_LIVE_TOOLS"] = "false"
        env["ORCHESTRATOR_HOST_PORT"] = "18080"
    return env


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_docker_compose_config_fails_without_explicit_modes():
    result = subprocess.run(
        ["docker", "compose", "-p", "synagentics-a2a-v1", "-f", str(_COMPOSE), "config", "--quiet"],
        cwd=str(_EXEC_DIR),
        env=_compose_env(explicit_modes=False),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_docker_compose_config_is_valid_with_explicit_modes():
    result = subprocess.run(
        ["docker", "compose", "-p", "synagentics-a2a-v1", "-f", str(_COMPOSE), "config"],
        cwd=str(_EXEC_DIR),
        env=_compose_env(explicit_modes=True),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    resolved = yaml.safe_load(result.stdout)
    assert resolved["volumes"]["adc_local_store"]["name"] == (
        "synagentics-a2a-v1_adc_local_store"
    )
    assert resolved["networks"]["adc_a2a_net"]["name"] == (
        "synagentics-a2a-v1_adc_a2a_net"
    )
    assert "dify" not in result.stdout.lower()
    assert "ragflow" not in result.stdout.lower()


# ── 19-20. no A2A task dispatch / Mem0 / PostgreSQL integration introduced ────
def test_no_dispatch_or_memory_or_db_integration_added():
    # The executable Turn E files (code + compose + Dockerfile) must not add task
    # dispatch, a Mem0 client, or a DB client. The DEPLOYMENT doc is prose and is
    # allowed to *name* these as future/not-used, so it is checked separately.
    code_files = [
        _COMPOSE, _DOCKERFILE, _DOCKERIGNORE,
        _EXEC_DIR / "app" / "a2a" / "step5_worker_main.py",
        _EXEC_DIR / "app" / "a2a" / "step6_worker_main.py",
        _EXEC_DIR / "app" / "a2a" / "structure_worker_main.py",
    ]
    for path in code_files:
        low = path.read_text().lower()
        for banned in ("send_task_async", "workerexecutionrequest", "psycopg", "sqlalchemy", "mem0client"):
            assert banned not in low, f"{path.name} must not reference {banned}"
    # DEPLOYMENT doc explicitly records the future-only Mem0/Postgres boundary.
    doc = (_EXEC_DIR / "DEPLOYMENT_A2A.md").read_text()
    assert "not deployed or integrated in Turn E" in doc
    assert "http://mem0:8010" in doc
