"""Turn E — Docker Compose deployment skeleton tests.

Static + import-level checks over the real ``docker-compose.yml`` / ``Dockerfile``
/ ``.dockerignore`` and the worker entrypoint modules. These do NOT build images
or start containers (that real smoke is run separately); they lock the skeleton's
contract so it cannot silently drift:

- exactly the five business services, correct names;
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
from app.settings import Settings

_EXEC_DIR = Path(__file__).resolve().parents[2]
_COMPOSE = _EXEC_DIR / "docker-compose.yml"
_OPENAI_COMPOSE = _EXEC_DIR / "docker-compose.openai.yml"
_LIVE_TOOLS_COMPOSE = _EXEC_DIR / "docker-compose.live-tools.yml"
_DOCKERFILE = _EXEC_DIR / "Dockerfile"
_DOCKERIGNORE = _EXEC_DIR / ".dockerignore"

_BUSINESS_SERVICES = {
    "orchestrator",
    "step5-context-agent",
    "step6-developability-agent",
    "step7-9-structure-design-agent",
    "step13-14-patent-evidence-agent",
}
_WORKERS = {
    "step5-context-agent",
    "step6-developability-agent",
    "step7-9-structure-design-agent",
    "step13-14-patent-evidence-agent",
}
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
def test_exact_five_business_services(compose):
    assert set(compose["services"]) == _BUSINESS_SERVICES
    assert {
        name: service["container_name"]
        for name, service in compose["services"].items()
    } == {name: name for name in _BUSINESS_SERVICES}


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
    assert "LANGGRAPH_CHECKPOINT_DATABASE_URL" in _env(
        compose["services"]["orchestrator"]
    )
    for name in _WORKERS:
        assert "LANGGRAPH_CHECKPOINT_DATABASE_URL" not in _env(
            compose["services"][name]
        )


def test_no_pgvector_or_vector_db(compose):
    # Only the five business services exist; none uses a vector-db image.
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
    assert compose["services"]["step5-context-agent"]["expose"] == ["8005"]
    assert compose["services"]["step6-developability-agent"]["expose"] == ["8006"]
    assert compose["services"]["step7-9-structure-design-agent"]["expose"] == ["8009"]
    assert compose["services"]["step13-14-patent-evidence-agent"]["expose"] == ["8014"]


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
    assert env["STEP5_WORKER_URL"] == "http://step5-context-agent:8005"
    assert env["STEP6_WORKER_URL"] == "http://step6-developability-agent:8006"
    assert env["STRUCTURE_WORKER_URL"] == (
        "http://step7-9-structure-design-agent:8009"
    )
    assert env["PATENT_EVIDENCE_WORKER_URL"] == (
        "http://step13-14-patent-evidence-agent:8014"
    )
    assert "must be set explicitly" in env[
        "LANGGRAPH_CHECKPOINT_DATABASE_URL"
    ]
    assert "must be set explicitly" in env[
        "ORCHESTRATOR_WORKER_TIMEOUT_SECONDS"
    ]


def test_default_settings_use_production_container_dns_names(monkeypatch):
    for name in (
        "STEP5_WORKER_URL",
        "STEP6_WORKER_URL",
        "STRUCTURE_WORKER_URL",
        "PATENT_EVIDENCE_WORKER_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.step5_worker_url == "http://step5-context-agent:8005"
    assert settings.step6_worker_url == "http://step6-developability-agent:8006"
    assert settings.structure_worker_url == (
        "http://step7-9-structure-design-agent:8009"
    )
    assert settings.patent_evidence_worker_url == (
        "http://step13-14-patent-evidence-agent:8014"
    )


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


def test_application_source_does_not_invalidate_dependency_layer():
    text = _DOCKERFILE.read_text()
    dependency_install = text.index("'.[deployment,admet]'")
    dependency_guard = text.index("python /tmp/check_cpu_dependencies.py")
    app_copy = text.index("COPY app ./app")
    full_build_tools_copy = text.index("COPY build_tools ./build_tools")

    assert text.count("COPY app ./app") == 1
    assert text.count("COPY build_tools ./build_tools") == 1
    assert "COPY . ." not in text
    assert dependency_install < dependency_guard < app_copy
    assert dependency_guard < full_build_tools_copy
    assert text.rindex("python -m pip install") < app_copy
    assert "COPY pyproject.toml ./" in text[:dependency_install]
    assert "COPY app ./app" not in text[:dependency_install]
    assert "COPY build_tools ./build_tools" not in text[:dependency_install]
    assert "--mount=type=cache,id=synagentics-pip-cache" in text
    assert "target=/root/.cache/pip,sharing=locked" in text
    assert "PIP_NO_CACHE_DIR" not in text
    assert "--no-cache-dir" not in text


def test_openai_sdk_is_a_direct_pinned_project_dependency():
    pyproject = (_EXEC_DIR / "pyproject.toml").read_text()
    assert pyproject.count('"openai==2.45.0"') == 1


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
    import app.a2a.patent_evidence_worker_main as mpe
    import app.a2a.step5_worker_main as m5
    import app.a2a.step6_worker_main as m6
    import app.a2a.structure_worker_main as ms
    from app.a2a.patent_evidence_worker import run_patent_evidence_worker
    from app.a2a.step5_worker import run_step5_worker
    from app.a2a.step6_worker import run_step6_worker
    from app.a2a.structure_worker import run_structure_worker

    assert m5.run_step5_worker is run_step5_worker
    assert m6.run_step6_worker is run_step6_worker
    assert ms.run_structure_worker is run_structure_worker
    assert mpe.run_patent_evidence_worker is run_patent_evidence_worker
    for m in (m5, m6, ms, mpe):
        assert callable(m.main)


def test_worker_entrypoints_do_not_call_domain_agents_or_scan_ports():
    # Inspect real code (not docstrings): the entrypoints must not IMPORT domain
    # agents, dispatch tasks, or scan/switch ports. Docstrings may mention agent
    # names to explain what they do NOT do, so we check imports/behaviour, not
    # substrings of prose.
    import ast

    for fname in (
        "step5_worker_main.py",
        "step6_worker_main.py",
        "structure_worker_main.py",
        "patent_evidence_worker_main.py",
    ):
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
        assert any(
            entrypoint in src
            for entrypoint in (
                "run_step5_worker",
                "run_step6_worker",
                "run_structure_worker",
                "run_patent_evidence_worker",
            )
        )


def test_compose_advertised_url_matches_worker_bind_port(compose):
    for name, url_key in (
        ("step5-context-agent", "STEP5_WORKER_URL"),
        ("step6-developability-agent", "STEP6_WORKER_URL"),
        ("step7-9-structure-design-agent", "STRUCTURE_WORKER_URL"),
        ("step13-14-patent-evidence-agent", "PATENT_EVIDENCE_WORKER_URL"),
    ):
        env = _env(compose["services"][name])
        advertised = env[url_key]
        bind_port = int(env["WORKER_BIND_PORT"])
        assert effective_url_port(advertised) == bind_port
        assert env["WORKER_BIND_HOST"] == "0.0.0.0"


def test_worker_commands_use_module_entrypoints_not_inline_python(compose):
    expected = {
        "step5-context-agent": ["python", "-m", "app.a2a.step5_worker_main"],
        "step6-developability-agent": [
            "python",
            "-m",
            "app.a2a.step6_worker_main",
        ],
        "step7-9-structure-design-agent": [
            "python",
            "-m",
            "app.a2a.structure_worker_main",
        ],
        "step13-14-patent-evidence-agent": [
            "python",
            "-m",
            "app.a2a.patent_evidence_worker_main",
        ],
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
        "step5-context-agent": (
            "http://127.0.0.1:8005/health",
            "step_05_candidate_context_agent",
            "step_05_candidate_context",
        ),
        "step6-developability-agent": (
            "http://127.0.0.1:8006/health",
            "step_06_developability_agent",
            "step_06_developability",
        ),
        "step7-9-structure-design-agent": (
            "http://127.0.0.1:8009/health",
            "structure_and_design_agent",
            "structure_design_workflow",
        ),
        "step13-14-patent-evidence-agent": (
            "http://127.0.0.1:8014/health",
            "patent_evidence_agent",
            "patent_evidence_workflow",
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
    env.pop("LANGGRAPH_CHECKPOINT_DATABASE_URL", None)
    env.pop("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_MODEL", None)
    if explicit_modes:
        env["LLM_PROVIDER"] = "mock"
        env["MCP_LIVE_TOOLS"] = "false"
        env["ORCHESTRATOR_HOST_PORT"] = "18080"
        env["LANGGRAPH_CHECKPOINT_DATABASE_URL"] = (
            "postgresql://checkpoint_user:test-only@checkpoint-db/adc_checkpoint"
        )
        env["ORCHESTRATOR_WORKER_TIMEOUT_SECONDS"] = "60"
    return env


def _openai_compose_env() -> dict[str, str]:
    env = _compose_env(explicit_modes=True)
    env["LLM_PROVIDER"] = "openai"
    env["OPENAI_MODEL"] = "gpt-5.5"
    env["OPENAI_API_KEY"] = "sk-fake-compose-sentinel"
    return env


def _live_tools_compose_env() -> dict[str, str]:
    env = _openai_compose_env()
    env["MCP_LIVE_TOOLS"] = "true"
    env["NVIDIA_API_KEY"] = "fake-nvidia-compose-sentinel"
    env["ESM_API_KEY"] = "fake-esm-compose-sentinel"
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
    assert set(resolved["services"]) == _BUSINESS_SERVICES
    assert {
        name: service["container_name"]
        for name, service in resolved["services"].items()
    } == {name: name for name in _BUSINESS_SERVICES}
    orchestrator_env = resolved["services"]["orchestrator"]["environment"]
    assert orchestrator_env["STEP5_WORKER_URL"] == (
        "http://step5-context-agent:8005"
    )
    assert orchestrator_env["STEP6_WORKER_URL"] == (
        "http://step6-developability-agent:8006"
    )
    assert orchestrator_env["STRUCTURE_WORKER_URL"] == (
        "http://step7-9-structure-design-agent:8009"
    )
    assert orchestrator_env["PATENT_EVIDENCE_WORKER_URL"] == (
        "http://step13-14-patent-evidence-agent:8014"
    )
    assert "ports" in resolved["services"]["orchestrator"]
    for name in _WORKERS:
        assert "ports" not in resolved["services"][name]
        assert resolved["services"][name]["expose"]
    assert resolved["volumes"]["adc_local_store"]["name"] == (
        "synagentics-a2a-v1_adc_local_store"
    )
    assert resolved["networks"]["adc_a2a_net"]["name"] == (
        "synagentics-a2a-v1_adc_a2a_net"
    )
    assert "dify" not in result.stdout.lower()
    assert "ragflow" not in result.stdout.lower()


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_docker_compose_config_services_are_exact():
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "synagentics-a2a-v1",
            "-f",
            str(_COMPOSE),
            "config",
            "--services",
        ],
        cwd=str(_EXEC_DIR),
        env=_compose_env(explicit_modes=True),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 5
    assert set(lines) == _BUSINESS_SERVICES


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_openai_compose_override_uses_one_secret_file_for_all_services():
    override = yaml.safe_load(_OPENAI_COMPOSE.read_text())
    assert "env_file" not in _COMPOSE.read_text()
    assert "env_file" not in _OPENAI_COMPOSE.read_text()
    assert set(override["services"]) == _BUSINESS_SERVICES
    assert override["secrets"] == {
        "openai_api_key": {"environment": "OPENAI_API_KEY"}
    }
    for service in override["services"].values():
        assert service["environment"] == {
            "OPENAI_API_KEY_FILE": "/run/secrets/openai_api_key",
            "OPENAI_MODEL": "${OPENAI_MODEL:?OPENAI_MODEL must be set explicitly}",
        }
        assert service["secrets"] == ["openai_api_key"]

    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "synagentics-a2a-v1",
            "-f",
            str(_COMPOSE),
            "-f",
            str(_OPENAI_COMPOSE),
            "config",
        ],
        cwd=str(_EXEC_DIR),
        env=_openai_compose_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "sk-fake-compose-sentinel" not in result.stdout
    assert "sk-fake-compose-sentinel" not in result.stderr
    resolved = yaml.safe_load(result.stdout)
    assert set(resolved["services"]) == _BUSINESS_SERVICES
    for name, service in resolved["services"].items():
        assert service["container_name"] == name
        assert service["environment"]["LLM_PROVIDER"] == "openai"
        assert service["environment"]["OPENAI_API_KEY_FILE"] == (
            "/run/secrets/openai_api_key"
        )
        assert service["environment"]["OPENAI_MODEL"] == "gpt-5.5"
        assert "OPENAI_API_KEY" not in service["environment"]
        assert service["secrets"] == [
            {
                "source": "openai_api_key",
                "target": "/run/secrets/openai_api_key",
            }
        ]
    assert "ports" in resolved["services"]["orchestrator"]
    for name in _WORKERS:
        assert "ports" not in resolved["services"][name]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_openai_compose_override_requires_explicit_model():
    env = _openai_compose_env()
    env.pop("OPENAI_MODEL")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "synagentics-a2a-v1",
            "-f",
            str(_COMPOSE),
            "-f",
            str(_OPENAI_COMPOSE),
            "config",
        ],
        cwd=str(_EXEC_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "OPENAI_MODEL must be set explicitly" in result.stderr
    assert "sk-fake-compose-sentinel" not in result.stderr


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_live_tools_overlay_is_structure_only_and_secret_file_based():
    override = yaml.safe_load(_LIVE_TOOLS_COMPOSE.read_text())
    assert "env_file" not in _LIVE_TOOLS_COMPOSE.read_text()
    assert set(override["services"]) == {"step7-9-structure-design-agent"}
    structure = override["services"]["step7-9-structure-design-agent"]
    assert structure["environment"] == {
        "NVIDIA_API_KEY_FILE": "/run/secrets/nvidia_api_key",
        "ESM_API_KEY_FILE": "/run/secrets/esm_api_key",
    }
    assert set(structure["secrets"]) == {"nvidia_api_key", "esm_api_key"}
    assert override["secrets"] == {
        "nvidia_api_key": {"environment": "NVIDIA_API_KEY"},
        "esm_api_key": {"environment": "ESM_API_KEY"},
    }

    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "synagentics-a2a-v1",
            "-f",
            str(_COMPOSE),
            "-f",
            str(_OPENAI_COMPOSE),
            "-f",
            str(_LIVE_TOOLS_COMPOSE),
            "config",
        ],
        cwd=str(_EXEC_DIR),
        env=_live_tools_compose_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for sentinel in (
        "fake-nvidia-compose-sentinel",
        "fake-esm-compose-sentinel",
        "sk-fake-compose-sentinel",
    ):
        assert sentinel not in result.stdout
        assert sentinel not in result.stderr

    resolved = yaml.safe_load(result.stdout)
    assert set(resolved["services"]) == _BUSINESS_SERVICES
    for name, service in resolved["services"].items():
        environment = service["environment"]
        assert "NVIDIA_API_KEY" not in environment
        assert "ESM_API_KEY" not in environment
        secret_sources = {
            item["source"] if isinstance(item, dict) else item
            for item in service.get("secrets", [])
        }
        if name == "step7-9-structure-design-agent":
            assert environment["NVIDIA_API_KEY_FILE"] == (
                "/run/secrets/nvidia_api_key"
            )
            assert environment["ESM_API_KEY_FILE"] == "/run/secrets/esm_api_key"
            assert {"nvidia_api_key", "esm_api_key"} <= secret_sources
        else:
            assert "NVIDIA_API_KEY_FILE" not in environment
            assert "ESM_API_KEY_FILE" not in environment
            assert "nvidia_api_key" not in secret_sources
            assert "esm_api_key" not in secret_sources


# ── 19-20. no Mem0 or Compose-managed database service introduced ──────
def test_no_memory_or_compose_database_service_added():
    code_files = [
        _COMPOSE, _DOCKERFILE, _DOCKERIGNORE,
        _EXEC_DIR / "app" / "a2a" / "step5_worker_main.py",
        _EXEC_DIR / "app" / "a2a" / "step6_worker_main.py",
        _EXEC_DIR / "app" / "a2a" / "structure_worker_main.py",
        _EXEC_DIR / "app" / "a2a" / "patent_evidence_worker_main.py",
    ]
    for path in code_files:
        low = path.read_text().lower()
        for banned in ("sqlalchemy", "mem0client"):
            assert banned not in low, f"{path.name} must not reference {banned}"
    doc = (_EXEC_DIR / "DEPLOYMENT_A2A.md").read_text()
    assert "external LangGraph Postgres checkpointer is integrated" in doc
    assert "creates **no** database service" in doc
    assert "http://mem0:8010" in doc
