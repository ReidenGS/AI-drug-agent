"""Static isolation contract for the ephemeral Postgres checkpoint profile."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
COMPOSE = ROOT / "docker-compose.checkpoint-test.yml"


def test_checkpoint_profile_has_no_host_ports_and_is_test_only():
    parsed = yaml.safe_load(COMPOSE.read_text())
    services = parsed["services"]
    assert set(services) == {"checkpoint-postgres", "checkpoint-test-runner"}
    for service in services.values():
        assert service["profiles"] == ["checkpoint-test"]
        assert "ports" not in service
        assert service["networks"] == ["checkpoint_test_net"]
    assert services["checkpoint-postgres"]["image"] == "postgres:16-alpine"
    assert services["checkpoint-test-runner"]["build"] == "."
    assert services["checkpoint-test-runner"]["image"] == (
        "synagentics-adc-a2a:checkpoint-test"
    )
    assert "tmpfs" in services["checkpoint-postgres"]


def test_checkpoint_test_credentials_are_not_production_compose_defaults():
    test_text = COMPOSE.read_text()
    production_text = (ROOT / "docker-compose.yml").read_text()
    assert "checkpoint_test_only" in test_text
    assert "checkpoint_test_only" not in production_text
    assert (
        "${LANGGRAPH_CHECKPOINT_DATABASE_URL:?"
        "LANGGRAPH_CHECKPOINT_DATABASE_URL must be set explicitly}"
    ) in production_text
