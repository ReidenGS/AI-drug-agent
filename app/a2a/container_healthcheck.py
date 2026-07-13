"""Compact container health probe for one A2A worker.

This module performs one finite-timeout HTTP GET against ``/health`` and
validates the exact worker identity/capability set. It never imports domain
agents, constructs A2A tasks, invokes LLM/MCP, scans ports, or rewrites URLs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HealthcheckResult:
    ok: bool
    code: str


def check_worker_health(
    *,
    url: str,
    expected_agent_id: str,
    expected_capability_ids: Sequence[str],
    timeout: float,
) -> HealthcheckResult:
    """Return a compact result without exposing response or endpoint details."""
    expected = {str(value) for value in expected_capability_ids}
    if not url or not expected_agent_id or not expected:
        return HealthcheckResult(False, "invalid_configuration")
    if not math.isfinite(timeout) or timeout <= 0:
        return HealthcheckResult(False, "invalid_timeout")

    try:
        response = urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
    except (urllib.error.URLError, TimeoutError, OSError):
        return HealthcheckResult(False, "connection_failed")
    except Exception:  # noqa: BLE001 - compact failure, never raw exception
        return HealthcheckResult(False, "request_failed")

    try:
        status_code = getattr(response, "status", None)
        if status_code != 200:
            return HealthcheckResult(False, "http_status_mismatch")
        try:
            body = json.load(response)
        except Exception:  # noqa: BLE001 - compact malformed-response code
            return HealthcheckResult(False, "response_not_json")
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    if not isinstance(body, dict):
        return HealthcheckResult(False, "response_not_object")
    if body.get("status") != "ok":
        return HealthcheckResult(False, "status_not_ok")
    if body.get("agent_id") != expected_agent_id:
        return HealthcheckResult(False, "agent_id_mismatch")
    capabilities = body.get("capabilities")
    if not isinstance(capabilities, list):
        return HealthcheckResult(False, "capabilities_not_list")
    if {str(value) for value in capabilities} != expected:
        return HealthcheckResult(False, "capabilities_mismatch")
    return HealthcheckResult(True, "ok")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--url", default=os.environ.get("HEALTHCHECK_URL", ""))
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("HEALTHCHECK_EXPECTED_AGENT_ID", ""),
    )
    parser.add_argument(
        "--capability",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("HEALTHCHECK_TIMEOUT_SECONDS", "3")),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capabilities = args.capability
    if capabilities is None:
        capabilities = [
            value.strip()
            for value in os.environ.get(
                "HEALTHCHECK_EXPECTED_CAPABILITIES",
                "",
            ).split(",")
            if value.strip()
        ]
    result = check_worker_health(
        url=args.url,
        expected_agent_id=args.agent_id,
        expected_capability_ids=capabilities,
        timeout=args.timeout,
    )
    if not result.ok:
        print(f"container_healthcheck_failed:{result.code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - container CLI boundary
    raise SystemExit(main())


__all__ = ["HealthcheckResult", "check_worker_health", "main"]
