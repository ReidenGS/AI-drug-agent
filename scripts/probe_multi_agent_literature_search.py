"""Probe ToolUniverse's MultiAgentLiteratureSearch live path.

Uses the local environment's GEMINI_API_KEY (the wrapper's adapter
hydrates it from `app.settings` into `os.environ` so TU's AgenticTool
sub-tools can see it). NEVER prints the key, full payload, prompts, or
raw search results — only:
  - whether the key is available (bool, NOT the value)
  - status / runtime / top-level envelope keys
  - nested `payload.results.{total_papers, search_plans, user_intent, success}`
    counts and lengths

Run:
  /Users/jackiewen/Desktop/desk/实习工作/国外ai医药/测试文件/.venv/bin/python \
      scripts/probe_multi_agent_literature_search.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hydrate the env BEFORE we even import the wrapper so a startup probe of
# key availability is honest about what TU will see at call time.
from app.mcp import tooluniverse_adapter  # noqa: E402

tooluniverse_adapter._hydrate_env_from_settings()

from app.mcp.tools.evidence import MultiAgentLiteratureSearch  # noqa: E402


QUERY = "HER2 antibody drug conjugate"
MAX_ITERATIONS = 1
QUALITY_THRESHOLD = 0.5


def _payload_size_hint(payload: object) -> str:
    if payload is None:
        return "None"
    if isinstance(payload, dict):
        return f"dict({len(payload)} keys)"
    if isinstance(payload, list):
        return f"list({len(payload)} items)"
    return type(payload).__name__


def main() -> int:
    # Truth-only signal about key availability — bool, never the value.
    print(f"gemini_key_available = {bool(os.environ.get('GEMINI_API_KEY'))}")

    started = time.monotonic()
    envelope = MultiAgentLiteratureSearch(
        query=QUERY,
        max_iterations=MAX_ITERATIONS,
        quality_threshold=QUALITY_THRESHOLD,
        _live=True,
    )
    elapsed = time.monotonic() - started

    status = envelope.get("status", "?")
    payload = envelope.get("payload")
    top_level_keys = sorted(envelope.keys())

    print(f"status              = {status}")
    print(f"runtime_seconds     = {elapsed:.2f}")
    print(f"top_level_keys      = {top_level_keys}")
    print(f"payload_type        = {_payload_size_hint(payload)}")

    if status == "upstream_error":
        msg = (envelope.get("error_message") or "")[:200]
        print(f"error_message       = {msg}")
        return 1

    # Nested-result digging: the TU ComposeTool wraps real fields under
    # payload.results.{...}. Stay defensive — TU may rearrange across
    # versions.
    results = None
    if isinstance(payload, dict):
        candidate = payload.get("results")
        if isinstance(candidate, dict):
            results = candidate

    if not isinstance(results, dict):
        print("payload_results     = <missing or non-dict>")
        return 0

    success = results.get("success")
    total_papers = results.get("total_papers")
    search_plans = results.get("search_plans")
    user_intent = results.get("user_intent")

    print(f"success             = {success}")
    print(f"total_papers        = {total_papers}")
    if isinstance(search_plans, list):
        print(f"search_plans_count  = {len(search_plans)}")
    else:
        print(f"search_plans_count  = <missing>")
    if isinstance(user_intent, str):
        print(f"user_intent_length  = {len(user_intent)}")
    elif isinstance(user_intent, dict):
        print(f"user_intent_length  = dict({len(user_intent)} keys)")
    else:
        print(f"user_intent_length  = <missing>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
