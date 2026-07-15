"""Compact secret-file resolution helpers for optional live-tool credentials."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


OptionalSecretName = Literal["nvidia_api_key", "esm_api_key"]


def resolve_optional_secret(
    *,
    direct_value: str | None,
    file_path: str | None,
    secret_name: OptionalSecretName,
) -> str:
    """Resolve an optional credential without exposing its value or source path.

    A non-empty direct value wins. An explicitly configured file is otherwise
    required to be readable and non-empty. When neither source is configured,
    the empty string preserves the live tool's missing-credential semantics.
    """

    direct = (direct_value or "").strip()
    if direct:
        return direct

    configured_path = (file_path or "").strip()
    if not configured_path:
        return ""
    try:
        resolved = Path(configured_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise ValueError(f"{secret_name}_file_unreadable") from None
    if not resolved:
        raise ValueError(f"{secret_name}_file_empty")
    return resolved
