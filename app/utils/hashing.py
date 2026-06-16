"""SHA-256 hashing helpers."""

from __future__ import annotations

import hashlib
from typing import Any
import json


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))
