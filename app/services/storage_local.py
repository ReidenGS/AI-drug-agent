"""Local filesystem storage adapter for dev mode."""

from __future__ import annotations

import json
from pathlib import Path


class LocalStorage:
    def __init__(self, root: str, prefix: str) -> None:
        self.root = Path(root).resolve()
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_json(self, key: str, payload: dict) -> str:
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
        return self.write_bytes(key, text.encode("utf-8"))

    def read_json(self, key: str) -> dict:
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_prefix(self, key_prefix: str) -> list[str]:
        base = self._path(key_prefix)
        if not base.exists():
            return []
        return [str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()]

    def delete(self, key: str) -> bool:
        path = self._path(key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def run_key(self, run_id: str, *parts: str) -> str:
        return "/".join([self.prefix, "runs", run_id, *parts])
