"""Local filesystem storage adapter for dev mode."""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from collections.abc import Callable
from fcntl import LOCK_EX, LOCK_UN, flock
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
        temporary_dir = self.root / ".adc_tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        temporary = temporary_dir / uuid.uuid4().hex
        try:
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return str(path)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_json(self, key: str, payload: dict) -> str:
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
        return self.write_bytes(key, text.encode("utf-8"))

    def read_json(self, key: str) -> dict:
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def atomic_update_json(
        self,
        key: str,
        update: Callable[[dict], dict],
    ) -> dict:
        """Serialize one key's full read-modify-write across local processes.

        POSIX ``flock`` is shared by containers mounting the same filesystem.
        The lock path is derived from the run-scoped JSON key, so unrelated
        runs and documents do not share a global lock.
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_dir = self.root / ".adc_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        lock_path = lock_dir / f"{lock_name}.lock"
        with lock_path.open("a+b") as lock_stream:
            flock(lock_stream.fileno(), LOCK_EX)
            try:
                current = json.loads(path.read_bytes().decode("utf-8"))
                updated = update(current)
                if not isinstance(updated, dict):
                    raise TypeError("atomic_json_update_must_return_dict")
                self.write_json(key, updated)
                return updated
            finally:
                flock(lock_stream.fileno(), LOCK_UN)

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
