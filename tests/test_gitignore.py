from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_exists_and_covers_generated_artifacts():
    path = REPO_ROOT / ".gitignore"
    assert path.exists(), ".gitignore is missing"
    contents = path.read_text()
    for entry in (".DS_Store", "__pycache__/", ".pytest_cache/", ".localstore/"):
        assert entry in contents, f".gitignore missing entry: {entry}"
