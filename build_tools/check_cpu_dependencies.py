"""Fail a container build when resolved packages include GPU runtimes."""

from __future__ import annotations

import importlib.metadata
import re


_FORBIDDEN_NAME = re.compile(
    r"^(?:nvidia-|cuda)|(?:cublas|cudnn|nccl|nvshmem)|^triton$",
    re.IGNORECASE,
)


def forbidden_gpu_packages(package_names: list[str]) -> list[str]:
    """Return normalized forbidden package names without exposing versions."""
    return sorted({name.lower() for name in package_names if _FORBIDDEN_NAME.search(name)})


def installed_forbidden_gpu_packages() -> list[str]:
    names = [distribution.metadata["Name"] for distribution in importlib.metadata.distributions()]
    return forbidden_gpu_packages([name for name in names if name])


def main() -> int:
    forbidden = installed_forbidden_gpu_packages()
    if forbidden:
        print("forbidden_gpu_dependency_detected", flush=True)
        return 1
    print("cpu_dependency_guard_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
