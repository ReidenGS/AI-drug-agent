from __future__ import annotations

from build_tools.check_cpu_dependencies import forbidden_gpu_packages


def test_nvidia_cublas_fails_guard() -> None:
    assert forbidden_gpu_packages(["torch", "nvidia-cublas"]) == ["nvidia-cublas"]


def test_triton_fails_guard() -> None:
    assert forbidden_gpu_packages(["torch", "triton"]) == ["triton"]


def test_cpu_torch_passes_guard() -> None:
    assert forbidden_gpu_packages(["torch"]) == []
